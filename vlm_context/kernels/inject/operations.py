"""Mapping-specific task vectors with disjoint extraction and fixed development selection."""
from __future__ import annotations
import math
import random
import time
from ..gate.common import digest, episode, publish, read, require, score, sha
from .readers import reference, verify_ref
from .hooks import HiddenEdit, capture
from .readers import reader
from .cache import SupportCache
from ..gate.policy import eligibility, mean_difference, mean_state

READERS=('connector_episode_remap','uncapped_all_episode_remap','frozen')
LAYERS=(6,13,20)
SCALES=(.25,.5,1.0)
METHODS=('explicit','explicit_cached_KV','zero_shot','difference','historical_raw_mean','zero_vector','shuffled_vector')


def blocks(session):
    return session.blocks


def encoded(session,ep,loader):
    from vlm_context.qwen import render_messages
    messages,images=render_messages(ep,loader)
    return session.evaluator.encode(messages,images,ep['label_space'])


def extract(session,block,manifest,loader,reader_identity):
    pairs={layer:[] for layer in LAYERS};supported={layer:[] for layer in LAYERS};cost=0.;tokens=0
    for index,query in enumerate(block['extraction_probe_ids']):
        raw={}
        for k,key in ((0,'zero_shot'),(8,'supported')):
            session.evaluator.synchronize();start=time.monotonic()
            ep=episode(block,manifest,query,k,'episode_remap','controller_development')
            inputs,metadata=encoded(session,ep,loader);position=len(metadata['input_token_ids'])-1
            states,_=capture(session.evaluator,inputs,LAYERS,[position])
            session.evaluator.synchronize();elapsed=time.monotonic()-start
            cost+=elapsed;tokens+=metadata['input_token_count']
            raw[key]={'episode':ep,'encoding':metadata,'position':position,'states':states,'seconds':elapsed}
        require(raw['zero_shot']['encoding']['candidate_tokens']==raw['supported']['encoding']['candidate_tokens'],
                'extraction candidate mapping changed')
        require(raw['zero_shot']['encoding']['input_token_ids'][-4:]==raw['supported']['encoding']['input_token_ids'][-4:],
                'semantic last-prefill boundary differs')
        for layer in LAYERS:
            a=raw['supported']['states'][str(layer)][str(raw['supported']['position'])]
            b=raw['zero_shot']['states'][str(layer)][str(raw['zero_shot']['position'])]
            pairs[layer].append((a,b));supported[layer].append(a)
        session.save(f'extract_{block["task"]}_s{block["seed"]}_{index:02d}',{'block_id':block['block_id'],
                     'mapping_sha256':block['mapping_sha256'],'reader_identity':reader_identity,'raw':raw,
                     'reference_position':'last assistant-generation prefill token; verified identical4-token suffix'})
    vectors={str(layer):{'difference':mean_difference(pairs[layer]),'historical_raw_mean':mean_state(supported[layer])} for layer in LAYERS}
    artifact={'block_id':block['block_id'],'mapping_sha256':block['mapping_sha256'],
              'support_ids':block['nested_support_ids']['8'],'probe_ids':block['extraction_probe_ids'],
              'vectors':vectors,'extraction_forwards':16,'extraction_seconds':cost,'extraction_prefix_tokens':tokens,
              'reader_identity':reader_identity,'aggregation':'float64 math.fsum across8 disjoint paired probes'}
    publish(session.output/'vectors'/f'{block["task"]}_s{block["seed"]}.json',artifact)
    return artifact


def injected(session,ep,loader,layer,vector,scale,native=False):
    session.evaluator.synchronize();started=time.monotonic()
    inputs,metadata=encoded(session,ep,loader)
    n=len(metadata['input_token_ids']);values=[v*scale for v in vector]
    with HiddenEdit(session.evaluator,layer,n,n-1,values,'add') as hook:
        result=score(session,ep,loader,native=native,prepared=(inputs,metadata))
    session.evaluator.synchronize();result['seconds']=time.monotonic()-started
    require(hook.calls>0,'Inject hook did not execute')
    return result


def smoke(session):
    session.start_model();block=blocks(session)[0];manifest,loader=session.loader(block['task'])
    # Verify the accepted reader reload and the frozen base independently.
    for name in ('frozen','connector_episode_remap'):
        with reader(session,name,0) as identity:
            query=block['development_query_ids'][0]
            zero=episode(block,manifest,query,0,'episode_remap','development')
            support=episode(block,manifest,query,8,'episode_remap','development')
            a=score(session,zero,loader,native=True);b=score(session,support,loader,native=True)
            session.save(f'{name}_zero',{'result':a,'reader_identity':identity})
            session.save(f'{name}_support',{'result':b,'reader_identity':identity})
            inputs,meta=encoded(session,support,loader);cache=SupportCache(session.evaluator,inputs,meta,8)
            cached=cache.score(inputs,meta,support)
            require(max(abs(x-y) for u,v in zip(b['per_token_label_log_probabilities'],cached['per_token_label_log_probabilities']) for x,y in zip(u,v))<=.05,
                    'cached explicit support differs from ordinary native scoring')
            require(cached['scores']['sum_prediction']==b['scores']['sum_prediction'] and
                    cached['greedy_token_ids']==b['greedy_token_ids'],'cached explicit argmax/greedy differs')
            session.save(f'{name}_cached_support',{'result':cached,'native_reference':f'{name}_support','reader_identity':identity})
            del cache
            for layer in LAYERS:
                z=injected(session,zero,loader,layer,[0.]*session.evaluator.model.config.hidden_size,1,native=True)
                require(z['per_token_label_log_probabilities']==a['per_token_label_log_probabilities'] and
                        z['greedy_token_ids']==a['greedy_token_ids'],'zero vector changed native computation')
                session.save(f'{name}_L{layer}_zero_vector',{'result':z,'reader_identity':identity})
                # Nonzero native hook trace uses only reserved extraction query.
                xi,xm=encoded(session,support,loader);zi,zm=encoded(session,zero,loader)
                xs,_=capture(session.evaluator,xi,[layer],[len(xm['input_token_ids'])-1])
                zs,_=capture(session.evaluator,zi,[layer],[len(zm['input_token_ids'])-1])
                vector=[x-y for x,y in zip(xs[str(layer)][str(len(xm['input_token_ids'])-1)],zs[str(layer)][str(len(zm['input_token_ids'])-1)])]
                v=injected(session,zero,loader,layer,vector,.25,native=True)
                session.save(f'{name}_L{layer}_difference_native',{'result':v,'development_only_vector':vector,
                             'reader_identity':identity,'used_for_confirmation':False})
    return session.finish({'development_only':True,'cache_and_zero_vector_noop_passed':True})


def eligibility_run(session):
    require(session.args.method in READERS,'unregistered prerequisite reader')
    session.start_model();rows=[]
    for block in blocks(session):
        manifest,loader=session.loader(block['task'])
        with reader(session,session.args.method,block['seed']) as identity:
            outcomes={0:[],8:[]}
            costs={0:0.,8:0.}
            for index,query in enumerate(block['development_query_ids']):
                for k in (0,8):
                    ep=episode(block,manifest,query,k,'episode_remap','development')
                    result=score(session,ep,loader)
                    session.save(f'{block["task"]}_s{block["seed"]}_K{k}_{index:03d}',{'result':result,'reader_identity':identity})
                    outcomes[k].append(result['scores']['sum_prediction']==ep['query']['class_id']);costs[k]+=result['seconds']
            rows.append({'block_id':block['block_id'],'task':block['task'],'seed':block['seed'],
                         **eligibility(outcomes[0],outcomes[8]),'reader_identity':identity,'selection_seconds':sum(costs.values())})
    publish(session.output/'eligibility.json',{'kind':'inject-reader-eligibility-v1','bundle_sha256':session.bundle_sha,
            'runtime_sha256':session.runtime_sha,'reader':session.args.method,'blocks':rows,'final_queries_used':False,
            'selection_policy':'choose reader by task using mean across3 seeds; threshold .60 accuracy and .10 gain; tie fewer parameters then declared order'})
    return session.finish({'eligibility_sha256':sha(session.output/'eligibility.json')})


def choose_readers(project,reviews,output):
    artifacts=[];bindings=[]
    for path in reviews:
        artifact=read(path)
        require(artifact.get('kind')=='inject-reader-eligibility-v1' and artifact['final_queries_used'] is False,
                'Reader eligibility must use development queries only')
        artifacts.append(artifact);bindings.append(reference(project,path))
    require(len(artifacts)==len(READERS) and {r['reader'] for r in artifacts}==set(READERS),'all declared readers must be retained')
    require(len({(a['bundle_sha256'],a['runtime_sha256']) for a in artifacts})==1,'prerequisite model/source mismatch')
    selected={};candidates={}
    for task in ('shapes4','cifar4'):
        rows=[]
        for artifact in artifacts:
            cohort=[r for r in artifact['blocks'] if r['task']==task]
            require(len(cohort)==3 and {r['seed'] for r in cohort}=={0,1,2},'incomplete prerequisite fitting seed grid')
            supported=sum(r['supported_correct'] for r in cohort);zero_count=sum(r['zero_correct'] for r in cohort)
            acc=supported/120;zero=zero_count/120
            rows.append({'reader':artifact['reader'],'supported_accuracy':acc,'zero_accuracy':zero,
                         'supported_correct':supported,'zero_correct':zero_count,'queries':120,
                         'eligible':supported>=72 and supported-zero_count>=12,
                         'trainable_parameters':cohort[0]['reader_identity']['trainable_parameters'],
                         'selection_seconds':sum(r['selection_seconds'] for r in cohort)})
        eligible=[r for r in rows if r['eligible']]
        selected[task]=min(eligible,key=lambda r:(-r['supported_accuracy'],r['trainable_parameters'],READERS.index(r['reader'])))['reader'] if eligible else None
        candidates[task]=rows
    result={'kind':'inject-reader-selection-v1','bundle_sha256':artifacts[0]['bundle_sha256'],
            'runtime_sha256':artifacts[0]['runtime_sha256'],'reader_by_task':selected,'candidates':candidates,
            'prerequisite_evidence':bindings,'final_queries_used':False,
            'negative_outcome_is_valid':True}
    if output is not None: publish(output,result)
    return result


def reviewed_readers(session):
    artifact=read(session.args.checkpoint)
    require(artifact.get('kind')=='inject-reader-selection-v1' and artifact.get('bundle_sha256')==session.bundle_sha
            and artifact.get('runtime_sha256')==session.runtime_sha, 'Reader selection model/data mismatch')
    require(artifact['final_queries_used'] is False and set(artifact['reader_by_task'])=={'shapes4','cifar4'},
            'Invalid task reader selection')
    for ref in artifact['prerequisite_evidence']: verify_ref(session.project,ref)
    # Recompute development-only selection; never select from confirmation outcomes.
    checked=choose_readers(session.project,[ref['path'] for ref in artifact['prerequisite_evidence']],None)
    require(checked==artifact,'Reader selection differs from the frozen rule')
    return artifact


def development(session):
    selection=reviewed_readers(session);rows=[]
    if any(selection['reader_by_task'].values()):session.start_model()
    for block in blocks(session):
        name=selection['reader_by_task'][block['task']]
        if name is None:continue
        manifest,loader=session.loader(block['task'])
        with reader(session,name,block['seed']) as identity:
            artifact=extract(session,block,manifest,loader,identity)
            for layer in LAYERS:
                for scale in SCALES:
                    outcomes=[]
                    for index,query in enumerate(block['development_query_ids']):
                        ep=episode(block,manifest,query,0,'episode_remap','development')
                        result=injected(session,ep,loader,layer,artifact['vectors'][str(layer)]['difference'],scale)
                        session.save(f'dev_{block["task"]}_s{block["seed"]}_L{layer}_a{scale}_{index:03d}',
                                     {'result':result,'reader_identity':identity,'layer':layer,'scale':scale})
                        outcomes.append(result['scores']['sum_prediction']==ep['query']['class_id'])
                    rows.append({'task':block['task'],'seed':block['seed'],'layer':layer,'scale':scale,'correct_count':sum(outcomes),'accuracy':sum(outcomes)/40})
    choices={}
    for task,name in selection['reader_by_task'].items():
        if name is None:choices[task]=None;continue
        candidates=[{'layer':layer,'scale':scale,'accuracy':sum(r['correct_count'] for r in rows if r['task']==task and r['layer']==layer and r['scale']==scale)/120} for layer in LAYERS for scale in SCALES]
        choices[task]=min(candidates,key=lambda r:(-r['accuracy'],r['layer'],r['scale']))
    artifact={'kind':'inject-development-selection-v1','bundle_sha256':session.bundle_sha,'runtime_sha256':session.runtime_sha,
              'reader_selection':reference(session.project,session.args.checkpoint),'choices_by_task':choices,
              'development_rows':rows,'layer_grid':list(LAYERS),'scale_grid':list(SCALES),
              'final_queries_used':False,
              'status':'no_eligible_prerequisite_reader' if not any(choices.values()) else 'development_selection_complete'}
    publish(session.output/'selection.json',artifact)
    return session.finish({'selection_sha256':sha(session.output/'selection.json'),'bounded_negative_outcome':not any(choices.values())})


def confirmation(session):
    readers=reviewed_readers(session);selected=read(session.args.selection)
    require(selected.get('kind')=='inject-development-selection-v1' and
            selected.get('bundle_sha256')==session.bundle_sha and selected.get('runtime_sha256')==session.runtime_sha,
            'Vector selection model/data mismatch')
    require(selected['final_queries_used'] is False and
            selected['reader_selection']==reference(session.project,session.args.checkpoint), 'Reader/vector selection mismatch')
    for task,name in readers['reader_by_task'].items():
        if name is None:
            require(selected['choices_by_task'][task] is None, 'Ineligible reader selected')
            continue
        rows_for_task=[r for r in selected['development_rows'] if r['task']==task]
        require(len(rows_for_task)==27 and {(r['seed'],r['layer'],r['scale']) for r in rows_for_task}==
                {(s,l,a) for s in (0,1,2) for l in LAYERS for a in SCALES}, 'Incomplete development grid')
        candidates=[{'layer':l,'scale':a,'accuracy':sum(r['correct_count'] for r in rows_for_task
                    if r['layer']==l and r['scale']==a)/120} for l in LAYERS for a in SCALES]
        expected=min(candidates,key=lambda r:(-r['accuracy'],r['layer'],r['scale']))
        require(selected['choices_by_task'][task]==expected,'Vector selection differs from frozen tie rule')
    rows=[]
    if any(selected['choices_by_task'].values()):session.start_model()
    for block in blocks(session):
        choice=selected['choices_by_task'][block['task']]
        if choice is None:continue
        layer,scale=choice['layer'],choice['scale'];name=readers['reader_by_task'][block['task']]
        require(layer in LAYERS and scale in SCALES and name in READERS,'nonprespecified vector configuration')
        manifest,loader=session.loader(block['task'])
        with reader(session,name,block['seed']) as identity:
            vectors=extract(session,block,manifest,loader,identity)
            difference=vectors['vectors'][str(layer)]['difference'];historical=vectors['vectors'][str(layer)]['historical_raw_mean']
            shuffled=list(difference);random.Random(int(digest(['inject-coordinate-shuffle',block['block_id']]),16)).shuffle(shuffled)
            cache=None;outcomes={method:[] for method in METHODS}
            for index,query in enumerate(block['final_query_ids']):
                zero=episode(block,manifest,query,0,'episode_remap','confirmation_query')
                explicit=episode(block,manifest,query,8,'episode_remap','confirmation_query')
                current={'zero_shot':score(session,zero,loader),'explicit':score(session,explicit,loader)}
                session.evaluator.synchronize();encoding_start=time.monotonic()
                inputs,meta=encoded(session,explicit,loader)
                session.evaluator.synchronize();encoding_seconds=time.monotonic()-encoding_start
                if cache is None:cache=SupportCache(session.evaluator,inputs,meta,8)
                current['explicit_cached_KV']=cache.score(inputs,meta,explicit)
                current['explicit_cached_KV']['seconds']+=encoding_seconds
                for method,vector,amplitude in [('difference',difference,scale),('historical_raw_mean',historical,1.),
                                                ('zero_vector',[0.]*len(difference),scale),('shuffled_vector',shuffled,scale)]:
                    current[method]=injected(session,zero,loader,layer,vector,amplitude)
                for method,result in current.items():
                    session.save(f'final_{block["task"]}_s{block["seed"]}_{method}_{index:03d}',{'result':result,
                                 'method':method,'reader_identity':identity,'layer':layer,'scale':scale,
                                 'vector_sha256':digest(vectors),'mapping_sha256':block['mapping_sha256']})
                    outcomes[method].append({'query_id':query,'correct':result['scores']['sum_prediction']==zero['query']['class_id'],
                                             'seconds':result['seconds'],'prefix_tokens':result['input_token_count']})
            rows.append({'block_id':block['block_id'],'reader_identity':identity,'layer':layer,'scale':scale,
                         'extraction_seconds':vectors['extraction_seconds'],'extraction_forwards':16,
                         'extraction_prefix_tokens':vectors['extraction_prefix_tokens'],
                         'explicit_cache_setup_seconds':cache.seconds,'explicit_cached_prefix_tokens':cache.prefix_tokens,
                         'methods':{m:{'accuracy':sum(r['correct'] for r in rr)/80,'seconds':math.fsum(r['seconds'] for r in rr),
                                       'prefix_tokens':sum(r['prefix_tokens'] for r in rr),'paired_rows':rr} for m,rr in outcomes.items()},
                         'fixed_reuse_counts':[1,8,40,80],
                         'cost_rule':'per-block extraction plus selected-method query cost; explicit cached reference adds one support-cache setup; reader/layer selection costs reported separately'})
            del cache
    publish(session.output/'summary.json',{'kind':'inject-descriptive-summary-v1','blocks':rows,
            'status':'no_eligible_prerequisite_reader' if not rows else 'confirmation_complete',
            'reader_selection':readers,'vector_selection':selected,'population_claim':False,
            'inference_unit':'80 fixed queries per task reused across3 fitting/context seeds; descriptive only',
            'negative_interpretation':'absence of a qualifying reader bounds this test; vector failure would not prove an absent integration computation'})
    return session.finish({'summary_sha256':sha(session.output/'summary.json'),'bounded_negative_outcome':not rows})
