"""Prospective first-crossing execution and separate development selection."""
from collections import defaultdict
import math
from .common import episode,publish,read,require,score,sha
from .policy import KS,select_threshold

def selected_blocks(session):
    return session.blocks

def reference(project,path):
    return {'path':str(path),'sha256':sha(path)}

def reviewed_selection(session):
    selection=read(session.args.selection)
    require(selection['final_queries_used'] is False,'Development selection used final labels')
    require(selection['model_id']==session.runtime['model_id'],'Selected threshold belongs to another model')
    require(selection['bundle_sha256']==session.bundle_sha,'Threshold development episodes/configuration differ')
    for condition,blocks in selection['blocks_by_condition'].items():
        require(select_threshold(blocks)==selection['threshold_by_model'][selection['model_id']][condition],
                'Threshold cannot be independently recomputed from development outcomes')
    return selection

def accuracy(records):
    require(bool(records),'empty accuracy cohort')
    return sum(r['scores']['sum_prediction']==r['episode']['query']['class_id'] for r in records)/len(records)


def probe_counts(records, endpoint_class):
    require(len(records)==40,'Gate requires40 probes including invalid predictions')
    return {'copied':sum(r['parse']['prediction']==endpoint_class for r in records),'total':40}


def run_cohort(session, block, manifest, loader, condition, ids, k, role, namespace, native=False):
    rows=[]
    for index,query in enumerate(ids):
        ep=episode(block,manifest,query,k,condition,role)
        result=score(session,ep,loader,native=native)
        key=f'{block["task"]}_s{block["seed"]}_{condition}_{namespace}_K{k}_{index:03d}'
        session.save(key,{'result':result,'cohort':namespace})
        rows.append(result)
    return rows


def smoke(session):
    session.start_model()
    blocks=selected_blocks(session)
    traces=[]
    for task in ('shapes4','cifar4'):
        block=next(b for b in blocks if b['task']==task and b['seed']==0)
        manifest,loader=session.loader(task)
        for condition in ('standard','episode_remap'):
            for k in (0,1,16):
                ep=episode(block,manifest,block['development_query_ids'][0],k,condition,'development')
                result=score(session,ep,loader,native=True)
                key=f'{task}_{condition}_K{k}'
                session.save(key,{'result':result,'native_gate_scope':'unmodified nested policy scorer'})
                traces.append(key)
    return session.finish({'development_only':True,'expected_traces':traces,'gate_math':'separately CPU tested'})


def development(session):
    session.start_model()
    groups=defaultdict(list)
    for block in selected_blocks(session):
        manifest,loader=session.loader(block['task'])
        for condition in ('standard','episode_remap'):
            counts,acc,costs={}, {}, {}
            for k in KS:
                rows=run_cohort(session,block,manifest,loader,condition,block['development_query_ids'],k,'development','development')
                acc[str(k)]=accuracy(rows)
                costs[f'development_K{k}']={'calls':len(rows),'seconds':math.fsum(r['seconds'] for r in rows),
                                            'prefix_tokens':sum(r['input_token_count'] for r in rows)}
                if k:
                    probes=run_cohort(session,block,manifest,loader,condition,block['gate_probe_ids'],k,
                                      'controller_development','probe')
                    endpoint=manifest['images'][block['nested_support_ids'][str(k)][-1]]['class_id']
                    counts[str(k)]=probe_counts(probes,endpoint)
                    costs[f'probe_K{k}']={'calls':len(probes),'seconds':math.fsum(r['seconds'] for r in probes),
                                         'prefix_tokens':sum(r['input_token_count'] for r in probes)}
            groups[condition].append({'block_id':block['block_id'],'copy_counts':counts,
                                      'development_accuracy':acc,'development_cost':costs})
    selection={condition:select_threshold(rows) for condition,rows in sorted(groups.items())}
    result={'kind':'gate-development-selection-v1','bundle_sha256':session.bundle_sha,
            'runtime_sha256':session.runtime_sha,'model_id':session.runtime['model_id'],
            'selection_roles':['controller_development','development'],
            'threshold_by_model':{session.runtime['model_id']:selection},'blocks_by_condition':dict(groups),
            'final_queries_used':False,'selection_uses_final_queries':False}
    publish(session.output/'selection.json',result)
    return session.finish({'selection_sha256':sha(session.output/'selection.json')})


def confirmation(session):
    selection=reviewed_selection(session)
    session.start_model()
    require(selection['model_id']==session.runtime['model_id'],'wrong selected model')
    summaries=[]
    for block in selected_blocks(session):
        manifest,loader=session.loader(block['task'])
        for condition in ('standard','episode_remap'):
            threshold=selection['threshold_by_model'][session.runtime['model_id']][condition]['threshold']
            selected,observations,probe_cost=0,[],{'calls':0,'seconds':0.0,'prefix_tokens':0}
            # Prospective dispatch actually stops. No later probe context is evaluated.
            for k in KS[1:]:
                probes=run_cohort(session,block,manifest,loader,condition,block['gate_probe_ids'],k,
                                  'controller_development','prospective_probe')
                endpoint=manifest['images'][block['nested_support_ids'][str(k)][-1]]['class_id']
                count=probe_counts(probes,endpoint)
                rejected=count['copied']/count['total']>threshold
                observations.append({'k':k,**count,'rejected':rejected})
                probe_cost['calls']+=len(probes);probe_cost['seconds']+=math.fsum(r['seconds'] for r in probes)
                probe_cost['prefix_tokens']+=sum(r['input_token_count'] for r in probes)
                if rejected:
                    break
                selected=k
            # Fixed-K/oracle evaluation follows the prospective decision. All 80
            # query identities are identical; its labels cannot alter that decision.
            policy={'block_id':block['block_id'],'condition':condition,'threshold':threshold,
                    'selected_k':selected,'observations':observations,'probe_cost':probe_cost}
            publish(session.output/'decisions'/f'{block["task"]}_s{block["seed"]}_{condition}.json',policy)
            evaluated={}
            for k in [selected]+[k for k in KS if k!=selected]:
                rows=run_cohort(session,block,manifest,loader,condition,block['final_query_ids'],k,
                                'confirmation_query','final')
                evaluated[str(k)]={'accuracy':accuracy(rows),'calls':len(rows),
                                  'seconds':math.fsum(r['seconds'] for r in rows),
                                  'prefix_tokens':sum(r['input_token_count'] for r in rows)}
            oracle=min(KS,key=lambda k:(-evaluated[str(k)]['accuracy'],k))
            chosen=evaluated[str(selected)]
            summaries.append({**policy,'fixed_k':evaluated,'actual_gate_accuracy':chosen['accuracy'],
                'actual_gate_total_seconds':chosen['seconds']+probe_cost['seconds'],
                'actual_gate_total_prefix_tokens':chosen['prefix_tokens']+probe_cost['prefix_tokens'],
                'oracle_k_analysis_only':oracle,'oracle_accuracy':evaluated[str(oracle)]['accuracy'],
                'accuracy_regret':evaluated[str(oracle)]['accuracy']-chosen['accuracy'],
                'stopped_before_final_accuracy_optimal_k':selected<oracle,
                'admitted_k_after_final_accuracy_optimal_k':selected>oracle,
                'zero_shot_accuracy':evaluated['0']['accuracy'],
                'fixed_two_support_accuracy':evaluated['2']['accuracy'],
                'two_support_reference':'raw labeled supports without vector injection',
                'cost_reference':'all charged probes plus chosen query context; candidate likelihood diagnostic cost included equally'})
    publish(session.output/'summary.json',{'kind':'gate-descriptive-summary-v1','blocks':summaries,
          'inference_unit':'80 fixed queries per task repeated across3 seeds; descriptive only',
          'development_selection':reference(session.project,session.args.selection),'population_claim':False})
    return session.finish({'summary_sha256':sha(session.output/'summary.json')})
