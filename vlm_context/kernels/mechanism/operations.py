"""Accepted paired residual and attention interventions."""
import copy,math
from vlm_context.kernels.gate.common import episode,score
from vlm_context.io import write_json as publish,read_json as read,sha256 as sha,require
from vlm_context.kernels.inject.hooks import AttentionCut,HiddenEdit,aligned,capture,classification_renderer,encode,segmentation
def blocks(session): return [b for b in session.blocks if b["task"]=="shapes4"]


LAYERS=(0,6,13,20,27)



METHODS=('clean','shuffled','self_patch','clean_patch','unrelated_patch',
         'label_cut','image_cut','instruction_text_cut')



def episodes(block,manifest,query,role):
    clean=episode(block,manifest,query,8,'episode_remap',role)
    corrupt=copy.deepcopy(clean)
    # A fixed derangement applied to displayed labels only. The image's true
    # class/content identity stays intact, and no final outcome chooses the map.
    for support in corrupt['supports']:
        support['display_class_id']=(support['class_id']+1)%4
    corrupt['support_label_permutation']=[1,2,3,0]
    return clean,corrupt



def trace(session,block,manifest,loader,query,role,layer,namespace,native=False,all_controls=True):
    clean,corrupt=episodes(block,manifest,query,role)
    ci,cm=encode(session.evaluator,clean,loader);ri,rm=encode(session.evaluator,corrupt,loader)
    cs=segmentation(session.evaluator,cm,clean);rs=segmentation(session.evaluator,rm,corrupt)
    align=aligned(cm,rm,cs,rs)
    n=align['prefix_length'];p=cs['reference_position'];u=cs['unrelated_patch_position']
    donor,_=capture(session.evaluator,ci,[layer],[p,u])
    recipient,_=capture(session.evaluator,ri,[layer],[p,u])
    vectors={'clean_reference':donor[str(layer)][str(p)],'shuffled_reference':recipient[str(layer)][str(p)],
             'clean_unrelated':donor[str(layer)][str(u)]}
    results={}
    def evaluate(method,ep,hook=None):
        if hook:
            with hook:
                result=score(session,ep,loader,native=native,renderer=classification_renderer)
            require(hook.calls>0,'hook did not execute')
        else:
            result=score(session,ep,loader,native=native,renderer=classification_renderer)
        results[method]=result
        session.save(f'{namespace}_L{layer}_{method}',{'result':result,'method':method,'layer':layer,
            'alignment':align,'donor_states':vectors if method in ('clean_patch','self_patch','unrelated_patch') else None,
            'hook_prefix_only':True,'candidate_tokens_do_not_change_selection':True})
    evaluate('clean',clean);evaluate('shuffled',corrupt)
    evaluate('clean_patch',corrupt,HiddenEdit(session.evaluator,layer,n,p,vectors['clean_reference']))
    if all_controls:
        evaluate('self_patch',corrupt,HiddenEdit(session.evaluator,layer,n,p,vectors['shuffled_reference']))
        evaluate('unrelated_patch',corrupt,HiddenEdit(session.evaluator,layer,n,u,vectors['clean_unrelated']))
        for name,columns in [('label_cut',cs['label_columns']),('image_cut',cs['image_control_columns']),
                             ('instruction_text_cut',cs['text_control_columns'])]:
            evaluate(name,clean,AttentionCut(session.evaluator,layer,n,cs['query_rows'],columns))
        require(max(abs(a-b) for ra,rb in zip(results['shuffled']['per_token_label_log_probabilities'],
                    results['self_patch']['per_token_label_log_probabilities']) for a,b in zip(ra,rb))<=.05,
                'self-patch is not a native no-op')
        require(results['shuffled']['greedy_token_ids']==results['self_patch']['greedy_token_ids'],'self-patch generation changed')
    if native:
        with AttentionCut(session.evaluator,layer,n,cs['query_rows'],[]) as hook:
            zero=score(session,clean,loader,native=True,renderer=classification_renderer)
        require(zero['per_token_label_log_probabilities']==results['clean']['per_token_label_log_probabilities'] and
                zero['greedy_token_ids']==results['clean']['greedy_token_ids'],'empty attention cut changed native output')
        session.save(f'{namespace}_L{layer}_empty_cut',{'result':zero,'method':'empty_cut','alignment':align})
    gold=clean['query']['class_id']
    return {method:{'correct':r['scores']['sum_prediction']==gold,'correct_label_logp':r['scores']['sum'][gold],
                    'seconds':r['seconds'],'prefix_tokens':r['input_token_count']} for method,r in results.items()}



def smoke(session):
    session.start_model();block=blocks(session)[0];manifest,loader=session.loader('shapes4')
    for layer in LAYERS:
        trace(session,block,manifest,loader,block['development_query_ids'][0],'development',layer,'native',native=True)
    return session.finish({'development_only':True,'layers':list(LAYERS),'methods':list(METHODS),
                           'fixed_scope':'Shapes K8 mapping controls; residual output last-prefill position'})



def development(session):
    session.start_model();manifest,loader=session.loader('shapes4');rows=[]
    for block in blocks(session):
        # Three IDs per class from the frozen CPU02 development stream. Selection
        # depends only on the registered role/class metadata, never model output.
        queries=[q for c in range(4) for q in [i for i in block['development_query_ids'] if manifest['images'][i]['class_id']==c][:3]]
        require(len(queries)==12,'balanced development subset incomplete')
        for index,query in enumerate(queries):
            for layer in LAYERS:
                outcome=trace(session,block,manifest,loader,query,'development',layer,
                              f'dev_s{block["seed"]}_q{index:03d}',all_controls=False)
                rows.append({'block_id':block['block_id'],'query_id':query,'layer':layer,'outcomes':outcome})
    means={layer:math.fsum(r['outcomes']['clean_patch']['correct_label_logp']-
                         r['outcomes']['shuffled']['correct_label_logp'] for r in rows if r['layer']==layer)/36 for layer in LAYERS}
    chosen=min(LAYERS,key=lambda layer:(-means[layer],layer))
    selection={'kind':'mechanism-development-selection-v1','bundle_sha256':session.bundle_sha,
               'runtime_sha256':session.runtime_sha,'layer':chosen,'layer_grid':list(LAYERS),
               'mean_correct_label_logp_restoration':means,'rows':rows,'final_queries_used':False,
               'selection_rule':'largest mean development correct-label logp restoration; ties lower layer',
               'attention_layer':'same selected residual layer; all heads, no per-head outcome selection',
               'selection_scope':'development_only'}
    publish(session.output/'selection.json',selection)
    return session.finish({'selection_sha256':sha(session.output/'selection.json')})



def attention_descriptive(session):
    """All-head statistics across K; no head is selected using final outcomes."""
    session.start_model();manifest,loader=session.loader('shapes4');all_layers=list(range(28))
    for block in blocks(session):
        for k in (1,2,4,8,16):
            for index,query in enumerate(block['final_query_ids']):
                ep=episode(block,manifest,query,k,'standard','confirmation_query')
                inputs,metadata=encode(session.evaluator,ep,loader)
                seg=segmentation(session.evaluator,metadata,ep)
                _,attentions=capture(session.evaluator,inputs,[],[],attention=True)
                stats=[]
                for layer,attention in enumerate(attentions):
                    rows=attention[0,:,seg['query_rows'],:].float()
                    images=[i for s in seg['segments'][:-1] for i in s['image']]
                    labels=seg['label_columns']
                    stats.append({'layer':layer,'query_to_demo_image_mass':rows[:,:,images].sum(-1).mean(-1).cpu().tolist(),
                                  'query_to_demo_label_mass':rows[:,:,labels].sum(-1).mean(-1).cpu().tolist(),
                                  'image_token_count':len(images),'label_token_count':len(labels),
                                  'query_row_count':len(seg['query_rows'])})
                del attentions
                session.save(f'attention_s{block["seed"]}_K{k}_q{index:03d}',{'episode':ep,'encoding':metadata,
                    'segments':seg,'head_statistics':stats,'descriptive_only':True})
    return session.finish({'raw_attention_saved':False,'all_heads_reported':True,'interpretation':'descriptive'})
