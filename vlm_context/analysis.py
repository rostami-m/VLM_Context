"""Recompute the paper's numerical analyses from public per-example evidence.

The bundled paper summaries are comparison targets and metadata containers.
They never supply bootstrap endpoints or observed means to a reducer.
"""
from copy import deepcopy
from pathlib import Path
import math
import platform
from .io import read_json, require, sha256, source_path, write_json
from .analysis_statistics import paired_family
from .analysis_core import analyze_core
from .analysis_q11 import analyze_q11
from .protocol import parse_diagnostics


def compare_numbers(observed, reference, path='', tolerance=1e-10):
    """Check every overlapping numerical leaf; extra explanatory metadata is allowed."""
    count=0
    if isinstance(observed,dict):
        require(isinstance(reference,dict),'Reference shape differs at '+path)
        for key,value in observed.items():
            if key in reference:count+=compare_numbers(value,reference[key],path+'/'+str(key),tolerance)
    elif isinstance(observed,list):
        require(isinstance(reference,list)and len(observed)==len(reference),'Reference list differs at '+path)
        for i,(a,b)in enumerate(zip(observed,reference)):count+=compare_numbers(a,b,path+'/'+str(i),tolerance)
    elif isinstance(observed,(int,float))and not isinstance(observed,bool):
        require(isinstance(reference,(int,float))and math.isfinite(observed)and abs(observed-reference)<=tolerance,'Recomputed value differs at '+path+f': {observed} vs {reference}')
        count=1
    return count


def recompute_families():
    requests=read_json(source_path('data/analysis/paired_families.json.gz'))
    destinations=read_json(source_path('data/analysis/paired_family_destinations.json.gz'))
    computed=[]
    for req in requests:
        args={k:v for k,v in req.items()if k!='cells'}
        if args.get('batch_calibration'):
            args['batch_calibration']={m:[{'weight':c['weight'],'logits':{(i,s):v for i,s,v in c['logits']}}for c in parts]for m,parts in args['batch_calibration'].items()}
        cells={m:{(i,s):{'class_id':c,'value':v}for i,s,c,v in rows}for m,rows in req['cells'].items()}
        require(all(len(cells[m])==len(req['cells'][m])for m in cells),'Duplicate released family observations')
        computed.append(paired_family(cells,**args))
    intervention=deepcopy(read_json(source_path('data/paper/intervention_results.json.gz')))
    bridge=deepcopy(read_json(source_path('data/paper/bridge_analysis.json')))
    def family_paths(value,path=()):
        if isinstance(value,dict):
            if {'reference','methods','bootstrap_unit','bootstrap_draws'}<=set(value):return {path}
            return set().union(*(family_paths(v,path+(k,))for k,v in value.items())) if value else set()
        return set()
    expected={('bridge',)+p for p in family_paths(bridge)}|family_paths(intervention)
    observed={tuple(r['destination'])for r in destinations if r['destination'][0]!='Q08_frozen_vision'}
    require(expected==observed,'Released paired-family destination coverage differs')
    checked=0
    for item in destinations:
        dest=item['destination'];root=bridge if dest[0]=='bridge'else intervention
        if dest[0]=='Q08_frozen_vision':continue
        path=dest[1:]if dest[0]=='bridge'else dest
        cursor=root
        for key in path[:-1]:cursor=cursor[key]
        fresh=computed[item['request']];checked+=compare_numbers(fresh,cursor[path[-1]],'/'.join(dest));cursor[path[-1]]=fresh
    return bridge,intervention,{'paired_families':len(requests),'paired_family_numeric_checks':checked}


def baseline_diagnostics():
    raw_path=source_path('data/analysis/intervention_raw_scores.json.gz')
    manifest=read_json(source_path('data/analysis/intervention_manifest.json'))
    require(sha256(raw_path)==manifest['raw_sha256'],'Intervention raw-score archive hash differs')
    records=read_json(raw_path);cells={};checked=0
    require(len(records)==manifest['records'],'Intervention raw-score inventory differs')
    schedule=read_json(source_path('data/protocols/interventions/baseline_schedule.json.gz'))
    allowed={(e['task'],e['condition'],e['k'],e['query']['image_id'],e['seed']):e['query']['class_id']for p in schedule['panels']for e in p['episodes']}
    seen=set();outcomes={};class_by_image={};raw_logits={};neutral_logits={};intervention_values={}
    for r in records:
        token_ids=r['candidate_token_ids'];logps=r['token_logps']
        require(len(token_ids)==len(logps)==len(r['labels'])==4,'Four intervention candidates required')
        for ids,seq in zip(token_ids,logps):
            require(len(ids)==len(seq)>0 and ids[-1]==151645 and ids.count(151645)==1,'Intervention answer/end-token span')
            require(all(math.isfinite(v)and v<=1e-7 for v in seq),'Nonfinite intervention score')
        checked+=1
        sums=[math.fsum(v)for v in logps]
        if r['phase']not in ('baseline-confirmation','bridge'):
            if r['gold']is not None:
                pred=max(range(4),key=lambda j:sums[j]);value=(float(pred==r['gold']),sums[r['gold']])
                if r['phase']=='mechanism-confirmation':identity=('Q09',r['method'],r['image_id'],r['seed'])
                elif r['phase']=='inject-confirmation':identity=('Q16',r['task'],r['method'],r['image_id'],r['seed'])
                elif r['phase']=='gate-confirmation':identity=('Q15',r['selector']['model'],r['task'],r['condition'],r['k'],r['image_id'],r['seed'])
                else:continue
                require(identity not in intervention_values,'Duplicate intervention query/context/condition');intervention_values[identity]=value
            continue
        method=r.get('method')or r['selector']['method'];key=method,r['task'],r['condition'],r['k']
        episode=r['task'],r['condition'],r['k'],r['image_id'],r['seed']
        if episode not in allowed:
            require(r.get('record_id','').endswith('_neutral.json')and method=='frozen','Unexpected baseline query outside declared schedule')
            neutral=(r['task'],r['condition'],r['k'],r['seed']);require(neutral not in neutral_logits,'Duplicate content-free baseline context');neutral_logits[neutral]=sums
            continue
        require(r['gold']==allowed[episode],'Baseline gold differs from frozen episode schedule')
        identity=(method,)+episode;require(identity not in seen,'Duplicate baseline query/context/method');seen.add(identity)
        require(class_by_image.setdefault((r['task'],r['image_id']),r['gold'])==r['gold'],'Baseline image class changed')
        if key not in cells:cells[key]={'method':method,'task':r['task'],'condition':r['condition'],'k':r['k'],'n':0,'strict_confusion_with_invalid_column':[[0]*5 for _ in range(4)],'sum_correct':0,'mean_correct':0,'sum_mean_disagreements':0}
        cell=cells[key];pred=parse_diagnostics(r['greedy_answer'],r['labels'])['prediction'];cell['strict_confusion_with_invalid_column'][r['gold']][4 if pred is None else pred]+=1;cell['n']+=1
        sums=[math.fsum(s)for s in logps];means=[math.fsum(s)/len(s)for s in logps];a=max(range(4),key=lambda j:sums[j]);b=max(range(4),key=lambda j:means[j]);cell['sum_correct']+=a==r['gold'];cell['mean_correct']+=b==r['gold'];cell['sum_mean_disagreements']+=a!=b;outcomes[identity]=float(a==r['gold']);raw_logits[identity]=sums
    for cell in cells.values():
        require(cell['n']==240,'Baseline cell must retain80queries by3contexts')
        matrix=cell['strict_confusion_with_invalid_column'];cell['strict_invalid_rate']=sum(r[-1]for r in matrix)/cell['n'];cell['strict_accuracy']=sum(matrix[i][i]for i in range(4))/cell['n'];cell['sum_accuracy']=cell['sum_correct']/cell['n'];cell['mean_accuracy']=cell['mean_correct']/cell['n']
    requests=read_json(source_path('data/analysis/paired_families.json.gz'));destinations=read_json(source_path('data/analysis/paired_family_destinations.json.gz'));bound=0
    for dest in destinations:
        path=dest['destination']
        if path[:2]!=['bridge','paired_accuracy_relative_to_joint']:continue
        task,condition,k=path[2].split('/');k=int(k)
        for method,rows in requests[dest['request']]['cells'].items():
            if method in ('cc','bc','vision_1nn'):continue
            for image,seed,gold,value in rows:
                identity=(method,task,condition,k,image,seed)
                require(identity in outcomes and value==outcomes[identity],'Baseline raw score differs from released paired outcome: '+str((identity,value,outcomes.get(identity))))
                bound+=1
        if 'cc'in requests[dest['request']]['cells']:
            for image,seed,gold,value in requests[dest['request']]['cells']['cc']:
                scores=raw_logits['frozen',task,condition,k,image,seed];neutral=neutral_logits[task,condition,k,seed];prediction=max(range(4),key=lambda j:scores[j]-neutral[j]);require(value==float(prediction==gold),'CC outcome differs from content-free recalibration');bound+=1
        for component in (requests[dest['request']].get('batch_calibration')or {}).get('bc',[]):
            for image,seed,logits in component['logits']:
                require(logits==raw_logits['frozen',task,condition,k,image,seed],'BC bootstrap logits differ from raw frozen candidate scores');bound+=1
    existing=read_json(source_path('data/paper/intervention_results.json.gz'));intervention_bound=0
    for dest in destinations:
        path=dest['destination']
        if path[0]not in ('Q09','Q15','Q16'):continue
        for method,rows in requests[dest['request']]['cells'].items():
            for image,seed,gold,value in rows:
                if path[0]=='Q09':identity=('Q09',method,image,seed);metric=1 if path[1].startswith('logp')else 0
                elif path[0]=='Q16':identity=('Q16',path[2],method,image,seed);metric=0
                else:
                    model=path[2];task,condition=path[4].split('/')
                    if method=='Gate':
                        choices=[b['selected_k']for b in existing['Q15']['models'][model]['blocks']if(b['task'],b['condition'],b['seed'])==(task,condition,seed)];require(len(choices)==1,'Unique saved Gate decision required');k=choices[0]
                    else:k=int(method.removeprefix('fixed_K'))
                    identity=('Q15',model,task,condition,k,image,seed);metric=0
                require(identity in intervention_values and abs(value-intervention_values[identity][metric])<1e-12,'Intervention raw score differs from elementary bootstrap outcome: '+str(identity));intervention_bound+=1
    return [cells[k]for k in sorted(cells)],{'score_spans_checked':checked,'baseline_raw_outcomes_bound_to_bootstrap':bound,'intervention_raw_outcomes_bound_to_bootstrap':intervention_bound}


def analyze(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    image,seeds=analyze_core()
    checked=compare_numbers(image,read_json(source_path('data/analysis/core_image_reference.json.gz')),'core_image_analysis')
    checked+=compare_numbers(image['cells'],read_json(source_path('data/paper/core_cells.json')),'core_cells')
    checked+=compare_numbers(seeds['cells'],read_json(source_path('data/paper/core_seed_cells.json.gz')),'core_seed_cells')
    checked+=compare_numbers(seeds['paired_contrasts'],read_json(source_path('data/paper/core_seed_contrasts.json.gz')),'core_seed_contrasts')
    q11=analyze_q11();checked+=compare_numbers(q11,read_json(source_path('data/paper/q11_results.json')),'q11')
    bridge,interventions,family_validation=recompute_families();diagnostics,raw_checked=baseline_diagnostics()
    result={'core_cells':image['cells'],'core_seed_cells':seeds['cells'],'core_seed_contrasts':seeds['paired_contrasts'],'bridge_analysis':bridge,'q11_results':q11,'intervention_results':interventions,'baseline_diagnostics':diagnostics}
    # Stream reducers are kept separate because their sampling units and RNG differ.
    from .continual_analysis import recompute_continual_controller
    stream=recompute_continual_controller()
    result.update(stream)
    files=sorted(source_path('data/analysis').glob('*.gz'))
    validation={'raw_core_records':59760,'core_cells':len(image['cells']),'core_image_contrasts':len(image['paired_contrasts']),'native_q11_records':3200,'numeric_reference_checks':checked,**family_validation,**raw_checked,'baseline_diagnostic_cells':len(diagnostics),'full_model_inference_rerun':False,'bootstrap_scope':'Fixed-panel descriptive image resampling and separate paired fitting/context-seed intervals; no paper-wide multiplicity correction.','inputs':{str(p.relative_to(source_path('data/analysis').parent.parent)):sha256(p)for p in files},'python':platform.python_version()}
    result['analysis_validation']=validation
    write_json(output/'core_image_analysis.json',image,overwrite=True)
    for name,value in result.items():write_json(output/(name+'.json'),value,overwrite=True)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('runs/analysis'));args=parser.parse_args();analyze(args.output)
