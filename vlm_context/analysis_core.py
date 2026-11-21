"""Portable original core image-cluster estimator and raw-score reduction."""
from collections import defaultdict
import itertools
import json
import math
import random
import statistics
import hashlib
from types import SimpleNamespace
from .io import require, read_json, source_path
from .protocol import parse_diagnostics

def config_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
p=SimpleNamespace(require=require,config_hash=config_hash)
c=SimpleNamespace(VERSION="repair-execution-20261106-1")
def paired_image_values(left,right,field):
    key=lambda r:(r['image_id'],r['seed'])
    l={key(r):r for r in left};r={key(x):x for x in right}
    p.require(len(l)==len(left) and len(r)==len(right) and set(l)==set(r),'paired image/seed cells differ or duplicate')
    values=defaultdict(list);classes={}
    for k in sorted(l):
        p.require(l[k]['class_id']==r[k]['class_id'],'paired image class differs')
        values[k[0]].append(l[k][field]-r[k][field]);classes[k[0]]=l[k]['class_id']
    p.require(len(values)==80 and all(len(v)==3 for v in values.values()),'80 images ×3 repeated seeds required')
    return {i:(classes[i],statistics.mean(v)) for i,v in values.items()}

def bootstrap_images(values,seed,draws=2000):
    strata=defaultdict(list)
    for cls,value in values.values():strata[cls].append(value)
    p.require({k:len(v) for k,v in strata.items()}=={k:20 for k in range(4)},'four balanced20image strata required')
    rng=random.Random(seed);samples=[]
    groups=[strata[k] for k in range(4)]
    for _ in range(draws):samples.append(math.fsum(group[rng.randrange(20)] for group in groups for _ in range(20))/80)
    samples.sort()
    return {'estimate':statistics.mean(v for _,v in values.values()),
        'descriptive_percentile_95':[samples[int(.025*(draws-1))],samples[int(.975*(draws-1))]],
        'image_clusters':80,'seeds_per_image':3,'bootstrap_draws':draws,
        'population_criterion_status':'unsupported_sampling_assumptions'}

def reduce_all(rows,draws=2000):
    cells=defaultdict(list)
    for row in rows:cells[(row['arm'],row['task'],row['k'],row['condition'])].append(row)
    summaries=[]
    for key,values in sorted(cells.items()):
        p.require(len(values)==240 and len({(r['image_id'],r['seed']) for r in values})==240,'incomplete/duplicated repeated-image cell')
        confusion=[[0]*5 for _ in range(4)]
        for r in values:confusion[r['reference_target']][4 if r['strict_prediction'] is None else r['strict_prediction']]+=1
        summaries.append({'arm':key[0],'task':key[1],'k':key[2],'condition':key[3],
            'n_images':80,'n_repeated_records':240,
            **{field:statistics.mean(r[field] for r in values) for field in ['reference_accuracy','displayed_accuracy','mean_reference_accuracy','strict_reference_accuracy','legacy_reference_accuracy','invalid']},
            'strict_confusion_with_invalid_column':confusion,
            'consistent_class_mapping_records':sum(r['consistent_class_mapping'] for r in values)})
    contrasts=[];seen=set()
    def add(kind,left,right,field='reference_accuracy'):
        identity=(kind,left,right,field)
        if identity in seen or left not in cells or right not in cells:return
        seen.add(identity);value=paired_image_values(cells[left],cells[right],field)
        seed=int(p.config_hash([c.VERSION,identity]),16)
        contrasts.append({'kind':kind,'left':left,'right':right,'metric':field,**bootstrap_images(value,seed,draws)})
    for arm,task,k,condition in sorted(cells):
        key=(arm,task,k,condition)
        if k and condition=='standard':add('standard_decline',(arm,task,0,condition),key)
        if k and condition.startswith('remap_'):add('remap_gain',key,(arm,task,0,condition))
        if arm!='frozen_joint':add('adapter_minus_frozen',key,('frozen_joint',task,k,condition))
        if arm.endswith('_episode_remap') and not arm.startswith('q05_'):
            add('episodic_minus_natural',key,(arm.removesuffix('_episode_remap')+'_natural',task,k,condition))
        if arm.startswith('joint_uncapped_r16_'):
            objective=arm.removeprefix('joint_uncapped_r16_')
            add('matched_joint_minus_late',key,('late_parameter_matched_r34_35_'+objective,task,k,condition))
            for omitted in ['connector','early','mid']:add('full_joint_minus_ablation',key,('joint_without_'+omitted+'_r16_'+objective,task,k,condition))
        if arm=='joint_uncapped_r16_natural':add('context_training_minus_no_demo',key,('joint_natural_no_demonstrations',task,k,condition))
        if arm=='unrestricted_q05_natural':add('unrestricted_natural_minus_episodic',key,('q05_uncapped_all_episode_remap',task,k,condition))
        if arm.startswith('depth_window_'):add('window_minus_connector_different_budget',key,('depth_connector_r8_separate_budget',task,k,condition))
        if condition in ['intact_pair_order_shuffle','independently_shuffled_support_labels','deliberately_inconsistent_support_labels']:
            add('corrupted_minus_intact',key,(arm,task,k,'remap_unseen_label_strings'))
    windows=sorted({key[0] for key in cells if key[0].startswith('depth_window_')})
    for left,right in itertools.combinations(windows,2):
        for a,task,k,condition in sorted(cells):
            if a==left:add('window_pairwise',(left,task,k,condition),(right,task,k,condition))
    counterfactual=[]
    for arm,task,k,condition in sorted(cells):
        if condition!='counterfactual_consistent_derangement':continue
        baseline=(arm,task,k,'remap_unseen_label_strings')
        # The counterfactual is a derangement of the unseen-label variant.
        # Compare label strings: a correct response can keep the same class
        # index while changing the corresponding label token.
        key=(arm,task,k,condition)
        if baseline not in cells:continue
        left={(r['image_id'],r['seed']):r for r in cells[baseline]};right={(r['image_id'],r['seed']):r for r in cells[key]}
        p.require(set(left)==set(right),'counterfactual pair inventory differs')
        counterfactual.append({'arm':arm,'task':task,'k':k,
            'both_demonstrated_targets_correct':statistics.mean(left[i]['displayed_accuracy']*right[i]['displayed_accuracy'] for i in left),
            'prediction_label_switch':statistics.mean(left[i]['predicted_label']!=right[i]['predicted_label'] for i in left),
            'scope':'descriptive; paired exact label strings, not class-index switching'})
    return {'cells':summaries,'paired_contrasts':contrasts,'counterfactual_response':counterfactual,
        'population_certification_enabled':False,}

def observations(records):
    rows=[]
    for raw in records:
        labels=raw['labels'];logps=raw['token_logps'];tokens=raw['candidate_token_ids']
        require(len(labels)==len(logps)==len(tokens)==4,'Core candidate coverage')
        for seq,ids in zip(logps,tokens):
            require(len(seq)==len(ids)>0 and ids[-1]==151645 and ids.count(151645)==1,'Core answer/end span')
            require(all(math.isfinite(x) and x<=1e-7 for x in seq),'Nonfinite core score')
        sums=[math.fsum(x)for x in logps];means=[sum(x)/len(x)for x in logps]
        pred=max(range(4),key=lambda j:sums[j]);mpred=max(range(4),key=lambda j:means[j])
        parsed=parse_diagnostics(raw['greedy_answer'],labels)
        reference=labels.index(raw['reference_target_label']);target=labels.index(raw['target_label'])
        row={k:raw[k]for k in ('arm','seed','task','k','condition','image_id','class_id','consistent_class_mapping')}
        row.update(sum_prediction=pred,mean_prediction=mpred,predicted_label=labels[pred],strict_prediction=parsed['prediction'],legacy_prediction=parsed['legacy_prediction'],reference_target=reference,displayed_target=target,reference_accuracy=int(pred==reference),displayed_accuracy=int(pred==target),mean_reference_accuracy=int(mpred==reference),strict_reference_accuracy=int(parsed['prediction']==reference),legacy_reference_accuracy=int(parsed['legacy_prediction']==reference),invalid=int(parsed['prediction'] is None))
        rows.append(row)
    return rows


def analyze_core():
    from .analysis_core_seeds import reduce_rows
    rows=observations(read_json(source_path('data/analysis/core_raw_scores.json.gz')))
    image=reduce_all(rows)
    declared=read_json(source_path('data/analysis/core_contrasts.json.gz'))
    require([{k:r[k]for k in ('kind','left','right','metric')}for r in image['paired_contrasts']]==[{**r,'left':tuple(r['left']),'right':tuple(r['right'])}for r in declared],'Core declared contrast family changed')
    seeds=reduce_rows(rows,image)
    return image,seeds
