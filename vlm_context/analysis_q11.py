"""Recompute native open-task scores and paired image/context summaries on CPU."""
from collections import defaultdict
import hashlib
import json
import math
import random
import statistics
from .io import require, read_json, source_path
from .open_tasks import vqa_score


def bootstrap(values, seed=20261106, draws=10000):
    require(bool(values), 'Empty image bootstrap')
    rng=random.Random(seed); n=len(values)
    samples=sorted(sum(rng.choice(values)for _ in values)/n for _ in range(draws))
    return {'mean':sum(values)/n,'low95':samples[int(.025*draws)],'high95':samples[int(.975*draws)],'images':n,'draws':draws,'sampling_unit':'query image, seeds averaged within image'}


def image_summary(records):
    by=defaultdict(list)
    for (image,seed),value in records.items():by[image].append(value)
    return bootstrap([sum(v)/len(v)for v in by.values()])


def seed_summary(left,right,reuse=False):
    ids=sorted({i for i,s in left});require({s for i,s in left}=={0,1,2},'Need three fitted/context replicates')
    cohorts=[{i for i,s in left if s==seed}for seed in range(3)]
    require(bool(cohorts[0])and all(c==cohorts[0]for c in cohorts),'Fitted/context image cohorts differ')
    require(set(right)==set(left)or(reuse and set(right)=={(i,0)for i in ids}),'Reference must be exact paired cohort or explicitly complete seed0 baseline')
    values={s:math.fsum(left[i,s]-right[(i,s)if(i,s)in right else(i,0)]for i in ids)/len(ids)for s in range(3)}
    xs=list(values.values());mean=statistics.mean(xs);sd=statistics.stdev(xs);critical=math.sqrt(2*.95**2/(1-.95**2));half=critical*sd/math.sqrt(3)
    return {'paired_replicate_estimates':{str(k):v for k,v in values.items()},'replicates':3,'mean':mean,'sample_sd':sd,'t_degrees_of_freedom':2,'t_critical_975':critical,'t_interval_95':[mean-half,mean+half],'range':[min(xs),max(xs)],'interval_not_clipped_to_parameter_range':True,'conditioning':'fixed query panel; paired fitting and context replicate varies jointly','population_certified':False,'unique_query_images':len(ids),'fixed_deterministic_reference_reused_across_replicates':reuse,'query_image_ids_sha256':hashlib.sha256(json.dumps(ids,sort_keys=True,separators=(',',':')).encode()).hexdigest()}


def collect(rows):
    cells=defaultdict(dict);pairs=defaultdict(dict);seen=set()
    for r in rows:
        if r['candidates'] is not None:
            sums=[]
            for c in r['candidates']:
                ids=c['scored_token_ids'];logps=c['token_log_probabilities']
                require(len(ids)==len(logps)>0 and ids[-1]==c['native_end_token'] and ids.count(c['native_end_token'])==1,'Native answer/end span')
                require(all(math.isfinite(v) and v<=1e-7 for v in logps),'Nonfinite native scores')
                value=math.fsum(logps);require(abs(value-c['sum_logp'])<1e-10,'Native candidate reduction mismatch');sums.append(value)
            pred=r['candidates'][max(range(len(sums)),key=lambda i:sums[i])]['label'];score=float(pred==r['answers'][0])
            require(pred==r['sum_prediction'],'Native prediction mismatch')
        else:score=vqa_score(r['generated'],r['answers'])
        require(abs(score-r['score'])<1e-12,'Native task score mismatch')
        key=(r['method'],r['task'],r['condition'],r['k']);identity=r['image_id'],r['seed'];unique=key+identity+(r['rule'],)
        require(unique not in seen,'Duplicate Q11 observation');seen.add(unique)
        if r['task']=='rule':pairs[key,identity][r['rule']]=score
        else:cells[key][identity]=score
    for (key,identity),sides in pairs.items():
        require(set(sides)=={'left','right'},'Incomplete rule pair');cells[key][identity]=sides['left']*sides['right']
    return cells


def analyze_q11():
    cells=collect(read_json(source_path('data/analysis/q11_outcomes.json.gz')))
    meta=read_json(source_path('data/analysis/q11_methods.json.gz'));out={}
    for task,info in meta.items():
        methods=info['methods'];summaries={};contrasts={};paired={};seeds={};frozen=next(m for m,v in methods.items()if v['arm']=='frozen')
        for key,values in sorted(cells.items()):
            method,t,condition,k=key
            if t!=task:continue
            name='/'.join(map(str,key));summary={'method':method,'task':task,'condition':condition,'k':k,'records':len(values),'accuracy':image_summary(values)}
            if task=='rule':summary['outcome']='both counterfactual sides correct on same image/seed'
            summaries[name]=summary;zero=cells.get((method,task,condition,0))
            if k and zero:contrasts[name]={'paired_K0_minus_K':image_summary({i:zero[i]-v for i,v in values.items()}),'prespecified_drop_margin':.10}
            if method!=frozen:
                base=cells[frozen,task,condition,k];reuse=set(base)!=set(values)
                require(not reuse or task in ('textvqa','scienceqa'),'Undeclared frozen reference reuse')
                delta={i:v-base[i if i in base else(i[0],0)]for i,v in values.items()}
                paired[name]={'adapted_minus_frozen':image_summary(delta),'frozen_method':frozen,'same_image_seed_pairs':not reuse,'fixed_deterministic_K0_baseline_reused_across_fit_seeds':reuse}
                seeds[name+'/adapted_minus_frozen']=seed_summary(values,base,reuse)
                if k and zero:seeds[name+'/K_minus_K0']=seed_summary(values,zero)
        out[task]={'methods':methods,'cells':summaries,'paired_K_contrasts':contrasts,'paired_checkpoint_contrasts':paired,'paired_fitting_context_seed_contrasts':seeds}
    return out
