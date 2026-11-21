"""Prespecified descriptive paired reductions; images and fitted seeds are distinct units."""
from __future__ import annotations
import math
import hashlib
from .io import require as need

T_DF2 = 4.3026527299
_BC_CACHE = {}


def weighted_bc_predictions(logits, weights):
    """Re-estimate unlabeled deployment bias; gold labels never enter this step."""
    need(len(logits)==len(weights)>0 and all(type(w) is int and w>=0 for w in weights) and sum(weights)>0,
         'BC resample weight inventory')
    need(all(len(r)==4 and all(math.isfinite(v) and v<=0 for v in r) for r in logits),'BC finite four-logit rows')
    bias=[math.fsum(w*r[j] for w,r in zip(weights,logits))/sum(weights) for j in range(4)]
    return [max(range(4),key=lambda j:r[j]-bias[j]) for r in logits]


def bc_draws(logits, counts, gold):
    """Each bootstrap sample recalibrates before computing weighted correctness."""
    import numpy as np
    matrix=np.asarray(logits,dtype='float64');gold=np.asarray(gold,dtype='int64')
    key=hashlib.sha256(matrix.tobytes()+counts.tobytes()+gold.tobytes()).hexdigest()
    if key not in _BC_CACHE:
        out=np.zeros(len(counts),dtype='float64')
        for i,weights in enumerate(counts):
            w=[int(x) for x in weights]
            predictions=weighted_bc_predictions(logits,w)
            out[i]=sum(n for n,p,g in zip(w,predictions,gold) if p==g)/sum(w)
        _BC_CACHE[key]=out
    return _BC_CACHE[key]


def seed_interval(values):
    need(set(values) == {0, 1, 2} and all(math.isfinite(v) for v in values.values()), 'three finite seed estimates required')
    vals = [values[s] for s in range(3)]; mean = math.fsum(vals)/3
    sd = math.sqrt(math.fsum((v-mean)**2 for v in vals)/2)
    half = T_DF2*sd/math.sqrt(3)
    return {'seed_estimates': vals, 'mean': mean, 'sample_sd': sd, 'range': [min(vals), max(vals)],
            't95_df2': [mean-half, mean+half], 'critical_value': T_DF2,
            'estimand': 'paired fitting/context replicate contrasts conditional on fixed query images; n=3; fitting initialization and support/context seed vary together, not isolated training variability or population certification'}


def paired_family(cells, reference, *, fitted=False, draws=10000, rng_seed=271828, batch_calibration=None):
    """cells[method][(image_id, context_seed)] = {class_id, value}.

    Sampling whole images within class retains all three contexts and every method
    together. Max absolute centered deviation gives a joint descriptive 95% band
    over the explicitly returned contrasts in this call, not all paper analyses.
    """
    import numpy as np
    need(reference in cells and len(cells) >= 1, 'reference missing')
    keys = set(cells[reference]); need(keys, 'empty family')
    ids = sorted({k[0] for k in keys})
    need(len(ids) == 80 and keys == {(q,s) for q in ids for s in range(3)}, 'requires exact80 images x3 contexts')
    classes = {}
    for name, rows in cells.items():
        need(set(rows) == keys, 'paired image/seed coverage differs: '+str(name))
        for key, row in rows.items():
            need(math.isfinite(row['value']) and type(row['class_id']) is int and row['class_id'] in range(4), 'invalid outcome')
            need(classes.setdefault(key[0], row['class_id']) == row['class_id'], 'class changed across contexts/methods')
    groups = [[i for i,q in enumerate(ids) if classes[q] == c] for c in range(4)]
    need(all(len(g) == 20 for g in groups), 'fixed class balance differs')
    methods = sorted(cells)
    values = np.array([[math.fsum(cells[m][(q,s)]['value'] for s in range(3))/3 for m in methods] for q in ids], dtype='float64')
    rng = np.random.default_rng(rng_seed)
    counts = np.zeros((draws,80), dtype='float64')
    for group in groups:
        counts[:,group] = rng.multinomial(20, [1/20]*20, size=draws)
    sampled = counts @ values / 80; means = values.mean(axis=0)
    for method,components in (batch_calibration or {}).items():
        need(method in cells and components,'BC override method/contract missing')
        j=methods.index(method);component_draws=[];observed={key:0. for key in keys}
        for component in components:
            weight=component['weight'];logits=component['logits']
            need(weight in (-1,1) and set(logits)==keys,'BC signed component/pair coverage')
            seed_draws=[]
            for seed in range(3):
                raw=[logits[(q,seed)] for q in ids]
                predictions=weighted_bc_predictions(raw,[1]*80)
                for q,pred in zip(ids,predictions):observed[(q,seed)]+=weight*float(pred==classes[q])
                seed_draws.append(bc_draws(raw,counts,[classes[q] for q in ids]))
            component_draws.append(weight*np.mean(seed_draws,axis=0))
        sampled[:,j]=np.sum(component_draws,axis=0)
        need(all(observed[key]==cells[method][key]['value'] for key in keys),'observed BC values disagree with unlabeled raw-logit recentering')
    ref = methods.index(reference)
    deltas = means-means[ref]; bd = sampled-sampled[:,ref,None]
    bound = float(np.quantile(np.max(np.abs(bd-deltas), axis=1), .95, method='linear'))
    out = {}
    for j, method in enumerate(methods):
        item = {'mean': float(means[j]), 'pointwise_image_cluster_percentile95': np.quantile(sampled[:,j],[.025,.975]).tolist(),
                'paired_difference_from_reference': float(deltas[j]),
                'paired_pointwise_image_cluster_percentile95': np.quantile(bd[:,j],[.025,.975]).tolist(),
                'paired_simultaneous_family_band95': [0.,0.] if method==reference else [float(deltas[j]-bound),float(deltas[j]+bound)]}
        if fitted and method != reference and (fitted is True or method in fitted):
            item['paired_fitted_seed_interval'] = seed_interval({s: math.fsum(cells[method][(q,s)]['value']-cells[reference][(q,s)]['value'] for q in ids)/80 for s in range(3)})
        out[method] = item
    return {'reference': reference, 'methods': out, 'images':80, 'contexts_per_image':3, 'bootstrap_draws':draws,
            'bootstrap_seed':rng_seed, 'bootstrap_unit':'class-stratified whole image, retaining all contexts and methods',
            'batch_calibration_bootstrap':'weighted unlabeled raw-logit centering recomputed per draw and context before gold comparison; signed K components recalibrated separately' if batch_calibration else None,
            'family':'only the contrasts in this object; no adjustment across other task/K/condition/metric objects',
            'interpretation':'fixed-panel descriptive resampling; finite nonrandom query panel does not certify population claims'}


def observation(result, prediction=None, metric='accuracy'):
    ep = result['episode']; q = ep['query']; pred = result['scores']['sum_prediction'] if prediction is None else prediction
    return ((q['image_id'],ep['seed']), {'class_id':q['class_id'],
            'value':float(pred == q['class_id']) if metric == 'accuracy' else result['scores']['sum'][q['class_id']]})


def insert(cell, result, prediction=None, metric='accuracy'):
    key, value = observation(result,prediction,metric)
    need(key not in cell, 'duplicate image/context observation')
    cell[key] = value


def cost(results):
    rows = list(results)
    need(all(isinstance(r['seconds'],(int,float)) and math.isfinite(r['seconds']) and r['seconds'] >= 0 and
             type(r['input_token_count']) is int and r['input_token_count'] == len(r['input_token_ids']) > 0 for r in rows), 'invalid timing/token cost')
    return {'calls':len(rows),'seconds':math.fsum(r['seconds'] for r in rows),
            'input_tokens':sum(r['input_token_count'] for r in rows),
            'scope':'actual scoring calls including candidate likelihood and greedy diagnostic; GPU peak unavailable'}
