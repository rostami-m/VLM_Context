"""Paired fitted/context-seed estimates from released core observations."""
from collections import Counter,defaultdict
import itertools
import math
import statistics
from .io import require
T_DF2=4.3026527299
METRICS=('reference_accuracy','displayed_accuracy','mean_reference_accuracy','strict_reference_accuracy','legacy_reference_accuracy','invalid')
def triplet(values, fitted=True):
    require(set(values) == {0,1,2} and all(type(v) in (int,float) and math.isfinite(v) for v in values.values()), 'three finite distinct replicate estimates required')
    xs = [values[i] for i in range(3)]; mean = statistics.mean(xs); sd = statistics.stdev(xs)
    radius = T_DF2 * sd / math.sqrt(3)
    return {'paired_replicate_estimates': [{'seed': i, 'estimate': values[i]} for i in range(3)],
            'estimate': mean, 'sample_sd': sd, 'range': [min(xs),max(xs)],
            'unadjusted_t95_df2': [mean-radius,mean+radius], 'df': 2,
            'replicate_scope': 'fitting/context jointly vary' if fitted else 'context/support variation only; no fitted model',
            'interval_scope': 'conditional on fixed query images; n=3 and normal-replicate approximation, unadjusted descriptive interval, not population certification'}

def index_rows(rows):
    cells = defaultdict(dict)
    for r in rows:
        key = (r['arm'],r['task'],r['k'],r['condition']); identity = (r['image_id'],r['seed'])
        require(identity not in cells[key], 'duplicate image/seed row')
        require(type(r['seed']) is int and r['seed'] in (0,1,2) and r['class_id'] in (0,1,2,3), 'invalid seed/class')
        require(all(type(r[m]) in (int,float) and math.isfinite(r[m]) and 0 <= r[m] <= 1 for m in METRICS), 'invalid finite accuracy')
        cells[key][identity] = r
    for key, values in cells.items():
        require(len(values) == 240, 'exact 80 images times three seeds required')
        images = [{i for i,seed in values if seed == s} for s in range(3)]
        require(len(images[0]) == 80 and images[0] == images[1] == images[2], 'replicate query-image panels differ')
        for seed in range(3): require(Counter(r['class_id'] for (i,s),r in values.items() if s == seed) == {c:20 for c in range(4)}, 'class balance differs')
        require(all(len({values[i,s]['class_id'] for s in range(3)}) == 1 for i in images[0]), 'same image class differs across seeds')
    return cells

def definitions(keys):
    """The same declared contrast family as the frozen original image reducer."""
    keys = set(keys); out = []; seen = set()
    def add(kind, left, right):
        value = (kind,left,right,'reference_accuracy')
        if left in keys and right in keys and value not in seen: seen.add(value); out.append(value)
    for arm,task,k,condition in sorted(keys):
        key = arm,task,k,condition
        if k and condition == 'standard': add('standard_decline',(arm,task,0,condition),key)
        if k and condition.startswith('remap_'): add('remap_gain',key,(arm,task,0,condition))
        if arm != 'frozen_joint': add('adapter_minus_frozen',key,('frozen_joint',task,k,condition))
        if arm.endswith('_episode_remap') and not arm.startswith('q05_'): add('episodic_minus_natural',key,(arm.removesuffix('_episode_remap')+'_natural',task,k,condition))
        if arm.startswith('joint_uncapped_r16_'):
            objective = arm.removeprefix('joint_uncapped_r16_')
            add('matched_joint_minus_late',key,('late_parameter_matched_r34_35_'+objective,task,k,condition))
            for region in ('connector','early','mid'): add('full_joint_minus_ablation',key,('joint_without_'+region+'_r16_'+objective,task,k,condition))
        if arm == 'joint_uncapped_r16_natural': add('context_training_minus_no_demo',key,('joint_natural_no_demonstrations',task,k,condition))
        if arm == 'unrestricted_q05_natural': add('unrestricted_natural_minus_episodic',key,('q05_uncapped_all_episode_remap',task,k,condition))
        if arm.startswith('depth_window_'): add('window_minus_connector_different_budget',key,('depth_connector_r8_separate_budget',task,k,condition))
        if condition in ('intact_pair_order_shuffle','independently_shuffled_support_labels','deliberately_inconsistent_support_labels'):
            add('corrupted_minus_intact',key,(arm,task,k,'remap_unseen_label_strings'))
    for left,right in itertools.combinations(sorted({k[0] for k in keys if k[0].startswith('depth_window_')}),2):
        for arm,task,k,condition in sorted(keys):
            if arm == left: add('window_pairwise',(left,task,k,condition),(right,task,k,condition))
    return out

def paired(left, right, metric):
    require(set(left) == set(right), 'paired image/seed inventories differ')
    require(all(left[i]['class_id'] == right[i]['class_id'] for i in left), 'paired class differs')
    return {seed: statistics.mean(left[i][metric]-right[i][metric] for i in left if i[1] == seed) for seed in range(3)}

def reduce_rows(rows, original):
    cells = index_rows(rows); family = definitions(cells)
    identity = lambda r: (r['kind'],tuple(r['left']),tuple(r['right']),r['metric'])
    original_contrasts = {identity(r):r for r in original['paired_contrasts']}
    require(len(original_contrasts) == len(original['paired_contrasts']) and set(original_contrasts) == set(family), 'original declared image-contrast family differs')
    original_cells = {(r['arm'],r['task'],r['k'],r['condition']):r for r in original['cells']}
    require(len(original_cells) == len(original['cells']) and set(original_cells) == set(cells), 'original image-cell family differs')
    summaries = []; contrasts = []
    for key, values in sorted(cells.items()):
        metrics = {}
        for metric in METRICS:
            stats = triplet({seed:statistics.mean(r[metric] for (i,s),r in values.items() if s == seed) for seed in range(3)},key[0] != 'frozen_joint')
            require(abs(stats['estimate']-original_cells[key][metric]) <= 1e-12, 'independent raw cell mean differs from image analysis')
            metrics[metric] = stats
        summaries.append({'arm':key[0],'task':key[1],'k':key[2],'condition':key[3],'unique_query_images':80,'metrics':metrics})
    for kind,left,right,metric in family:
        stats = triplet(paired(cells[left],cells[right],metric), any(key[0] != 'frozen_joint' for key in (left,right)))
        require(abs(stats['estimate']-original_contrasts[kind,left,right,metric]['estimate']) <= 1e-12, 'independent paired mean differs from image analysis')
        contrasts.append({'kind':kind,'left':left,'right':right,'metric':metric,'unique_query_images':80,**stats})
    return {'cells':summaries,'paired_contrasts':contrasts,'image_estimator_unchanged':True,
            'unadjusted_multiplicity':True,'seed_and_image_intervals_are_different_estimands':True,
            'population_certification_enabled':False}
