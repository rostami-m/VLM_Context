"""Accepted support-order experiment and exact within-query swap constraints."""
from collections import Counter
import hashlib,json,math,random
from vlm_context.io import require


CONDITIONS = ['balanced_2_2_2_2', 'majority_5_1_1_1']



VARIANTS = ['baseline', 'wrong_endpoint', 'interior_sham']



VERSION = 'position-native-v1'



def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()



def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()



def validate_triplet(t, manifest, role):
    images = manifest['images']
    query = t['query_id']
    require(query in manifest['roles'][role], 'incorrect query role')
    require(t['seed'] in (0, 1, 2), 'unknown seed')
    require(t['condition'] in CONDITIONS, 'unknown frequency condition')
    require(sorted(t['option_order']) == [0,1,2,3], 'option order not a permutation')
    variants = t['support_variants']
    require(set(variants) == set(VARIANTS), 'variant inventory')
    base, endpoint, sham = [variants[v] for v in VARIANTS]
    require(len(base) == len(set(base)) == 8, 'eight unique support images required')
    require(query not in base, 'query support leakage')
    require(set(base) <= set(manifest['roles']['evaluation_support']), 'wrong support role')
    require(all(sorted(v) == sorted(base) and v[0] == base[0] for v in variants.values()),
            'support multiset or first image changed')
    counts = Counter(images[i]['class_id'] for i in base)
    require({str(k):v for k,v in counts.items()} == {str(k):v for k,v in t['support_class_counts'].items()}, 'count metadata')
    true = images[query]['class_id']
    wrong = t['wrong_class']
    require(wrong in range(4) and wrong != true, 'invalid wrong class')
    require(images[base[-1]]['class_id'] == true and images[endpoint[-1]]['class_id'] == wrong,
            'endpoint class constraint')
    require(images[base[0]]['class_id'] not in (true, wrong), 'first role collision')
    donor, interior = t['donor_original_position'], t['sham_position']
    require(donor in range(1,7) and interior in range(1,7) and donor != interior, 'swap locations')
    expected = list(base)
    expected[donor], expected[-1] = expected[-1], expected[donor]
    require(endpoint == expected, 'endpoint is not the exact designated swap')
    expected = list(base)
    expected[donor], expected[interior] = expected[interior], expected[donor]
    require(sham == expected and sham[-1] == base[-1], 'sham not exact designated swap')
    require(counts == Counter({c:2 for c in range(4)}) if t['condition'] == CONDITIONS[0]
            else counts == Counter({c:5 if c == wrong else 1 for c in range(4)}), 'condition counts')
    require(t['majority_classes'] == [c for c in range(4) if counts[c] == max(counts.values())], 'majority ties')



def episode(t, manifest, variant, role):
    labels = manifest['class_names']
    clean = lambda i: {k: manifest['images'][i][k] for k in ('image_id','class_id','content_sha256')}
    return {'protocol_version': VERSION, 'stage': role, 'k': 8,
            'query': clean(t['query_id']), 'supports': [clean(i) for i in t['support_variants'][variant]],
            'seed': t['seed'], 'seed_roles': {'supports':t['seed'],'mapping':None,'options':t['seed']},
            'sampler':'frozen_CPU02_position_triplet', 'condition':'standard',
            'frequency_condition':t['condition'], 'position_variant':variant,
            'label_space':labels, 'option_order':t['option_order'],
            'displayed_options':[labels[i] for i in t['option_order']],
            'native_scoring_validation':'independent full and stepwise paths'}



def expand(triplets, manifest, role):
    records = []
    for t in triplets:
        validate_triplet(t, manifest, role)
        identity = digest(t)
        order = list(VARIANTS)
        random.Random(int(digest(['execution-order',identity]),16)).shuffle(order)
        for variant in order:
            records.append({'record_id':digest([identity,variant]), 'triplet_id':identity,
                            'triplet':t, 'variant':variant,
                            'episode':episode(t,manifest,variant,role)})
    return records



def extended_validation(result):
    verification = result['independent_validation']
    require(verification is not None,'native smoke verification absent')
    require(verification['generation_matches_transformers_generate'] is True,'native generation equivalence absent')
    arrays = [result['per_token_label_log_probabilities'], verification['full_forward_token_logps'],verification['independent_stepwise_token_logps']]
    expected = [len(c['scored_token_ids']) for c in result['candidate_tokens']]
    require(all([len(v) for v in source] == expected for source in arrays),'candidate or token trace truncated')
    require(all(math.isfinite(v) for source in arrays for trace in source for v in trace),'nonfinite score')
    sums = [[math.fsum(v) for v in source] for source in arrays]
    predictions = [max(range(4),key=lambda i:source[i]) for source in sums]
    require(len(set(predictions)) == 1,'cached/full/stepwise primary argmax mismatch')
    return {'candidate_lengths':expected,'primary_argmax_all_three_agree':True,
            'all_trace_values_finite':True,'generation_matches_transformers_generate':verification['generation_matches_transformers_generate']}



def validate_result(result, record, smoke=False):
    from vlm_context.protocol import parse_diagnostics
    require(result['status']=='ok' and result['episode']==record['episode'],'record/episode mismatch')
    require(len(result['candidate_tokens']) == len(result['per_token_label_log_probabilities']) == 4,'candidate count')
    for candidate, values in zip(result['candidate_tokens'],result['per_token_label_log_probabilities']):
        require(len(values)==len(candidate['scored_token_ids']) and all(math.isfinite(v) and v<=0 for v in values),'scorer trace length or finite failure')
        require(candidate['scored_token_ids'] == candidate['label_token_ids']+[result['eos_token_id']],'EOS convention changed')
    expected_sums=[math.fsum(v) for v in result['per_token_label_log_probabilities']]
    require(len(result['scores']['sum'])==len(result['scores']['mean'])==4,'sum/mean candidate inventory')
    require(all(abs(a-b)<1e-10 for a,b in zip(expected_sums,result['scores']['sum'])),'primary score reduction')
    require(result['scores']['sum_prediction']==max(range(4),key=lambda i:expected_sums[i]),'primary argmax')
    expected_means=[v/len(trace) for v,trace in zip(expected_sums,result['per_token_label_log_probabilities'])]
    require(all(abs(a-b)<1e-10 for a,b in zip(expected_means,result['scores']['mean'])),'secondary mean reduction')
    require(result['scores']['mean_prediction']==max(range(4),key=lambda i:expected_means[i]),'secondary mean argmax')
    require(result['parser']==parse_diagnostics(result['greedy_answer'],record['episode']['label_space'],
                                               natural_labels=record['episode']['label_space']),'strict/legacy parsing changed')
    require(0<len(result['greedy_token_ids'])==len(result['greedy_token_log_probabilities'])<=8,'greedy token/logp trace count')
    require(all(type(v) is int and v>=0 for v in result['greedy_token_ids']) and
            all(math.isfinite(v) and v<=0 for v in result['greedy_token_log_probabilities']),'invalid greedy token/logp trace')
    require(len(result['per_image_visual_token_counts']) == 9,'visual image count')
    if smoke:
        extended_validation(result)
