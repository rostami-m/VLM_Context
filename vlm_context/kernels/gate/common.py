"""Shared explicit episode/scoring helpers for Gate and Inject."""
from __future__ import annotations
import hashlib,json,math,time
from vlm_context.io import read_json as read, write_json as publish, sha256 as sha
from .policy import require

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def validate_score(result, episode, native=False):
    require(result['episode'] == episode and result['status'] == 'ok', 'episode or status drift')
    rows = result['per_token_label_log_probabilities']
    require(len(rows) == len(episode['label_space']) == 4, 'candidate coverage')
    for c, values in zip(result['candidate_tokens'], rows):
        require(len(c['scored_token_ids']) == len(values) and values and
                all(math.isfinite(v) and v <= 0 for v in values), 'invalid token logps')
    sums = [math.fsum(r) for r in rows]
    means = [v / len(r) for v, r in zip(sums, rows)]
    require(all(abs(a-b) < 1e-6 for a,b in zip(sums,result['scores']['sum'])), 'sum scoring mismatch')
    require(all(abs(a-b) < 1e-6 for a,b in zip(means,result['scores']['mean'])), 'mean scoring mismatch')
    require(result['scores']['sum_prediction'] == max(range(4), key=lambda i: sums[i]), 'argmax mismatch')
    require(result['scores']['mean_prediction'] == max(range(4), key=lambda i: means[i]), 'mean argmax mismatch')
    require(len(result['greedy_token_ids']) == len(result['greedy_token_log_probabilities']) > 0 and
            all(math.isfinite(v) and v <= 0 for v in result['greedy_token_log_probabilities']), 'invalid greedy trace')
    if native:
        check = result['independent_validation']
        for key in ('full_forward_token_logps', 'stepwise_token_logps'):
            ref = check[key]
            require(len(ref) == 4, 'native reference incomplete')
            require(all(len(a) == len(b) and all(math.isfinite(v) and v <= 0 for v in b) for a,b in zip(rows,ref)), 'native trace shape')
            require(max(abs(x-y) for a,b in zip(rows,ref) for x,y in zip(a,b)) <= .05, 'native trace mismatch')
            require(max(range(4), key=lambda i: math.fsum(ref[i])) == result['scores']['sum_prediction'], 'native argmax mismatch')
        require(check['native_greedy_token_ids'] == result['greedy_token_ids'], 'native generation mismatch')


def episode(block, manifest, query_id, k, condition, role):
    require(k in (0,1,2,4,8,16) and condition in ('standard','episode_remap'), 'invalid policy episode')
    labels = manifest['class_names'] if condition == 'standard' else block['mapping']
    def row(i):
        return {k:manifest['images'][i][k] for k in ('image_id','class_id','content_sha256')}
    require(query_id in manifest['roles'][role], 'query outside declared role')
    supports = block['nested_support_ids'][str(k)]
    require(query_id not in supports and set(supports) <= set(manifest['roles']['evaluation_support']), 'support leakage')
    return {'task':block['task'],'seed':block['seed'],'k':k,'condition':condition,
            'query':row(query_id),'supports':[row(i) for i in supports],
            'label_space':labels,'option_order':block['option_order'],
            'displayed_options':[labels[c] for c in block['option_order']],
            'query_role':role,'block_id':block['block_id']}


def score(session, ep, loader, native=False, renderer=None, prepared=None):
    from vlm_context.qwen import render_messages
    from vlm_context.qwen import native_control
    from vlm_context.protocol import parse_diagnostics
    ev = session.evaluator
    ev.synchronize();start=time.monotonic()
    if prepared is None:
        messages,images=(renderer or render_messages)(ep,loader)
        inputs,meta=ev.encode(messages,images,ep['label_space'])
    else:
        inputs,meta=prepared
    require(meta['input_token_count']<=2048,'prespecified bounded context exceeded')
    result=native_control(ev,inputs,meta,verify=native)
    ev.synchronize()
    text=ev.processor.tokenizer.decode(result['greedy_token_ids'],skip_special_tokens=True)
    result={'status':'ok','episode':ep,**meta,**result,'greedy_text':text,
            'parse':parse_diagnostics(text,ep['label_space']), 'seconds':time.monotonic()-start}
    validate_score(result,ep,native)
    return result
