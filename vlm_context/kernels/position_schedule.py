"""Deterministic position/frequency triplet generation."""
import random
from vlm_context.kernels.gate.common import digest as canonical_hash


def position_triplet(images, support_pool, query, seed, condition):
    rng = random.Random(int(canonical_hash(["position-v1", query, seed, condition]), 16))
    true = images[query]["class_id"]
    wrong = rng.choice([c for c in range(4) if c != true])
    counts = {c: (5 if c == wrong else 1) if condition == "majority_5_1_1_1" else 2 for c in range(4)}
    pool = {c: sorted(i for i in support_pool if images[i]["class_id"] == c) for c in range(4)}
    chosen = {c: rng.sample(pool[c], counts[c]) for c in range(4)}
    # Baseline endpoint is query class; hold first fixed across endpoint/sham swaps.
    last = chosen[true].pop()
    first_class = rng.choice([c for c in range(4) if c not in (true, wrong)])
    first = chosen[first_class].pop()
    donor = chosen[wrong].pop()
    middle = [i for ids in chosen.values() for i in ids]
    rng.shuffle(middle)
    middle.insert(rng.randrange(len(middle) + 1), donor)
    baseline = [first, *middle, last]
    donor_index = baseline.index(donor)
    endpoint = list(baseline)
    endpoint[donor_index], endpoint[-1] = endpoint[-1], endpoint[donor_index]
    sham_index = rng.choice([i for i in range(1, 7) if i != donor_index])
    sham = list(baseline)
    sham[donor_index], sham[sham_index] = sham[sham_index], sham[donor_index]
    variants = {"baseline": baseline, "wrong_endpoint": endpoint, "interior_sham": sham}
    if not all(sorted(ids) == sorted(baseline) and ids[0] == first for ids in variants.values()):
        raise ValueError("intervention changes support multiset or first position")
    if len(set(baseline)) != 8 or query in baseline:
        raise ValueError("intervention support duplication or query leakage")
    if images[endpoint[-1]]["class_id"] == true or sham[-1] != baseline[-1]:
        raise ValueError("endpoint/sham semantic failure")
    return {"query_id": query, "seed": seed, "condition": condition,
            "support_variants": variants, "support_class_counts": counts,
            "wrong_class": wrong, "majority_classes": [c for c, n in counts.items() if n == max(counts.values())],
            "donor_original_position": donor_index, "sham_position": sham_index,
            "option_order": rng.sample(list(range(4)), 4),
            "nearest_image_class": None,
            "nearest_status": "requires frozen-model vision features; preserve collisions"}
