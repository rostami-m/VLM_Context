# Release validation

Checks completed on 2026-11-19:

- All 47 CPU tests passed, covering answer/end-token scoring, strict parsing, calibration, frozen schedules, target budgets, dataset-role separation, policy selection, continual replay access, controller triggers, and checkpoint lineage.
- All package modules compile and import. All 23 experiment configurations resolve to callable runners.
- All 117 released evidence/protocol files match their recorded output checksums; compressed payloads also match their decompressed checksums.
- All 57,600 Q05 per-image records reproduce the 2,880 archived scoring/confusion cells exactly. The 747 absolute core replicate metrics reproduce their 249 aggregate means.
- All 17 corresponding manuscript table comparisons match at printed precision, covering 129 printed numerical rows. Both curve figures regenerate; the core figure includes the actual individual-replicate traces.
- Exact reconstruction was checked for 1,304 Shapes images, 2,360 continual-learning CIFAR-100 images, 816 different-rule images, and 66 required VQAv2 images. Decoded pixels and, where specified by the protocol, encoded PNG hashes match.
- A local MPS check loaded the pinned Qwen2-VL-2B snapshot, ran frozen inference, completed two connector-adapter updates, saved/reloaded the adapter, and evaluated K=0 standard and K=16 remapped episodes with independent full-forward, stepwise and native-generation checks. Complete-answer loss, frozen-base integrity and checkpoint checks passed. These are explicitly marked technical smoke runs.

These checks validate the release interfaces and deterministic reconstruction of the saved evidence. They do **not** constitute a fresh run of the full fitting/model grid. The full results remain those of the archived experiments; hardware-dependent new training or inference may differ. Saved resampling intervals are reproduced from the released analyses rather than regenerated from unavailable raw model outputs.

Install from a source checkout with `python -m pip install -e '.[analysis]'`, run `python -m unittest discover -s tests -v`, then `python -m vlm_context regenerate --output runs/paper`. The CPU workflow performs these checks in GitHub Actions. Model runs require the separately documented environments, data and weights.
