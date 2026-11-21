# Reproducing the classification and mechanistic experiments

Use a complete source checkout and an editable installation. The schedules and numeric evidence remain in `data/`; standalone wheels are intentionally disabled. Run commands from the repository root. `--output` must name a fresh directory: partial attempts and completed results are preserved. Models, image banks and run outputs are excluded from Git.

The release contains runnable model code and the original numeric evidence. The release checks include CPU checks, artifact reductions, image reconstruction, and the bounded native checks listed in the README. The complete training/inference grids have **not** been rerun as part of preparing this release. Hardware and numerical libraries can affect rerun outputs; the saved precision and independent scoring checks make those differences inspectable.

## Environment and inputs

Use Python 3.12 for the recorded Qwen environment. The environment file records the package versions in the accepted primary runtime. Install a PyTorch build appropriate to the accelerator, and preserve the recorded Transformers/PEFT versions for this implementation. Other model families have separate environments described in [NATIVE_MODELS.md](NATIVE_MODELS.md).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r environments/qwen2.txt
python -m pip install -e '.[analysis]'
python -m unittest discover -s tests -v

python -m vlm_context fetch-model --size 2B --cache-dir models
python -m vlm_context prepare-data --task shapes4 --data-root image_banks
python -m vlm_context prepare-data --task cifar4 --data-root image_banks
python -m vlm_context prepare-data --task fashion4 --data-root image_banks
```

`fetch-model` prints the snapshot path to use as `--model-dir`. The runner checks the complete file hashes in `configs/qwen2_snapshot.json`, including the tokenizer and weight shards. Keep the pinned snapshot intact. The 7B Gate experiment uses `fetch-model --size 7B` and its separate recorded hash inventory.

CIFAR-10 and Fashion-MNIST are fetched at the recorded dataset revisions. The materializer reconstructs the exact original rows, converts to RGB, resizes to 112×112 using the recorded transform, and checks every decoded image hash. Shapes is deterministic and its generated identities, role assignments and pixels must match the archived manifest. Query, training and support roles are checked for identity and content overlap.

For CUDA deterministic matrix operations, set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting Python. Select `--device cuda`, `mps`, or `cpu` explicitly; no device fallback is performed. FP32 evaluation of a 2B model requires substantial memory, and the 7B run requires correspondingly more. CPU execution is supported by the API but is not a practical recommendation for the complete model grid.

## Objective/location study (Q05)

Each of the two in-domain tasks (`shapes4`, `cifar4`) has nine adapter arms and three fitting seeds. Each fit uses 150 updates. Use these arm names:

- `connector_natural`, `connector_episode_remap`
- `early_episode_remap`, `middle_episode_remap`, `late_episode_remap`, `late_natural`
- `historical_capped_all_episode_remap`, `uncapped_all_episode_remap`, `matched_all_depth_episode_remap`

The capped arm is an explicit diagnostic target definition retained in the paper; it is not a default target-selection policy. All public fits use named target inventories and reject extra or missing trainable tensors.

```bash
python -m vlm_context run --config configs/q05.json --mode train \
  --task shapes4 --arm connector_episode_remap --seed 0 \
  --model-dir MODEL_SNAPSHOT --data-root image_banks --device cuda \
  --output runs/q05/shapes4/connector_episode_remap/seed_0

python -m vlm_context run --config configs/q05.json --mode evaluate \
  --task shapes4 --arm connector_episode_remap --seed 0 \
  --checkpoint runs/q05/shapes4/connector_episode_remap/seed_0/checkpoint \
  --model-dir MODEL_SNAPSHOT --data-root image_banks --device cuda \
  --output runs/q05_eval/shapes4/connector_episode_remap/seed_0
```

Repeat over both tasks, all nine arms and seeds 0, 1, 2. For the frozen reference, use `--arm frozen` without `--checkpoint`. Each task/seed evaluation contains 960 episodes: 80 fixed queries × six K values × two label conditions. The output preserves complete candidate-token probabilities, invalid generation diagnostics, prompts, image identities and independent scoring validation. The first scored episode checks cached, full-forward, stepwise and native generation paths.

Adapter training uses a BF16 base with FP32 trainable LoRA parameters. Evaluation first loads the same BF16 base values and then promotes them to FP32. This is distinct from directly loading the original weights in FP32. The optimizer is AdamW, learning rate 10⁻⁴, betas (0.9, 0.999), epsilon 10⁻⁸, weight decay 0.01, LoRA dropout 0.05, batch size one, and no gradient clipping or accumulation. Supervision covers the complete query label and exactly one end-of-turn token; prompt tokens are masked. Training averages cross-entropy across those supervised tokens, whereas classification sums candidate token log probabilities including the native terminator. Checkpoints are the fixed final updates, without outcome-based selection.

## Joint, component, capacity and depth study

`configs/core.json` trains Shapes source-only and evaluates the retained cells on Shapes, CIFAR and Fashion. The accepted retained joint fit names are:

- `joint_uncapped_r16_natural`, `joint_uncapped_r16_episode_remap`
- `late_parameter_matched_r34_35_natural`, `late_parameter_matched_r34_35_episode_remap`
- `joint_without_connector_r16_episode_remap`, `joint_without_early_r16_episode_remap`, `joint_without_mid_r16_episode_remap`
- `unrestricted_q05_natural` (the paired 150-update natural-objective unrestricted diagnostic fit)

The joint and parameter-matched late arms each contain 12,800,000 trainable parameters, with different target locations/ranks. Equal parameter count does not imply equal functional capacity. Joint fits use 250 updates; their exact 18/6 mapping schedules are supplied. The unrestricted natural supplement uses the original 150-update diagnostic schedule and target inventory.

```bash
python -m vlm_context run --config configs/core.json --mode train \
  --task shapes4 --arm joint_uncapped_r16_episode_remap --seed 0 \
  --model-dir MODEL_SNAPSHOT --data-root image_banks --device cuda \
  --output runs/core/joint_uncapped_r16_episode_remap/seed_0

python -m vlm_context run --config configs/core.json --mode evaluate \
  --task fashion4 --arm joint_uncapped_r16_episode_remap --seed 0 \
  --checkpoint runs/core/joint_uncapped_r16_episode_remap/seed_0/checkpoint \
  --model-dir MODEL_SNAPSHOT --data-root image_banks --device cuda \
  --output runs/core_eval/fashion4/joint_uncapped_r16_episode_remap/seed_0
```

Repeat for seeds 0–2 and each task. `core/retained_cells.json` specifies which condition/K cells belong to each reported comparison. The core runner selects those cells, including held-out mappings, unseen label strings, and support-label/order controls where retained. `--arm frozen` selects the frozen joint reference. The `q05_uncapped_all_episode_remap` and `q05_matched_all_depth_episode_remap` evaluation arms reuse the corresponding **Shapes** Q05 checkpoint, including for transfer to CIFAR/Fashion. Independent support-label shuffles do not enforce class consistency and may produce a coherent mapping by chance. Their reference target remains unchanged; coherent counterfactual controls are scored against the displayed alternative mapping.

For equal-budget depth, use `configs/depth.json` and arm names `window_0_3_r8`, `window_4_7_r8`, `window_8_11_r8`, `window_12_15_r8`, `window_16_19_r8`, `window_20_23_r8`, `window_24_27_r8`, or `connector_r8_separate_budget`. Each window has 1,318,912 parameters and uses the same first 150 remapping episodes; the connector is explicitly a separate budget. Train on Shapes and evaluate Shapes or CIFAR with the same `train`/`evaluate` command pattern. The depth schedule fixes the reported query/mapping/K grid.

## Shared-prompt baselines and main-checkpoint bridge

These comparisons use the same A-only classification prompt for all methods, distinct from the main classifier prompt. Use `configs/baselines.json`; `--arm` selects `frozen`, `cama`, `joint`, `late`, `connector_natural`, `connector_episode_remap`, `uncapped_all_episode_remap`, or `unrestricted_q05_natural`. Adapted methods require their matching source-only checkpoint. Run on `cifar4` and `fashion4`, seeds 0–2. Each partition has six panels (K=0,8,16 × standard/remapped labels), with 80 paired queries per panel.

```bash
python -m vlm_context run --config configs/baselines.json --arm frozen \
  --task cifar4 --seed 0 --model-dir MODEL_SNAPSHOT --data-root image_banks \
  --device cuda --output runs/baselines/frozen/cifar4/seed_0
```

Frozen standard-label panels also compute contextual calibration with one gray query per fixed context and batch calibration using the unlabeled deployment panel. Batch calibration is transductive; no ground-truth labels enter either calibration calculation. CAMA is an independent native-model adaptation with its own selector probe and prefill behavior, not a claim to reproduce official task scores.

For DARA, train `--config configs/dara.json --mode train --arm dara_episode_remap` (or `dara_natural`) on Shapes, for each seed, then evaluate on each transfer task with `--checkpoint TRAIN_OUTPUT/checkpoint.json`. Its 96 factors, first-layer slot rule, 150-step schedule, gradient scaling and optimizer are preserved. DARA training uses BF16 base computation with FP32 factors; evaluation uses the matched FP32 cohort.

`configs/vision_features.json` runs the frozen model’s postmerger feature comparator for both transfer tasks and all seeds. It uses the exact 16 supports, mean pools 16×1536 visual tokens in float64, normalizes them, and predicts from cosine nearest-neighbor similarity. K=0 is undefined for this comparator. The source vision model receives no fitted classifier.

## Order and mechanistic controls

- `configs/position.json`: all 480 frozen triplets / 1,440 outcomes; swaps a designated donor into the final position and compares an interior sham while preserving images, labels, counts and first position. This earlier accepted study retains BF16 inference.
- `configs/mechanism.json`: selects one layer from (0,6,13,20,27) using disjoint development queries, then runs clean/shuffled, self/clean/unrelated patches, and matched label/image/instruction attention cuts on fixed confirmation queries. It uses the later FP32 inference path.
- `configs/attention.json`: descriptive all-head attention statistics over 28 layers, 12 heads and K=1,2,4,8,16. It does not select heads from final outcomes.

In the mechanism experiment, `shuffled` applies the fixed cyclic permutation `c → (c+1) mod 4`, with zero-based class indices `c ∈ {0,1,2,3}`, to displayed support classes while retaining the clean query target. Clean/self patches replace the cyclically relabeled prompt’s layer-output residual at the final assistant prefill position, using the clean prompt’s state or the recipient’s own state, respectively. `unrelated_patch` copies the clean residual at the first selected nonspecial instruction token before the first image to that same position in the recipient; the prompts share the causal prefix there. Attention cuts act across heads at one layer, from query-image and `A:`-segment rows to selected support columns. Label cuts use support-label spans; image cuts use the first image tokens of each support, matching its label-token count; instruction cuts match the total count using nonspecial instruction tokens before the first image. The final assistant prefill token is outside these cut rows.

These use the generic `run --config ... --model-dir ... --data-root ... --device ... --output ...` command. `--smoke` requests a small technical check and labels its outputs accordingly. Smoke outputs are never evidence for a full experiment. Gate, Inject and continual/controller commands are described in [POLICIES.md](POLICIES.md).

## Saved evidence and regenerated results

```bash
python -m vlm_context regenerate --output runs/paper_artifacts
```

This CPU command consumes the included per-example evidence, reconstructs diagnostic confusion/accuracy records, repeats the recorded paired statistical analyses, and emits tables, figures and a manifest describing the source level used for each output. It does not run a model. Use `--no-figures` to perform the same numerical analysis without plotting. [The analysis guide](ANALYSIS.md) documents coverage, pairing, RNG settings and the separate image/seed uncertainty views. Model commands produce fresh raw token-level records and summaries in their output folders. To compare a rerun with the archived paper evidence, preserve the configured task/arm/seed and query identities; do not pool repeated contexts as independent images.
