# Gate, Inject and sequential experiments

Run commands below from the repository root after installing the main Qwen environment described in `REPRODUCING.md`. Set `MODEL_DIR` to the pinned local Qwen2-VL-2B snapshot and `DATA_ROOT` to your image bank directory. The examples use CUDA; the runtime also accepts `mps` and `cpu`, with memory and runtime depending on the device.

## Image banks

```bash
python -m vlm_context prepare-data --task shapes4 --data-root "$DATA_ROOT"
python -m vlm_context prepare-data --task cifar4 --data-root "$DATA_ROOT"
python -m vlm_context prepare-data --task continual --data-root "$DATA_ROOT"
```

Continual preparation retrieves pinned CIFAR-100 Parquet files and reconstructs the exact 2,360-image manifest, verifying raw pixels, resized pixels and PNG hashes. Existing official Parquet files may be supplied with `--parquet-root`. Reserved exclusion images are represented in the manifest; the runners restrict training and evaluation to the declared roles.

## Gate

```bash
python -m vlm_context run --config configs/gate_2b.json \
  --model-dir "$MODEL_DIR" --data-root "$DATA_ROOT" \
  --device cuda --output runs/gate_2b
```

The default run first evaluates the development grids, then selects a threshold separately for each label condition. Ties favor a smaller mean selected prefix, then the smaller threshold. Confirmation visits prefixes in the order 1, 2, 4, 8, 16 and stops at the first copy rate **strictly greater** than the selected threshold. It returns the preceding safe prefix, including zero. Each copy-rate denominator is 40, including invalid responses.

The final query labels are used only after the stopping decision, for the fixed-prefix and policy comparisons. An oracle prefix is a retrospective analysis column and cannot drive selection. Both probe and prediction costs are retained. The 7B command uses `configs/gate_7b.json` and the separately pinned 7B model snapshot.

## Inject

First train the two Shapes-only Q05 readers for all three seeds. These source fits are reused on CIFAR; the reader is not retrained on the confirmation task.

```bash
for ARM in connector_episode_remap uncapped_all_episode_remap; do
  for SEED in 0 1 2; do
    python -m vlm_context run --config configs/q05.json --mode train \
      --task shapes4 --arm "$ARM" --seed "$SEED" \
      --model-dir "$MODEL_DIR" --data-root "$DATA_ROOT" --device cuda \
      --output "runs/q05/shapes4/$ARM/seed_$SEED"
  done
done
python -m vlm_context run --config configs/inject.json \
  --model-dir "$MODEL_DIR" --data-root "$DATA_ROOT" \
  --device cuda --output runs/inject
```

`configs/inject.json` names these checkpoint paths. If you use another directory, copy the configuration and change its `reader_checkpoints`. A fit must have all 150 updates, the matching source task/arm/seed, its full checkpoint receipt, and unchanged checkpoint hashes. Smoke fits are rejected.

The complete runner evaluates the frozen, connector and uncapped readers on development queries. Qualification requires at least 60% supported accuracy and a ten-point gain over zero-shot, pooled over the three fixed seed panels. It selects the best qualified reader per task, with parameter-count and declared-order tie breaks. No qualifying reader is a valid bounded negative outcome.

For each block, eight disjoint extraction probes provide supported-minus-zero-shot states at the last prefill position. Layer candidates are 6, 13 and 20; scale candidates are 0.25, 0.5 and 1. Development accuracy selects the configuration, breaking ties by lower layer and then lower scale. Confirmation retains difference, raw-mean, zero and shuffled vectors, zero-shot, ordinary explicit context, and explicit support with native KV caching. The raw-mean control uses scale one. Extraction and cache setup costs are saved separately from query costs.

The reported layer 20 and scale 0.25 are outcomes of the archived development selection, not values chosen using final queries. Fresh model runs execute the selection again and preserve their own results.

## Continual learning and the controller

```bash
python -m vlm_context run --config configs/continual.json \
  --model-dir "$MODEL_DIR" --data-root "$DATA_ROOT" \
  --device cuda --output runs/continual
python -m vlm_context run --config configs/controller.json \
  --model-dir "$MODEL_DIR" --data-root "$DATA_ROOT" \
  --device cuda --output runs/controller
```

The continual command runs joint, late, all-depth and late-plus-replay adapters over three orders and three initialization seeds: 36 streams. Each adapter has exactly 12.8M trainable parameters. Five tasks reuse labels A–D without a rendered task identifier. Each task releases 192 training images and receives 80 updates. Replay retains at most 16 images and occupies 16 updates when memory is available. AdamW state resets at task boundaries; adapter weights persist. The base and inference remain BF16, and LoRA values remain FP32.

The controller is a separate nine-stream experiment: prompt-only Gate, fixed consolidation at prefix eight, and consolidation after two consecutive Gate decisions underuse the available prefix. Each policy runs order zero with three seeds. There is at most one 80-update event per task, using only the released prefix and retained replay. Gate reruns after an update. Query outcomes never drive the trigger. Training uses the BF16 base and FP32 adapters; inference upcasts the represented base values to FP32 without changing adapter values.

Online accuracy permits the policy-selected current-task support. All retained-task matrices are zero-shot. Final average accuracy and backward transfer are calculated from those matrices. Probe, prediction, reload and training times are recorded separately. The available prefix budget must not be described as the selected number of demonstrations.

Use a fresh output root. `--smoke` shortens the technical checks; it does not reproduce the full stream grid or support an efficacy claim.
