# Native model and open-task experiments

The native entry points retain each model's processor, prompt template, complete answer plus native terminator scoring, and greedy decoding. They load explicit local model directories and verify the recorded snapshot hashes. There is no model substitution or context truncation. The bundled schedules determine the images, labels, option order, and fitting episodes.

Use a separate environment for each incompatible Transformers version. The files `environments/qwen25.txt`, `internvl.txt`, `idefics.txt`, and `openflamingo.txt` list the packages recorded in the accepted runs. Qwen2.5, InternVL and IDEFICS2/Mantis used Python 3.12; OpenFlamingo used Python 3.10. These are recorded runtime versions, not a claim that every platform supplies identical wheels. The public refactor has CPU protocol checks; the full model grid has not been rerun as part of preparing this release.

Prepare downloaded dataset banks in the primary Qwen environment (`environments/qwen2.txt`, which includes `datasets`), then reuse the same `--data-root` across native model environments. In particular, keep OpenFlamingo's older model dependencies separate from the modern dataset-preparation environment. Synthetic Shapes preparation only needs Pillow.

## Models and environments

| Configuration | Model | Revision | Recorded evaluation dtype |
|---|---|---|---|
| `replication_qwen25.json` | `Qwen/Qwen2.5-VL-3B-Instruct` | `66285546d2b821cf421d4f5eb2576359d3770cd3` | BF16 base, FP32 adapters |
| `replication_internvl.json` | `OpenGVLab/InternVL3-2B-hf` | `cb57a075cb75a2e6d1b668b128d48bb00ae321d2` | BF16 base, FP32 adapters |
| `native_idefics2.json` | `HuggingFaceM4/idefics2-8b` | `2c42686c57fe21cf0348c9ce1077d094b72e7698` | BF16 |
| `native_mantis.json` | `TIGER-Lab/Mantis-8B-Idefics2` | `a1ae928477b92a51b14d619b0932740cb122115b` | FP32 |
| `native_openflamingo.json` | `openflamingo/OpenFlamingo-4B-vitl-rpj3b` | `df8d3f7e75bcf891ce2fbf5253a12f524692d9c2` | FP32 |

For example, install one native environment, then download that exact model snapshot using `huggingface_hub.snapshot_download(repo_id, revision=...)`. Pass the returned directory as `--model-dir`. Downloading and use of model weights remain subject to their original terms. Weights and dataset images are not redistributed here.

```bash
python -m pip install -r environments/qwen25.txt
python -m pip install -e . --no-deps
python -m vlm_context prepare-data --task shapes4 --data-root /data/vlm-context
python -m vlm_context run --config configs/replication_qwen25.json \
  --mode train --arm connector --seed 0 --model-dir /models/qwen25 \
  --data-root /data/vlm-context --output runs/qwen25-connector-seed0 --device cuda
python -m vlm_context run --config configs/replication_qwen25.json \
  --mode evaluate --arm connector --seed 0 --model-dir /models/qwen25 \
  --checkpoint runs/qwen25-connector-seed0/checkpoint \
  --data-root /data/vlm-context --output runs/qwen25-connector-seed0-eval --device cuda
```

Q13 uses 150 updates: 75 natural-label and 75 remap episodes, rank 8, alpha 16, zero dropout, zero weight decay, AdamW learning rate `1e-4`, and gradient clipping at 1. The late arm adapts the last four language blocks. Qwen2.5 connector/late parameter counts are 139,264/1,662,976; InternVL counts are 69,632/1,318,912. Run each adapted arm and frozen baseline with seeds 0, 1, 2. The fitter verifies the masked complete-answer loss, finite gradients, exact frozen-base hashes, and saved-adapter reload behavior.

The frozen family panel uses CIFAR4, K=0 and 16, natural/remapped labels, 20 fixed query images and three context seeds. Prepare CIFAR4 with the shared data command, then select `configs/native_idefics2.json` or `native_mantis.json` with `run --seed 0` (and repeat seeds 1 and 2). IDEFICS2-based scoring validates causal prefixes by slicing the already processed native input, preserving image padding and patch masks.

### OpenFlamingo dependencies

OpenFlamingo requires its pinned upstream implementation, language model, and OpenAI CLIP archive in addition to `--model-dir`:

```text
/data/vlm-context/openflamingo/
  open_flamingo_source/  # Git checkout of mlfoundations/open_flamingo
  language/             # RedPajama snapshot below
  ViT-L-14.pt            # Official OpenAI CLIP archive below
```

Use upstream OpenFlamingo commit `655f693fbfa04cd6e9a987d960654624d48917cf`; install its recorded environment without upgrading Transformers. Download `togethercomputer/RedPajama-INCITE-Base-3B-v1` revision `094fbdd0c911feb485ce55de1952ab2e75277e1e` into `language/`. The exact OpenAI archive URL and SHA256 are in `data/protocols/families/openflamingo_runtime.json`. That file also records the language-model and code-file hashes. The loader preserves OpenAI QuickGELU preprocessing, adds only verified same-storage checkpoint aliases, and preserves the native `<|endofchunk|>` completion convention. On MPS, the pinned GPTNeoX integer position calculation is performed on CPU; its algorithm is unchanged.

## Q11 VQA, retention, and paired different-rule evaluation

Q11 uses the primary Qwen2-VL snapshot and the Shapes-trained core joint adapter for the corresponding seed. The base is loaded in BF16 and then upcast to FP32 before the FP32 adapter is loaded. Exact visual attention is split into query chunks of 256 while retaining all original keys within each image segment. Pixels, prompts and tokens remain unchanged.

```bash
python -m vlm_context prepare-data --task vqav2 --data-root /data/vlm-context
python -m vlm_context prepare-data --task textvqa --data-root /data/vlm-context
python -m vlm_context prepare-data --task scienceqa --data-root /data/vlm-context
python -m vlm_context prepare-data --task rule --data-root /data/vlm-context
python -m vlm_context run --config configs/q11_vqav2.json \
  --arm joint --seed 0 --model-dir /models/qwen2 \
  --checkpoint runs/core-joint-seed0/checkpoint \
  --data-root /data/vlm-context --output runs/q11-vqav2-joint-seed0 --device cuda
```

The data preparer downloads pinned source revisions and verifies the exact saved PNG hashes. VQAv2 uses two recorded validation parquet shards; `--parquet-root` can reuse existing copies after SHA256 verification. No new query selection occurs. TextVQA and ScienceQA preserve their original row indices and native image sizes. The different-rule images are reconstructed from frozen circle positions, colors and backgrounds, and are checked against their saved hashes.

Choose the other `q11_*.json` configurations for TextVQA, ScienceQA and the rule task, and run frozen and joint arms with seeds 0, 1, 2. VQAv2 has 40 confirmation images at K=0/16 per seed, TextVQA and ScienceQA have 100 images at K=0, and the rule task has 40 paired scenes per factorial cell at K=16. VQA uses the leave-one-annotator-out soft score. The rule summary requires both the left-rule and right-rule answers to be correct; individual answer accuracy is not substituted.

All runs require a fresh output directory and save the full episode, native scores/output, runtime and summary. Adapter evaluation requires the complete source fit directory (checkpoint, run metadata, runtime, and fit receipt), with the matching model revision, arm, seed and source task. Checkpoint bytes are verified against the fit receipt; Q13 also binds its recipe and reload evidence by SHA256. Frozen runs reject checkpoints. Smoke-trained adapters can only be evaluated with `--smoke`. `--smoke` evaluates a small technical subset and labels the output accordingly. It is not a complete paper rerun. Q13 fitting and the native family panels require considerably more memory than CPU artifact regeneration; hardware and floating-point kernels can affect numerical trajectories even with the same protocol.
