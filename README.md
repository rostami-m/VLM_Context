# VLM Context

Code and numerical evidence for the revised study of multimodal in-context collapse and CircA adaptation.

The release supports two workflows:

- **Regenerate the reported analyses on CPU.** The per-example records in `data/analysis/` supply generated answers, candidate scores, and paired outcomes. The reducers recalculate predictions, tables, curves, and uncertainty estimates and compare them with the reference summaries in `data/paper/`. This requires neither model weights nor access to the original experiment folders.
- **Run the model experiments.** Portable runners use the frozen episode schedules, image manifests, exact adapter targets, and native scoring implementations in this repository. These runs require the specified model weights, source datasets, and sufficient accelerator memory. Retraining is a new scientific run; identical outputs across hardware are not assumed.

## Start with the numerical evidence

Use Python 3.10 or newer from a source checkout:

```bash
git clone https://github.com/rostami-m/VLM_Context.git
cd VLM_Context
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[analysis]'
python -m vlm_context regenerate --output runs/paper
python -m unittest discover -s tests -v
```

Each output directory preserves its evidence. Choose a new directory for a changed configuration. A `--smoke` model run is a small technical check and is never a substitute for a reported experiment.

## Experiments and paper correspondence

| Experiment | Configuration / entrypoint | Paper evidence |
| --- | --- | --- |
| Natural versus episodic objectives and sparse adaptation locations | `configs/q05.json` | Objective/location curves, exact target budgets, scoring and invalid-output diagnostics |
| Shapes-trained joint adapter, matched late adapter, transfer and mapping controls | `configs/core.json` | Main classification transfer, held-out mappings and tokens, counterfactual and component controls |
| Four-block depth scan | `configs/depth.json` | Finer adaptation-location comparison |
| Calibration, CAMA, DARA, source-only LoRA and frozen visual features | `configs/baselines.json`, `dara.json`, `vision_features.json` | Separate paired baseline episodes and budget comparison |
| Native attention, activation/path interventions and support position | `configs/attention.json`, `mechanism.json`, `position.json` | Descriptive attention, bounded interventions, intact labeled-pair reorderings |
| VQAv2, TextVQA, ScienceQA and the different-rule task | `configs/q11_*.json` | Bounded nonclassification transfer and retention panels |
| Qwen2.5-VL and InternVL3 adaptation | `configs/replication_*.json` | Three-seed connector/late comparison with the balanced training mixture |
| IDEFICS2, Mantis and OpenFlamingo | `configs/native_*.json` | Frozen CIFAR endpoint probes and native output diagnostics |
| Gate | `configs/gate_2b.json`, `gate_7b.json` | Development-selected first-crossing policy and confirmation costs |
| Inject | `configs/inject.json` | Reader qualification, disjoint difference-vector extraction, controls and cached explicit context |
| Continual adaptation | `configs/continual.json` | Four exactly matched 12.8M-parameter arms, three orders and three seeds |
| Prompting/consolidation controller | `configs/controller.json` | Three policies, one task order and three seeds |

See [the CPU analysis guide](docs/ANALYSIS.md) for released inputs, statistical units, bootstrap settings, and analysis coverage. [The reproduction instructions](docs/REPRODUCING.md) describe model environments, data reconstruction, training and evaluation commands. [Native model instructions](docs/NATIVE_MODELS.md) cover the additional model families and open tasks. [Policy experiments](docs/POLICIES.md) describes Gate, Inject and the sequential studies. Numerical source identities and transformations are recorded in [the data provenance manifest](data/PROVENANCE.json).

The [release validation record](docs/VALIDATION.md) describes CPU tests, reconstruction from raw numerical records, numerical table checks, and earlier image reconstruction and bounded native training/reload checks. The analysis release does not rerun the full model grid.

## Evaluation contract

Candidate likelihood scores include the entire answer and exactly one native end token. Sum likelihood, mean-token likelihood, strict generation parsing, and the diagnostic permissive parser are separate measures. Every classification model uses the same strict grammar: the entire answer must be an allowed label after case and whitespace normalization, optionally followed by `.`, `!`, or `?`. Arbitrary prose, multiple answers, and option echoes remain invalid. Native results additionally report case/punctuation-sensitive exact-string compliance as a separate diagnostic. Invalid generated answers stay in the denominator. Natural-label robustness and learning arbitrary demonstrated labels are separate outcomes.

The main fixed panels reuse query images across contexts and fitting seeds. Resampling intervals describe these panels; repeated contexts do not create additional independent images. Three-seed variation is reported separately. Baseline episodes, main transfer episodes, and task-specific objective experiments are kept distinct. CPU regeneration rebuilds the analyses from released records with the original resampling settings, including refitting batch calibration inside each bootstrap sample. Reference summaries check the reconstruction rather than supply precomputed interval endpoints. The analysis guide identifies each source level and its limits.

Precision is experiment specific. Most final Qwen2-VL classification, Gate, Inject and intervention evaluations use a BF16-loaded base upcast to FP32 while retaining trained adapter values. The matched continual study uses BF16 inference. The separate controller uses BF16 training and value-preserving FP32 inference. Native family settings are specified separately. A global dtype substitution changes the experiment.

This release contains the retained implementations and evidence. It omits obsolete campaign launchers, local machine paths, internal review gates, model weights, and raw third-party datasets. Dataset and model access follow the original providers' terms. The code is distributed under the [MIT license](LICENSE).
