# CPU analysis from saved experiment outputs

Run the numerical analysis from a source checkout, with no model downloads or GPU:

```bash
python -m vlm_context analyze --output runs/analysis
python -m vlm_context regenerate --output runs/paper --no-figures
```

The `analyze` command recomputes the core, baseline, intervention, open-task,
continual and controller analyses. The `regenerate` command additionally recomputes
the task-specific adaptation, native-model and position analyses before creating
all paper tables. For figures,
omit `--no-figures` and install the `analysis` extra. These commands replay the saved
experiment outputs; they do not rerun pretrained models or refit adapters.

## Evidence and estimators

| Evidence | Public input | CPU computation |
|---|---|---|
| Main task-specific adaptation | `data/analysis/q05_raw.jsonl.gz` | Complete-answer token sums and means, normalized strict parsing, all per-example predictions, 2,880 cells and confusion matrices |
| Core transfer, depth and component controls | `data/analysis/core_raw_scores.json.gz` | 59,760 complete-answer score traces and generated answers; 249 cells, 615 paired image contrasts, per-seed estimates and t intervals, counterfactual response statistics |
| Matched baseline comparison | `data/analysis/paired_families.json.gz`, `intervention_raw_scores.json.gz` | Paired endpoint and K-gain families; all three contexts remain with each query; raw candidate logits drive batch-calibration refits inside every draw |
| Frozen interventions, Gate and Inject | Same paired-family inputs and score archive | Per-query accuracy/log-score outcomes, paired image bands and fitted/context-seed intervals where defined |
| Open tasks and different-rule tests | `data/analysis/q11_outcomes.json.gz` | 3,200 saved answers rescored from generation plus annotator answers or candidate token log probabilities; paired rule sides, image bootstraps and seed intervals |
| Continual learning | `data/analysis/continual_raw_scores.json.gz` | 43,200 matrix predictions, all 36 streams, AA/BWT, six paired image intervals, and order-by-seed variation |
| End-to-end controller | `data/analysis/controller_raw_scores.json.gz`, `controller_streams.json.gz` | 32,400 matrix/online predictions; nine streams, policy means, paired differences, event counters and summed recorded costs |
| Additional native-model families and replication | `data/analysis/native_records.json.gz` | Native token-score classification, one shared generated-answer grammar, separately named exact-string compliance and confusion matrices |
| Position interventions | `data/analysis/position_raw.json.gz` | 1,440 saved predictions and candidate scores; position contrasts and copy diagnostics |

The paired-family files contain elementary image/context outcomes, not confidence
intervals. `paired_family_destinations.json.gz` identifies the explicitly defined
comparison family for each calculation. Candidate traces are included separately
for auditing scoring and generation. Baseline generation diagnostics cover the
actual baseline comparison episodes, including the joint and matched-late bridge;
calibrated and nearest-neighbor predictors have no invented generated answers.
The score archive contains 42,578 original records, including 18 content-free
calibration records excluded from classification denominators. Its baseline
diagnostics contain 100 method/task/condition/K cells, each with 80 queries and
three contexts, and are written to `analysis/baseline_diagnostics.json` under a
regeneration output directory (or `baseline_diagnostics.json` under an `analyze`
output directory). Frozen and CAMA generations were decoded from their saved
token IDs using the original tokenizer; `intervention_manifest.json` records the
decoder revision, hash and settings. Other records retain the saved generated text.

The bundled `data/paper/` summaries are numerical comparison targets and containers
for fixed protocol descriptions and recorded measurements.
`core_image_reference.json.gz` is also a comparison target, not an estimator input.
The analysis replaces
the recomputed result fields before table generation and fails on disagreement.
Its output includes every recomputed family, diagnostics, input SHA256 hashes and
verification counts. Timing, cache setup and extraction costs remain recorded
measurements; CPU analysis does not measure model speed again.

## Exact resampling conventions

* Baseline, frozen-intervention, Gate and Inject families: 10,000 NumPy multinomial
  draws with RNG seed 271828. Each draw samples 20 images within each of four
  classes, retaining the same three contexts and all methods. A 95th-percentile
  maximum absolute centered deviation gives each named family's simultaneous
  band. Each family is separate; this is not paper-wide multiplicity control.
* Batch calibration: subtract the weighted unlabeled mean candidate log score
  anew in every resample and context. Signed K-gain components are calibrated
  separately. Gold labels are used only after calibration to score predictions.
* Core: 2,000 stratified image draws using Python's `random.Random`; each contrast's
  RNG seed is the SHA256 of the original estimator version and contrast identity.
  The 80 images retain all three contexts. The discrete percentile indices and
  paired contrast definitions preserve the original estimator.
* Open tasks: 10,000 image draws, Python RNG seed 20261106. Scores are averaged
  across contexts within each image before resampling. Original registered episode
  order is preserved because it determines the deterministic RNG replay. Rule
  success requires both sides on the same image/context. TextVQA and ScienceQA
  explicitly reuse one deterministic frozen K=0 baseline across fitted seeds.
* Continual learning: 2,000 draws within 20 task/class strata, with 20 images per
  stratum. The same 400 images retain all nine order/seed streams. Six comparison
  families use the original RNG seeds 0 through 5.
* Additional native families and replication: 10,000 unstratified image draws
  with Python RNG seed 20261106. The three contexts are averaged within each
  image before resampling; paired comparisons retain the same image/context
  identities. The original episode order and percentile indices are recorded in
  `data/analysis/native_manifest.json`.
* Fitting/context variability: paired sample-SD Student t intervals across the
  three coupled replicates, with two degrees of freedom. These intervals are
  intentionally not clipped to feasible accuracy bounds.

These are descriptive analyses of fixed, deterministically selected panels.
Repeated contexts and streams do not create new independent images. Image
resampling and fitted/context-seed intervals address different uncertainty
sources. The released controller analysis makes no additional population
confidence-interval claim. New training, different hardware, new datasets or
broader architecture coverage are separate scientific reruns.

## Integrity and boundaries

Compact records retain experiment identity, query IDs, seeds, conditions, target
labels, complete candidate token scores or native task answers, and source-record
hashes. The export excludes image pixels, pretrained weights, adapter checkpoints,
private machine paths, transport logs and reviewer correspondence. Public frozen
schedules and dataset manifests identify the corresponding images and episodes.

The original selected records were checked against their source receipts during
export. Public input hashes bind the portable records; source-record hashes refer
to the original archived bytes and do not imply that those private wrappers are
included. Public verification therefore uses the portable inputs and released
reducers, rather than requiring the original machine or execution environment.

The release supports reanalysis of the retained experiment grid. It does not claim
new model inference, complete historical traces from discarded development runs,
or universal transfer beyond the datasets, models and task conditions reported
in the paper.
