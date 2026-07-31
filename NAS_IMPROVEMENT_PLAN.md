# NAS Improvement Plan

**Status:** All rule-neutral implementation work in Priorities 0-5, the
31 July controller-correction pass, and the post-run architecture-robustness
pass is complete and locally tested. Two
deliberately conditional experiments remain disabled: train-plus-validation
refit pending broad historical ablation, and an ensemble pending explicit
organizer confirmation. The corrected controller still requires a fresh
multi-seed historical/CUDA rerun.

**Date:** 31 July 2026

## 0. Current implementation completion record

This record supersedes prospective descriptions later in the document where
they describe the repository as it existed before implementation.

- [x] Byte-bounded fingerprinting, normalization, shift/flip samples, train
  cache, and validation cache; source dtype is preserved until per-example
  conversion.
- [x] Prior-preserving selection validation and disjoint confirmation
  validation with ordinary and balanced accuracy tracked separately.
- [x] Loss, probability margin, examples/data coverage, deterministic seeds,
  LR, throughput, time, and CUDA peak-memory candidate measurements.
- [x] Logical-batch gradient accumulation with reusable OOM-calibrated search
  microbatches and progressive data cursors.
- [x] Incumbent-safe two-stage recipe race with loss slope,
  train-validation-gap, and confirmation acceptance.
- [x] AdamW controls plus zero-smoothing SGD/Nesterov/cosine.
- [x] Functional augmentation-policy comparison, byte-bounded moment shift,
  and a cheap held-out domain-classification probe.
- [x] Hierarchical family probes and uncertainty-aware promotion before macro
  search.
- [x] Categorical/dense sequence, multi-view, volumetric, coordinate,
  spatial-axis, pre-activation grouped/wide, and dense feature-reuse anchors.
- [x] Scalable Tier-1 search budget and a clock-driven final action queue
  covering every recipe, one tied/efficient challenger, fresh-seed
  continuations, EMA,
  checkpoint averaging, and validated-flip TTA.
- [x] Focused controller, objective, architecture, trainer, data, memory, and
  accelerated end-to-end regression tests.
- [x] Fix the CUDA representation-probe regression exposed by the first
  post-implementation three-dataset run: every controller evaluation now
  places a fresh or CPU-parked model on the controller device before the
  forward pass.
- [x] Make controller validation recursively split an oversized batch after
  CUDA OOM while preserving sample order, and contain a malformed
  representation family instead of aborting the entire dataset.
- [x] Preserve and restore each architecture finalist's best refinement model,
  optimizer, step, and data-cursor state.
- [x] Make confirmation accuracy dominant in final architecture selection;
  adaptive history is now a statistical-tie-only signal.
- [x] Replace the one-example recipe threshold with a bounded
  uncertainty-aware margin on both selection and confirmation.
- [x] Restore the complete feasible size range after family promotion and add
  a genuinely wide deterministic anchor for full-grid families.
- [x] Rotate a final-training attempt early when it fails to recover the
  immutable baseline, while retaining a longer guarded runway for productive
  slow recoveries.
- [x] Prevent SGD cosine decay from rebounding after its initial horizon and
  label scheduler diagnostics accurately.
- [x] Remove candidate-list position from representation, macro-initialization,
  fidelity-round, and refinement seeds; paired architectures now use common
  deterministic seeds and data order at equal fidelity.
- [x] Reserve feasible compact/central/capacity reference points inside family
  quotas and give one still-learning capacity anchor a bounded late-fidelity
  repechage.
- [x] Use train-validation separation and absolute cross-entropy only as
  bounded late-fidelity/tie risk evidence; they cannot overturn a clear
  confirmation-accuracy winner.
- [x] Retain one materially smaller architecture as a dormant sequential
  challenger when the winner has strong overfit/brittleness evidence and the
  smaller model is within eight raw percentage points of accuracy.
- [ ] Run the original three and broader historical datasets with multiple
  seeds after the 31 July corrections; their arrays are not present locally.
- [ ] Complete a successful real-CUDA peak-memory/OOM validation after the
  fix; the local PyTorch runtime is CPU-only.
- [ ] Enable train-plus-validation refit only if leave-one-dataset-out
  ablation shows a general benefit.
- [ ] Enable a complementary-model ensemble only after organizer confirmation.

The exact active architecture is documented in
`submission/NAS_ARCHITECTURE.md`.

**Primary evidence:** The current repository, the competition evaluator, the
original successful three-dataset baseline log at:

`C:\Users\Lennart\.codex\attachments\4a0d3e89-fc84-49a2-802e-4255e45f0318\pasted-text.txt`

the invalid-runtime failure log at:

`C:\Users\Lennart\.codex\attachments\548f438a-769b-4a6f-9a48-37f91aefaa3c\pasted-text.txt`

and the latest successful pre-correction run at:

`C:\Users\Lennart\.codex\attachments\d2ca2821-4622-492f-b61d-bd7cac87de6d\pasted-text.txt`

and the latest supplied post-correction run at:

`C:\Users\Lennart\.codex\attachments\892d552c-77de-4c02-9e05-8b2206a898e9\pasted-text.txt`

The middle log used the wrong runtime and is not evidence of a submission
defect. Device-safe evaluation and OOM splitting remain useful defensive
invariants, but the failure must not be used to justify performance decisions.

The latest successful run scored **4.233 adjusted points**:

| Dataset | Test accuracy | Adjusted score | Runtime |
|---|---:|---:|---:|
| AddNIST / Adaline | 93.280% | +3.379 | 1767.9s |
| Gutenberg | 41.950% | +0.164 | 1775.3s |
| Language / LaMelo | 86.220% | +0.689 | 1773.9s |

Compared with the earlier 5.135 baseline, AddNIST changed by -0.810 raw
points, Gutenberg by -5.717, and Language by +1.280. A single deterministic
run is not sufficient to establish expected performance, but it exposed six
controller-level problems addressed by the checked items above.

### 0.1 Evidence-to-change map for the 31 July correction pass

The following observations come from the latest successful **pre-correction**
run. They are the empirical baseline for the current code; no post-correction
competition-style accuracy run has yet been supplied.

| Weakness | Exact run-log evidence | Implemented correction |
|---|---|---|
| Finalist refinement returned the endpoint instead of the best checkpoint | AddNIST spatial fell from 75.50% to 72.14%; Gutenberg pyramid from 41.26% to 40.96%; Language pyramid from 79.55% to 78.35% | Snapshot and restore each finalist's best model, matching optimizer, step count, and data cursor |
| Adaptive rank history could override both independent holdouts | On AddNIST, spatial was selected at 72.14% selection / 71.26% confirmation over pyramid at 74.94% / 74.68% | Make confirmation accuracy decisive outside a bounded uncertainty band; use loss, combined holdout score, and historical rank only inside that band |
| A statistically tiny recipe gain could replace the incumbent | Gutenberg selected SGD on a gain of only 0.20 percentage points | Require `max(configured floor, 1/N, min(1pp, 0.75 * pooled SE))` on selection and confirmation |
| Proxy subsampling could remove useful capacity before label training | The 12 Gutenberg label candidates contained no C64 spatial model although the earlier 47.667% winner was C64 spatial | Restore every feasible macro size in promoted families, add a deterministic C64 anchor, and select label candidates across log-parameter strata |
| A failed optimizer could consume most final-training time | Gutenberg SGD used 105 of 154 logged epochs without recovering the 41.23% warm baseline; regularized received only 21 epochs | Rotate a sufficiently overfit, stalled attempt early when it has not recovered its starting global best; prefer a regularized alternative |
| Productive attempts could run long after their best, and cosine SGD could rebound | Language peaked at 86.43% around epoch 51 and continued for about 83 epochs; AddNIST trained beyond a cosine horizon of 49 and its LR rose from about `2e-4` to `5.9e-4` | Give productive attempts a guarded long runway, then rotate on a late plateau; freeze cosine at `T_max` and clamp every time-based LR update monotonically |

Expected benefits are lower architecture-selection regret, less destructive
overtraining, more reliable recipe replacement, and better coverage of
unknown capacity requirements. The main risk is spending label-training work
on weak size strata or rotating a genuinely slow recovery. Those risks are
bounded by family promotion, fixed finalist count, conservative rotation
conditions, an immutable global best checkpoint, and the external clock.

Local CPU verification on 31 July passed:

- bytecode compilation for `submission/` and the affected tests;
- `test_controller_refinements.py`, `test_search_objective.py`,
  `test_trainer_anytime.py`, `test_architecture_fixes.py`,
  `test_data_processor.py` (7/7), and `test_memory_safety.py`;
- the accelerated Tier-1 end-to-end pipeline, including complete ordered
  prediction;
- a short real-clock Tier-3 `run_pipeline()` smoke test.

The Miniconda runtime does not include pytest, so the repository's executable
assertion-based test entry points were used. PyTorch emitted existing AMP API
deprecation warnings only. These checks establish regression safety, not
post-correction dataset accuracy.

### 0.2 Evidence-to-change map for the post-correction robustness pass

The latest supplied run improved the total adjusted score to **6.628**, the
best supplied total so far, but exposed a remaining architecture-selection
failure on Gutenberg:

| Dataset | Test accuracy | Adjusted score | Runtime |
|---|---:|---:|---:|
| AddNIST / Adaline | 95.710% | +5.773 | 1775.6s |
| Gutenberg | 43.833% | +0.483 | 1776.5s |
| Language / LaMelo | 85.750% | +0.372 | 1773.9s |

The earlier controller corrections worked as intended: Gutenberg restored a
42.30% search checkpoint, preserved the incumbent against a marginal SGD
result, rotated failed recovery attempts, and reached 42.67% validation with
EMA. The residual problem occurred earlier:

| Weakness | Exact evidence | Implemented correction |
|---|---|---|
| Architecture and RNG trial were coupled to list position | The historical 47.667% `spatial S2 C64 B3 basic K3 SE0 stem3` model moved from candidate position 4 to position 12; the old code added candidate index to initialization, round, and refinement seeds | All architectures now receive a common initialization seed and a common seed per equal-work round/pass; representation probes follow the same rule |
| A known-feasible slow starter could be eliminated before meaningful fidelity | The exact historical winner was present, scored 20.29% then 21.72%, ranked sixth after round 2, and was eliminated before the later full-data trajectory that had previously reached 47.667% test accuracy | Family quotas explicitly reserve feasible anchor endpoints; one capacity reference may replace the weakest survivor while within a learning/uncertainty allowance, then receive at most two extra full-data passes |
| The selected architecture showed severe brittleness | The selected `spatial S4 C32 B2 basic K5 SE1 stem7` model had 7,701,478 parameters, selection cross-entropy 2.2761 (worse than `log(6)=1.792`), and reached 100% training accuracy by final-training epoch 14 while validation remained near 42% | Cross-entropy excess and train-validation gap are bounded late risk terms and tie-breakers; confirmation accuracy remains primary |
| Only a statistically tied runner-up could reach Trainer | Recipe changes could not escape the selected Gutenberg architecture, even though a much smaller diverse architecture could generalize differently | When the winner is risky, retain at most one architecture with no more than 75% of its parameters and no more than an eight-point holdout deficit; Trainer tries it sequentially after distinct primary recipes, never as an ensemble |

The repechage is generic and clock-bounded. It uses only architecture role,
observed learning gain, sampling uncertainty, parameter count, validation
accuracy/loss, and train accuracy. It does not inspect a dataset name. Earlier
rounds keep their scheduled survivor count; only the last round may carry one
extra reference, and that reference receives at most two equal full-data
passes before the best two continue. A reference outside the bounded
allowance receives no insurance.

Local regression tests prove that reversing candidate order preserves an
existing specification's initialization and round seed, feasible anchor
endpoints survive quota construction, repechage is bounded, late risk acts
only inside a confirmation tie, and a materially smaller checkpoint is
retained only for a risky winner. These tests establish controller behavior;
the three competition-style datasets still require a fresh rerun to measure
the accuracy effect.

Post-pass CPU verification completed bytecode compilation,
`test_controller_refinements.py`, `test_search_objective.py`,
`test_trainer_anytime.py`, `test_architecture_fixes.py`,
`test_data_processor.py` (7/7), `test_memory_safety.py`, and the accelerated
Tier-1 processing/search/training/prediction integration. The unaccelerated
`test_full_pipeline.py` was started but is not counted as a pass: its long
synthetic-clock training exceeded the 240-second command timeout without an
assertion failure. Existing AMP deprecation warnings were the only warnings.

## 1. Executive summary

The current submission is a safety-conscious hierarchical NAS portfolio. It
profiles the input, activates several independent architecture families,
pre-screens candidates with zero-cost proxies, compares a family-balanced set
with label-based multi-fidelity training, refines the best two plus at most one
bounded reference, chooses a training recipe, and then continues training the
selected model under a wall-clock guard.

The foundation is sound and should be retained:

- no dataset-codename routing;
- independent PyTorch models rather than a weight-sharing supernet;
- full test-set prediction in fixed order;
- time and prediction reserves;
- warm-starting from search;
- best-checkpoint restoration;
- CUDA OOM fallbacks.

The dominant performance problems were not a lack of additional zero-cost
proxies. Historical runs and the latest successful pre-correction run exposed:

1. best refinement checkpoints being replaced by worse endpoints;
2. historical rank overriding clearly better selection and confirmation
   accuracy;
3. recipe replacement on gains too small to distinguish from validation
   noise;
4. promoted families losing their wide capacity options before label
   training;
5. failed or exhausted final-training attempts monopolizing the clock;
6. cosine SGD increasing its learning rate after the intended decay horizon.

The supplied 6.628 run then exposed four later architecture-selection issues:
candidate-position-dependent seeds, premature elimination of a feasible slow
starter, severe selected-model brittleness, and challenger retention limited
to statistical ties.

All ten are now corrected in code and covered by focused regression tests. The
remaining gap is empirical rather than architectural: the corrected controller
still needs multi-seed runs on the original three datasets, broader historical
datasets, and a competition-compatible CUDA runtime.

The implemented direction is:

> Keep the independent-model portfolio, but reorganize it around small
> label-aware representation probes, meaningful minimum fidelity, an
> incumbent-safe multi-stage recipe/optimizer race, and a genuinely anytime
> controller. Add new representation families only after the controller can
> measure them reliably.

The first implementation step was a **controller-only change**:
replace the one-stage recipe tournament with an incumbent-preserving,
two-stage recipe race evaluated on the same validation data, while leaving the
architecture families unchanged. This step is now implemented. Optimizer
expansion, architecture-family changes, augmentation changes, and later
controller corrections were implemented as separately testable follow-up
changes rather than folded into that first step.

## 2. Competition objective and constraints

The local [`README.md`](README.md) and evaluator define the relevant
constraints:

- `DataProcessor.process()` must return train, validation, and test
  DataLoaders.
- `NAS.search()` must return a PyTorch model.
- `Trainer.train()` must return the trained model.
- `Trainer.predict()` must return predictions for every test example.
- The available time is dataset-specific and can change between evaluation
  phases.
- The external evaluator owns the real time limit. Attempts to bypass it are
  disqualifying.
- The final datasets are hidden and must not be inferred or downloaded.
- Runtime or memory failure can produce a score of `-10` for a dataset.

The scoring implementation in [`evaluation/score.py`](evaluation/score.py) is:

```text
scaling_factor = 10 / (100 - benchmark)
adjusted_score = (raw_accuracy - benchmark) * scaling_factor
adjusted_score = max(-10, adjusted_score)
```

Important consequences:

- Within a dataset, maximizing adjusted score is equivalent to maximizing raw
  accuracy.
- Parameter count and successful runtime are printed but are not direct score
  penalties.
- Speed, parameter limits, and latency should be treated as feasibility and
  time-to-accuracy constraints, not as independent optimization objectives.
- Reliability is score-critical because a single crash can erase gains from
  several successful datasets.
- The benchmark can be used generically as a target because it is supplied in
  metadata. It must not be used together with the codename to hard-code a
  dataset-specific policy.

The official competition information states that tasks use four-dimensional
image-shaped tensors, but historical tasks demonstrate that tensor axes can
encode depth, time, tokens, embedding dimensions, board pieces, or independent
views rather than ordinary RGB image semantics.

The supplied README and currently published public rules do not explicitly
forbid a model that internally contains multiple branches or submodels.
However, the current submission deliberately promises no final ensemble.
Ensembling should therefore remain optional and should be confirmed with the
organizers before being enabled. None of the core recommendations below
requires an ensemble.

## 3. Concise description of the pre-improvement baseline

This section intentionally records the architecture that produced the latest
three-dataset evidence. It is retained so the rationale is reproducible; it is
not the active post-implementation architecture. The active implementation is
summarized in Section 0 and documented in
[`submission/NAS_ARCHITECTURE.md`](submission/NAS_ARCHITECTURE.md). The code
remains the final authority.

### 3.1 Data processing

[`submission/data_processor.py`](submission/data_processor.py) performs:

- conversion of NumPy arrays to PyTorch tensors;
- insertion of a missing channel dimension for three-dimensional arrays;
- content fingerprinting through `helpers.inspect_data_properties()`;
- channel-wise normalization;
- class-count, imbalance, and smoothed inverse-frequency statistics;
- conservative augmentation selected before NAS begins;
- batch-size estimation from input resolution and available GPU memory;
- deterministic train shuffling and ordered validation/test loading.

The fingerprint in [`submission/helpers.py`](submission/helpers.py) examines a
deterministic sample of at most 256 training examples and produces hypotheses
for:

- generic spatial processing;
- position-sensitive processing;
- width-sequence processing;
- height-sequence processing;
- factorized processing.

It pays particular attention to sparse binary grids with approximately
one-hot rows or columns.

### 3.2 Architecture portfolio

[`submission/search_space.py`](submission/search_space.py) defines independent
models from these active families:

- `spatial`: residual 2D CNN with global average pooling;
- `spatial_pyramid`: residual 2D CNN with global plus 2x2 pooled layout;
- `factorized`: depthwise-separable residual 2D CNN with GroupNorm;
- `axis_width`: 1D residual encoder along width;
- `axis_height`: 1D residual encoder along height;
- `dual_axis`: independent width and height encoders followed by feature
  fusion.

The common macro dimensions are:

- two to four stages when resolution permits;
- 16, 32, or 64 initial channels;
- one to three blocks per stage;
- basic or bottleneck residual blocks;
- kernel size 3 or 5;
- optional squeeze-and-excitation;
- stem kernel 3, 5, or 7.

Axis models add sinusoidal absolute position features and retain four or fewer
ordered pooling bins. Spatial-pyramid models retain only a global bin and a
2x2 layout.

### 3.3 Search controller

[`submission/nas.py`](submission/nas.py) implements three time tiers:

- Tier 1, at least 15 minutes: 54 proxy candidates and 12 label-trained
  finalists;
- Tier 2, 5-15 minutes: 30 proxy candidates and 7 label-trained finalists;
- Tier 3, under 5 minutes: direct portfolio anchor.

Tier 1 assigns 22% of remaining time to search and scales its ceiling from six
minutes up to 30 minutes on long runs. Tier 2 remains capped at 12% or 120
seconds.

The Tier 1 pipeline is:

1. build a parameter-filtered candidate pool;
2. sample architectures across active families and parameter-size strata;
3. compute SynFlow, Jacobian correlation, NASWOT, and proxy latency;
4. run equal-work, label-aware representation-family probes;
5. promote the best families with uncertainty-aware ties;
6. restore every feasible macro size for promoted families and choose 12
   label-aware candidates with family quotas and log-parameter coverage;
7. train them with progressive data coverage and multi-fidelity halving;
8. adaptively refine the final two while preserving each one's best matching
   model and optimizer state;
9. compare them on prior-preserving selection and disjoint confirmation
   splits, with confirmation dominance and historical rank used only for
   statistical ties;
10. compare the unchanged architecture checkpoint with every active recipe in
    a fair two-stage race;
11. return a recipe checkpoint only when its uncertainty-bounded improvement
    also confirms; otherwise return the incumbent.

The final architecture search uses one neutral `architecture_probe` recipe.
The active final recipes are:

- `stable`;
- `regularized`;
- `balanced` when class imbalance is detected;
- otherwise `fast_fit` when examples per class are high;
- zero-smoothing SGD with Nesterov momentum and monotonic cosine decay.

### 3.4 Final trainer

[`submission/trainer.py`](submission/trainer.py) performs:

- real train/validation throughput calibration;
- measured prediction-time reservation;
- optimizer-matched continuation from the selected search state;
- AMP on CUDA;
- gradient clipping;
- label smoothing, optional MixUp, and optional class weighting;
- warmup, validation-driven AdamW reduction, or no-restart cosine SGD;
- full-validation checkpoint selection;
- clock-driven independent recipe attempts from a common NAS checkpoint;
- early rotation of failed recoveries and guarded rotation of late productive
  plateaus;
- preference for regularized recipes after a strongly overfit failure;
- one optional statistically tied or materially smaller risk-insurance
  architecture challenger, fresh-seed continuations, EMA, and
  same-architecture checkpoint averaging;
- recursive prediction-batch splitting after CUDA OOM;
- restoration of the globally best checkpoint before prediction.

The external clock and measured prediction reserve are the effective stopping
conditions. A very high 100,000-epoch ceiling exists only as protection
against a broken synthetic clock; exhausting a fixed retry count no longer
ends an otherwise safe run.

## 4. Original successful baseline facts

This section records the first successful run used to design Priorities 0-5.
It is historical evidence, not the latest executable result; Section 0 records
the latest successful pre-correction run.

All three local datasets lacked a `time_limit` metadata field, so the evaluator
used its 30-minute default.

| Dataset / codename | Input | NAS winner | Search | Trainer best validation | Test accuracy | Adjusted score | Total runtime | Unused time |
|---|---|---|---:|---:|---:|---:|---:|---:|
| AddNIST / Adaline | 3x28x28, 20 classes | `spatial_pyramid`, S3 C32 B3 basic K5, `fast_fit`, 2,999,476 params | 5m05s | 93.84% | 94.090% | +4.177 | 1763.0s | about 37s |
| Gutenberg | 1x27x18, 6 classes | `spatial`, S2 C64 B3 basic K3, `regularized`, 1,044,422 params | 2m36s | 47.17% | 47.667% | +1.133 | 1051.4s | about 749s |
| Language / LaMelo | 1x24x24, 10 classes | `dual_axis`, S3 C32 B2 basic K3, `stable`, 497,098 params | 3m26s | 85.03% | 84.940% | -0.176 | 1284.1s | about 516s |

Total adjusted score: **5.135**.

Score sensitivity per additional raw-accuracy percentage point was:

- AddNIST: about `+0.985` adjusted points per raw point;
- Gutenberg: about `+0.169`;
- Language: about `+0.676`.

This does not change model ordering within a dataset, but it emphasizes that a
small regression on a high-benchmark dataset can erase a large gain on a
low-benchmark dataset.

## 5. Original weaknesses and historical run evidence

The implementation descriptions in this section intentionally describe the
old controller that produced the 5.135 baseline. The completion record and
Section 0.1 state how each applicable weakness is handled now.

### 5.1 Early architecture fidelity is not predictive enough

**Implementation involved**

- `NAS._cache_search_batches()`
- `NAS._train_low_fidelity()`
- `NAS._candidate_utility()`
- `NAS._successive_halving()`

The first three rounds use at most a cached partial epoch and repeatedly start
at the beginning of that cache. Accuracy is the dominant promotion signal.

**Original-run evidence**

On AddNIST, the eventual winner had:

- 7.69% validation after round 1;
- 15.53% after round 2;
- 22.80% after round 3;
- 38.09%, 44.68%, 55.49%, 64.33%, and 72.31% during full-data
  refinement;
- 72.82% on full validation before recipe selection;
- 93.84% as the final trainer best.

The first-round leader was a different architecture at 12.06%, and the
round-two leader was also a different architecture at 17.92%. The ordering
changed as soon as candidates received enough distinct data.

On Language, the eventual dual-axis winner started behind both the spatial and
width-axis candidates:

- spatial: 49.17% after round 1;
- width-axis: 45.48%;
- eventual dual-axis winner: 36.25%.

It became competitive only after 135-263 total search steps.

**Why it matters**

The controller can eliminate a slow-starting but superior architecture. More
proxies do not solve this because the problem occurs after proxy screening.
Promotion needs validation loss, learning slope, uncertainty, and a minimum
meaningful exposure to distinct data.

### 5.2 The recipe tournament is too short and can damage the incumbent

**Implementation involved**

- `NAS._recipe_tournament()`
- `Trainer._select_alternative_recipe()`
- the independent-retry section of `Trainer.train()`

Each recipe receives a single equal segment, normally at most one epoch, from
the common architecture checkpoint. The best endpoint accuracy is selected.
The unchanged common checkpoint is not included as a competing incumbent.

**Original-run evidence**

AddNIST:

- tournament: `fast_fit` 77.95%, `stable` 77.34%, `regularized` 74.44%;
- eventual best: `fast_fit`, so the tournament was correct.

Gutenberg:

- tournament: `regularized` 38.94%, `stable` 34.79%, `fast_fit` 33.86%;
- final `regularized` attempt peaked at 46.51%;
- later `fast_fit` attempt reached 47.17%;
- the one-stage tournament therefore selected the wrong eventual recipe.

Language:

- architecture checkpoint full validation: 84.00%;
- tournament: `stable` 83.06%, `fast_fit` 82.64%,
  `regularized` 81.64%;
- trainer baseline after accepting `stable`: 82.89%, below the 84.00%
  architecture checkpoint;
- final `stable` attempt peaked at 84.23%;
- `fast_fit` reached 84.76%;
- `regularized` reached 85.03% and was the eventual best.

The recipe tournament was therefore directionally wrong on two of three
datasets and degraded the Language incumbent before final training.

**Why it matters**

Recipe learning curves cross after more than one epoch. The tournament should
be multi-stage, should compare all trials on exactly the same validation
source, and should never replace the common checkpoint unless a trial improves
it beyond noise.

### 5.3 The controller is not genuinely anytime

**Implementation involved**

- the fixed Tier 1/Tier 2 budgets in `NAS.search()`;
- the 360-second search cap in `NAS._search_pipeline()`;
- `Trainer.max_epochs = 400`;
- the retry-count and recipe-exhaustion exit in `Trainer.train()`.

**Original-run evidence**

- AddNIST used 97.9% of the 30-minute budget.
- Gutenberg ended with about 12m29s remaining and used only 58.4%.
- Language ended with about 8m37s remaining and used only 71.3%.

Gutenberg stopped after `regularized` and one `fast_fit` retry even though
`stable` had not received a full trainer attempt and substantial safe time
remained. Language exhausted three recipes but then stopped rather than using
the remaining time for a new seed, EMA/SWA, a challenger architecture, or an
incumbent-safe continuation.

For a multi-hour Phase 3 run, search was still capped at six minutes and the
trainer was still capped at 400 epochs or a small number of recipe attempts.
That controller therefore did not scale its exploration with the unknown final
time allocation.

**Why it matters**

Unused per-dataset time cannot improve another dataset. Once the best
checkpoint and prediction reserve are protected, safe extra trials have
positive option value and no accuracy downside.

### 5.4 Validation sampling and selection uncertainty do not match the score

**Implementation involved**

- `NAS._cache_validation_batches()`
- `NAS._candidate_utility()`
- `NAS._final_selection_scores()`

The cached validation subset deliberately allocates approximately equal
samples per class. On an imbalanced dataset, its ordinary accuracy is already
close to balanced accuracy, even though competition scoring uses population
accuracy. The utility then mixes in balanced accuracy again.

Final selection blends:

- 65% full-validation accuracy;
- 25% preceding fidelity;
- 10% historical rank.

It does not estimate whether finalist differences are statistically
meaningful.

**Original-run evidence**

Gutenberg finalists had:

- spatial: 38.31% full validation;
- spatial-pyramid: 38.28% full validation.

The 0.03-point difference is far below normal sampling noise. Historical rank
then helped turn an effective tie into a hard decision. No paired-error or
confidence analysis was performed.

All three logged datasets were nearly class-balanced, so the class-prior
mismatch did not trigger visibly in this run. The log therefore supplies
evidence for the uncertainty problem but not an empirical imbalance failure.
The latter is a code-level risk that must be exercised on imbalanced
historical or synthetic data.

### 5.5 The proxy ranking is biased toward one family and has little observed
agreement with the winners

**Implementation involved**

- [`submission/zero_cost_proxies.py`](submission/zero_cost_proxies.py)
- `NAS._proxy_screen()`
- `NAS._select_label_aware_entries()`

Family quotas successfully prevent total family elimination, which is a
strength. However, proxy ranking still decides which parameter strata and
which individual candidates enter label-aware training.

**Original-run evidence**

For all three datasets, all proxy top-three candidates were `factorized`.
None of the three final winners was factorized:

- AddNIST: `spatial_pyramid`;
- Gutenberg: `spatial`;
- Language: `dual_axis`.

This does not prove that proxies are harmful because the quota system kept
other families alive. It does show that their cross-family ranking has almost
no positive evidence in the original run. Proxy effort should not be increased
until its selected-versus-final rank correlation has been measured.

### 5.6 Input fingerprinting does not cover enough axis semantics

**Implementation involved**

- `helpers.inspect_data_properties()`
- `SearchSpace.__init__()`
- `SearchSpace._build_axis_encoder()`
- `SearchSpace._build_spatial_model()`

The old content fingerprint was strongest for sparse one-hot grids. It did
not explicitly detect:

- independent per-channel views;
- a channel axis representing physical depth or ordered time;
- a short sequence of high-dimensional continuous embeddings;
- small position-sensitive boards whose values are not binary;
- fixed-coordinate scientific data where translation/crop augmentation is
  invalid.

**Original-run evidence**

The fingerprint did correctly identify two one-hot structures:

- Gutenberg: one-hot-column ratio 1.000 and
  `sequence_width = 1.0`;
- Language: one-hot-row ratio 1.000 and
  `sequence_height = 1.0`.

Even so:

- Gutenberg's two width-axis candidates were eliminated by the end of the
  second fidelity round;
- Language's directionally useful axis models needed substantially more steps
  before becoming competitive;
- AddNIST exposed only spatial, spatial-pyramid, and factorized models, even
  though its three channels are semantically separate digit views rather than
  ordinary RGB color channels.

Published historical datasets make the missing semantics concrete:

- Voxel is `(N, 20, 20, 20)`, with the channel axis acting as a third spatial
  dimension.
- Cryptic is `(N, 1, 6, 768)`, representing six continuous word embeddings;
  convolution across embedding coordinates is not ordinary spatial
  convolution.
- Chesseract is a position-sensitive 12-channel 8x8 board.
- Windspeed uses channels for multiple variables over several time steps.

**Why it matters**

The final datasets are specifically designed to look image-like while
representing unusual tasks. A successful generic NAS must infer useful
operations from labels and learning behavior, not only from tensor shape and
binary sparsity.

### 5.7 Augmentation and normalization are selected too early

**Implementation involved**

- `DataProcessor._estimate_flip_safety()`
- `build_augmentation_pipeline()`
- `DataProcessor._compute_normalization_stats()`

The DataProcessor commits to one augmentation policy before any architecture
or recipe training. Recipes vary MixUp and regularization but cannot compare
identity, crop, translation, or flip policies.

The old structural rules also conflated modality with grayscale/color:

- grayscale data bypasses the label-aware flip probe;
- standardized or structured multi-channel data can still receive random
  crop;
- one-hot sequences receive no geometry, which is appropriate for the logged
  language tasks but is not a general solution.

**Original-run evidence**

- AddNIST received random crop and no flips.
- Gutenberg and Language received normalization only.
- The log contains no augmentation ablation, so it cannot establish whether
  these choices helped or hurt.

This weakness is therefore evidenced by missing comparative evidence rather
than a directly observed failure. Augmentation should become a small searched
policy dimension, not a hard preprocessing commitment.

### 5.8 Final training overfits and explores too narrow an optimizer space

**Implementation involved**

- `NAS._training_recipes()`
- `Trainer._make_optimizer_and_scheduler()`
- `Trainer.train()`

All recipes in that run used AdamW. The recipe space varied only LR scale,
weight decay, label smoothing, MixUp, and class weights. Architecture dropout
was tied to channel count rather than searched. There was no EMA, SWA,
optimizer alternative, stochastic depth, or optional post-selection refit.

**Original-run evidence**

Language repeatedly reached approximately 100% training accuracy while
validation stayed around 83-85%. Three optimizer restarts changed the path but
did not remove the generalization gap:

- `stable` best: 84.23%;
- `fast_fit` best: 84.76%;
- `regularized` best: 85.03%.

Gutenberg similarly reached 100% training accuracy while validation remained
around 46-47%.

This indicates that additional epochs in the same basin are not enough. The
system needs representation changes, stronger or different regularization,
weight averaging, and optimizer diversity.

### 5.9 RAM safety is insufficiently validated

**Implementation involved**

- `NASDataset.__init__()`;
- `DataProcessor._compute_normalization_stats()`;
- `NAS._cache_validation_batches()`;
- storage of several CPU models and AdamW optimizer states during search.

Potential large allocations include:

- converting an entire non-float32 array to float32;
- converting up to 10,000 large images to a temporary float32 NumPy array;
- caching up to 4,096 validation samples without a byte cap;
- keeping multiple models and two-moment AdamW states.

**Original-run evidence**

The latest inputs were only 24-28 pixels per side and all used batch size 128.
No RAM or VRAM failure occurred. Consequently, the run does not validate the
large-input path at all. Historical data includes 128x128 images and
multi-gigabyte datasets, so absence of a failure in this log must not be
interpreted as evidence of safety.

## 6. Prioritized improvement plan

### Priority 0: Establish trustworthy measurements and memory invariants

**Goal:** Ensure later architecture changes are judged correctly and cannot
introduce avoidable `-10` failures.

Changes:

1. [x] Preserve class priors in the main search-validation sample.
2. [x] Track a separate balanced metric instead of balancing the sample itself.
3. [x] Split validation deterministically into selection and confirmation subsets.
4. [x] Log validation loss, accuracy, balanced accuracy, margin, examples seen,
   data coverage, seed, LR, time, and peak memory for every candidate.
5. [x] Use the same stochastic seed schedule and data order for paired candidate
   comparisons.
6. [x] Retain at most one alternate architecture: either a statistically tied
   finalist or a materially smaller competitive model when the winner has
   strong late brittleness evidence.
7. [x] Convert normalization statistics to a byte-bounded streaming calculation.
8. [x] Add byte caps to both train and validation caches.
9. [x] Avoid whole-dataset dtype conversion.
10. [x] Calibrate microbatch size before architecture racing and use gradient
    accumulation where necessary.

Expected benefit:

- lower selection noise;
- correct alignment with competition accuracy;
- reliable experiments for subsequent changes;
- substantially lower RAM failure risk.

Primary risk:

- changing validation sampling can change candidate ordering. That is intended,
  but it must be tested first on the current three datasets and synthetic
  imbalanced data.

### Priority 1: Make recipe selection incumbent-safe and multi-fidelity

**Goal:** Correct the clearest empirically demonstrated controller failure.

Changes:

1. [x] Evaluate the common architecture checkpoint on the exact same validation
   source used for recipe trials.
2. [x] Include the unchanged common checkpoint as the incumbent.
3. [x] Give each recipe a short first stage.
4. [x] Complete the full planned promotion-signal bundle.
   - [x] Promote two recipes using best validation accuracy and bounded
     validation-loss slope.
   - [x] Add an explicit train-validation-gap term after it can be measured
     fairly without increasing tournament data passes.
5. [x] Preserve the best evaluated stage state within each recipe trial, not
   only its last state.
6. [x] Accept a recipe-trained state only if it beats the incumbent beyond a
   configurable noise margin.
7. [x] Pass the winning optimizer state into Trainer only when it belongs to the
   returned winning checkpoint.
8. [x] Add SGD with Nesterov and cosine decay as a genuinely different optimizer
   path; retain AdamW recipes.
9. [x] Add a zero-smoothing or very-low-smoothing recipe.

Completed implementation scope:

- `submission/nas.py` now measures unsmoothed validation cross-entropy,
  evaluates the incumbent and all trials on the identical source, stages
  recipe promotion, restores the best evaluated stage, and applies an
  acceptance margin of
  `max(0.10pp configured floor, one validation example,
  min(1pp, 0.75 * pooled standard error))` separately on selection and
  confirmation.
- `test_controller_refinements.py` covers incumbent preservation, identical
  initial states and seeds, equal stage work, loss-slope ranking, checkpoint
  restoration, optimizer-state ownership, rejection of a Gutenberg-sized
  marginal gain, and short-budget fallback.
- This first step did not change any independent architecture, augmentation,
  or final-Trainer behavior. Those were added and tested in later steps.

Expected benefit:

- prevents the observed Language checkpoint regression;
- increases the probability that the initially selected recipe is the eventual
  best;
- reduces time lost retraining the wrong recipe;
- adds optimizer diversity against the observed overfitting.

Primary risk:

- a longer recipe race consumes time that would otherwise go to final
  training. Because recipe training warm-starts the returned model, most of
  this work is not wasted, but the 30-minute budget split must be tested.

### Priority 2: Replace fixed low-fidelity halving with hierarchical
representation racing

**Goal:** Compare architecture families only after they have a minimally
informative learning curve.

Changes:

1. [x] Create one compact, calibrated probe model for each plausible
   representation family.
2. [x] Train probes on the same distinct data coverage for approximately one
   epoch, or until loss slope is measurable.
3. [x] Promote the best two families with uncertainty-aware ties.
4. [x] Search macro dimensions only inside the promoted families.
5. [x] Use progressive data coverage rather than repeatedly replaying the
   beginning of a fixed cache.
6. [x] Treat zero-cost proxies as ordering or duplicate-reduction tools only.
7. [x] Promote using loss, accuracy, slope, and an optimistic confidence bound.
8. [x] Continue tied full-validation finalists rather than blending an arbitrary
   0.03-point difference with old ranks.
9. [x] Retain the second finalist's specification and optional checkpoint for an
   anytime fallback.
10. [x] Restore the best adaptive-refinement checkpoint and its matching
    optimizer/data-stream state before final holdout comparison.
11. [x] Restore the complete feasible macro pool after family promotion and
    choose within-family candidates across log-parameter strata, including a
    deterministic C64 anchor for full-grid families.
12. [x] Remove candidate-index-dependent architecture seeds and use common
    deterministic seeds for equal-work representation, fidelity, and
    refinement comparisons.
13. [x] Reserve feasible deterministic anchor endpoints inside family quotas.
14. [x] Give one plausibly still-learning capacity anchor a bounded late
    repechage without increasing earlier-round survivor counts.
15. [x] Use train-validation gap and absolute cross-entropy as bounded late
    risk/tie evidence, never as a substitute for a clear holdout win.
16. [x] Preserve one materially smaller, sufficiently competitive architecture
    as sequential overfit insurance when the winner is brittle.

Expected benefit:

- sharply lower architecture-selection regret;
- fairer treatment of slow-starting axis or larger models;
- less budget spent training many near-chance variants from the same family.

Primary risk:

- fewer architectures receive label training. This is acceptable only if
  functional family probes have been validated across many historical
  datasets.

### Priority 3: Make time allocation genuinely scalable and anytime

**Goal:** Use safely available per-dataset time without risking prediction.

Changes:

1. [x] Replace the fixed six-minute Tier 1 search cap with a budget curve based on:
   - total remaining time;
   - measured epoch time;
   - number of plausible families;
   - whether architecture ranking has stabilized.
2. [x] Remove `max_epochs = 400` as a hard multi-hour ceiling; retain only a very
   high safety ceiling plus the clock.
3. [x] Replace retry-count termination with an ordered anytime action queue:
   - [x] continue the best checkpoint with EMA;
   - [x] run every untried optimizer/recipe;
   - [x] run a new seed from the best checkpoint;
   - [x] continue one dormant architecture when statistically tied or when a
     materially smaller competitive model insures a brittle winner;
   - [x] try same-architecture checkpoint averaging;
   - [ ] optionally refit after locking the policy, pending broad ablation.
4. [x] Keep the global best validation checkpoint immutable.
5. [x] Estimate prediction reserve from measured batches with a conservative upper
   bound rather than an unconditional 180-second maximum.
6. [x] Rotate a failed recovery early only after enough epochs, validation
   stalls, failure to recover the attempt's starting best, and a large
   train-validation gap.
7. [x] Give productive slow recoveries a longer runway, then rotate only after
   a sustained late plateau.
8. [x] Stop cosine SGD at its initial `T_max` and prevent all time-based LR
   cooldowns from increasing the current learning rate.

A reasonable initial 30-minute split to test is:

| Stage | Initial budget |
|---|---:|
| Profiling and memory calibration | 30 seconds |
| Representation-family probes | 2 minutes |
| Within-family architecture race | 3 minutes |
| Recipe/optimizer race | 2 minutes |
| Final training and safe improvement attempts | about 21 minutes |
| Prediction reserve | measured, initially about 90 seconds |

This is a starting policy, not a fixed rule. Search training warm-starts the
final model, so its entire budget should not be treated as discarded compute.

Expected benefit:

- addresses the 29% and 42% unused budget observed on the original Language
  and Gutenberg run;
- adapts to multi-hour final runs;
- permits additional seeds or representations without risking the best model;
- avoids spending most of the budget on a failed optimizer while preserving
  genuinely productive slow recoveries.

Primary risk:

- more trials increase validation-selection pressure. Confirmation validation
  and uncertainty thresholds are therefore prerequisites.

### Priority 4: Expand representation families

Add families in this order so each addition can be ablated.

#### [x] 4.1 Categorical sequence family

For an approximately one-hot axis:

- decode or softly project categorical tokens;
- use a learned embedding;
- compare bag-of-token/frequency features;
- add multi-kernel TCN/TextCNN blocks;
- add a small attention encoder;
- retain explicit positional features.

Likely coverage:

- Language-like and Gutenberg-like data;
- other symbolic sequences encoded as grids.

#### [x] 4.2 Dense embedding sequence family

For one short axis and one large continuous feature axis:

- treat the short axis as sequence length;
- apply per-token LayerNorm and linear projection;
- use attention, TCN, or pooled token MLP across sequence positions;
- do not convolve as if adjacent embedding dimensions were image pixels.

Likely coverage:

- Cryptic-like pretrained embeddings;
- spectrogram or feature-vector sequences when detected by probes.

#### [x] 4.3 Channel-independent and multi-view family

- shared or separate 2D stems per channel or channel group;
- late feature fusion;
- relation/counting head;
- optional permutation-aware or permutation-invariant fusion selected by a
  probe.

Likely coverage:

- AddNIST/MultNIST-like independent channel images;
- multi-sensor inputs.

#### [x] 4.4 Volumetric and temporal-spatial family

- small Conv3D or axial 3D convolutions when channel and spatial dimensions
  appear exchangeable;
- temporal depthwise convolution plus spatial 2D backbone for ordered
  multi-time inputs;
- channel attention for grouped scientific variables.

Likely coverage:

- Voxel-like data;
- Windspeed-like temporal scientific grids.

#### [x] 4.5 Position-sensitive board and hybrid families

- CoordConv or explicit normalized coordinates;
- 4x4 or adaptive multi-level spatial-pyramid pooling;
- attention pooling over spatial tokens;
- a compact spatial-plus-axis fusion model.

Likely coverage:

- Sudoku and Chesseract-like boards;
- ambiguous data where both local 2D structure and absolute position matter.

#### [x] 4.6 Stronger generic image anchors

Add a small number of structurally different, well-tested anchors:

- pre-activation WideResNet;
- lightweight ResNeXt/cardinality blocks;
- DenseNet-style feature reuse.

Do not add dozens of cosmetic variants at once. The benefit comes from
different inductive biases, not a larger count of nearly homogeneous residual
CNNs.

Expected benefit:

- highest upside for genuinely obscure hidden modalities;
- closes known blind spots in axis semantics;
- improves the chance that family search has a suitable candidate.

Primary risks:

- larger code and testing surface;
- more families can dilute fidelity if the controller is not hierarchical;
- Conv3D and attention can increase memory usage;
- hard routing could overfit historical datasets, so label-aware functional
  probes and leave-one-dataset-out evaluation are mandatory.

### Priority 5: Search augmentation and improve final generalization

Changes:

1. [x] Keep an identity/evaluation transform available to every task.
2. [x] Compare a very small augmentation portfolio with a compact anchor:
   - identity;
   - translation/crop;
   - verified horizontal/vertical flips;
   - light erasing or CutMix for natural-image-like data.
3. [x] Do not infer flip safety solely from grayscale/color.
4. [x] Detect train-validation distribution shift through simple per-axis moments
   and a cheap domain-classification probe.
5. [x] Add EMA during every final attempt.
6. [x] Test same-architecture checkpoint averaging when time remains.
7. [x] Search dropout strength with the recipe.
8. [ ] After architecture, optimizer, and epoch policy are locked, test a
   train-plus-validation refit as a separate ablation. It must not use test
   labels or select on test predictions.
9. [x] Use test-time augmentation only for transformations already validated
   as label-preserving.

Expected benefit:

- lower train-validation gaps;
- better checkpoint stability;
- safer use of remaining time.

Primary risks:

- augmentation can destroy labels in synthetic, scientific, or
  position-sensitive tasks;
- train-plus-validation refit removes the ordinary stopping holdout and must
  use a policy fixed beforehand.

### Priority 6: Optional complementary-model ensemble

**Status: [ ] deliberately disabled pending organizer confirmation.**

Only after organizer confirmation:

- retain two finalists only when their validation accuracies are tied and their
  per-example errors are complementary;
- wrap them in one API-compatible `nn.Module`;
- confirm training and prediction memory;
- disable automatically under short budgets.

The rule-neutral alternative is EMA, SWA, or same-architecture checkpoint
averaging.

## 7. Exact files and components that would need to change

Implementation status: all rule-neutral responsibilities below are active.
`submission/ensemble.py` remains inactive, and train-plus-validation refit
remains an empirical follow-up rather than enabled competition behavior.

| File | Components | Planned responsibility |
|---|---|---|
| `submission/nas.py` | validation splitting/evaluation, common architecture seed helpers, `_expand_promoted_macro_pool`, `_select_label_aware_entries`, `_select_halving_survivors`, `_architecture_risk`, refinement checkpoint helpers, `_final_selection_scores`, `_order_final_results`, `_select_retained_architecture_challenger`, `_recipe_acceptance_margin`, `_recipe_tournament`, `_successive_halving`, `_search_pipeline`, time-tier configuration | Prior-preserving holdouts, full promoted-family capacity coverage, order-independent paired trials, bounded anchor insurance, best refinement restoration, confirmation-dominant selection, late brittleness evidence, uncertainty-aware incumbent replacement, scalable budgets, one dormant tied/efficient challenger |
| `submission/trainer.py` | challenger handoff, recipe application, optimizer/scheduler construction, `_step_scheduler`, `_apply_time_cooldown`, `_attempt_rotation_reason`, `_select_alternative_recipe`, `train`, `predict` | Sequential tied/efficient architecture recovery, SGD/Nesterov, monotonic no-restart cosine, failed-attempt and late-plateau rotation, regularized fallback, EMA/checkpoint averaging, anytime action queue, measured prediction reserve, optional refit/TTA |
| `submission/data_processor.py` | `NASDataset`, normalization statistics, augmentation construction, DataLoader construction | Lazy/per-batch dtype conversion, streaming byte-bounded statistics, searchable augmentation policies, safer large-data loading |
| `submission/helpers.py` | `inspect_data_properties`, batch-size helper, possible new memory/time helpers | Richer axis/modality fingerprints, memory estimates, train-validation shift statistics |
| `submission/search_space.py` | `ArchSpec`, blocks, model builders, analytical parameter counts | Token/embedding sequence, channel-independent, volumetric/temporal, position-sensitive, hybrid, WideResNet/ResNeXt/DenseNet-lite families |
| `submission/zero_cost_proxies.py` | proxy normalization and aggregation | Keep proxies weak; add diagnostics for rank correlation; avoid cross-family scale bias |
| `submission/ensemble.py` | currently inactive | Optional only after rule confirmation; otherwise leave unused |
| `submission/NAS_ARCHITECTURE.md` | active-controller section, pipeline, family and trainer descriptions | Must be updated in the same change as any implementation modification |
| `submission/README.md` | runtime modes and feature summary | Update after behavior changes |
| `test_controller_refinements.py` | controller unit/regression tests | Incumbent preservation, staged recipe promotion, uncertainty margins, order-independent architecture seeds, anchor reservation/repechage, best refinement restore, confirmation dominance, late-risk-only ties, efficient challenger retention, promoted-family capacity coverage |
| `test_architecture_fixes.py` | search-space tests | New family activation, shapes, parameter counts, forward compatibility |
| `test_data_processor.py` | data-processing tests | No full dtype copy, streaming stats, byte caps, augmentation policy behavior |
| `test_full_pipeline.py` | end-to-end synthetic tiers | Full prediction count, short-budget fallbacks, no idle/crash behavior |
| `test_memory_safety.py` | implemented | Large synthetic shapes/dtypes, cache-byte ceilings, logical microbatches |
| `test_search_objective.py` | implemented | Prior-preserving validation, confirmation, confidence, loss/gap promotion, and common representation-stage seeds |
| `test_trainer_anytime.py` | implemented | SGD/no-restart cosine, attempt-rotation guards, regularized priority, TTA, dormant challenger, clock-driven training |
| `test_accelerated_pipeline.py` | implemented | Fast Tier-1 processing/search/training/prediction integration |

Files that must **not** be modified for a submission:

- `evaluation/main.py`;
- `evaluation/score.py`.

They are organizer-owned and replaced during evaluation.

Packaging must continue to include only the submission implementation and must
not bundle evaluation files, test data, or hidden-dataset logic.

## 8. Risks, expected benefits, and competition-rule constraints

| Change | Expected benefit | Main risk | Rule/safety constraint |
|---|---|---|---|
| Prior-preserving, split validation | More accurate model selection; less overfitting | Smaller selection subsets can be noisy | Never inspect test labels |
| Incumbent-safe staged recipe race with uncertainty margin | Fixes an observed 2/3 recipe-selection failure and rejects marginal replacement | May preserve an actually better recipe on a noisy/small holdout | Must preserve prediction reserve and use training labels only |
| Best refinement restoration | Prevents later search updates from erasing a better finalist | Extra CPU checkpoint memory | Optimizer and data-cursor state must match restored weights |
| Confirmation-dominant final selection | Prevents adaptive history from overriding independent evidence | Small confirmation subsets can reorder true near-ties | Historical rank is permitted only inside the bounded tie band |
| Full promoted-family size coverage | Recovers compact-to-wide capacity options | A weak size stratum consumes one finalist slot | Fixed label-candidate count and parameter/memory guards remain active |
| Common architecture seeds | Removes candidate-order RNG confounding and makes paired comparisons reproducible | Correlated trials may favor a particular shared stochastic path | Hidden-data decisions still require multi-seed reruns; no candidate-specific seed tuning |
| Bounded capacity-anchor insurance | Gives a slow-starting generic reference meaningful fidelity | Can spend two passes on a weak model | One slot, observed-gain/uncertainty gate, eight-point cap, clock reserve, and no codename logic |
| Late architecture-risk evidence | Prefers a less brittle model inside a true holdout tie and identifies when recovery is warranted | Cross-entropy can be noisy while underfit | Activated only after refinement; cannot overturn a clear confirmation winner |
| Efficient dormant challenger | Offers a different generalization path after primary recipes fail | Extra CPU memory and final-training time | At most one, at least 25% smaller, within eight accuracy points, trained sequentially, never ensembled |
| Minimum meaningful architecture fidelity | Lower selection regret | Fewer candidates evaluated | Remain dataset-agnostic |
| Scalable anytime controller | Uses otherwise wasted compute and long Phase 3 budgets | More validation adaptivity | Hard external clock checks remain mandatory |
| Guarded attempt rotation | Redirects time from failed/overfit attempts | A very slow recovery may be interrupted | Immutable global best and minimum-runway conditions remain active |
| Monotonic no-restart cosine | Avoids late destructive LR rebound | May remove a useful restart on rare tasks | No new clock, data, or test-label dependency |
| Lazy conversion and byte caps | Prevents RAM failure and `-10` | Per-batch CPU overhead | Full test ordering and coverage must remain unchanged |
| New semantic families | Large upside on obscure data | Search dilution and code complexity | Activate from data/labels, never codename |
| SGD/EMA/SWA | Better generalization and stable checkpoints | Additional state and tuning | Return one valid PyTorch model |
| Train+validation refit | More labeled training data | No ordinary held-out stopper | Policy must be fixed before refit; no test-label use |
| Ensemble | Potential complementary-error gain | Double compute/memory; rule ambiguity | Obtain organizer confirmation first |
| Pretrained external weights | Potential large natural-image gain | Rule/licensing/package ambiguity; domain mismatch | Exclude unless explicitly confirmed as allowed |

## 9. Recommended first implementation step — completed

Implemented a **controller-only incumbent-safe recipe race**, without adding
new architecture families or changing augmentation.

### Scope

Primary files:

- `submission/nas.py`;
- `test_controller_refinements.py`;
- `submission/NAS_ARCHITECTURE.md`;
- `submission/README.md`.

`submission/trainer.py` should change only if the new recipe representation
needs to pass optimizer type or scheduler information.

### Required behavior

1. [x] Save the architecture checkpoint and evaluate it on the exact recipe
   tournament validation source.
2. [x] Treat that checkpoint as the incumbent.
3. [x] Clone all recipe trials from the same state and use the same data order and
   stochastic seed.
4. [x] Give every recipe a short first segment.
5. [x] Promote at least two recipes to a second segment when affordable.
6. [x] Rank using best validation accuracy plus validation-loss slope, not only
   last endpoint accuracy.
7. [x] Preserve the best evaluated stage checkpoint within every trial.
8. [x] Return the incumbent unchanged if no recipe trial improves it beyond a
   small, tested noise threshold.
9. [x] Attach only the optimizer state corresponding to the returned checkpoint.
10. [x] Preserve existing clock, OOM, API, and prediction invariants.

### Why this should be first

- It directly fixes a failure observed on two of three datasets.
- It prevents the concrete Language regression from 84.00% to an 82.89%
  baseline.
- It is isolated from the larger architecture redesign.
- It creates infrastructure that later optimizer and regularization recipes
  can reuse.
- Its effect can be measured on the existing three datasets before downloading
  or preparing the full historical suite.

### Acceptance criteria for the first step

- [x] On Language-like regression tests, no recipe trial may replace a stronger
  common checkpoint.
- [x] All trials must start from byte-identical weights.
- [x] Equal-stage trials must see the same example count and deterministic
  order.
- [x] The returned optimizer state must match the returned model state.
- [x] Short budgets must fall back to the incumbent safely.
- [x] Full prediction count and order must remain unchanged.
- [ ] In historical-dataset reruns, compare tournament selection with the
  eventual best recipe, not
  only its immediate endpoint.

### Baseline and post-change comparison

The focused pre-change characterization used an 84.00% incumbent and three
weaker recipe endpoints:

- baseline controller result: 80.00%, so the incumbent regressed by 4.00
  percentage points;
- baseline stage counts: one segment for each recipe.

The identical post-change characterization produced:

- returned result: 84.00%, byte-identical to the incumbent;
- stage counts: all three recipes received stage 1 and the best two received
  an equal stage 2;
- optimizer-state marker: the incumbent's optimizer state was retained.

A real-PyTorch CPU integration check with actual AdamW updates produced:

- incumbent: 56.64% validation accuracy and 0.7691 loss;
- returned staged-race model: 59.77% and 0.7100 loss;
- the selected model carried the optimizer state for its returned checkpoint.

Regression status:

- controller regression script: passed;
- submission compile check: passed;
- architecture regression script: passed;
- data-processor regression script: 7/7 passed;
- Tier 3 end-to-end pipeline: passed with complete predictions;
- Tier 2 end-to-end pipeline: passed with complete predictions.

The three historical datasets from the original log were not rerun because
their data is not present in this workspace. Their eventual-recipe comparison
therefore remains an explicit unchecked acceptance item.

## 10. Validation strategy for general rather than three-dataset improvement

Architecture and controller thresholds must not be selected from the current
three datasets alone.

### Dataset protocol

Use all available historical competition datasets, including:

- AddNIST;
- Language;
- MultNIST;
- CIFARTile;
- Gutenberg;
- GeoClassing;
- Chesseract;
- Sudoku;
- Voxel;
- Myofibre;
- GameOfLife;
- Cryptic;
- Windspeed;
- Isabella if it can be reproduced under its data license.

Use leave-one-dataset-out calibration:

1. select thresholds and policies on all but one historical dataset;
2. evaluate the held-out dataset without changing the policy;
3. rotate the held-out dataset;
4. optimize median, lower-tail, and failure rate, not the best individual
   dataset.

### Time protocol

Test at:

- under 5 minutes;
- 5-15 minutes;
- 30 minutes;
- 60 minutes;
- at least one multi-hour simulated budget.

Short tests must verify graceful fallback. Long tests must verify that fixed
caps do not leave most of the budget idle.

### Seed protocol

- At least three seeds for final comparisons.
- One paired seed for broad racing.
- A second seed only for finalists or uncertain decisions.

### Metrics

Record:

- raw and adjusted test score;
- median, lower quartile, and worst-case adjusted score;
- failure rate;
- selected-versus-best-tried architecture regret;
- selected-versus-best-tried recipe regret;
- low-fidelity versus later rank correlation;
- peak RAM and VRAM;
- search, training, validation, and prediction time;
- unused safe time;
- train-validation gap;
- validation-selection versus confirmation gap.

### Promotion gate

A change should be promoted only if:

- it causes zero incomplete-prediction, OOM, or timeout failures in the stress
  suite;
- median adjusted score improves;
- lower-tail performance does not materially regress;
- improvements remain when each historical dataset is held out from
  calibration;
- any added search time produces lower selection regret or better final
  accuracy;
- prediction finishes with a conservative measured reserve.

Recommended ablation order:

1. current baseline;
2. measurement and memory invariants only;
3. incumbent-safe recipe race;
4. hierarchical architecture racing;
5. new representation families one group at a time;
6. optimizer/EMA/SWA improvements;
7. optional refit or ensemble.

## 11. Important conclusions for a fresh Codex session

A new session should not have to rediscover the following:

1. **The active architecture description and current code are authoritative.**
   Use the first "active controller" section of
   `submission/NAS_ARCHITECTURE.md`; later historical material is retained only
   to explain how the design evolved.
2. **The latest supplied score is 6.628 adjusted points:** AddNIST `+5.773`
   at 95.710%, Gutenberg `+0.483` at 43.833%, and Language `+0.372` at
   85.750%. The immediate pre-correction run scored 4.233; the older 5.135
   result remains a separate historical baseline.
3. **Those runs used the evaluator's 30-minute default** because all three
   metadata files lacked `time_limit`.
4. **The three-dataset failure log used the wrong runtime.** It is not evidence
   that the submission was broken. Device-safe evaluation, OOM splitting, and
   malformed-family containment remain worthwhile defensive invariants.
5. **The 4.233 pre-correction run returned worse refinement endpoints for
   several finalists.** Current code restores the best matching model,
   optimizer, step, and data-cursor state before holdout comparison.
6. **Historical rank selected the clearly worse AddNIST finalist.** Current
   code makes confirmation accuracy dominant; loss, combined holdout score,
   and rank can act only inside a bounded confirmation uncertainty band.
7. **A 0.20-point Gutenberg recipe gain was too small to trust.** Recipe
   replacement now must clear an accuracy-dependent uncertainty margin on both
   selection and confirmation; the unchanged architecture checkpoint remains
   the incumbent.
8. **Capacity coverage, not more proxy weight, is the appropriate proxy fix.**
   Current code restores the full feasible macro pool for promoted families,
   uses compact-to-wide parameter strata, and includes a C64 anchor for
   full-grid families.
9. **Final-training allocation was pathological on the 4.233 run.** Current
   code distinguishes a failed overfit recovery from a productive slow
   recovery, rotates the former early, and rotates the latter only after a
   guarded late plateau.
10. **Cosine SGD must not restart accidentally.** Its scheduler now freezes at
    `T_max`, and time-based cooldown is clamped so LR cannot increase.
11. **The controller is now prior-preserving and uncertainty-aware.** Search
    selection and confirmation are deterministic disjoint splits, and
    ordinary accuracy remains distinct from balanced-accuracy diagnostics.
12. **Optimizer and representation diversity are already implemented.**
    AdamW and zero-smoothing SGD/Nesterov are active; categorical and dense
    sequences, channel-independent views, volume/time, coordinate-sensitive,
    grouped/wide, and dense-reuse families are in the portfolio.
13. **Shape alone is still not a semantic routing signal.** Historical
    four-dimensional tensors can be non-image, so activation must remain based
    on content statistics and measured label-aware learning.
14. **Local memory regression coverage exists, but real CUDA behavior is still
    unproven after the correction.** The latest inputs were only 24-28 pixels
    per side; run a competition-compatible CUDA/OOM stress test before
    submission.
15. **No post-robustness-pass accuracy claim is justified yet.** The 6.628 run
    predates common architecture seeds, anchor insurance, late risk evidence,
    and efficient challenger retention. The next empirical task is a paired,
    multi-seed rerun on the original three and broader historical datasets,
    recording raw accuracy, selection regret, runtime, peak memory, and
    failure rate.
16. **The default `C:\Python314` environment lacks PyTorch and pytest.**
    `C:\Users\Lennart\miniconda3\python.exe` contains CPU PyTorch and runs the
    executable regression scripts; CUDA and historical arrays require Colab or
    the competition-compatible environment.
17. **Do not tune against codenames or touch organizer files.** `Adaline` is
    AddNIST and `LaMelo` is Language in the log, but decisions must depend only
    on allowed tensor, label, learning, memory, and clock measurements.
    `evaluation/main.py` and `evaluation/score.py` are organizer-controlled.
18. **Refit and ensembling remain deliberately disabled.** Enable
    train-plus-validation refit only after broad leave-one-dataset-out evidence,
    and do not enable `submission/ensemble.py` without organizer confirmation.
19. **Before any future edit, recheck `git status` and preserve unrelated user
    changes.**
20. **The remaining Gutenberg regression was an early architecture-selection
    failure.** The exact older 47.667% architecture was in the race but was
    eliminated after 20.29% and 21.72% low-fidelity results; changing final
    recipes alone could not recover it.
21. **Candidate order must never choose an architecture's stochastic trial.**
    Representation probes, model initialization, fidelity rounds, and
    refinement passes now use common deterministic paired seeds. Reordering
    and insertion behavior is regression-tested.
22. **Anchor insurance is bounded evidence gathering, not hard routing.** One
    capacity reference can survive only while its deficit is supported by
    uncertainty or recent gain, and it receives at most two insured full-data
    passes before it must rank in the top two on merit. A clear confirmation
    winner remains decisive.
23. **The efficient challenger is sequential, not an ensemble.** At most one
    materially smaller checkpoint is kept when the winner is demonstrably
    brittle. It is tried only after the primary architecture's distinct
    recipes and never participates in the same forward pass.

## 12. Reference links

Local:

- [`README.md`](README.md)
- [`evaluation/main.py`](evaluation/main.py)
- [`evaluation/score.py`](evaluation/score.py)
- [`submission/NAS_ARCHITECTURE.md`](submission/NAS_ARCHITECTURE.md)
- [`submission/data_processor.py`](submission/data_processor.py)
- [`submission/helpers.py`](submission/helpers.py)
- [`submission/search_space.py`](submission/search_space.py)
- [`submission/nas.py`](submission/nas.py)
- [`submission/trainer.py`](submission/trainer.py)
- [`submission/zero_cost_proxies.py`](submission/zero_cost_proxies.py)

Public primary sources consulted during analysis:

- Competition rules: <https://www.nascompetition.com/rules>
- Competition technical information: <https://www.nascompetition.com/info>
- Historical dataset study: <https://arxiv.org/abs/2404.02189>
- Voxel dataset: <https://data.ncl.ac.uk/articles/dataset/Voxel_Dataset/26970223>
- Cryptic dataset:
  <https://datashare.ed.ac.uk/items/6ac0fdaf-3d56-4760-915f-6036103867aa>
