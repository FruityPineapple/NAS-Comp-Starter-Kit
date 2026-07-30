# NAS Improvement Plan

**Status:** The recommended first implementation step is complete. The
incumbent-safe staged recipe race is implemented and tested; all other
priorities remain pending unless explicitly checked below.

**Date:** 30 July 2026

**Primary evidence:** The current repository, the competition evaluator, and
the latest three-dataset run log at:

`C:\Users\Lennart\.codex\attachments\4a0d3e89-fc84-49a2-802e-4255e45f0318\pasted-text.txt`

## 1. Executive summary

The current submission is a safety-conscious hierarchical NAS portfolio. It
profiles the input, activates several independent architecture families,
pre-screens candidates with zero-cost proxies, compares a family-balanced set
with label-based multi-fidelity training, refines two finalists, chooses a
training recipe, and then continues training the selected model under a
wall-clock guard.

The foundation is sound and should be retained:

- no dataset-codename routing;
- independent PyTorch models rather than a weight-sharing supernet;
- full test-set prediction in fixed order;
- time and prediction reserves;
- warm-starting from search;
- best-checkpoint restoration;
- CUDA OOM fallbacks.

The dominant performance problems are not a lack of additional zero-cost
proxies. The latest run shows that:

1. architecture candidates are compared before their learning curves are
   informative;
2. the one-epoch recipe tournament selected the eventual best recipe on only
   one of three datasets and can return a model worse than its input
   checkpoint;
3. the controller is not truly anytime and left 8-12 minutes unused on two
   30-minute datasets;
4. the search objective and uncertainty handling are imperfect;
5. the architecture portfolio still assumes too much about the semantic
   meaning of channels, height, and width;
6. large-input RAM safety has not been exercised by the latest run.

The recommended direction is therefore:

> Keep the independent-model portfolio, but reorganize it around small
> label-aware representation probes, meaningful minimum fidelity, an
> incumbent-safe multi-stage recipe/optimizer race, and a genuinely anytime
> controller. Add new representation families only after the controller can
> measure them reliably.

The recommended first implementation step was a **controller-only change**:
replace the one-stage recipe tournament with an incumbent-preserving,
two-stage recipe race evaluated on the same validation data, while leaving the
architecture families unchanged. This step is now implemented. Optimizer
expansion, architecture-family changes, augmentation changes, and the other
priorities in this document have deliberately not been bundled with it.

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

## 3. Concise description of the current NAS architecture

The active implementation is documented in
[`submission/NAS_ARCHITECTURE.md`](submission/NAS_ARCHITECTURE.md). Its first
section explicitly overrides contradictory historical descriptions later in
that file. The code remains the final authority.

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

Tier 1 search is capped at 18% of remaining time or 360 seconds. Tier 2 is
capped at 12% or 120 seconds.

The Tier 1 pipeline is:

1. build a parameter-filtered candidate pool;
2. sample architectures across active families and parameter-size strata;
3. compute SynFlow, Jacobian correlation, NASWOT, and proxy latency;
4. assign equal family quotas to 12 label-aware entries;
5. train them for three successive-halving rounds on a cached partial epoch;
6. retain AdamW optimizer state between rounds;
7. refine the final two on a deterministic full-data stream;
8. evaluate both on full validation;
9. select using full validation, preceding fidelity, and rank history;
10. clone the winning checkpoint and give each active recipe one equal
    training segment;
11. return the best recipe trial as the warm-started model.

The final architecture search uses one neutral `architecture_probe` recipe.
The active final recipes are:

- `stable`;
- `regularized`;
- `balanced` when class imbalance is detected;
- otherwise `fast_fit` when examples per class are high.

### 3.4 Final trainer

[`submission/trainer.py`](submission/trainer.py) performs:

- real train/validation throughput calibration;
- measured prediction-time reservation;
- AdamW continuation from the selected search state;
- AMP on CUDA;
- gradient clipping;
- label smoothing, optional MixUp, and optional class weighting;
- warmup, validation-driven LR reduction, and a wall-clock cosine LR ceiling;
- full-validation checkpoint selection;
- independent recipe attempts from a common NAS checkpoint;
- recursive prediction-batch splitting after CUDA OOM;
- restoration of the globally best checkpoint before prediction.

The trainer currently allows up to one independent retry when the benchmark is
already met and up to two retries while it remains unmet. It can stop after
available recipes are exhausted even when substantial safe time remains.

## 4. Evidence from the latest run: baseline facts

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

## 5. Most important weaknesses and latest-run evidence

### 5.1 Early architecture fidelity is not predictive enough

**Implementation involved**

- `NAS._cache_search_batches()`
- `NAS._train_low_fidelity()`
- `NAS._candidate_utility()`
- `NAS._successive_halving()`

The first three rounds use at most a cached partial epoch and repeatedly start
at the beginning of that cache. Accuracy is the dominant promotion signal.

**Latest-run evidence**

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

**Latest-run evidence**

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

**Latest-run evidence**

- AddNIST used 97.9% of the 30-minute budget.
- Gutenberg ended with about 12m29s remaining and used only 58.4%.
- Language ended with about 8m37s remaining and used only 71.3%.

Gutenberg stopped after `regularized` and one `fast_fit` retry even though
`stable` had not received a full trainer attempt and substantial safe time
remained. Language exhausted three recipes but then stopped rather than using
the remaining time for a new seed, EMA/SWA, a challenger architecture, or an
incumbent-safe continuation.

For a multi-hour Phase 3 run, search is still capped at six minutes and the
trainer is still capped at 400 epochs or a small number of recipe attempts.
The current code therefore does not scale its exploration with the unknown
final time allocation.

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

**Latest-run evidence**

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

**Latest-run evidence**

For all three datasets, all proxy top-three candidates were `factorized`.
None of the three final winners was factorized:

- AddNIST: `spatial_pyramid`;
- Gutenberg: `spatial`;
- Language: `dual_axis`.

This does not prove that proxies are harmful because the quota system kept
other families alive. It does show that their cross-family ranking has almost
no positive evidence in the latest run. Proxy effort should not be increased
until its selected-versus-final rank correlation has been measured.

### 5.6 Input fingerprinting does not cover enough axis semantics

**Implementation involved**

- `helpers.inspect_data_properties()`
- `SearchSpace.__init__()`
- `SearchSpace._build_axis_encoder()`
- `SearchSpace._build_spatial_model()`

The current content fingerprint is strongest for sparse one-hot grids. It does
not explicitly detect:

- independent per-channel views;
- a channel axis representing physical depth or ordered time;
- a short sequence of high-dimensional continuous embeddings;
- small position-sensitive boards whose values are not binary;
- fixed-coordinate scientific data where translation/crop augmentation is
  invalid.

**Latest-run evidence**

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

The current structural rules also conflate modality with grayscale/color:

- grayscale data bypasses the label-aware flip probe;
- standardized or structured multi-channel data can still receive random
  crop;
- one-hot sequences receive no geometry, which is appropriate for the logged
  language tasks but is not a general solution.

**Latest-run evidence**

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

All active recipes use AdamW. The recipe space varies only LR scale, weight
decay, label smoothing, MixUp, and class weights. Architecture dropout is tied
to channel count rather than searched. There is no EMA, SWA, optimizer
alternative, stochastic depth, or optional post-selection refit.

**Latest-run evidence**

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

**Latest-run evidence**

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

1. Preserve class priors in the main search-validation sample.
2. Track a separate balanced metric instead of balancing the sample itself.
3. Split validation deterministically into selection and confirmation subsets.
4. Log validation loss, accuracy, balanced accuracy, margin, examples seen,
   data coverage, seed, LR, time, and peak memory for every candidate.
5. Use the same stochastic seed schedule and data order for paired candidate
   comparisons.
6. Evaluate two seeds only for final tied candidates.
7. Convert normalization statistics to a byte-bounded streaming calculation.
8. Add byte caps to both train and validation caches.
9. Avoid whole-dataset dtype conversion.
10. Calibrate microbatch size before architecture racing and use gradient
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
4. [ ] Complete the full planned promotion-signal bundle.
   - [x] Promote two recipes using best validation accuracy and bounded
     validation-loss slope.
   - [ ] Add an explicit train-validation-gap term after it can be measured
     fairly without increasing tournament data passes.
5. [x] Preserve the best evaluated stage state within each recipe trial, not
   only its last state.
6. [x] Accept a recipe-trained state only if it beats the incumbent beyond a
   configurable noise margin.
7. [x] Pass the winning optimizer state into Trainer only when it belongs to the
   returned winning checkpoint.
8. [ ] Add SGD with Nesterov and cosine decay as a genuinely different optimizer
   path; retain AdamW recipes.
9. [ ] Add a zero-smoothing or very-low-smoothing recipe.

Completed implementation scope:

- `submission/nas.py` now measures unsmoothed validation cross-entropy,
  evaluates the incumbent and all trials on the identical source, stages
  recipe promotion, restores the best evaluated stage, and applies an
  acceptance margin of at least 0.10 percentage points or one validation
  example.
- `test_controller_refinements.py` covers incumbent preservation, identical
  initial states and seeds, equal stage work, loss-slope ranking, checkpoint
  restoration, optimizer-state ownership, and short-budget fallback.
- No optimizer family, architecture family, augmentation, or final-Trainer
  behavior was changed in this step.

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

1. Create one compact, calibrated probe model for each plausible
   representation family.
2. Train probes on the same distinct data coverage for approximately one
   epoch, or until loss slope is measurable.
3. Promote the best two families with uncertainty-aware ties.
4. Search macro dimensions only inside the promoted families.
5. Use progressive data coverage rather than repeatedly replaying the
   beginning of a fixed cache.
6. Treat zero-cost proxies as ordering or duplicate-reduction tools only.
7. Promote using loss, accuracy, slope, and an optimistic confidence bound.
8. Continue tied full-validation finalists rather than blending an arbitrary
   0.03-point difference with old ranks.
9. Retain the second finalist's specification and optional checkpoint for an
   anytime fallback.

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

1. Replace the fixed six-minute Tier 1 search cap with a budget curve based on:
   - total remaining time;
   - measured epoch time;
   - number of plausible families;
   - whether architecture ranking has stabilized.
2. Remove `max_epochs = 400` as a hard multi-hour ceiling; retain only a very
   high safety ceiling plus the clock.
3. Replace retry-count termination with an ordered anytime action queue:
   - continue the best checkpoint with EMA;
   - run an untried optimizer/recipe;
   - run a new seed from the best common checkpoint;
   - continue the second architecture finalist;
   - try SWA or same-architecture checkpoint averaging;
   - optionally refit after locking the policy.
4. Keep the global best validation checkpoint immutable.
5. Estimate prediction reserve from measured batches with a conservative upper
   bound rather than an unconditional 180-second maximum.

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

- recovers the 29% and 42% unused budget observed on Language and Gutenberg;
- adapts to multi-hour final runs;
- permits additional seeds or representations without risking the best model.

Primary risk:

- more trials increase validation-selection pressure. Confirmation validation
  and uncertainty thresholds are therefore prerequisites.

### Priority 4: Expand representation families

Add families in this order so each addition can be ablated.

#### 4.1 Categorical sequence family

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

#### 4.2 Dense embedding sequence family

For one short axis and one large continuous feature axis:

- treat the short axis as sequence length;
- apply per-token LayerNorm and linear projection;
- use attention, TCN, or pooled token MLP across sequence positions;
- do not convolve as if adjacent embedding dimensions were image pixels.

Likely coverage:

- Cryptic-like pretrained embeddings;
- spectrogram or feature-vector sequences when detected by probes.

#### 4.3 Channel-independent and multi-view family

- shared or separate 2D stems per channel or channel group;
- late feature fusion;
- relation/counting head;
- optional permutation-aware or permutation-invariant fusion selected by a
  probe.

Likely coverage:

- AddNIST/MultNIST-like independent channel images;
- multi-sensor inputs.

#### 4.4 Volumetric and temporal-spatial family

- small Conv3D or axial 3D convolutions when channel and spatial dimensions
  appear exchangeable;
- temporal depthwise convolution plus spatial 2D backbone for ordered
  multi-time inputs;
- channel attention for grouped scientific variables.

Likely coverage:

- Voxel-like data;
- Windspeed-like temporal scientific grids.

#### 4.5 Position-sensitive board and hybrid families

- CoordConv or explicit normalized coordinates;
- 4x4 or adaptive multi-level spatial-pyramid pooling;
- attention pooling over spatial tokens;
- a compact spatial-plus-axis fusion model.

Likely coverage:

- Sudoku and Chesseract-like boards;
- ambiguous data where both local 2D structure and absolute position matter.

#### 4.6 Stronger generic image anchors

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

1. Keep an identity/evaluation transform available to every task.
2. Compare a very small augmentation portfolio with a compact anchor:
   - identity;
   - translation/crop;
   - verified horizontal/vertical flips;
   - light erasing or CutMix for natural-image-like data.
3. Do not infer flip safety solely from grayscale/color.
4. Detect train-validation distribution shift through simple per-axis moments
   and a cheap domain-classification probe.
5. Add EMA during every final attempt.
6. Test SWA or same-architecture checkpoint averaging when time remains.
7. Search dropout/stochastic-depth strength with the recipe.
8. After architecture, optimizer, and epoch policy are locked, test a
   train-plus-validation refit as a separate ablation. It must not use test
   labels or select on test predictions.
9. Consider test-time augmentation only for transformations already validated
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

Only after organizer confirmation:

- retain two finalists only when their validation accuracies are tied and their
  per-example errors are complementary;
- wrap them in one API-compatible `nn.Module`;
- confirm training and prediction memory;
- disable automatically under short budgets.

The rule-neutral alternative is EMA, SWA, or same-architecture checkpoint
averaging.

## 7. Exact files and components that would need to change

First-step status: `submission/nas.py` and
`test_controller_refinements.py` have been modified, and
`submission/NAS_ARCHITECTURE.md`, `submission/README.md`, and this plan have
been updated accordingly. The remaining files in this table are still
prospective.

| File | Components | Planned responsibility |
|---|---|---|
| `submission/nas.py` | `_cache_validation_batches`, `_train_low_fidelity`, `_candidate_utility`, `_final_selection_scores`, `_recipe_tournament`, `_successive_halving`, `_search_pipeline`, time-tier configuration | Prior-preserving validation, loss/margin metrics, uncertainty-aware racing, incumbent-safe recipe ASHA, scalable budgets, retained challenger |
| `submission/trainer.py` | recipe application, optimizer construction, scheduler construction, retry logic, time-cooldown logic, `train`, `predict` | SGD/Nesterov recipe, EMA/SWA, anytime action queue, no idle safe time, improved prediction reserve, optional refit/TTA |
| `submission/data_processor.py` | `NASDataset`, normalization statistics, augmentation construction, DataLoader construction | Lazy/per-batch dtype conversion, streaming byte-bounded statistics, searchable augmentation policies, safer large-data loading |
| `submission/helpers.py` | `inspect_data_properties`, batch-size helper, possible new memory/time helpers | Richer axis/modality fingerprints, memory estimates, train-validation shift statistics |
| `submission/search_space.py` | `ArchSpec`, blocks, model builders, analytical parameter counts | Token/embedding sequence, channel-independent, volumetric/temporal, position-sensitive, hybrid, WideResNet/ResNeXt/DenseNet-lite families |
| `submission/zero_cost_proxies.py` | proxy normalization and aggregation | Keep proxies weak; add diagnostics for rank correlation; avoid cross-family scale bias |
| `submission/ensemble.py` | currently inactive | Optional only after rule confirmation; otherwise leave unused |
| `submission/NAS_ARCHITECTURE.md` | active-controller section, pipeline, family and trainer descriptions | Must be updated in the same change as any implementation modification |
| `submission/README.md` | runtime modes and feature summary | Update after behavior changes |
| `test_controller_refinements.py` | controller unit/regression tests | Incumbent preservation, staged recipe promotion, time-budget behavior, tied finalists |
| `test_architecture_fixes.py` | search-space tests | New family activation, shapes, parameter counts, forward compatibility |
| `test_data_processor.py` | data-processing tests | No full dtype copy, streaming stats, byte caps, augmentation policy behavior |
| `test_full_pipeline.py` | end-to-end synthetic tiers | Full prediction count, short-budget fallbacks, no idle/crash behavior |
| New `test_memory_safety.py` | proposed | Large synthetic shapes/dtypes, cache-byte ceilings, OOM fallback |
| New `test_search_objective.py` | proposed | Prior-preserving validation, confidence ties, loss/slope promotion |

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
| Incumbent-safe staged recipe race | Fixes an observed 2/3 recipe-selection failure | Uses more early time | Must preserve prediction reserve |
| Minimum meaningful architecture fidelity | Lower selection regret | Fewer candidates evaluated | Remain dataset-agnostic |
| Scalable anytime controller | Uses otherwise wasted compute and long Phase 3 budgets | More validation adaptivity | Hard external clock checks remain mandatory |
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

1. **The current code is more authoritative than old prose.** The first
   "active controller" section of `submission/NAS_ARCHITECTURE.md` overrides
   historical sections later in that file.
2. **The latest run used the evaluator's 30-minute default** because the three
   metadata files lacked `time_limit`.
3. **The current total adjusted score is 5.135:** AddNIST `+4.177`,
   Gutenberg `+1.133`, Language `-0.176`.
4. **The recipe tournament was reliable on only one of three tasks.**
   Gutenberg eventually preferred `fast_fit`, and Language eventually
   preferred `regularized`, despite different one-stage tournament winners.
5. **Language entered Trainer below its pre-tournament architecture
   checkpoint:** 84.00% before the tournament versus an 82.89% trainer
   baseline.
6. **Early architecture accuracy is not reliable at the current fidelity.**
   The AddNIST winner moved from 7.69% in round 1 to 93.84% final validation.
7. **The controller left substantial time unused:** about 749 seconds on
   Gutenberg and 516 seconds on Language.
8. **All three proxy top-three lists were factorized, but no final winner was
   factorized.** Family quotas are useful; additional proxy weight is not
   currently justified.
9. **Gutenberg's final architecture comparison was an effective tie:**
   38.31% versus 38.28%. The current selector has no uncertainty model.
10. **The current search validation sample changes class priors.** This has not
    yet failed visibly because the three logged datasets were balanced.
11. **All final recipes use AdamW.** Optimizer diversity is currently absent.
12. **The architecture portfolio handles sparse one-hot axes better than
    continuous embedding sequences, channel-independent views, volume, time,
    or exact board position.**
13. **Historical four-dimensional tensors can be semantically non-image.**
    Shape alone is not a safe routing signal.
14. **Large-input memory safety is unproven.** The latest inputs were only
    24-28 pixels per side. Normalization and validation caching still have
    large temporary-allocation risks.
15. **The default `C:\Python314` environment has neither PyTorch nor pytest.**
    `C:\Users\Lennart\miniconda3\python.exe` does contain CPU PyTorch and was
    used to run the repository's executable regression scripts. CUDA and
    historical-dataset accuracy validation still require Colab or the
    competition-compatible environment.
16. **Do not tune against codenames.** `Adaline` is AddNIST and `LaMelo` is
    Language in the log, but implementation decisions must remain based on
    tensor content, labels, measured learning, memory, and time.
17. **Do not modify `evaluation/main.py` or `evaluation/score.py`.** They are
    organizer-controlled and overwritten during evaluation.
18. **The existing `submission/ensemble.py` is inactive.** Ensembling is not
    required by this plan and should not be enabled without organizer
    confirmation.
19. **The first implementation step is complete.** `submission/nas.py` now
    contains the incumbent-safe staged recipe race, with regression coverage
    in `test_controller_refinements.py`. No independent architecture,
    augmentation, or Trainer change was included.
20. **Before any future edit, recheck `git status` and preserve unrelated user
    changes.**

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
