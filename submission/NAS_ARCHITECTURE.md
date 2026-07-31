# NAS Architecture

Status: 31 July 2026

This document describes the active competition implementation. The code is
authoritative if a future change makes this document stale.

## 1. Competition interface and invariants

The evaluator calls:

- `DataProcessor.process()` and expects ordered train, validation, and test
  loaders;
- `NAS.search()` and expects one `torch.nn.Module`;
- `Trainer.train()` and expects one trained `torch.nn.Module`;
- `Trainer.predict()` and expects one class prediction for every test example.

The controller never routes on the dataset codename and never reads test
labels. It uses tensor contents, shapes, class statistics, validation
measurements, resource measurements, and the supplied clock. The active
submission contains no DARTS, einspace grammar, supernet weight sharing,
external pretrained weights, or final ensemble.

Organizer-owned files under `evaluation/` are not part of the implementation
and must not be modified or packaged as submission code.

## 2. Data processing and memory behavior

`data_processor.py` and `helpers.py` build a deterministic data profile.
Fingerprint sample counts are constrained by both an example limit and a byte
limit. The profile records:

- actual channels, height, width, value range, moments, and spatial variance;
- sparse/binary and row/column one-hot structure;
- categorical-width, categorical-height, dense-sequence, position-sensitive,
  board, channel-independent, volumetric, temporal, factorized, and
  natural-image confidence;
- channel correlation, approximate value cardinality, class imbalance, and
  smoothed inverse-frequency class weights;
- train/validation moment shift;
- label-aware horizontal/vertical flip safety.

`NASDataset` keeps the source storage dtype. A `uint8` or `float64` dataset is
not converted wholesale to float32; each requested example is converted just
before its transform. Channel normalization is computed with float64
accumulators over deterministic float32 chunks capped at 16 MiB by default.
The data fingerprint, shift sample, and flip-safety sample also have byte
ceilings.

The test loader is never shuffled and never drops its last batch.

## 3. Augmentation portfolio

Every dataset exposes an identity/evaluation transform. Depending on the
fingerprint, the safe training portfolio can also contain:

- `conservative`: small crop or translation, verified flips, and cautious
  erasing for suitable natural-image inputs;
- `safe_flips`: only flips that passed the label-aware invariance check.

Detected train/validation shift makes identity the initial incumbent.
Before proxy screening starts, NAS trains byte-identical compact anchors with
the same examples and seed under every available policy. A policy replaces
the incumbent only after a validation-accuracy gain beyond the sampling margin
or an accuracy tie with a material validation-loss improvement. This occurs
before worker processes have consumed the training dataset, so the selected
policy is the one seen by later search and training.

## 4. Architecture portfolio

All candidates are independent modules. The always-available safety families
are:

| Family | Inductive bias |
|---|---|
| `spatial` | residual 2D CNN with global pooling |
| `spatial_pyramid` | residual CNN retaining global and 2x2 layout |
| `factorized` | depthwise-separable residual CNN with GroupNorm |

The content profile can activate:

| Family | Inductive bias |
|---|---|
| `axis_width` | ordered 1D residual encoder along width |
| `axis_height` | ordered 1D residual encoder along height |
| `dual_axis` | fused width and height encoders |
| `categorical_sequence` | soft token projection, multi-kernel TCN, attention, ordered pooling |
| `dense_sequence` | per-token LayerNorm/projection, TCN, attention, ordered pooling |
| `multiview` | shared per-channel 2D encoder with invariant and ordered late fusion |
| `volumetric` | compact Conv3D over channel/depth and spatial dimensions |
| `coord_spatial` | explicit normalized coordinates and 1x1/2x2/4x4 pyramid |
| `spatial_axis` | local 2D features fused with raw row/column summaries |
| `wide_residual` | GroupNorm pre-activation wide/grouped residual anchor |
| `dense_reuse` | compact DenseNet-style feature reuse |

Axis encoders add absolute sinusoidal position features and keep multiple
ordered bins. Semantic families have only three calibrated anchor
specifications rather than the full residual Cartesian grid. This adds
different inductive biases without consuming hundreds of near-duplicate
slots. Exact module parameter counts are used for semantic anchors; the legacy
families retain their tested analytical counts.

## 5. Validation and search measurements

Validation is sampled in original class proportions, not balanced by
resampling. NAS tracks ordinary accuracy as the competition objective and
balanced accuracy as a separate diagnostic.

The byte-bounded validation cache is deterministically split into:

- a 75% selection subset used by adaptive search decisions;
- a disjoint 25% confirmation subset used for finalist and recipe acceptance.

The default maximum is 4,096 examples and 64 MiB. Tiny splits safely fall back
to one selection set.

Candidate state records unsmoothed cross-entropy, accuracy, balanced accuracy,
top-two probability margin, examples seen, progressive data cursor, seed
schedule, final learning rate, elapsed time, examples/second, and CUDA peak
allocation. Accuracy remains the dominant ranking signal. Loss slope,
uncertainty, train/validation gap, and efficiency are bounded near-tie terms.

## 6. Hierarchical NAS controller

### 6.1 Time tiers

| Tier | Remaining time | Controller |
|---|---:|---|
| 1 | at least 15 minutes | 54 proxy candidates, family probes, 12 macro finalists |
| 2 | 5–15 minutes | 30 proxy candidates, 7 macro finalists |
| 3 | under 5 minutes | direct robust anchor |

Tier 2 protects final training after at most 12% or 120 seconds of search.
Tier 1 uses 22% of the remaining time with a scalable cap: at least the former
360-second ceiling is available on long runs, and the cap grows to at most
1,800 seconds on multi-hour runs. Every phase also checks the protected
remaining-time deadline.

### 6.2 Proxy pre-screen

SynFlow, Jacobian correlation, NASWOT, and measured proxy latency only order
or reduce the initial pool. Proxy values do not enter later label-based
utility. Deterministic family anchors are considered first, and sampling spans
family and parameter-size strata. The generic full-grid families include a
genuinely wide C64 anchor when it survives the parameter guard.

### 6.3 Representation probe race

Before macro dimensions compete, one compact anchor from every represented
family is evaluated untrained and receives an equal progressive training
segment. Accuracy, validation loss improvement, uncertainty, and
train/validation gap determine promotion. The best two families advance; a
third advances when its uncertainty interval overlaps. Promoted probes receive
a second equal segment on new data positions.

Only macro candidates from promoted families enter successive halving. If the
clock or evidence is insufficient, all represented families remain eligible.
After promotion, the controller restores every feasible specification from
the promoted families before selecting macro finalists. Deterministic
log-parameter quantiles provide compact, medium, and high-capacity coverage;
proxy rank breaks only same-size ties. The number of label-trained finalists
does not increase.

Every controller evaluation first places its model on the controller device;
this includes newly constructed probes and probes parked on CPU between
stages. A runtime-invalid family is skipped locally. If fewer than two probes
remain, successful families are retained; if none remain, the conservative
spatial/factorized family set is restored rather than aborting the dataset.

### 6.4 Macro architecture race

The train cache is capped at 128 MiB and one epoch. Its first batch is sliced
if necessary, so even an oversized first batch cannot exceed the ceiling.
CUDA performs one compact calibration update before the race. Logical batches
are retained while OOM recursively halves a reusable microbatch and accumulates
gradients.

Successive-halving survivors retain optimizer state and use a progressive data
cursor instead of replaying the start of a cache. Finalists then receive equal
deterministic full-data passes while they still improve and the recipe reserve
is safe. Each finalist keeps its best evaluated model state together with the
matching optimizer state, step count, and data cursor. That state is restored
before final selection, so a later partial or degrading pass cannot erase a
stronger refinement checkpoint.

The last architectures are compared on both validation splits. A candidate
that clearly wins confirmation accuracy cannot be overturned by adaptive
history. When confirmation accuracies overlap within a bounded sampling
interval, confirmation loss resolves the tie, followed by the combined
selection/confirmation score and finally historical rank. Historical rank is
therefore strictly a tie-break rather than an additive accuracy surrogate.

The runner-up specification is always recorded. Its CPU checkpoint is retained
only for a statistical tie. It is stored in a dormant plain Python bundle, so
it is not a registered branch, does not change the returned model's parameter
count or forward pass, and cannot act as an ensemble.

## 7. Recipe and optimizer race

Architecture comparison uses one neutral zero-smoothing AdamW probe. After an
architecture wins, byte-identical checkpoint clones enter an incumbent-safe
two-stage recipe race:

| Recipe | Main distinction |
|---|---|
| `stable` | AdamW, zero smoothing, reduced dropout |
| `regularized` | AdamW, stronger decay/smoothing/dropout and optional MixUp |
| `balanced` | AdamW with class weights when imbalance is material |
| `fast_fit` | AdamW with higher LR and reduced regularization on large datasets |
| `sgd_nesterov` | SGD, momentum 0.9, Nesterov, cosine decay, zero smoothing |

Only `balanced` or `fast_fit` is added for the applicable condition, so a run
normally races three or four recipes.

All recipes receive the same stage-one work, data order, and seed. The best
two receive equal stage-two work. Promotion uses best accuracy plus bounded
loss slope and train/validation-gap terms. Each trial restores its best
evaluated stage and matching optimizer state.

A trial must improve both selection and confirmation accuracy by at least:

`max(0.001, 1/N, min(0.01, 0.75 * pooled_standard_error))`

The margin is recomputed for each incumbent/challenger comparison. Otherwise
the unchanged architecture incumbent and its matching optimizer state are
returned.

## 8. Final anytime trainer

The trainer benchmarks real train and validation steps before optimization.
The prediction reserve is:

`max(20 seconds, 1.75 * measured_prediction_time + 15 seconds)`

and includes every enabled test-time view. It has no arbitrary 180-second cap.
The hard epoch ceiling is 100,000 and exists only as protection against a
broken synthetic clock; the supplied competition clock is the normal stop.

AdamW recipes use validation-driven `ReduceLROnPlateau`. SGD uses Nesterov and
a single monotonic cosine decay. Once the initial cosine horizon is exhausted,
the LR remains at its floor; exceeding an early epoch estimate cannot create
an accidental warm restart. Both paths use warmup, AMP on CUDA, gradient
clipping, and a monotonic wall-clock cooldown ceiling.

The ordered anytime queue is:

1. continue the NAS-selected recipe and preserve its warm start;
2. run every untried recipe/optimizer from the common NAS checkpoint with a
   new deterministic seed;
3. when retained, train the tied second architecture as a single challenger;
4. restart from the immutable global-best model with progressively smaller
   learning rates and fresh seeds while another full epoch is safe;
5. evaluate the current attempt's EMA and the top three same-architecture
   checkpoint average if a complete validation pass still fits.

An attempt can rotate before reaching its LR floor when it has received a
minimum runway, remains materially below the immutable starting checkpoint,
has stopped improving, and has developed a large train/validation gap. An
attempt that already improved the checkpoint receives a much longer runway
and rotates only after a long plateau at a substantially decayed LR. A large
gap prioritizes the regularized alternative. These conditions are based only
on validation behavior and resource measurements.

The global best validation checkpoint never regresses. If a different
architecture wins, Trainer returns it and updates the model used by
`predict()`.

EMA runs during every attempt. Test-time augmentation is enabled only when the
functional augmentation probe selected `safe_flips`; prediction averages the
original logits and only the verified horizontal/vertical views.

## 9. Failure behavior

- Search and validation materialization are byte bounded.
- Controller evaluation is a device-safe boundary: the model is moved before
  inputs, including for fresh representation probes on CUDA.
- Validation CUDA OOM recursively splits the current batch, concatenates CPU
  logits in original order, and leaves metric accumulation on CPU.
- CUDA search OOM halves the microbatch and retries the same logical batch.
- A candidate is rejected only if a one-example microbatch still fails.
- A failed representation-family evaluation is contained to that family and
  cannot by itself terminate the dataset.
- Calibration OOM rebuilds final loaders at half batch size.
- In-epoch OOM preserves the global best and can reduce the loader.
- Prediction OOM recursively splits a batch while preserving order.
- Missing/failed proxy values cannot terminate the whole pipeline.

## 10. Rule-gated and empirical items intentionally disabled

- A complementary-model ensemble remains disabled until organizers explicitly
  confirm it is allowed. `metadata["nas_ensemble"]` is always `False`.
- Train-plus-validation refit remains disabled until leave-one-dataset-out
  experiments show that losing the stopping holdout is beneficial.
- No test-label adaptation, codename routing, downloaded hidden-data logic, or
  external pretrained weights are present.

## 11. Local verification

The current regression suite covers:

- all legacy and semantic family activations, forward shapes, and exact
  parameter counts;
- source-dtype preservation, streaming statistics, train/validation byte caps,
  disjoint prior-preserving confirmation, and logical microbatch accumulation;
- incumbent preservation, staged recipe fairness, loss/gap/confirmation
  objectives, uncertainty-aware acceptance, best-refinement restoration,
  holdout-dominant ordering, promoted-family capacity strata, and
  optimizer-state ownership;
- SGD/Nesterov/cosine construction, no-rebound scheduling,
  incumbent-relative attempt rotation, incompatible-state rejection,
  EMA/state helpers, safe-flip TTA, and dormant challenger registration;
- an accelerated Tier-1 end-to-end flow through processing, NAS, anytime
  training, and complete ordered prediction.

CPU tests pass locally. The original three datasets and a CUDA GPU are not
available in this workspace, so multi-seed historical accuracy and real CUDA
peak-memory comparisons remain required before submission.
