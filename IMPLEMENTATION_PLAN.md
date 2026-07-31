# Robust Anytime NAS - Implementation Plan

## 1. Executive recommendation

Build a new submission directory, `submission_robust_nas`, containing an anytime,
failure-aware neural architecture search pipeline for arbitrary 4-D classification data.
The core method should be:

1. Profile and sanitize the dataset without copying it unnecessarily.
2. Construct a guaranteed-safe fallback network and run a one-batch preflight.
3. Search a small, conditional family of compact residual CNNs using a portfolio of
   strong defaults plus a few random mutations.
4. Allocate training with gentle successive halving, using optimizer steps as the
   fidelity rather than assuming that an epoch has comparable cost across datasets.
5. Continue training the best completed candidate while continuously reserving time
   for inference.
6. Return valid predictions even if search, training, CUDA, or a candidate fails.

This is intentionally a conservative NAS design. The course emphasizes that search
space design matters more than an elaborate search strategy, random search is a strong
baseline, and multi-fidelity assumptions can fail. The competition makes the same
trade-off unusually sharp: a timeout or memory crash is worth -10 for the dataset.

### Proposed research question

> Can a clock-aware, failure-aware, multi-fidelity search over a compact conditional
> CNN space outperform a fixed ResNet baseline on unseen 4-D classification tasks while
> completing every run within an unknown time and memory budget?

### Optimization priority

Use a lexicographic objective during development:

1. Zero failed datasets.
2. Zero invalid or incomplete prediction arrays.
3. Best worst-dataset adjusted score.
4. Best total adjusted score.
5. Lower runtime and memory among statistically indistinguishable configurations.

Do not trade a material increase in failure probability for a small mean accuracy gain.

## 2. Authoritative constraints and rule interpretation

The implementation must honor the following competition contract:

- The evaluator imports `DataProcessor`, `NAS`, and `Trainer`, then calls
  `process()`, `search()`, `train()`, and `predict()` in that order.
- Test data are 4-D arrays in NCHW form. The final tasks are novel classification
  datasets and must not be downloaded or otherwise probed.
- The test loader must use `shuffle=False` and `drop_last=False`.
- Predictions must contain exactly one class label per test example, in original order.
- The score is accuracy normalized around the supplied benchmark. Parameter count and
  runtime are reported but have no direct score term; their practical cost is slower
  training and a higher chance of the -10 failure penalty.
- The live `clock.check()` value is authoritative. A cached `metadata["time_remaining"]`
  value is only a snapshot.
- A timeout or RAM/VRAM crash receives -10 for that dataset.
- The 2026 starter-kit note defines time per dataset and applies 0.5 hours when
  `metadata["time_limit"]` is missing. The older README runtime paragraph and the
  website's older 24-hour wording conflict with this. The implementation must not rely
  on either fixed duration; it must adapt to the live clock.
- Phase 2 permits at most seven submissions. The last working Phase 2 submission is
  reused for Phase 3, so the first priority is a working, conservative baseline.
- Do not include `main.py` or `score.py` in the submission. Do not alter or bypass the
  evaluator clock.
- Avoid downloads, pretrained weights, network calls, and undeclared libraries.
  Depend only on Python's standard library, NumPy, PyTorch, and optionally components
  known to exist in the supplied environment. In practice, the core submission should
  not need scikit-learn or torchvision.

Official references:

- [Competition rules](https://www.nascompetition.com/rules)
- [Technical details](https://www.nascompetition.com/info)
- [Starter kit](https://github.com/Towers-D/NAS-Comp-Starter-Kit)

### Local evaluator issue to resolve before testing

The checked-in `evaluation/main.py` currently defines:

```python
def is_out_of_time(clock: Clock, metadata, grace_time: bool):
```

but calls it three times without `grace_time`. As written, the local evaluator raises a
`TypeError` before the submission code runs and records a failed dataset. Confirm the
latest evaluator with the organizers. For local testing only, the likely correction is
`grace_time: bool = False`. Never ship a modified `main.py`; the official evaluator will
overwrite it and the rules explicitly exclude it from submissions.

## 3. Why this method fits the lecture material

| Course topic | Design decision |
| --- | --- |
| Full-workflow AutoML and CASH | Search architecture, optimizer, regularization, and a small number of safe preprocessing/training choices jointly. |
| Evaluation and meta-overfitting | Limit the number of trials, separate search and confirmation validation when feasible, compare multiple seeds locally, and never use test data. |
| Random/local/evolutionary search | Start with a robust portfolio and random conditional samples; use one-field mutations only when extra budget exists. Avoid grid search. |
| Bayesian optimization | Do not use BO in the default 30-minute path: there are too few reliable observations for a mixed-space surrogate, and a good initial design alone would consume much of the budget. A surrogate is an optional later extension only. |
| Multi-fidelity HPO | Use successive halving over optimizer steps and validation sample size. Promotion is gentle because low-fidelity rankings can be wrong. |
| Warmstarting and priors | Seed search with task-agnostic anchor architectures and select additional anchors from simple dataset meta-features. Never specialize on a codename. |
| NAS search spaces | Use a compact chain/cell-inspired residual space with conditional blocks, downsampling, pooling, and positional options. |
| Performance estimation | Use short real training runs. Parameter count is a feasibility filter, not a quality proxy. Do not rely on zero-cost proxies or learning-curve extrapolation. |
| One-shot NAS and DARTS | Exclude them from the primary implementation because of supernet memory cost, weak inherited-weight ranking correlation, and DARTS brittleness. |
| Reproducibility and anytime behavior | Seed every source of randomness, log incumbent quality over elapsed time, and make all loops interruptible at batch boundaries. |

## 4. Submission layout

Create the following directory by copying the API shape of `submission_template`:

```text
submission_robust_nas/
  data_processor.py   # profiling, normalization, datasets, loaders
  nas.py              # portfolio, search, promotion, model selection
  trainer.py          # final training, checkpointing, prediction
  models.py           # safe model builder and residual blocks
  helpers.py          # clock budget, seeds, metrics, OOM and finite checks
  README.md            # method, assumptions, and reproducible test commands
```

Use imports such as `from models import build_model`, because the evaluator copies the
files into a flat execution directory. Do not create any file named `main.py` or
`score.py`.

## 5. Shared safety layer

Implement these primitives first because every later component depends on them.

### 5.1 `BudgetManager`

The class wraps the provided clock but never modifies it. It should:

- Capture the remaining seconds after the evaluator has loaded the NumPy arrays.
- Call `clock.check()` whenever a decision is made; never count down locally as the
  sole source of truth.
- Maintain named reserves for final training and prediction.
- Record an exponential moving average of batch, validation, and inference durations.
- Answer `can_start(estimated_cost, reserve)` before every rung, epoch, validation pass,
  and expensive model construction.
- Stop loops at minibatch boundaries once the remaining time reaches the relevant
  reserve. An internal stop condition must be handled locally, not raised into
  `evaluation/main.py`.

Prediction reserve should be:

```text
max(90 seconds,
    12% of the initial remaining budget,
    2.5 * measured validation inference time scaled to n_test)
```

Cap it at 25% of very short budgets so some training is still possible. The 2.5 factor
covers loader overhead, a slower test set, output checks, and allocator variance.

### 5.2 Reproducibility

- Seed Python, NumPy, CPU PyTorch, and all CUDA devices.
- Derive candidate seeds from a fixed base seed and candidate index, not from the
  hidden dataset codename.
- Use an explicit `torch.Generator` for every shuffled training loader.
- Prefer standard deterministic operators. Do not enable a strict mode that can throw
  on an unsupported CUDA kernel.
- Log every sampled configuration, seed, fidelity, score, runtime, and failure reason.

### 5.3 Failure containment

No individual candidate failure may abort `NAS.search()`:

- Catch CUDA out-of-memory errors, release references, call `torch.cuda.empty_cache()`,
  halve the microbatch, and retry once.
- If model weights themselves do not fit, reject that configuration and build a
  narrower fallback.
- Reject a candidate after repeated non-finite losses or logits.
- Catch ordinary candidate-construction/training exceptions, log them, and continue.
- Construct and preflight a safe fallback before sampling any risky candidate.
- `Trainer.train()` must return the best usable state it has if later training fails.
- `Trainer.predict()` must have a final majority-class path that returns the correct
  number of original-label predictions even if model inference becomes impossible.

The majority fallback may score poorly, but it prevents a pipeline exception and is
strictly preferable to an avoidable failed-run penalty.

## 6. `DataProcessor` plan

### 6.1 Profile before transforming

Inspect and store the following in the mutable metadata dictionary:

- train, validation, and test counts;
- channels, height, width, input dtype, and bytes per sample;
- class labels, counts, imbalance ratio, and majority label;
- finite-value rate, per-channel mean, and per-channel standard deviation;
- detected batch-size tier and available CUDA memory tier;
- whether the data are tiny, high resolution, highly imbalanced, or spatially
  degenerate (`H == 1` or `W == 1`).

Compute statistics from deterministic chunks or a capped sample, not by converting the
entire dataset to a second float tensor. Bound the profiling work by both a scalar-count
cap and the data-processing time allowance.

### 6.2 Shape and label handling

- Trust the NCHW contract for 4-D data.
- If a defensive 3-D input `(N, H, W)` appears, add a singleton channel dimension.
- Reject no valid size merely because `H`, `W`, or `N` is small.
- Accept bool, integer, float32, and float64 inputs; convert each batch to float32
  lazily.
- Convert one-hot labels to indices if encountered.
- Map the union of train and validation label values to contiguous indices for
  cross-entropy, store the inverse mapping, and map predictions back before returning
  them.
- Handle the one-class case explicitly with a constant classifier path.

### 6.3 Numerical preprocessing

- Derive normalization statistics from training inputs only.
- Replace NaN and infinities per batch using finite training statistics.
- Use per-channel standardization with an epsilon floor; set the scale of constant
  channels to one.
- Avoid unconditional per-image normalization, clipping, resizing, geometric flips,
  rotations, and crops. These can destroy global intensity, text, ordering, or exact
  symbolic structure in an obscure dataset.
- Implement optional mixup and label smoothing in the training loop, where they can be
  searched and disabled. The protected fallback uses neither.

### 6.4 Memory-safe datasets and loaders

- Keep references to the NumPy arrays and convert samples/batches lazily. Do not use
  `torch.tensor(full_array)`, which copies the complete dataset.
- Use `num_workers=0` as the reliable default. It avoids worker startup, pickling,
  duplicated array mappings, and hidden worker failures.
- Use `pin_memory=True` only on CUDA and only if the batch estimate is safe.
- Set `drop_last=False` on all loaders. This is mandatory for test and prevents empty
  training epochs on tiny datasets.
- Set `shuffle=True` only for training. Validation and test must preserve order.
- Ensure the test dataset yields an input tensor directly, not an `(input, label)` or
  one-element tuple.
- Select a conservative power-of-two loader batch from input elements and spatial
  resolution, capped at 128. Training code must still support smaller microbatches.

Before returning, assert that the test sampler is not random, `drop_last` is false, and
the loader exposes exactly `n_test` samples.

## 7. Architecture search space

The space should be expressive enough to cover texture, object, symbolic, positional,
and 1-D-like image tasks, but every sampled architecture must remain buildable.

### 7.1 Common model skeleton

```text
optional coordinate channels
  -> 1x1 or 3x3 input adapter
  -> 2-4 residual stages
  -> adaptive grid pooling (1x1, 2x2, or 4x4)
  -> dropout
  -> linear classifier
```

Use odd kernels with explicit integer padding and `AdaptiveAvgPool2d`, so every legal
height and width works. Downsample an axis only while its current size is large enough.
For `H == 1` or `W == 1`, use an anisotropic kernel/stride that does not invent repeated
downsampling along the singleton axis.

### 7.2 Conditional configuration space

| Choice | Values / condition | Rationale |
| --- | --- | --- |
| Block family | basic residual; depthwise-separable residual | Basic blocks are a quality anchor; separable blocks protect memory on large inputs. |
| Base width | 16, 24, 32, 48 | Small enough for unknown hardware; upper values enabled only after preflight. |
| Stages | 2, 3, 4 | Conditional on spatial size and memory. |
| Blocks per stage | 1 or 2, with an optional third block only on small inputs | Provides genuine depth search without a huge graph space. |
| Kernel | 3; optional 5 in a depthwise block | Larger receptive field without quadratic standard-convolution cost. |
| Normalization | GroupNorm default; BatchNorm only when effective batch is at least 16 | GroupNorm remains valid at microbatch 1. |
| Activation | SiLU or ReLU | Both are available in the supplied PyTorch generation. |
| Downsampling | strided convolution; max-pool stem only for high resolution | Prevents aggressive information loss on small symbolic grids. |
| Pooling grid | 1, 2, or 4 per non-singleton axis | Grid pooling preserves coarse absolute layout when global pooling is too invariant. |
| Coordinate channels | off/on | Gives positional tasks an explicit absolute-position option. |
| Channel attention | off; lightweight squeeze-excitation when memory allows | Conditional refinement, not required by the fallback. |
| Dropout | 0, 0.1, 0.25 | Scaled up for small data. |
| Optimizer | AdamW default; one SGD anchor for sufficiently large datasets | AdamW converges quickly under short budgets; SGD is a diversity candidate. |
| Learning rate | log-spaced values around `3e-4` to `3e-3` for AdamW | CASH rather than architecture-only tuning. |
| Weight decay | `1e-5`, `1e-4`, `1e-3` | Conditional on model/data size. |
| Mixup | off or mild | No geometric assumption; fallback remains off. |
| Class weighting | off/on only for strong imbalance | Accuracy remains the selection metric, so weighting must prove itself on validation. |

Apply both parameter and activation guards before training. Suggested parameter caps are
approximately 0.75M for fewer than 2,000 training examples, 2M for 2,000-20,000, and
5M above that, reduced for high-resolution inputs or low-memory devices. These are
safety priors, not scoring penalties.

### 7.3 Protected anchor portfolio

Always generate these first:

1. **Fallback tiny residual:** width 16, two stages, GroupNorm, no attention, no mixup.
2. **Balanced residual:** width 24/32, three stages, basic blocks, global or 2x2 pooling.
3. **Position-aware residual:** coordinate channels and 4x4 pooling when dimensions
   permit.
4. **Efficient large-input model:** depthwise-separable blocks, early safe
   downsampling, 1x1/2x2 pooling.

Meta-features decide the order and whether an anchor is applicable, but never select a
model by codename. Fill remaining slots with condition-aware random samples. When the
budget is long, mutate one field of a high-performing configuration at a time, similar
to local or aging evolutionary search.

## 8. `NAS.search()` plan

### 8.1 Preflight

Build the fallback first and run one forward/backward update plus validation inference.
This establishes:

- that input and output shapes are valid;
- a safe starting microbatch;
- approximate train and inference throughput;
- an initial parameter/activation feasibility result;
- a usable model to return if every later step fails.

If CUDA preflight fails, shrink width and microbatch. Use CPU only if CUDA is absent;
switching an entire run to CPU after a late CUDA failure is unlikely to meet the clock.

### 8.2 Validation discipline

Use only train and validation labels during search.

- If validation has enough examples per class, split it deterministically and
  stratifiably into a search half and a confirmation half.
- Use the search half for early rungs.
- Evaluate only the top two completed candidates once on the confirmation half.
- If validation is too small for this split, use it whole but reduce the trial count.
- Rank primarily by accuracy, break ties with cross-entropy, and select the smaller
  model when accuracy is within a small practical tolerance (for example 0.5 percentage
  points).

This does not create a fully nested benchmark inside the competition, but it reduces
meta-overfitting without spending the budget of cross-validation.

### 8.3 Multi-fidelity scheduler

Use optimizer updates, not raw epochs, as the main fidelity:

- Rung 0: 16-64 updates, capped at one pass over the chosen train subset.
- Rung 1: twice the cumulative updates and a larger train/validation subset.
- Rung 2: four times the initial updates and full validation.
- Keep roughly the best half at each rung (`eta = 2`).

Resume promoted candidates rather than starting them again. Use the same number of
examples and deterministic data order for candidates within a rung. Keep candidate
states on CPU between rungs and cap total checkpoint memory; discard eliminated states
immediately.

Protect the fallback until at least one other candidate has completed a comparable
fidelity. Do not promote exclusively from a zero-cost proxy or an extrapolated learning
curve. Low-fidelity score, parameter count, and stability are evidence, not guarantees.

### 8.4 Budget tiers

Choose behavior from the remaining live budget after data loading:

| Remaining time | Search behavior |
| --- | --- |
| Under 5 minutes | No architecture search. Preflight the fallback and train it; keep at least 20% for prediction. |
| 5-15 minutes | Three anchors, one short rung, optionally continue the top two. Use at most 15-20% for search. |
| 15-45 minutes | Four anchors plus two random/mutated candidates, two or three rungs. Use about 25% for search. |
| 45-120 minutes | Eight candidates, three rungs, then a few local mutations. Use at most 30-35% for search. |
| Over 120 minutes | Add Hyperband-style bracket diversity or aging evolution, but retain the same hard prediction reserve. |

For the 30-minute default, target approximately:

- at most 45 seconds for profiling and preflight;
- about 7 minutes for search;
- about 18 minutes for final training;
- at least 4 minutes for prediction, validation overhead, and variance.

These are ceilings, not timers. Measured throughput and `clock.check()` decide whether
the next unit of work is safe.

### 8.5 Search return value

Return the best completed, finite model state, with its configuration and search history
attached as attributes and/or metadata. If no challenger completed safely, return the
preflighted fallback. Never return an uninitialized or known-invalid model.

## 9. `Trainer` plan

### 9.1 Final optimization

- Continue from the selected search checkpoint by default; this avoids throwing away
  scarce compute.
- Use mixed precision only on CUDA, with `GradScaler`.
- Use gradient accumulation so loader batch size and optimizer effective batch size are
  decoupled.
- Clip gradient norm and reject/skip non-finite batches.
- Use cosine decay with a short warm-up, parameterized by update count rather than a
  fixed epoch assumption.
- Evaluate only when the estimated validation cost fits outside the prediction reserve.
- Keep the best finite state dictionary in CPU memory.
- Stop at minibatch boundaries if the next evaluation/training block is unsafe.

When sufficient time remains, restore the best validation checkpoint and fine-tune for
a small, predetermined number of low-learning-rate updates on a concatenation of train
and validation data. This follows the course recommendation to refit after selection.
Skip this step whenever it endangers the inference reserve.

### 9.2 OOM-adaptive microbatching

Each loader batch should be splittable into microbatches. On a CUDA OOM:

1. Clear partial gradients and temporary tensors.
2. Halve the microbatch.
3. Retry the full logical batch once.
4. If microbatch one still fails, restore the best state and stop training.

Do not rebuild a larger model after any OOM. Record the recovered microbatch in metadata
so prediction starts conservatively.

### 9.3 Prediction guarantee

`predict(test_loader)` must:

- use `eval()`, `inference_mode()` or `no_grad()`, and CUDA autocast when safe;
- preserve loader order;
- concatenate class indices, inverse-map them to original labels, and return exactly
  `len(test_loader.dataset)` values;
- halve the inference microbatch after an OOM;
- replace non-finite logits with safe finite values before `argmax`;
- assert output length and label membership before returning;
- on any unrecoverable inference exception, return the stored training-majority label
  repeated exactly `n_test` times.

The prediction path should receive the largest time reserve because the evaluator only
scores a completed prediction file.

## 10. Testing and evaluation plan

### 10.1 Synthetic contract tests

Generate local datasets that cover at least:

- `(N, 1, 1, 128)`, `(N, 1, 4, 4)`, `(N, 3, 32, 32)`,
  `(N, 3, 224, 224)`, and `(N, 64, 8, 8)`;
- `N < batch_size`, a one-class problem, 50+ classes, non-contiguous labels, and a
  validation split missing a training class;
- bool, uint8, float32, and float64 inputs;
- constant channels, NaNs, positive/negative infinities, and extreme scale;
- severe class imbalance;
- a test size not divisible by any common batch size;
- CPU-only execution and CUDA execution when available.

Every test must assert loader order, finite logits/loss, valid model output shape,
prediction dtype/domain, and exact prediction length.

### 10.2 Failure-injection tests

- Force an oversized candidate and verify that search rejects it without escaping.
- Inject a candidate-construction exception.
- Inject a non-finite loss.
- Set very small `time_limit` values and verify early return with predictions.
- Simulate inference failure and verify the majority fallback length and label mapping.
- Run with no CUDA.

### 10.3 Public-data benchmark

Use a diverse selection of the public historical datasets linked in the README, plus
synthetic tasks not resembling them. Do not tune special cases for dataset names.

Compare at least:

1. the supplied fixed ResNet-18 example;
2. the protected tiny fallback;
3. portfolio random search without multi-fidelity;
4. the complete portfolio plus successive-halving method.

Run multiple seeds where feasible. Record failure count, adjusted competition score,
worst-dataset score, accuracy, time, peak VRAM, peak RAM, parameter count, candidate
history, and right-continuous anytime incumbent curves. For final method comparisons,
use paired dataset results; if sample size permits, use the Wilcoxon signed-rank test for
two methods or Friedman followed by corrected pairwise tests for more methods.

### 10.4 Runtime matrix

Exercise at least these per-dataset limits:

- 1-2 minutes for smoke/fallback behavior;
- 5-10 minutes for short Phase 2-like behavior;
- 30 minutes for the documented 2026 default;
- 60+ minutes to verify that extra time results in more search/training rather than a
  fixed-duration run.

Require a positive reserve at the start of prediction in every case.

### 10.5 Packaging and evaluator tests

Run the complete official pipeline in a Linux/WSL/container environment because the
provided Makefile uses Unix commands. Before each submission:

- perform an import smoke test using the competition's PyTorch generation;
- run the evaluator from a clean directory;
- inspect the zip contents;
- verify there are no datasets, checkpoints, caches, `main.py`, or `score.py`;
- verify all imports work without network access;
- verify every test loader is ordered and complete;
- verify the output for every dataset contains both stats and predictions.

## 11. Implementation sequence and acceptance gates

### Milestone 0 - Clarify evaluator discrepancy

- Confirm the missing default argument in `is_out_of_time` with the organizers or
  obtain their latest evaluator.
- Keep any local evaluator fix outside the submission.

**Gate:** the supplied example can reach `DataProcessor.process()` locally.

### Milestone 1 - Safe fixed baseline

- Implement `BudgetManager`, dataset profiling, lazy loaders, the tiny residual model,
  time-aware training, and prediction fallback.
- Do not implement search yet.

**Gate:** all synthetic contract and tiny-time tests finish with exact-length
predictions and no uncaught exception. This is the first candidate for a Phase 2
submission because a working submission must be established before risky changes.

### Milestone 2 - Conditional architecture space

- Add residual/separable blocks, positional channels, adaptive pooling grids, resource
  guards, and the four anchor configurations.

**Gate:** every legal sampled configuration passes randomized shape fuzzing and
one-batch forward/backward preflight.

### Milestone 3 - Multi-fidelity NAS

- Add deterministic candidate generation, step-based rungs, promotion, CPU checkpoint
  storage, validation confirmation, and winner handoff.

**Gate:** the NAS path beats or matches the fixed fallback on aggregate public-data
validation without introducing any new failure.

### Milestone 4 - Hardening

- Add OOM retries, non-finite handling, one-class and non-contiguous-label paths,
  majority prediction fallback, exact output validation, and full logging.

**Gate:** all injected failures are contained and every time-budget test begins
prediction with reserve remaining.

### Milestone 5 - Empirical refinement

- Compare anchor portfolio, search fraction, halving aggressiveness, pooling grid,
  positional channels, mixup, and final refit.
- Use ablations to remove choices that add complexity without robust benefit.

**Gate:** zero failures across the complete public/synthetic suite, improved worst-case
score, and no dependence on codename-specific logic.

## 12. Final pre-submission checklist

- [ ] No attempt to inspect or download final datasets.
- [ ] No clock modification or evaluator override.
- [ ] No `main.py` or `score.py` in the zip.
- [ ] No imports outside the guaranteed environment.
- [ ] No pretrained-weight or network dependency.
- [ ] Live clock checked inside search, training, validation, and prediction loops.
- [ ] Prediction reserve computed from live and measured values.
- [ ] A preflighted fallback exists before architecture search starts.
- [ ] Candidate OOM/error/NaN cannot escape `NAS.search()`.
- [ ] Test loader is ordered and never drops examples.
- [ ] Predictions have exact length and original label values.
- [ ] Tiny datasets, singleton spatial axes, large channels, one class, and CPU-only
      execution have been tested.
- [ ] Default 30-minute and shorter budgets have been tested end to end.
- [ ] Peak RAM/VRAM remains below the smallest tested environment.
- [ ] The final zip passes a clean, offline evaluator run.
- [ ] A known-working Phase 2 submission is preserved before using another of the seven
      allowed submissions.

## 13. Explicitly deferred features

Do not add these until the complete robust pipeline passes all gates:

- DARTS or another differentiable supernet;
- full one-shot weight sharing;
- a Gaussian-process or random-forest BO stack;
- aggressive learning-curve extrapolation;
- zero-cost proxy-only selection;
- large ensembles at prediction time;
- dataset-name-specific rules;
- geometric augmentation enabled by default;
- multiprocessing data loaders;
- external downloads or pretrained models.

Each may be defensible in another setting, but here it increases implementation,
memory, ranking, or runtime risk before the failure-safe baseline is secure.
