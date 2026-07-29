# NAS Unseen-Data Challenge 2026: Project Proposal (v2)

## 1. Research Question

**"How can a fully time-adaptive NAS pipeline—combining multi-proxy architecture screening, a macro-level search space, and API-compatible ensembling—robustly maximize adjusted score across unseen datasets with unknown, strict time constraints?"**

The competition evaluates submissions on entirely novel datasets under per-dataset time limits that are unknown in advance (Phase 2: 1 hour; Phase 3: unknown). The adjusted scoring formula heavily penalizes failures (-10 per crashed dataset), meaning **reliability is as important as peak accuracy**. We propose a tiered pipeline that dynamically selects its operating mode based on the available time budget, guaranteeing predictions are always produced while maximizing performance when time permits.

## 2. Scoring-Aware Design Philosophy

The adjusted score formula is:

```
scaling_factor = 10 / (100 - benchmark)
adj_score = (raw_score - benchmark) * scaling_factor   # clamped to min -10
```

Key implications that drive our design:
- **Never crash.** A single failed dataset (-10) can erase gains from multiple successful ones.
- **Beat the benchmark, even slightly.** Going from benchmark to benchmark+1% is more valuable per-point on hard datasets (high benchmark) than easy ones.
- **Always produce predictions.** Even a mediocre model is vastly better than a failure.

## 3. Time Budget Allocation

Every decision is gated on `clock.check()`. We define the following budget split as a percentage of the per-dataset `time_limit`:

| Phase | Budget | Purpose |
|-------|--------|---------|
| Data Processing | ~3% | Load, normalize, build dataloaders |
| NAS Search | ~12% | Multi-proxy screening + short eval |
| HPO (conditional) | ~5% | Only if >60% budget remains after NAS |
| Training (primary model) | ~65% | Full training of the best architecture |
| Training (ensemble members) | ~10% | Only if >30% budget remains after primary training |
| Safety margin | ~5% | Buffer to guarantee prediction + save |

These are soft targets. The pipeline continuously monitors `clock.check()` and will skip or truncate any phase that threatens the safety margin.

## 4. Required Steps

### Step 1: Adaptive DataProcessor

**Goal:** Build train/valid/test PyTorch dataloaders that handle arbitrary data domains.

- Read `metadata['input_shape']` to extract `[n, channels, height, width]`.
- Handle the edge case of **3D input arrays** (missing channel dimension) by reshaping `[n, H, W]` → `[n, 1, H, W]`.
- Compute per-channel mean and std from training data for normalization.
- Apply **data-adaptive augmentation**:
  - Inspect spatial resolution: skip geometric augmentations (flip, crop, rotate) if the input is very small (e.g., ≤8×8), since they may destroy structural patterns (Sudoku, GameOfLife).
  - Inspect channel count: skip color jitter for single-channel data.
  - For medium/large images (≥16×16): apply horizontal flip, small random crop with padding, and optional Cutout/CutMix.
  - Avoid heavy augmentations like RandAugment by default — the domain is unknown, and aggressive augmentation on structured data (chess boards, voxels) can be destructive.
- Dynamically adjust batch size based on input dimensions and available GPU memory (start with 128, halve on OOM).

### Step 2: Macro-Level Search Space

**Goal:** Define a flexible but simple-to-implement search space.

Rather than a full cell-based DARTS/NASNet space (which requires weeks of careful engineering and is fragile to implement correctly), we use a **parameterized macro-architecture** search over a ResNet-like backbone:

- **Searchable hyperparameters:**
  - Number of stages: {2, 3, 4}
  - Channels per stage: {16, 32, 64, 128} (first stage), with 2× growth per stage
  - Blocks per stage: {1, 2, 3}
  - Block type: {BasicBlock, Bottleneck}
  - Kernel size in conv layers: {3, 5}
  - Whether to use Squeeze-and-Excitation (SE) attention: {True, False}
  - Stem convolution kernel: {3, 5, 7}
  - Global pooling type: {AvgPool, MaxPool, AdaptiveAvg}

This yields ~2,500–5,000 discrete candidate architectures — large enough to explore meaningfully, small enough to evaluate rapidly with zero-cost proxies.

**Why this over cell-based search:**
- Zero external dependencies (pure PyTorch `nn.Module`)
- Each candidate is a standalone model — no mixed-ops, no discretization step
- Robust to implementation bugs
- Transfers well across data domains since depth/width/kernel flexibility adapts capacity

### Step 3: Multi-Proxy Architecture Screening

**Goal:** Rapidly rank candidate architectures without training.

1. **Sample 500 architectures** uniformly from the search space.
2. **Evaluate each with 3 zero-cost proxies** (implemented from scratch — no external libraries needed):
   - `synflow` — data-agnostic, measures parameter saliency via gradient flow
   - `jacob_cov` — data-dependent, measures the covariance of the Jacobian of the network output w.r.t. inputs (uses a single mini-batch)
   - `naswot` — data-dependent, measures the correlation of activation patterns across different inputs (higher entropy = better)
3. **Rank-aggregate** the three proxy scores using Borda count (average rank across proxies). This is more robust than relying on any single proxy, especially on unfamiliar data domains.
4. **Select top-20 architectures** from the rank aggregation.
5. **Short learning-curve evaluation** (if time permits — check `clock.check()`):
   - Train each top-20 candidate for 1–2 epochs on a 25% subsample of training data.
   - Rank by validation accuracy after this short training.
   - Select the **top-3** architectures for final training/ensembling.
   - If time is too short, skip this step and take the top-3 from the proxy ranking directly.

### Step 4: Fallback Strategy

**Goal:** Guarantee a valid prediction under any time budget.

> **This is the most critical engineering decision in the submission.**

The `NAS.search()` method implements a **tiered fallback**:

```
if time_remaining < 5 minutes:
    → TIER 3: Return a pre-configured EfficientNet-B0
              (adjusted first conv for input channels, adjusted FC for num_classes)
              No search performed.

elif time_remaining < 15 minutes:
    → TIER 2: Evaluate 100 architectures with ZCPs only (no learning curve).
              Return the top-1 architecture.

else:
    → TIER 1: Full pipeline — 500 candidates, ZCPs, rank aggregation,
              learning curve evaluation on top-20, return top-1 (or ensemble wrapper).
```

The TIER 3 fallback uses `torchvision.models.efficientnet_b0` (available in torchvision, no external deps) with modified stem/head. This guarantees a reasonable model is always available — EfficientNet-B0 is a strong general-purpose architecture that outperforms ResNet-18 on most domains.

### Step 5: API-Compatible Ensemble Wrapper

**Goal:** Ensemble multiple architectures through the single-model API.

The evaluation pipeline expects `search()` to return one `nn.Module` and `train()` to return one trained model. We work within this constraint by building an **`EnsembleModule`**:

```python
class EnsembleModule(nn.Module):
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)  # registers all params
        self.weights = weights or [1.0 / len(models)] * len(models)

    def forward(self, x):
        outputs = [m(x) for m in self.models]
        weighted = sum(w * o for w, o in zip(self.weights, outputs))
        return weighted
```

- `nn.ModuleList` ensures `general_num_params()` correctly counts all parameters.
- `forward()` produces a single output tensor, so `Trainer.predict()` works unchanged.
- **Ensemble is conditional**: only activated if TIER 1 completes and `clock.check()` shows enough time to train 2-3 models. Otherwise, `search()` returns a single model (no wrapper needed).

**Weight optimization (when time permits):**
After training all ensemble members, use the validation set to optimize ensemble weights via:
- Simple grid search over weight combinations (fast, no external deps), OR
- Greedy Ensemble Selection: iteratively add the model that most improves validation accuracy.

No CMA-ES (avoids the `cma` dependency). Greedy selection is comparably effective and trivial to implement.

### Step 6: Time-Aware Trainer

**Goal:** Train the model fully while guaranteeing completion before the time limit.

- **Before training starts**: estimate time-per-epoch by running a single forward+backward pass on one mini-batch and extrapolating by the number of batches.
- **Compute max safe epochs**: `max_epochs = floor((time_remaining - safety_margin) / estimated_time_per_epoch)`
- **Training loop**:
  - Use AdamW optimizer (lr=1e-3, weight_decay=1e-2) — more robust than SGD across unknown domains.
  - Cosine annealing LR schedule with warmup (5% of epochs).
  - Check `clock.check()` after every epoch. If remaining time < `2 × time_per_epoch`, stop immediately.
  - Track validation accuracy each epoch; keep a checkpoint of the best model (by val acc).
  - If validation accuracy plateaus for 3 consecutive epochs, stop early.
- **After training**: return the best checkpoint (not necessarily the last epoch).
- **Prediction**: always runs within the safety margin. Uses `model.eval()` + `torch.no_grad()` for speed.

### Step 7: Conditional HPO (Luxury Phase)

**Goal:** Tune training hyperparameters only when time is abundant.

This phase **only activates** if >60% of the time budget remains after NAS completes. If activated:

- Use **successive halving** over a small grid:
  - Learning rate: {3e-4, 1e-3, 3e-3}
  - Weight decay: {1e-3, 1e-2, 5e-2}
  - Batch size: {64, 128}
- Train each config for 3 epochs on a 30% subsample, halve the configs, double the epochs, repeat.
- Total configs: 18 → 9 → 4 → 2 → 1 winner.
- Estimated overhead: ~10-15% of total time budget.

If time does not permit, use the default config (AdamW, lr=1e-3, wd=1e-2, bs=128) which is a solid general-purpose default.

## 5. Dependency Management

The server environment provides: `torch`, `torchvision`, `numpy`, `sklearn`. We will **not** rely on any external NAS libraries.

| Component | Implementation |
|-----------|----------------|
| Zero-cost proxies (synflow, jacob_cov, naswot) | Self-implemented (~50 lines each) |
| Search space | Pure PyTorch `nn.Module` construction |
| Rank aggregation | Simple Borda count (10 lines of Python) |
| Greedy Ensemble Selection | Self-implemented (~30 lines) |
| Successive Halving | Self-implemented (~40 lines) |
| EnsembleModule wrapper | `nn.Module` with `nn.ModuleList` |

All code lives within the submission directory: `data_processor.py`, `nas.py`, `trainer.py`, `helpers.py`.

## 6. Evaluation Strategy

### Baseline Comparison
- Compare our tiered pipeline against the example submission (ResNet-18, no search, 2-epoch training) to verify we consistently achieve higher adjusted scores.
- Compare TIER 1 (full pipeline) vs TIER 2 (ZCP-only) vs TIER 3 (fallback) to quantify the value of each stage.

### Dataset Diversity Testing
- Test locally on all 13 historical datasets (AddNIST, Language, MultNIST, CIFARTile, Gutenberg, GeoClassing, Chesseract, Sudoku, Voxel, Myofibre, GameOfLife, Cryptic, Windspeed).
- Pay special attention to structurally unusual datasets (Sudoku, GameOfLife) where augmentation and ZCP behavior may differ.
- Verify that the DataProcessor handles all input shapes and channel counts correctly.

### Time Constraint Stress Testing
- Using `make submission=our_submission all`, test with `time_limit` values of: 0.08 (5 min), 0.25 (15 min), 0.5 (30 min), 1.0 (1 hour), 2.0 (2 hours).
- **Success criterion**: 100% completion rate — no crashes, no time overruns, predictions always saved.
- Verify the fallback tiers activate at the correct thresholds.

### Scoring Validation
- Compute adjusted scores using `score.py` across all datasets.
- Target: positive adjusted score on every dataset (i.e., beat the benchmark on each).
- Track total score as the primary optimization target.

## 7. Project Timeline

| Week | Milestone |
|------|-----------|
| 1 | DataProcessor (adaptive augmentation, edge cases) + macro search space definition |
| 2 | ZCP implementations (synflow, jacob_cov, naswot) + rank aggregation + fallback tiers |
| 3 | Time-aware Trainer + EnsembleModule wrapper + Greedy Ensemble Selection |
| 4 | Integration testing across all 13 datasets + time stress testing |
| 5 | Conditional HPO + performance tuning + final validation |
| 6 | Buffer / ablation studies / submission preparation |

## 8. Risk Mitigation Summary

| Risk | Mitigation |
|------|-----------|
| Time overrun → dataset failure (-10 pts) | Safety margin (5% budget) + per-epoch clock checks + tiered fallback |
| ZCPs poorly correlated on exotic domains | Multi-proxy rank aggregation + learning-curve validation stage |
| Search space too narrow / too wide | Macro-level search (~3K candidates) balances coverage with tractability |
| External dependency missing on server | All algorithms self-implemented; only torch/torchvision/numpy/sklearn used |
| Augmentation destroys data structure | Data-adaptive augmentation based on resolution and channel inspection |
| Ensemble incompatible with API | `EnsembleModule(nn.Module)` with `nn.ModuleList` — transparent to the evaluation harness |
| Too little time for any search | TIER 3 fallback: return EfficientNet-B0 immediately, skip NAS entirely |
