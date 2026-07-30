# NAS Unseen-Data Challenge 2026 Submission

This submission implements a hierarchical architecture portfolio for unseen
classification datasets. It remains competition-compatible: `NAS.search()`
compares independent PyTorch candidates and returns exactly one selected model.
There is no DARTS, einspace, supernet weight sharing, or final ensemble.

The full design and invariants are documented in
[`NAS_ARCHITECTURE.md`](NAS_ARCHITECTURE.md).

## Runtime modes

| Tier | Remaining time | Strategy |
|---|---:|---|
| 1 | at least 15 min | 54 diverse architectures, 12 label-aware finalists, adaptive fidelity |
| 2 | 5-15 min | 30 candidates, 7 label-aware finalists, two fidelity rounds |
| 3 | under 5 min | one robust portfolio anchor |

Tier 1 can use at most 18 percent or 360 seconds for search. Tier 2 can use at
most 12 percent or 120 seconds. Training and prediction time remain protected.

## Main properties

- The data fingerprint emits several representation hypotheses instead of one
  brittle categorical/non-categorical route.
- Sparse grids are inspected along rows and columns.
- The portfolio contains spatial, position-preserving spatial-pyramid,
  depthwise-factorized, width-axis, height-axis, and dual-axis networks.
- Axis families are activated by confidence scores, never dataset codenames.
- Zero-cost proxies only pre-screen candidates. They have no influence on the
  label-aware utility after the first round.
- First-round finalists use explicit equal family quotas and parameter-size
  strata. Architectures all use the same neutral probe recipe.
- Low-fidelity candidates receive equal wall-time quanta and identical cached
  data. The final champion/challenger comparison uses full validation.
- The last two candidates receive equal full-data passes while either learning
  curve still improves. AdamW state persists across fidelity rounds.
- Full validation, preceding fidelity, and rank stability select the
  architecture. Recipes are then compared on clones of the same checkpoint.
- Axis encoders use absolute sinusoidal position features and coarse ordered
  pooling bins instead of destroying all order with a global mean.
- The trainer uses validation-driven LR reductions plus a monotonic
  wall-clock cooldown. A high-LR plateau never ends training immediately.
- At the first exhausted attempt, an independent attempt starts from the
  common NAS checkpoint with a new optimizer, seed, and alternative recipe.
  If the benchmark remains unmet, one further untried recipe may be attempted.
  The globally best checkpoint remains protected throughout.
- If validation remains below the supplied benchmark, unused time is
  prioritised for that independent attempt.
- Class imbalance is used by the balanced recipe.
- CUDA OOM handling reduces the final training batch and recursively splits
  prediction batches.
- The best validation checkpoint and a measured prediction-time reserve are
  always protected.

Only PyTorch, torchvision, and NumPy are required by the active pipeline.

## Files

- `data_processor.py` - normalization, conservative augmentation, loaders
- `helpers.py` - data hypotheses, time/device and batch-size helpers
- `search_space.py` - independent architecture portfolio
- `nas.py` - hierarchical multi-fidelity controller
- `zero_cost_proxies.py` - proxy features used only for pre-screening
- `trainer.py` - recipe-aware, wall-clock-controlled final training
