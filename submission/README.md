# NAS Unseen-Data Challenge 2026 Submission

This submission uses a deterministic, compute-aware residual-network search.
It is designed to spend a bounded part of each dataset's runtime on search and
protect the remaining budget for full training and prediction.

The complete architecture, design rationale, invariants, and change log are
maintained in [`NAS_ARCHITECTURE.md`](NAS_ARCHITECTURE.md).

## Pipeline

| Tier | Remaining time | Strategy |
|---|---:|---|
| 1 | at least 15 min | 72 unique candidates, fixed-batch proxy screening, successive halving |
| 2 | 5–15 min | 32 unique candidates and a shorter successive-halving search |
| 3 | under 5 min | compact input-adaptive residual fallback |

Key properties:

- Candidates are sampled without replacement and with a shape-derived seed.
- Parameter guards remove architectures that cannot receive enough training
  under the available budget.
- SynFlow, Jacobian correlation, and NASWOT use the same cached inputs.
- Low-fidelity candidates use identical cached training batches.
- The selected low-fidelity model is warm-started during final training.
- The trainer measures real train/validation throughput before selecting its
  epoch and monotonic warmup/cosine schedule.
- CUDA training and inference use automatic mixed precision.
- Data augmentation distinguishes structured/standardized inputs from
  natural-looking and low-variance RGB imagery.

## Files

- `data_processor.py` — data fingerprinting, normalization, augmentation, loaders
- `nas.py` — budgeted proxy screening and successive halving
- `search_space.py` — compact residual macro search space
- `zero_cost_proxies.py` — deterministic proxy implementations and ranking
- `trainer.py` — measured time-aware training and prediction
- `helpers.py` — shared time, device, batch-size, and data helpers
- `ensemble.py` — compatibility wrapper retained for optional future use

Only PyTorch, torchvision, and NumPy are required by the active pipeline.
