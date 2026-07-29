# NAS Unseen-Data Challenge 2026 — Submission

## Architecture

This submission implements a 3-tier NAS pipeline that adapts to the
available time budget:

| Tier | Time Budget | Strategy |
|------|------------|----------|
| TIER 1 | >15 min | Full: ZCP screening (300 archs) → learning curve (top 15) → ensemble (top 3) |
| TIER 2 | 5–15 min | ZCP-only: Screen 100 architectures, return best |
| TIER 3 | <5 min | Immediate ResNet-18 fallback (zero search overhead) |

## Files

- `data_processor.py` — Adaptive data pipeline (auto channel-dim fix, per-channel
  normalization, data-adaptive augmentation, dynamic batch sizing)
- `nas.py` — Tiered NAS search controller
- `trainer.py` — Time-aware training with LR warmup, gradient clipping, early stopping,
  and ensemble-aware sequential training
- `search_space.py` — Macro-level search space (BasicBlock, BottleneckBlock, SE attention)
- `zero_cost_proxies.py` — synflow, jacob_cov, naswot with Borda rank aggregation
- `ensemble.py` — API-compatible nn.Module ensemble wrapper
- `helpers.py` — Time display, device detection, data inspection utilities

## Dependencies

Only standard competition dependencies:
- `torch`, `torchvision`
- `numpy`
- `scikit-learn`
