# NAS Unseen-Data Challenge 2026 Submission

This submission implements a hierarchical, anytime architecture portfolio for
unseen classification datasets. `NAS.search()` compares independent PyTorch
models and returns one model. There is no codename routing, DARTS, einspace,
supernet, pretrained external model, or final ensemble.

See [`NAS_ARCHITECTURE.md`](NAS_ARCHITECTURE.md) for the complete active
design.

## Runtime modes

| Tier | Remaining time | Strategy |
|---|---:|---|
| 1 | at least 15 min | augmentation probe, semantic-family race, 54 proxy entries, 12 macro finalists, recipe race |
| 2 | 5–15 min | 30 proxy entries and 7 macro finalists |
| 3 | under 5 min | robust portfolio anchor |

Tier 2 search is limited to 12% or 120 seconds. Tier 1 uses 22% with a cap
that scales from the former six-minute horizon to at most 30 minutes on
multi-hour runs. Search and prediction deadlines remain protected.

## Main properties

- Byte-bounded fingerprinting, streaming normalization, caches, shift
  measurements, and per-example float32 conversion.
- Prior-preserving selection validation plus a disjoint confirmation split.
- Searchable identity/conservative/verified-flip augmentation policies.
- Spatial, factorized, axis, categorical/dense sequence, multi-view,
  volumetric, coordinate, hybrid, pre-activation wide/grouped, and dense
  feature-reuse families.
- Label-trained representation probes promote two families, with a third only
  on an uncertainty tie; macro search then restores full compact-to-wide size
  coverage inside the promoted families.
- Progressive data coverage, optimizer-state continuation, logical-batch
  microbatch accumulation, and OOM retry.
- Best-stage architecture restoration and confirmation-dominant finalist
  selection, with historical rank used only for statistical ties.
- Incumbent-safe two-stage recipe selection using uncertainty-aware
  confirmation evidence.
- AdamW controls plus zero-smoothing SGD/Nesterov with monotonic, no-restart
  cosine decay.
- Clock-driven final attempts across every untried recipe, an optional tied
  architecture challenger, fresh-seed continuations, EMA, and tested
  same-architecture checkpoint averaging.
- Early rotation of attempts that fail to recover the immutable baseline,
  while productive slow-recovery attempts retain a longer runway.
- TTA only for flip policies that passed both invariance and functional
  validation probes.
- Recursive prediction splitting preserves complete test order after OOM.

The active dependencies are PyTorch, torchvision, and NumPy.

## Files

- `data_processor.py` — bounded statistics, augmentation policies, loaders
- `helpers.py` — semantic fingerprints, device/time, batch sizing
- `search_space.py` — independent architecture families
- `nas.py` — hierarchical family/macro/recipe controller
- `zero_cost_proxies.py` — weak proxy pre-screen only
- `trainer.py` — optimizer-diverse anytime training and prediction
