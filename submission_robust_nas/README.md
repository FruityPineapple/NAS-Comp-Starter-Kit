# Robust Anytime NAS submission

This directory is a flat, offline competition submission implementing the pipeline
described in `IMPLEMENTATION_PLAN.md`. It imports only Python's standard library,
NumPy, and PyTorch. It does not use dataset names, pretrained weights, downloads,
network access, `torchvision`, or `scikit-learn`.

## Method

`DataProcessor` profiles a bounded deterministic sample of the training array, maps
the union of train/validation labels to contiguous indices, and exposes lazy NumPy
datasets. Non-finite values are replaced using finite training statistics and each
channel is standardized. The test loader is ordered, complete, and never drops a
sample.

`NAS` first constructs and forward/backward-preflights a protected tiny residual
network. Depending on the live clock it evaluates a portfolio of basic, positional,
and depthwise-separable residual CNNs plus conditional random candidates. Fidelity is
measured in optimizer updates; candidates are resumed between gentle successive
halving rungs. Candidate construction, OOM, non-finite output, and training failures
are contained locally.

`Trainer` continues the selected search checkpoint with AMP on CUDA, logical-batch
microbatching, gradient clipping, warm-up/cosine learning rates, validation
checkpoints, and optional low-rate train+validation refitting. Every loop consults the
live evaluator clock. Prediction preserves order and inverse-maps original labels. If
model inference cannot finish safely, an exact-length majority-label array is
returned.

The shared `BudgetManager` never modifies the evaluator clock. Its prediction reserve
is the maximum of 90 seconds, 12% of the initial remaining budget, and 2.5 times
measured validation inference scaled to test size, with a 25% cap for very short smoke
budgets.

## Files

- `data_processor.py`: profiling, label mapping, normalization, lazy datasets/loaders
- `models.py`: shape-safe residual/separable model space and resource guards
- `nas.py`: anchor portfolio, preflight, multi-fidelity scheduling, selection
- `trainer.py`: continuation training, checkpointing, OOM adaptation, prediction
- `helpers.py`: live-clock budgeting, seeding, logging, state and finite-value helpers

There is intentionally no `main.py` or `score.py`; the evaluator supplies those files.

## Reproducible local checks

From the repository root with NumPy and PyTorch installed:

```bash
python -m unittest discover -s tests -p "test_robust_nas.py" -v
python -m py_compile submission_robust_nas/*.py
make submission=submission_robust_nas zip
unzip -l submission.zip
```

For a full dataset run, first apply the known missing-default correction to a local
copy of `evaluation/main.py` (`grace_time=False`); never include that local evaluator
copy in the submission archive. Then run:

```bash
make submission=submission_robust_nas all
```

The synthetic tests cover singleton axes, high channel counts, non-contiguous labels,
one-hot/one-class labels, non-finite inputs, exact test length, ordered loaders, and the
majority fallback. CUDA paths are exercised automatically when CUDA is available.

