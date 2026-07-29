# NAS Unseen-Data Challenge 2026: Project Proposal

## 1. Research Question
**"How can I effectively adapt Neural Architecture Search to unseen datasets with unknown, strict time constraints by leveraging Zero-Cost Proxies (ZCPs), hybrid evaluation, and time-aware advanced ensembling?"**

Because the competition involves entirely novel datasets and an unknown runtime limit, traditional NAS approaches that rely on full or partial training for performance estimation are too slow and risky. I propose to evaluate whether a hybrid approach—using Zero-Cost Proxies (ZCPs) for rapid initial filtering followed by short learning curve extrapolation—can reliably identify high-performing architectures on unseen data domains. Furthermore, I investigate if advanced ensembling techniques (such as Greedy Ensemble Selection or CMA-ES weight optimization) applied to the top architectures can significantly improve generalization and robustness when time permits.

## 2. Required Steps

1. **DataProcessor Construction:** 
   Implement dynamic PyTorch dataloaders that read the dataset's metadata. The processor will automatically adjust batch sizes and apply generalized robust augmentations (like RandAugment) to handle varying domains, ensuring strong regularization without overfitting.
2. **Search Space Definition:** 
   Adopt a compact, cell-based search space (DARTS, NASnet). A cell-based approach transfers well across different datasets and keeps the macro-architecture complexity manageable. Maybe im going to use NASLib or other Libs but I have to check if that is allowed.
3. **Hybrid ZCP-Guided Search Strategy:** 
   Implement a Random Search or Evolutionary algorithm that uses Zero-Cost Proxies (e.g., `synflow`, `jacob_cov`) for initial performance estimation. This allows evaluating thousands of candidate architectures in just a few minutes. To mitigate the risk of ZCPs correlating poorly with specific novel datasets, we will aggressively filter down to the top 50 architectures, then use learning curve extrapolation (training for 1-2 epochs) to select the final top candidates.
4. **Fast Hyperparameter Optimization (HPO):**
   Once the top architectures are selected, allocate a small fraction of the time budget to a fast multi-fidelity HPO method (e.g., Successive Halving or BOHB) to tune training hyperparameters like learning rate and weight decay specifically for the unseen dataset.
5. **Time-Aware Trainer:** 
   Develop a training loop that constantly monitors the `TIME_LIMIT` clock provided in `main.py`. It will dynamically calculate the time taken per epoch and employ early stopping to ensure the model finishes training and saves its predictions before the environment terminates the process.
6. **Advanced Adaptive Ensembling:** 
   If the estimated remaining time is sufficient, the pipeline will fully train the top-2 or top-3 architectures. Instead of basic equal-weight averaging, we will use a small validation split (or Out-Of-Bag predictions) to quickly run **Greedy Ensemble Selection (GES)** or **CMA-ES weight optimization** to find optimal weights for these models, boosting final performance on the test set.

## 3. Evaluation Strategy

- **Baseline Comparison:** I will compare my hybrid ZCP + advanced ensembling approach against a standard Random Search baseline that relies purely on learning curve extrapolation, to verify which method yields better architectures within strict time limits.
- **Dataset Diversity Testing:** I will test my pipeline locally on the diverse historical datasets provided in the starter kit (AddNIST, Gutenberg, CIFARTile, GeoClassing, Sudoku, etc.) to ensure my `DataProcessor` and NAS strategy generalize across drastically different data modalities.
- **Time Constraint Stress Testing:** Using the provided `Makefile` (`make submission=$SUBMISSION_DIRECTORY all`), I will run extensive local evaluations with varying `time_limit` variables in the dataset metadata (e.g., 5 minutes, 30 minutes, 2 hours). I will measure the successful completion rate to ensure the time-aware trainer never crashes or exceeds the limit.
