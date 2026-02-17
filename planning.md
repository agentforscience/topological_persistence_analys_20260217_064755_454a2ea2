# Research Plan: Topological Persistence Analysis of LLM Training Dynamics for Scaling Prediction

## Motivation & Novelty Assessment

### Why This Research Matters
Training large language models is computationally expensive, costing millions of dollars. A key decision during training is whether to continue training the current model or scale up to a larger model. Currently, this decision relies on simple power-law extrapolation of loss curves, which provides limited insight into the underlying optimization dynamics. Topological data analysis (TDA) can capture structural properties of the optimization landscape that scalar metrics (loss, gradient norm) cannot—such as connectivity, number of basins, and topological complexity. If persistent topological features correlate with scaling efficiency, they could serve as early warning signals for when to scale, potentially saving significant compute.

### Gap in Existing Work
Based on the literature review:
1. **No TDA applied to LLM/transformer training dynamics**: Ballester et al. (2024) survey explicitly notes this gap—most TDA-for-NN work uses classical CNNs/FCNNs.
2. **No connection between topological features and scaling laws**: While TDA metrics correlate with generalization and training progress (Rieck et al. 2019), nobody has studied whether they predict scaling behavior across model sizes.
3. **Temporal topological analysis is underexplored**: Tracking topological features across training checkpoints (creating "topological time series") is largely unexplored.

### Our Novel Contribution
We will: (1) Train multiple small transformer models (1K–50K parameters) from scratch, (2) compute persistent homology on their weight spaces and loss surfaces across training, (3) identify topological signatures that correlate with training efficiency and scale predictably across model sizes, and (4) test whether topological features improve prediction of optimal training-to-scaling transitions.

### Experiment Justification
- **Experiment 1 (Train small LLMs)**: Needed to generate controlled training trajectories where we know the ground truth scaling behavior. Small models are computationally tractable.
- **Experiment 2 (Compute TDA features)**: Neural persistence and weight-space persistent homology capture structural complexity of the learned representations—the core measurement.
- **Experiment 3 (Correlation analysis)**: Tests whether topological features contain information about training efficiency beyond what loss/gradient norms provide.
- **Experiment 4 (Scaling prediction)**: The main hypothesis test—can TDA features predict when to scale?

## Research Question
Can persistent topological features extracted from the weight spaces and training trajectories of small transformer language models serve as early predictors of optimal scaling decisions (when to scale model size vs. continue training)?

## Background and Motivation
Neural scaling laws (Kaplan 2020, Hoffmann 2022) describe how LLM performance scales with compute, but rely on fitting power laws to loss curves. Topological data analysis offers a richer characterization of training dynamics through persistent homology, which tracks how topological features (connected components, loops) persist across filtration scales. Rieck et al. (2019) showed neural persistence correlates with training progress in CNNs, but no one has extended this to transformers or connected it to scaling predictions.

## Hypothesis Decomposition
1. **H1**: Neural persistence (H_0 of weight graphs) increases monotonically during training and its rate of change correlates with training efficiency.
2. **H2**: The topological complexity of weight-space point clouds (Betti numbers from Vietoris-Rips filtration) shows distinct phase transitions during training that coincide with diminishing returns on continued training.
3. **H3**: Topological features extracted from different model sizes follow predictable scaling relationships (e.g., power-law or log-linear).
4. **H4**: Topological features improve prediction of optimal stopping/scaling points beyond what loss curve extrapolation alone provides.

## Proposed Methodology

### Approach
Train a family of small GPT-style transformer models (1K, 5K, 10K, 25K, 50K parameters) on a text corpus, saving frequent checkpoints. At each checkpoint, extract: (a) neural persistence per layer, (b) persistent homology of weight-space point clouds, (c) loss and gradient statistics. Analyze temporal evolution, cross-size scaling, and predictive utility.

### Experimental Steps

1. **Data Preparation**: Use Wikitext-2 or a subset of The Pile as training corpus. Tokenize with a simple BPE tokenizer.

2. **Model Training** (GPU-accelerated):
   - Define 5 GPT-style models: ~1K, 5K, 10K, 25K, 50K parameters
   - Architecture: decoder-only transformer with varying hidden_dim, n_layers, n_heads
   - Train each for 10K steps with cosine LR schedule
   - Save checkpoints every 200 steps (50 checkpoints per model)
   - Record: loss, gradient norm, learning rate at each step

3. **TDA Feature Extraction** (per checkpoint):
   - **Neural Persistence**: Per-layer H_0 persistence of weight graphs (Rieck et al. method)
   - **Weight-Space PH**: Sample weight vectors, compute Vietoris-Rips persistence (H_0, H_1)
   - **Gradient-Space PH**: PH of gradient vectors at each checkpoint
   - Use ripser for fast computation

4. **Correlation Analysis**:
   - Spearman/Pearson correlation between TDA features and training metrics
   - Phase transition detection via changepoint analysis on topological time series
   - Cross-size comparison of topological feature trajectories

5. **Scaling Prediction**:
   - Use TDA features from early training (first 30% of steps) to predict final loss
   - Compare: TDA features alone, loss-only extrapolation, TDA + loss combined
   - Evaluate: MAE of predicted final loss, rank correlation across model sizes

### Baselines
1. **Loss curve extrapolation**: Fit power law L(t) = a*t^(-b) + c to early loss curve, predict final loss
2. **Gradient norm statistics**: Use mean/std of gradient norms as predictive features
3. **Weight norm tracking**: L2 norm of weights per layer (non-topological structural metric)

### Evaluation Metrics
- **Topological**: Neural persistence per layer, total persistence, number of persistence diagram points, Wasserstein/bottleneck distances between consecutive persistence diagrams
- **Prediction**: MAE of final loss prediction, Spearman rank correlation of predicted vs actual scaling
- **Correlation**: Spearman ρ between TDA features and training efficiency metrics

### Statistical Analysis Plan
- Spearman rank correlation with p-values for all feature-metric pairs
- Bootstrap confidence intervals (1000 samples) for correlation estimates
- Paired t-tests or Wilcoxon signed-rank for comparing prediction methods
- Multiple comparison correction (Bonferroni) where applicable
- Significance level: α = 0.05

## Expected Outcomes
- **Supporting hypothesis**: TDA features show strong correlation (ρ > 0.7) with training efficiency, exhibit detectable phase transitions, and improve prediction MAE by >10% vs loss-only baseline.
- **Refuting hypothesis**: TDA features are noisy, show no consistent scaling relationship, or provide no predictive advantage.

## Timeline and Milestones
| Phase | Time | Milestone |
|-------|------|-----------|
| Planning | 10 min | planning.md complete |
| Environment | 5 min | Packages verified, data loaded |
| Model Training | 20 min | 5 models trained with checkpoints |
| TDA Extraction | 15 min | Topological features computed |
| Analysis | 15 min | Correlations and predictions computed |
| Visualization | 10 min | Key plots generated |
| Documentation | 15 min | REPORT.md and README.md written |
| **Total** | **~90 min** | |

## Potential Challenges
1. **Computational cost of PH**: Mitigate by subsampling weights and using ripser's efficient C++ backend.
2. **Small model regime may not generalize**: Acknowledge as limitation; validate trends on medium models if time allows.
3. **Noisy topological features**: Use smoothing and aggregation (per-layer means, persistence images).
4. **Overfitting prediction model**: Use leave-one-model-out cross-validation.

## Success Criteria
1. At least one TDA feature shows statistically significant correlation (p < 0.05) with training efficiency across model sizes.
2. Topological feature trajectories show qualitatively different behavior across model sizes.
3. TDA-augmented predictions outperform loss-only baseline for at least 3/5 model sizes.
