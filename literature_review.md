# Literature Review: Topological Persistence Analysis of LLM Training Dynamics for Scaling Prediction

## Research Area Overview

This research investigates whether persistent topological features in the loss landscapes and parameter evolution trajectories of small, efficiently trained LLMs can serve as early predictors of scaling decisions -- specifically, when to scale model size versus continue training. The project sits at the intersection of three active research areas: (1) topological data analysis (TDA) applied to neural networks, (2) neural scaling laws and compute-optimal training, and (3) LLM training dynamics and checkpoint analysis.

---

## Key Papers

### Category 1: TDA Applied to Loss Landscapes and Neural Network Structure

#### Paper 1: Neural Persistence (Rieck et al., ICLR 2019)
- **Authors**: Bastian Rieck, Matteo Togninalli, Christian Bock, Michael Moor, Max Horn, Thomas Gumbsch, Karsten Borgwardt
- **Source**: ICLR 2019 (arXiv:1812.09764)
- **Key Contribution**: Introduced "neural persistence" -- a structural complexity measure for neural networks based on persistent homology of the weight graph. Proposed a weight-based filtration on the network's stratified graph (sorting by descending absolute weight) and computing 0-dimensional persistence diagrams per layer.
- **Methodology**: Represents NN as bipartite graph per layer, constructs descending filtration from weight magnitudes, computes H_0 persistent homology via union-find (nearly linear time).
- **Datasets Used**: MNIST, Fashion-MNIST, CIFAR-10, IMDB
- **Results**: Neural persistence increases during training; correlates with regularization quality (dropout > batch norm > none); enables data-free early stopping that saves 0.5-1.7 epochs with minimal accuracy loss.
- **Code Available**: Yes -- https://github.com/BorgwardtLab/Neural-Persistence
- **Relevance**: Foundational method for our project. Neural persistence can be computed on transformer linear layers. Key insight: focus on NP *dynamics* (rate of change) rather than absolute values.

#### Paper 2: Evaluating Loss Landscapes from a Topology Perspective (Xie et al., NeurIPS Workshop 2024)
- **Authors**: Tiankai Xie, Caleb Geniesse, Jiaqing Chen, Yaoqing Yang, Dmitriy Morozov, Michael Mahoney, Ross Maciejewski, Gunther Weber
- **Source**: NeurIPS 2024 Workshop (arXiv:2411.09807)
- **Key Contribution**: Pipeline for quantifying loss landscape topology using merge trees and persistence diagrams. Showed correlations between topological metrics (saddle point count, average persistence) and ML performance metrics (accuracy, Hessian eigenvalues).
- **Methodology**: Projects loss landscape into 2D subspace (random or Hessian-based), computes merge tree and 0-dim persistence diagram using Topology ToolKit (TTK). Prefers AkNN graph (k=8) for scalability.
- **Datasets Used**: CIFAR-10 (ResNet-20), 1D convection PDE (PINNs)
- **Results**: More saddle points = worse performance; higher average persistence = simpler landscape = better performance (context-dependent). Residual connections dramatically simplify loss landscape topology.
- **Code Available**: No explicit repo, but uses open tools (TTK, PyHessian)
- **Relevance**: Directly transferable pipeline. Track merge tree / persistence metrics across training checkpoints to detect phase transitions. Adapt for LLM loss landscapes using Hessian-based subspace construction.

#### Paper 3: Visualizing Loss Functions as Topological Landscape Profiles (Geniesse et al., 2024)
- **Authors**: Caleb Geniesse et al.
- **Source**: arXiv:2411.12136 (Nov 2024)
- **Key Contribution**: Higher-dimensional visualization of loss landscapes using TDA. Constructs filtrations along optimization trajectories to capture multi-scale topological features as "landscape profiles."
- **Methodology**: Samples points along optimization trajectories, builds filtrations at different scales, transforms persistence diagrams into interpretable landscape profiles.
- **Results**: Topological landscape profiles reveal structure invisible in standard 2D projections; complexity is greater near performance transitions.
- **Relevance**: The trajectory-based sampling approach is directly applicable to analyzing optimization paths during LLM training.

#### Paper 4: A Topological Description of Loss Surfaces Based on Betti Numbers (Ballarin et al., 2024)
- **Authors**: Ballarin et al.
- **Source**: Neural Networks (ScienceDirect), 2024 (arXiv:2401.00358)
- **Key Contribution**: Derives theoretical upper and lower bounds for Betti numbers of loss surfaces in multilayer NNs. Compares topological complexity of deep vs. shallow architectures.
- **Methodology**: Analytical derivation of Betti number bounds for loss surfaces as functions of network depth, width, activation function, and training data shape.
- **Results**: Loss surface complexity increases with depth and number of hidden units; sigmoidal activations lead to quantifiable complexity differences between deep and shallow nets.
- **Relevance**: Provides theoretical grounding for how topological complexity scales with model size -- directly relevant to predicting scaling behavior.

#### Paper 5: TDA for Neural Network Analysis: A Comprehensive Survey (Ballester et al., 2024)
- **Authors**: Rubén Ballester, Carles Casacuberta, Sergio Escalera
- **Source**: arXiv:2312.05840 (Jan 2024)
- **Key Contribution**: Comprehensive survey organizing TDA-for-NN work into four categories: (1) network structure, (2) decision regions/boundaries, (3) internal representations/activations, (4) training dynamics and loss functions.
- **Key Findings from Section 3.4 (Training Dynamics)**:
  - Loss function connectivity: sublevel set topology determines optimization feasibility (Nguyen 2019)
  - Fractal dimension of weight trajectories via persistent homology correlates with generalization (Birdal et al. 2021)
  - dimPH(Θ) = dimBox(Θ) -- persistent homology dimension equals box-counting fractal dimension
  - Generalization bound: gap ≤ O(√(dimPH * log²(mL²) / m))
- **Key Gap Identified**: Most experiments use classical CNNs/FCNNs, NOT transformers or LLMs. Extension to modern architectures is a critical open problem.
- **Relevance**: Confirms the novelty of applying TDA to LLM training dynamics. The Birdal et al. (2021) connection between persistent homology dimension of weight trajectories and generalization is directly applicable.

#### Paper 6: Visualizing the Loss Landscape of Neural Nets (Li et al., NeurIPS 2018)
- **Authors**: Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, Tom Goldstein
- **Source**: NeurIPS 2018 (arXiv:1712.09913)
- **Key Contribution**: Foundational work on filter-normalized loss landscape visualization. Showed skip connections produce dramatically smoother loss landscapes.
- **Methodology**: Random direction perturbation with filter normalization for fair comparison across architectures.
- **Code Available**: Yes -- https://github.com/tomgoldstein/loss-landscape
- **Relevance**: Standard method for generating the 2D loss landscape projections that TDA methods analyze.

#### Paper 7: Exploring the Geometry and Topology of Neural Network Loss Landscapes (Horoi et al., 2021)
- **Authors**: Horoi, Huang, Rieck, Lajoie, Wolf, Krishnaswamy
- **Source**: arXiv:2102.00485
- **Key Contribution**: "Jump and retrain" procedure for sampling loss landscape. Combined PHATE trajectory visualization with computational homology to quantify differences between generalizing and non-generalizing networks.
- **Relevance**: The trajectory-based approach to sampling loss landscapes is more informative than random projections and could be adapted for LLM checkpoint analysis.

### Category 2: Neural Scaling Laws and Training Dynamics

#### Paper 8: Scaling Laws for Neural Language Models (Kaplan et al., 2020)
- **Authors**: Jared Kaplan et al. (OpenAI)
- **Source**: arXiv:2001.08361
- **Key Contribution**: Established that LM performance scales as power laws with model size, dataset size, and compute. Showed smooth, predictable scaling across many orders of magnitude.
- **Relevance**: The baseline scaling law framework that our topological analysis aims to improve upon.

#### Paper 9: Training Compute-Optimal Large Language Models [Chinchilla] (Hoffmann et al., 2022)
- **Authors**: Jordan Hoffmann et al. (DeepMind)
- **Source**: arXiv:2203.15556
- **Key Contribution**: Showed previous LLMs were significantly undertrained. Derived optimal allocation between model size and training tokens for a given compute budget.
- **Results**: For compute-optimal training, model size and training tokens should scale roughly equally.
- **Relevance**: The scaling prediction problem our topological analysis addresses -- can TDA metrics predict when to scale vs. continue training?

#### Paper 10: A Dynamical Model of Neural Scaling Laws (Bordelon et al., 2024)
- **Authors**: Blake Bordelon, Alexander Atanasov, Cengiz Pehlevan
- **Source**: arXiv:2402.01092
- **Key Contribution**: Provides a theoretical dynamical model explaining scaling laws via random feature model + gradient descent analysis. Predicts asymmetric compute-optimal scaling: training steps should increase faster than model parameters.
- **Methodology**: Analyzes feature learning dynamics through eigenspectrum decomposition; different modes of the model are learned at different rates.
- **Key Predictions**: Power law exponents for performance vs. training time differ from those for performance vs. model size; asymmetric allocation is optimal.
- **Relevance**: The dynamical model's prediction of different learning phases (fast modes learned first, slow modes later) could produce detectable topological signatures.

#### Paper 11: Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations (Porian et al., NeurIPS 2024)
- **Authors**: Porian et al.
- **Source**: NeurIPS 2024 (arXiv:2405.18392)
- **Key Contribution**: Extends scaling laws to handle learning rate schedules and variable training durations. Validated at 1B and 8B parameter scales.
- **Relevance**: Practical framework for understanding how LR schedules affect scaling, which interacts with loss landscape topology.

#### Paper 12: LLMs on the Line: Data Determines Loss-to-Loss Scaling Laws (Gadre et al., 2025)
- **Authors**: Gadre et al.
- **Source**: arXiv:2502.12120
- **Key Contribution**: Studies loss-to-loss scaling curves across >6000 checkpoints. Shows pretraining data has the most substantial impact on scaling laws; intermediate checkpoints are valuable for fitting.
- **Relevance**: Demonstrates that intermediate training checkpoints contain rich scaling information -- exactly what we aim to enrich with topological features.

#### Paper 13: Multi-Power Law for Loss Curve Prediction (Luo et al., 2025)
- **Authors**: Luo et al. (Tsinghua)
- **Source**: arXiv:2501.02751
- **Key Contribution**: Proposes multi-power law (MPL) framework for predicting training loss curves including learning rate decay effects.
- **Relevance**: Loss curve prediction is the downstream task our topological features aim to improve.

#### Paper 14: Scaling Laws for Downstream Task Performance (Isik et al., 2024)
- **Authors**: Isik et al.
- **Source**: arXiv:2402.04177
- **Key Contribution**: Two-stage prediction framework: FLOPs → loss → downstream performance. Achieves 5-10% error margins for 7B/13B models using only ≤3B sampling models.
- **Relevance**: The two-stage prediction framework could be augmented with topological features for improved accuracy.

### Category 3: TDA Tools and Methods

#### Paper 15: giotto-tda (Tauzin et al., JMLR 2021)
- **Authors**: Tauzin et al.
- **Source**: JMLR 2021 (arXiv:2004.02551)
- **Key Contribution**: Scikit-learn compatible Python library for TDA with high-performance C++ backend. Provides VietorisRipsPersistence, persistence diagrams, persistence images, and landscapes.
- **Code Available**: Yes -- https://github.com/giotto-ai/giotto-tda
- **Relevance**: Primary computational tool for our persistent homology calculations.

#### Paper 16: On the Expressivity of Persistent Homology in Graph Learning (Horn et al., 2023)
- **Authors**: Horn et al.
- **Source**: arXiv:2302.09826
- **Key Contribution**: Studies what persistent homology can and cannot capture in graph classification, establishing theoretical expressivity results.
- **Relevance**: Provides theoretical understanding of PH's capabilities when applied to neural network weight graphs.

---

## Common Methodologies

### TDA Methods Applied to Neural Networks
- **Persistent Homology (H_0)**: Connected component tracking via weight-based or distance-based filtrations. Used in Rieck et al. (2019), Xie et al. (2024).
- **Merge Trees**: Track sublevel set component merges. Used in Xie et al. (2024).
- **Persistence Diagrams**: Summarize birth/death of topological features. Universal across all TDA papers.
- **Persistent Homology Dimension**: Links fractal dimension to PH, connecting to generalization bounds. Used in Birdal et al. (2021).
- **Representation Topology Divergence (RTD)**: Compares VR filtrations across different representations. Used in Barannikov et al. (2022).

### Loss Landscape Analysis Pipeline
1. Define 2D subspace (random or Hessian-based directions)
2. Evaluate loss at grid points in subspace
3. Construct graph representation (AkNN, Delaunay, etc.)
4. Compute topological invariants (merge tree, persistence diagram)
5. Extract summary statistics (saddle count, average persistence, total persistence)
6. Correlate with ML performance metrics

### Scaling Law Fitting
- Power law regression: L(C) = αC^(-β) + L∞
- Two-stage prediction: compute → loss → downstream performance
- Key variables: model parameters N, training tokens D, compute C = 6ND

---

## Standard Baselines

For our experiments, the literature suggests these baselines:
1. **Raw loss curve extrapolation**: Fit power law to loss curve, predict future loss
2. **Chinchilla-style scaling**: Predict optimal allocation from compute budget alone
3. **Multi-power law (MPL)**: Loss curve prediction with LR schedule awareness
4. **Gradient norm / Hessian-based metrics**: Track curvature without topological analysis
5. **Weight norm statistics**: Simple weight magnitude statistics (p-norms) without PH

---

## Evaluation Metrics

### Topological Metrics (from Literature)
- **Number of saddle points** (from merge tree)
- **Average persistence** (mean distance from diagonal in persistence diagram)
- **Total persistence** (sum of all persistence values, the p-norm)
- **Neural persistence** (normalized total persistence per layer)
- **Persistent homology dimension** (fractal dimension via PH)
- **Number of persistence diagram points** (topological feature count)

### ML Performance Metrics
- **Validation loss** (perplexity for LLMs)
- **Downstream task accuracy** (MMLU, HellaSwag, etc.)
- **Top-k Hessian eigenvalues** (sharpness)
- **Hessian trace** (total curvature)

### Scaling Prediction Metrics
- **Mean absolute error** of loss prediction at future checkpoints
- **Rank correlation** between predicted and actual scaling curves
- **Compute savings** achieved by early stopping based on topological signals

---

## Datasets in the Literature

### For TDA Experiments
- MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100 (standard benchmarks)
- IMDB sentiment (NLP baseline)
- 1D convection PDE (PINNs)

### For Scaling Law Studies
- The Pile / Pile-deduped (Pythia training data)
- Dolma (OLMo training data)
- C4, RedPajama (common LLM training corpora)

### For Our Experiments (Recommended)
- **Pythia checkpoints** (EleutherAI): 8 model sizes (14M to 12B), 154 checkpoints each, all trained on same data in same order. Ideal for controlled scaling analysis.
- **OLMo checkpoints** (AI2): Fully open models with training code, data, and intermediate checkpoints.

---

## Gaps and Opportunities

1. **No prior work applies TDA to LLM/transformer training dynamics**: The survey by Ballester et al. (2024) explicitly identifies this as a critical gap -- most TDA-for-NN work uses classical CNNs/FCNNs.

2. **No connection between topological features and scaling laws**: While TDA metrics correlate with generalization and training progress, no one has studied whether they predict scaling behavior across model sizes.

3. **Computational scalability**: Persistent homology is expensive for large networks. However, per-layer analysis (as in neural persistence) is tractable since individual layers have O(10^4-10^6) weights even in large LLMs.

4. **Temporal topological analysis**: Tracking topological features across training checkpoints (creating a "topological time series") is largely unexplored, with only Rieck et al. (2019) and Muller et al. (2024) touching on this.

5. **Higher-dimensional homology**: Nearly all work uses only H_0 (connected components). H_1 (loops/cycles in the loss landscape or weight graph) could capture richer structural information relevant to scaling.

---

## Recommendations for Our Experiment

### Recommended Datasets
1. **Primary: Pythia model suite** (14M, 70M, 160M, 410M, 1B, 1.4B checkpoints) -- controlled scaling, 154 checkpoints each
2. **Validation: OLMo checkpoints** -- independent confirmation with different architecture/data
3. **Synthetic: Generated loss curves** -- for method development and sanity checks

### Recommended Baselines
1. Raw loss curve power-law extrapolation
2. Weight norm statistics (non-topological baseline)
3. Hessian eigenvalue tracking (geometry without topology)
4. Chinchilla-style compute-optimal predictions

### Recommended Metrics
1. Neural persistence (per-layer, normalized)
2. Merge tree statistics (saddle count, average persistence)
3. Persistent homology dimension of weight trajectories
4. Correlation with downstream task performance at each checkpoint

### Methodological Considerations
- Use **Hessian-based subspace** construction for loss landscape analysis (more informative than random)
- Compute TDA metrics **per-layer** to maintain tractability at LLM scale
- Track **temporal evolution** of topological features, not just static snapshots
- Use **giotto-tda** and **ripser** for persistent homology computation
- Use **PyHessian** for Hessian computation (supports stochastic approximation)
- Compare topological features across **at least 3 model sizes** to identify scaling trends
