# Notes: Evaluating Loss Landscapes from a Topology Perspective

**Authors:** Tiankai Xie, Caleb Geniesse, Jiaqing Chen, Yaoqing Yang, Dmitriy Morozov, Michael W. Mahoney, Ross Maciejewski, Gunther H. Weber

**Affiliations:** Arizona State University, Lawrence Berkeley National Lab, Dartmouth College, ICSI/LBNL/UC Berkeley

**Venue:** Workshop on Scientific Methods for Understanding Deep Learning, NeurIPS 2024

**ArXiv:** 2411.09807v1 (14 Nov 2024)

---

## 1. Problem Statement

The paper addresses the gap between **visualizing** neural network loss landscapes and **quantitatively analyzing** them. While various methods exist to visualize the loss function with respect to model parameters (the "loss landscape"), these visualizations are typically explored only qualitatively -- researchers look at surface plots and make subjective comparisons. The authors argue that:

- Loss landscapes are inherently high-dimensional (as many dimensions as model parameters), making 2D visual inspection insufficient and potentially misleading.
- Existing visualization can hide important structure (e.g., multi-scale features invisible without zooming/clipping).
- There is no established framework for extracting **actionable, reproducible, quantitative** metrics from loss landscape topology that can be correlated with model performance.

The core contribution is a pipeline that uses topological data analysis (TDA) to **quantify the shape** of loss landscapes, producing metrics that correlate with standard ML performance measures (accuracy, error) and Hessian-based geometry measures (top eigenvalue, trace, spectral density).

---

## 2. Methodology

### 2.1 Loss Landscape Generation

Two approaches for projecting the high-dimensional loss function into a 2D subspace:

1. **Random projection method** (Goodfellow et al., 2014; Li et al., 2018): Sample two random orthogonal vectors of the same dimensionality as the model parameters. In high dimensions, random vectors are nearly orthogonal.

2. **Hessian-based directions** (Yao et al., 2020): Use the top-2 eigenvectors of the Hessian matrix as projection directions. Since these correspond to directions of maximum curvature (most variation in the loss function), they reveal more informative surfaces. Computed using PyHessian.

In both cases, the loss is evaluated as:

```
f(alpha_1, alpha_2) = L(theta + alpha_1 * delta_1 + alpha_2 * delta_2)
```

where `theta` is the trained model parameters, `delta_1` and `delta_2` are the two orthonormal directions, and `(alpha_1, alpha_2)` are coordinates in the 2D subspace.

### 2.2 Data Representations

Four representations of the sampled 2D loss landscape are explored:

1. **Image representation**: Loss values stored as pixel intensities in a 2D image grid.
2. **Unstructured grid (Delaunay triangulation)**: Vertices with scalar loss values connected via Delaunay triangulation.
3. **Unstructured grid (Gabriel graph)**: Vertices connected via Gabriel graph.
4. **Unstructured grid (Approximate k-Nearest Neighbor, AkNN)**: Uses a scalable approximate nearest neighbor algorithm (Dong et al., 2011) with k=8 to define vertex connectivity, mimicking pixel adjacency (left, right, top, bottom, and four diagonals). This was found to give good results and be much faster than triangulation approaches.

### 2.3 TDA Tools Applied

Two specific TDA methods, both capturing 0-dimensional persistent homology:

#### Merge Tree
- Tracks connected components of **sub-level sets** `L^-(v) = {x in D : x <= v}` as a threshold `v` is increased.
- As `v` increases: new connected components form at **local minima** and merge with neighboring components at **saddle points**.
- Encoding: Local minima are degree-1 nodes; saddle points are degree-3 nodes.
- Key advantage: Captures multi-scale structure by default, without manual tuning of visualization parameters. Can reveal hidden structure that is not visible in raw loss landscape plots.

#### Persistence Diagram (0-dimensional)
- Represents features (branches of the merge tree) as points in a 2D plane.
- Horizontal axis = **birth** (value of the minimum where the feature first appears).
- Vertical axis = **death** (value of the saddle where it merges into a more persistent feature).
- Distance from the diagonal `y = x` encodes **persistence** (how long the feature lasts in the filtration).
- Captures information about landscape ruggedness: depth of local valleys and height of barriers between them.

**Important note:** The 0-dimensional persistence diagram is exactly equivalent to the branches in the merge tree. The merge tree additionally encodes *which* component merges into which. The authors deliberately limited analysis to 0-dimensional features, leaving higher-dimensional holes (loops, voids) for future work.

---

## 3. Key Algorithms

### 3.1 Full Analysis Pipeline (Six Stages)

1. **Subspace Definition**: Define 2D subspace via random vectors or top-2 Hessian eigenvectors.
2. **Loss Computation**: Evaluate loss at discrete positions in the subspace via coordinate-based model perturbation: `L(theta + alpha_1 * delta_1 + alpha_2 * delta_2)`.
3. **Data Representation**: Transform loss landscape into image or unstructured grid (Delaunay, Gabriel, or AkNN with k=8).
4. **Topological Analysis**: Compute merge tree and 0-dimensional persistence diagram using the Topology ToolKit (TTK).
5. **Quantitative Evaluation**: Extract summary statistics -- number of saddle points from the merge tree; average persistence from the persistence diagram.
6. **Loss Landscape Property Evaluation**: Relate TDA-based metrics to traditional ML metrics and Hessian-based geometry metrics.

### 3.2 Hessian Computation

- Uses **PyHessian** (Yao et al., 2020) for efficient Hessian computation via randomized numerical linear algebra.
- Extracts: Top-1 Hessian eigenvalue, Hessian trace, eigenvalue spectral density, and top-2 eigenvectors for subspace construction.

### 3.3 Graph Construction

- For unstructured grid representations, approximate k-nearest neighbor graphs (k=8) were preferred over Delaunay/Gabriel triangulations for scalability and speed.

---

## 4. Datasets and Models

### 4.1 Image Pattern Recognition: ResNet-20 on CIFAR-10

- **Architecture**: ResNet-20, tested both with and without residual (skip) connections.
- **Dataset**: CIFAR-10.
- **Training**: Four separate runs per configuration, each with a unique random seed (0, 123, 123456, 2023).
- **Performance**: ResNet-20 with residual connections achieved ~92% accuracy; without residual connections achieved ~90% accuracy.

### 4.2 Scientific ML: Physics-Informed Neural Networks (PINNs)

- **Problem**: 1D convection equation (hyperbolic PDE for transport phenomena):
  ```
  du/dt + beta * du/dx = 0
  ```
- **Architecture**: Standard PINN with a loss combining data fitting, physics residual, and boundary terms.
- **Varied parameter**: Convection coefficient beta in {1.0, 3.0, 5.0, 7.0, 9.0} and more broadly beta in [1..10].
- **Training**: Fixed random seed (seed=0) for the PINN experiments.
- **Observation**: PINNs fail to accurately predict spatiotemporal patterns around beta >= 7.0.

---

## 5. Evaluation Metrics

### 5.1 TDA-Based Metrics (proposed)

| Metric | Source | Definition |
|--------|--------|------------|
| **Number of saddle points** | Merge tree | Count of degree-3 nodes (saddle points where components merge) |
| **Number of minima** | Merge tree | Count of degree-1 nodes (local minima where components are born) |
| **Average persistence** | Persistence diagram | Mean distance of all persistence pairs from the diagonal `y = x`; measures average "prominence" of topological features |

### 5.2 Traditional ML Metrics (for comparison)

| Metric | Description |
|--------|-------------|
| **Accuracy** | Classification accuracy (for ResNet experiments) |
| **Absolute Error** | Prediction error (for PINN experiments) |
| **Top-1 Hessian Eigenvalue** | Largest eigenvalue of the Hessian; measures sharpness |
| **Hessian Trace** | Sum of all eigenvalues; measures total curvature |
| **Hessian Eigenvalue Spectral Density** | Distribution of eigenvalues (plotted in log-log scale) |

---

## 6. Key Results

### 6.1 ResNet-20 (Image Classification)

- **Removing residual connections** produces a significantly more complex loss landscape topology:
  - More saddle points in the merge tree.
  - Lower average persistence.
  - More complicated branching structure visible in the merge tree (but hidden in raw surface plots).
- **With residual connections**: Simpler merge tree (single minimum, no saddle points), confirming the "smoothing" effect of skip connections on the loss landscape.
- **Quantitative correlations**:
  - **Inverse relationship** between number of saddle points and ML-based metrics (accuracy, Hessian eigenvalue, Hessian trace). More saddle points = worse performance.
  - **Direct relationship** between average persistence and ML-based metrics. Higher average persistence = better performance.
- The merge tree revealed structure that was **invisible in the standard loss landscape visualization** -- the raw plot for ResNet without residual connections appeared deceptively simple until zoomed/clipped.

### 6.2 PINNs (Scientific ML)

- As beta increases (harder optimization problem):
  - Loss landscapes become **increasingly complex** in both random projection and Hessian-based views.
  - Merge trees show **more minima and saddle points**.
  - Persistence diagrams show **higher persistence values**.
  - Hessian density indicates increasing volume and complexity.
- **Quantitative correlations** (beta in [1..10]):
  - Number of saddle points **increases** with beta.
  - Average persistence **increases** with beta.
  - Absolute Error, Top-1 Hessian Eigenvalue, and Hessian Trace all increase with beta.
  - All trends align: topological complexity tracks optimization difficulty.
- **Key insight for PINNs**: Unlike ResNet where saddle points and persistence are inversely related, for PINNs both increase together as the problem becomes harder. This reflects a fundamentally different failure mode -- the landscape becomes both more rugged AND features become more prominent.

### 6.3 Cross-Domain Observations

- The relationship between TDA metrics and ML metrics is **context-dependent**:
  - For ResNet: more saddle points = worse, lower persistence = worse (removing skip connections creates many shallow, insignificant features).
  - For PINNs: more saddle points = worse, higher persistence = worse (increasing beta creates deep, significant barriers).
- This suggests that TDA metrics capture richer information about landscape complexity than simple scalar summaries.

---

## 7. Limitations

1. **Only 0-dimensional persistence**: The analysis is restricted to 0-dimensional persistent homology (connected components). Higher-dimensional topological features (1-dimensional loops/cycles, 2-dimensional voids) are not analyzed. The authors explicitly note this as future work.

2. **2D projections only**: Loss landscapes are projected into 2D subspaces, which is an extreme reduction from the true parameter space (potentially millions or billions of dimensions). While Hessian-based directions capture the most informative 2D slice, substantial information is necessarily lost.

3. **Limited model architectures**: Only ResNet-20 (a relatively small, dated architecture) and PINNs are studied. No experiments with larger or more modern architectures (transformers, large language models).

4. **Limited datasets**: Only CIFAR-10 for the image classification task; only the 1D convection equation for PINNs.

5. **Small scale**: ResNet-20 on CIFAR-10 is a small-scale experiment by modern standards. Scalability to large models is discussed aspirationally but not demonstrated.

6. **Context-dependent interpretation**: The relationship between TDA metrics and performance differs between ResNet and PINN experiments (inverse vs. direct), complicating the development of universal guidelines.

7. **No formal statistical testing**: Correlations between TDA metrics and ML metrics are presented visually via plots across 4 seeds (ResNet) or a single seed with varying beta (PINNs), without formal correlation coefficients, confidence intervals, or statistical significance tests.

8. **Computational cost not discussed**: The paper does not discuss the computational cost of generating loss landscapes (which requires many forward passes) or computing TDA metrics, which is important for practical applicability.

9. **Workshop paper length**: As a NeurIPS workshop paper (10 pages including appendix and references), the analysis is necessarily concise and could benefit from deeper investigation.

---

## 8. Code and Tools

### Software Used

| Tool | Purpose | Reference |
|------|---------|-----------|
| **Topology ToolKit (TTK)** | Computing merge trees and persistence diagrams | Bin Masood et al., 2021 |
| **PyHessian** | Hessian computation (eigenvalues, eigenvectors, trace, spectral density) | Yao et al., 2020 |
| **PyTorch** | Neural network training and random vector sampling | -- |
| **Python** | Further analysis and visualization of TDA results | -- |
| **Approximate k-NN** | Graph construction for unstructured grid representations | Dong et al., 2011 |

### Code Repository

- No explicit code repository URL is mentioned in the paper. However, the tools they rely on are open source:
  - TTK: https://topology-tool-kit.github.io/
  - PyHessian: https://github.com/amirgholami/PyHessian
  - The loss landscape visualization approach follows Li et al. (2018), whose code is available at: https://github.com/tomgoldstein/loss-landscape

---

## 9. Relevance to Our Project: Analyzing LLM Training Dynamics for Scaling Prediction

### 9.1 Directly Transferable Ideas

1. **Merge tree and persistence diagram as quantitative loss landscape descriptors**: The core idea of replacing qualitative loss landscape visualization with quantitative TDA-based metrics (number of saddle points, average persistence) is directly applicable to studying LLM training dynamics. We could track how these metrics evolve during training to characterize different training phases.

2. **Hessian-based subspace construction**: Using top Hessian eigenvectors to define the most informative 2D (or higher-dimensional) subspace for loss landscape sampling is a principled approach we should adopt. For LLMs, this would capture the directions of greatest curvature in the training loss.

3. **Pipeline architecture**: Their six-stage pipeline (subspace definition, loss computation, data representation, topological analysis, quantitative evaluation, property evaluation) provides a clean template for our own analysis pipeline.

4. **TTK and PyHessian as computational tools**: Both are mature, open-source tools that we can directly integrate into our pipeline.

### 9.2 Required Adaptations for LLM Training Dynamics

1. **Temporal dimension**: Xie et al. analyze loss landscapes at fixed trained checkpoints. For studying training dynamics, we need to compute loss landscapes (and their TDA metrics) **at multiple checkpoints during training**, creating a time series of topological features. Tracking how the number of saddle points, average persistence, and other topological metrics evolve over training steps could reveal phase transitions and predict scaling behavior.

2. **Scale considerations**: LLMs have billions of parameters, making even 2D loss landscape generation (which requires many forward passes with perturbed parameters) computationally expensive. We may need:
   - Efficient sampling strategies (fewer grid points).
   - Layer-wise or block-wise loss landscape analysis (rather than full-model perturbation).
   - Stochastic approximations for Hessian eigenvectors (PyHessian already supports this).

3. **Higher-dimensional analysis**: The authors explicitly note that extending to higher-dimensional loss landscapes (3D to 10D, using top-k Hessian eigenvectors) could reveal additional structure. For LLMs, this is particularly relevant since the effective dimensionality of the loss landscape may be higher than 2, and scaling behavior may only be visible in higher-dimensional topological features.

4. **Higher-dimensional persistent homology**: Beyond 0-dimensional features (connected components), 1-dimensional features (loops/tunnels in the loss landscape) could be informative for understanding training dynamics -- e.g., the presence of circular paths around saddle points or barriers that optimization must navigate.

5. **Scaling law connection**: The key gap between Xie et al.'s work and our project is connecting topological metrics to **scaling laws**. Specifically:
   - Do topological features of the loss landscape evolve predictably with model scale (parameter count, data size, compute)?
   - Can early-training topological signatures predict final model performance at larger scale?
   - Does the number of saddle points or average persistence follow power-law scaling with model size?

6. **Cross-scale comparison**: Rather than comparing architectures (ResNet with/without skip connections) or problem difficulty (varying beta), we would compare models of different sizes trained on the same data, looking for topological signatures that predict scaling behavior.

### 9.3 Specific Hypotheses to Test

Based on the Xie et al. findings, we can formulate several hypotheses for LLM training:

- **H1**: The number of saddle points in the loss landscape merge tree decreases during LLM training, and the rate of decrease correlates with eventual model quality.
- **H2**: Average persistence of loss landscape features evolves non-monotonically during training, with phase transitions corresponding to capability emergence.
- **H3**: The topological complexity of the loss landscape at a given training step scales predictably with model size, enabling extrapolation of training dynamics to larger scales.
- **H4**: Topological features computed in higher-dimensional Hessian-based subspaces (3D-10D) provide stronger predictive signals for scaling than 2D projections.

### 9.4 Practical Considerations

- **Checkpoint frequency**: Need to balance computational cost of loss landscape generation with temporal resolution of topological feature tracking.
- **Batch dependence**: Loss landscape shape may depend on which data batch is used for evaluation; need to assess sensitivity.
- **Comparison across scales**: Must ensure consistent hyperparameters for loss landscape generation (grid resolution, perturbation range, number of Hessian eigenvectors) across models of different sizes to enable valid comparison.

---

## Key References from the Paper

- Li et al. (2018) -- "Visualizing the loss landscape of neural nets" (filter-normalized random direction method)
- Yao et al. (2020) -- "PyHessian: Neural networks through the lens of the Hessian" (Hessian-based directions)
- Carr et al. (2003) -- "Computing contour trees in all dimensions" (merge tree algorithm)
- Edelsbrunner & Harer (2008) -- "Persistent Homology -- a Survey" (persistence diagrams)
- Krishnapriyan et al. (2021) -- "Characterizing possible failure modes in physics-informed neural networks"
- Martin & Mahoney (2021) -- "Implicit Self-Regularization in Deep Neural Networks" (random matrix theory for DNN analysis)
- Yang et al. (2021) -- "Taxonomizing local versus global structure in neural network loss landscapes"
- Bin Masood et al. (2021) -- "An overview of the Topology ToolKit" (TTK software)
