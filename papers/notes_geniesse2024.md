# Notes: Geniesse et al. (2024) -- Visualizing Loss Functions as Topological Landscape Profiles

**Full Title:** Visualizing Loss Functions as Topological Landscape Profiles
**Authors:** Caleb Geniesse*, Jiaqing Chen*, Tiankai Xie*, Ge Shi, Yaoqing Yang, Dmitriy Morozov, Talita Perciano, Michael W. Mahoney, Ross Maciejewski, Gunther H. Weber
**Affiliations:** Lawrence Berkeley National Laboratory, Arizona State University, UC Davis, Dartmouth College, ICSI/LBNL/UC Berkeley
**Venue:** Under Review -- Symmetry and Geometry in Neural Representations (Proceedings Track), 2024
**arXiv:** 2411.12136v1 (19 Nov 2024)

---

## 1. Problem Statement

Standard loss landscape visualization methods project the extremely high-dimensional parameter space (with as many dimensions as model parameters) down to just one or two dimensions. This projection discards potentially critical information from additional dimensions. Specifically:

- **Goodfellow et al. (2014):** Interpolated along a single random direction (1D).
- **Im et al. (2016):** Extended to 2D via barycentric/bilinear interpolation.
- **Li et al. (2018):** Introduced filter-wise normalization for 2D landscapes.
- **Yao et al. (2020):** Used top-2 Hessian eigenvectors as directions (still 2D).

**The gap:** All prior approaches restrict sampling to 1 or 2 directions. By limiting to two dimensions, we ignore informative structure captured by the eigenvectors associated with additional dominant eigenvalues of the Hessian matrix. There is no established method to visualize and reason about loss landscapes in 3, 4, or higher dimensions.

---

## 2. Methodology: Constructing Topological Landscape Profiles

The pipeline has four stages:

### Stage 1: Direction Selection (Hessian Eigenvectors)
- Compute the top *n* Hessian eigenvectors using **PyHessian** (Yao et al., 2020).
- These eigenvectors define the *n*-dimensional subspace along which the loss landscape will be sampled.
- The top-*n* eigenvectors (by eigenvalue magnitude) capture the most significant local loss fluctuations.

### Stage 2: Loss Landscape Sampling
- Perturb trained model parameters theta along the *n* directions and evaluate loss:
  ```
  f(alpha_1, ..., alpha_n) = L(theta + sum_{i=1}^{n} alpha_i * delta_i)
  ```
  where delta_i is the i-th Hessian eigenvector direction, and alpha_i are coordinates in the subspace.
- Use an **equally spaced grid** with resolution r = 41 per dimension, centered on the original model (so that sum of alpha_i * delta_i = 0 at center).
- Represent sampled points as an **unstructured grid** where each vertex has *n* coordinates and a scalar loss value.

### Stage 3: Neighborhood Graph and Merge Tree
- Construct a **k-nearest neighbor graph** (symmetric version) to define spatial proximity of vertices, using approximate nearest neighbor search (Dong et al., 2011).
- k = 4 * n (e.g., k=8 for 2D, k=12 for 3D, k=16 for 4D), analogous to pixel connectivity in images.
- Symmetric version: an edge (u,v) is kept only if u is among k-nearest of v AND v is among k-nearest of u.
- Compute a **merge tree** using the **Topology ToolKit (TTK)** (Bin Masood et al., 2021).
- The merge tree tracks connected components of sub-level sets L^-(v) = {x in D : x <= v} as threshold v increases.
  - **Degree-1 nodes** = local minima
  - **Degree-3 nodes** = saddle points connecting two local minima

### Stage 4: Topological Landscape Profile Construction
- Convert the merge tree into a **topological landscape profile** following Oesterling et al. (2013), but adapted for loss functions:
  - Instead of representing maxima as hills (for density/clustering), they represent **minima as basins/valleys**.
  - Each branch ending in a local minimum becomes a **basin** in the profile.
  - Each sub-branch ending in a saddle point becomes a **sub-basin**.
  - Basin width encodes the **cumulative size** of the corresponding branch (number of points).
  - Basin depth encodes the **persistence** (difference between saddle point and minimum loss).
- **Color encoding:** Basins are colored by average loss -- darker blue = lower average loss (evoking ocean depth).
- **Critical point annotation:** Saddle points (orange dots) and minima (red dots) are overlaid on the profile.

---

## 3. Key Innovation

1. **Dimensionality:** The approach goes beyond 1D/2D projections, enabling visualization of 3D and 4D loss landscapes (and in principle arbitrary dimensions). Standard surface plots cannot render 3D or 4D scalar fields; the topological landscape profile re-represents n-dimensional topology in 2D.

2. **Topology over geometry:** Rather than showing raw loss values on a surface (which is inherently limited to 2D), they extract the **topological skeleton** (merge tree) that captures the number, depth, and nesting of minima and saddle points, then re-encode it as a readable 2D basin diagram.

3. **Separation of representation from sampling space:** The topological landscape profile is independent of the dimensionality in which the landscape was sampled. This decoupling means the same visual language works for 2D, 3D, 4D, or higher-dimensional landscapes.

4. **Basin metaphor for loss functions:** Previous work by Oesterling et al. (2013) used hills for density peaks in point cloud clustering. Geniesse et al. invert this to use valleys/basins, which is more natural for loss minimization landscapes.

---

## 4. Sampling Strategy

- **Directions:** Top-*n* Hessian eigenvectors (computed via PyHessian).
- **Grid:** Uniform/equally-spaced grid with resolution r = 41 per dimension, centered on the trained model weights.
- **Total points sampled:** 41^n (e.g., 41^2 = 1,681 for 2D; 41^3 = 68,921 for 3D; 41^4 = 2,825,761 for 4D).
- **Perturbation distance:** 0.01 (mentioned for UNet experiments); layerwise normalization adopted (Li et al., 2018).
- **Neighborhood connectivity:** k-NN graph with k = 4n, symmetric version (mutual nearest neighbors).
- **Practical limit:** While theoretically extendable to arbitrary dimensions, sampling cost grows exponentially. The paper limits to n = 3 and n = 4 in practice.

---

## 5. Topological Features Extracted

- **Zero-dimensional persistent homology (H0):** Connected components of sub-level sets, encoded via the merge tree.
  - **Local minima:** Unique parameter sets where loss is locally minimized (degree-1 nodes in merge tree).
  - **Saddle points:** Points where two basins merge (degree-3 nodes in merge tree).
  - **Persistence:** The "prominence" of a minimum -- the difference between its loss value and the loss value of the saddle point at which its basin merges with another.
  - **Basin size:** Number of grid points belonging to each connected component branch.
- **Qualitative shape descriptors derived from the profiles:**
  - **Funnel-like** landscapes: Deep, narrow basins with well-defined global minima (associated with better-performing models).
  - **Bowl-like** landscapes: Flat, wide, rough basins with many shallow local minima and saddle points (associated with worse-performing models).
  - **Spikiness:** In higher dimensions (4D vs 3D), many more critical points appear, creating spikier profiles.
- **Basin color (average loss):** Reincorporates discarded metric information back into the topological representation.
- **Critical point density:** Distribution of saddle points and minima reflects local sharpness vs. flatness.

---

## 6. Experiments

### Experiment 1: Physics-Informed Neural Networks (PINNs) -- Convection Problem (Section 4.1)
- **Model:** PINN solving 1D convection equation (du/dt + beta * du/dx = 0).
- **Dataset/Task:** Synthetic PDE solution. Varied physical wave speed parameter beta across a range (1 to 70).
- **Hyperparameters varied:** Wave speed beta and learning rate (5 learning rates from 0.0001 to 1.0).
- **Landscapes computed:** 3D and 4D Hessian-based loss landscapes.
- **Repetitions:** 5 random seeds per hyperparameter configuration.

### Experiment 2: UNet with CRF-RNN -- Image Segmentation (Section 4.2)
- **Model:** UNet with learnable CRF-RNN layer (Avaylon et al., 2022).
- **Dataset:** Oxford-IIIT Pet dataset (Parkhi et al., 2012) -- image segmentation task.
- **Hyperparameters varied:** 7 different learning rates (0.0001 to 0.01), trained for 30 epochs.
- **Landscapes computed:** 2D Hessian-based loss landscapes at each training checkpoint.
- **Repetitions:** 5 random seeds per learning rate; 3 seeds shown in detail.
- **Perturbation:** Distance of 0.01, with layerwise normalization.

---

## 7. Key Results

### PINN Convection Results
1. **Simpler topology = better performance:** Models with lower error (small beta) have **smoother, funnel-like** loss landscapes with well-defined global minima. Models with higher error (large beta) have **flatter, rougher, bowl-like** landscapes with many saddle points.
2. **Phase transition variability:** Near the transition from low to high error (intermediate beta values), loss landscape shapes are **highly variable across random seeds**. Some seeds find good solutions (funnel-like), while others fail (bowl-like). This variability is a signature of phase transitions in model performance.
3. **Landscape stability:** For clearly low-error and clearly high-error regimes, landscapes look consistent across random seeds.
4. **Higher-dimensional profiles reveal more structure:** 4D landscapes show many more critical points than 3D, with spikier basins. However, the global shape is preserved -- basins in 4D profiles can be mapped back to wider basins in 3D profiles for the same seed.

### UNet Image Segmentation Results
5. **Training dynamics in landscapes:** Early in training, loss landscapes are shallow with a high-loss global minimum. As training proceeds, the basin deepens (minimum drops) while edges remain at high loss, making the model more sensitive to weight perturbation. Later in training, the basin flattens (edges also decrease in loss), meaning the model becomes more stable.
6. **Learning rate effects:** Deeper basins observed when learning rate is too small or too large, indicating less stable models. Shallower basins correspond to better-performing, more stable models.
7. **Consistency across seeds:** Landscape shape variations across learning rates are reflected consistently across different random seeds.

### Overarching Findings
- **The topology of the loss landscape is simpler for better-performing models.**
- **Greater topological variability occurs near transitions from low to high model performance.**

---

## 8. Code/Tools

- **PyHessian** (Yao et al., 2020): Used to compute top-n Hessian eigenvectors for defining sampling directions.
- **Topology ToolKit (TTK)** (Bin Masood et al., 2021): Used to compute merge trees from the sampled loss landscapes.
- **Approximate k-NN:** Based on Dong et al. (2011) for scalable neighborhood graph construction.
- **No specific repository URL is provided** in the paper for the topological landscape profile code itself. The authors mention a "complementary visualization tool" but do not provide a public link.
- **Related prior work code reference:** Xie et al. (2024) -- "Evaluating loss landscapes from a topology perspective" (arXiv:2411.09807), which developed topology-based metrics for loss landscapes but limited to 2D sampling.

---

## 9. Relevance to Our Project: Topological Persistence Analysis of LLM Training Dynamics

### Direct Applications

1. **Tracking loss landscape topology across LLM training checkpoints:** Just as Section 4.2 tracks UNet loss landscapes over 30 epochs, we could compute topological landscape profiles at each LLM training checkpoint (e.g., every N steps). The evolution of basin depth, width, number of minima, and critical point density could serve as topological features for predicting scaling behavior.

2. **Detecting phase transitions during training:** The finding that topological variability increases near performance transitions is directly applicable to LLM training. We could look for sudden changes in merge tree complexity or basin structure as signals of emergent capabilities, grokking, or loss plateau transitions.

3. **Hessian-based subspace sampling for LLMs:** While computing full Hessian eigenvectors for billion-parameter LLMs is expensive, approximations exist (e.g., stochastic Lanczos quadrature, PyHessian-style top-k computation on subsets of parameters). We could compute top-3 or top-4 Hessian eigenvectors for specific layers or parameter groups and construct localized topological landscape profiles.

4. **Scaling prediction via topological simplicity:** The key finding -- simpler topology correlates with better performance -- suggests a predictive signal: if an LLM's loss landscape becomes topologically simpler at smaller scales, this may predict continued improvement at larger scales. Conversely, increasing topological complexity could signal diminishing returns.

5. **Merge tree features as persistence descriptors:** The merge tree encodes H0 persistence (birth-death of connected components). These persistence features (number of significant minima, total persistence, persistence entropy, max persistence) could be extracted at each checkpoint and used as features in scaling law models.

### Methodological Considerations for LLMs

- **Sampling cost:** 41^4 = ~2.8M forward passes per landscape. For LLMs, each forward pass is expensive. We may need to reduce resolution (e.g., r=11 or r=21) or limit to 2-3 dimensions.
- **Layer-wise analysis:** Rather than computing a single global Hessian, we could compute per-layer or per-block Hessian eigenvectors and construct layer-specific topological profiles, tracking how different parts of the network evolve.
- **Comparison across model scales:** Topological landscape profiles of the same architecture at different parameter counts (e.g., 125M, 350M, 1.3B) could reveal how loss landscape topology changes with scale, potentially informing scaling laws.
- **Alternative to Hessian directions:** For very large models where Hessian computation is infeasible, random directions (with filter-wise normalization) or gradient-based directions could be used as a more affordable alternative, though at some cost to the quality of the captured landscape structure.

### Key Takeaway for Our Project
The topological landscape profile framework provides a principled way to reduce high-dimensional loss landscape information to interpretable topological features (number of basins, persistence, basin size, critical point density). These features could serve as complementary signals -- alongside weight matrix spectral properties (alpha-hat from WeightWatcher) -- for building topology-aware scaling predictors for LLM training dynamics.

---

## Key References from This Paper

| Reference | Relevance |
|-----------|-----------|
| Yao et al. (2020) -- PyHessian | Hessian eigenvector computation tool |
| Li et al. (2018) -- Visualizing loss landscape of neural nets | Filter-wise normalization, 2D landscape baseline |
| Oesterling et al. (2013) -- Topological landscape profiles | Original landscape profile method (for point cloud density) |
| Bin Masood et al. (2021) -- Topology ToolKit (TTK) | Software for merge tree computation |
| Krishnapriyan et al. (2021) -- PINN failure modes | PINN convection experiments baseline |
| Xie et al. (2024) -- Evaluating loss landscapes from topology | Companion paper with topology-based metrics for loss landscapes (2D) |
| Martin & Mahoney (2021) -- Implicit self-regularization | Spectral analysis of weight matrices (connects to our WeightWatcher approach) |
| Martin et al. (2021) -- Predicting NN quality | Training-data-free quality metrics via spectral analysis |
| Goodfellow et al. (2014) -- 1D random direction | First loss landscape visualization |
| Sakarvadia et al. (2024) -- Mitigating memorization | Loss landscape analysis in language models |
