# Detailed Notes: Neural Persistence (Rieck et al., ICLR 2019)

**Full Title:** Neural Persistence: A Complexity Measure for Deep Neural Networks Using Algebraic Topology

**Authors:** Bastian Rieck, Matteo Togninalli, Christian Bock, Michael Moor, Max Horn, Thomas Gumbsch, Karsten Borgwardt (ETH Zurich / SIB Swiss Institute of Bioinformatics)

**Venue:** Published as a conference paper at ICLR 2019 (arXiv:1812.09764v3)

---

## 1. Problem Statement

The paper addresses the lack of **structural complexity measures** for deep neural networks. Existing approaches for understanding neural networks all rely on interrogating the network with input data (e.g., feature visualization, sensitivity analysis, information-theoretic descriptions). No measures existed for characterizing and monitoring the **structural properties** of a network (i.e., its weights and connectivity) independently of input data. The authors note that formal measures for assessing the generalization capabilities of deep neural networks had yet to be identified (citing Zhang et al., 2017). The key challenge is to provide meaningful insights while maintaining theoretical generality.

**Core gap:** There was no principled, data-independent, computationally efficient measure that integrates both network weights and connectivity to characterize neural network complexity.

---

## 2. Methodology: Defining Neural Persistence

### 2.1 Stratified Graph Representation

The neural network is rephrased as a **stratified graph** -- a multipartite graph G = (V, E) where:
- V = V_0 ⊔ V_1 ⊔ ... (disjoint vertex sets per layer)
- Edges only connect adjacent vertex sets: if u in V_i, v in V_j, and (u,v) in E, then j = i+1
- The k-th layer is the subgraph G_k := (V_k ⊔ V_{k+1}, E_k)

### 2.2 Filtration on the Network Graph

A novel **weight-based filtration** is constructed:

1. **Weight normalization:** Given all weights W for one training step, compute w_max = max|w|. Transform weights to W' = {|w|/w_max : w in W}, yielding values in [0, 1] indexed in non-ascending order: 1 = w'_0 >= w'_1 >= ... >= 0.

2. **Descending filtration:** For the k-th layer, define G_k^(0) ⊆ G_k^(1) ⊆ ..., where G_k^(i) includes all edges whose transformed weight >= w'_i. This means edges with the **largest absolute weights are added first**, mimicking the intuition that large weights indicate stronger connections.

3. **Scale invariance:** Since transformed weights are in [0,1], the filtration is invariant to global weight scaling, enabling comparison across different networks.

**Key design choice:** The filtration starts from the strongest connections and progressively adds weaker ones. Activation functions are not explicitly modeled, but they influence weight evolution during training.

### 2.3 Neural Persistence Definition

**Definition 2 (Neural Persistence):** The neural persistence of the k-th layer G_k is the p-norm of its persistence diagram D_k:

```
NP(G_k) := ||D_k||_p := ( sum_{(c,d) in D_k} pers(c,d)^p )^{1/p}
```

For p=2, this captures the Euclidean distance of persistence diagram points to the diagonal.

**Normalized Neural Persistence (Definition 3):** NP(G_k) divided by its upper bound, allowing comparison of layers with different numbers of neurons.

**Mean Normalized Neural Persistence (Definition 4):** The average of normalized NP values across all layers:

```
NP_bar(G) := (1/l) * sum_{k=0}^{l-1} NP_tilde(G_k)
```

---

## 3. Key Algorithms

### 3.1 Algorithm 1: Neural Persistence Calculation

```
Input: Neural network with l layers and weights W
1. w_max <- max_{w in W} |w|                    # Find largest absolute weight
2. W' <- {|w|/w_max : w in W}                   # Transform weights to [0,1]
3. For k in {0, ..., l-1}:
4.     F_k <- filtration of k-th layer           # Sort edges by transformed weight
5.     D_k <- PersistentHomology(F_k)            # Compute persistence diagram
6. Return {||D_0||_p, ..., ||D_{l-1}||_p}       # Neural persistence per layer
```

**Computational complexity:**
- Filtration (sorting weights): O(n log n)
- Persistent homology (union-find): O(n * alpha(n)), where alpha is the inverse Ackermann function
- Overall: **nearly linear** in the number of weights

**What is computed:** Only **0-dimensional persistent homology** (connected components). The filtration contains at most 1-simplices (edges), so they track how connected components are created and merged. This is structurally equivalent to computing a **maximum spanning tree**.

All vertices are present at the start (assigned weight 1), producing |V_k x V_{k+1}| connected components initially. Persistence diagram entries are of the form (1, x) with x in W'.

### 3.2 Algorithm 2: Early Stopping Based on Neural Persistence

```
Input: Network N, patience g, delta_min
1. P <- 0, G <- 0                               # Init best value and counter
2. At each epoch:
3.     P' <- NP_bar(N)                           # Compute mean normalized NP
4.     If P' > P + delta_min:                    # Improvement detected
5.         P <- P', G <- 0                       # Update and reset patience
6.     Else:
7.         G <- G + 1                            # Increment patience counter
8.     If G >= g:                                # Patience exhausted
9.         Stop training, return P
```

### 3.3 Algorithm 3: Approximating Neural Persistence for Convolutional Layers

A closed-form approximation is derived for convolutional layers by exploiting the structure of unrolled weight matrices. Each output neuron shares the same filter weights, so destruction events in the persistence diagram simplify. The approximation is ~23,000x faster than naive exact computation (0.00038s vs. 8.77s per filter per evaluation step).

---

## 4. Experiments

### 4.1 Networks and Datasets

| Dataset | Architecture | Optimizer | Epochs | Runs |
|---------|-------------|-----------|--------|------|
| MNIST | Perceptron | Minibatch SGD (lr=0.5) | 10 | 100 |
| MNIST | [650, 650] | Adam (lr=0.0003) | 40 | 50 |
| (Fashion-)MNIST | [50,50,20], [300,100], [20,20,20] | Adam (lr=0.0003) | 40 | 100 |
| CIFAR-10 | [800, 300, 800] | Adam (lr=0.0003) | 80 | 10 |
| IMDB | [128, 64, 16] | Adam (lr=1e-5) | 25 | 5 |
| Fashion-MNIST (CNN) | LeNet-like (2 conv + 1 FC) | -- | 20 | 100 |
| Fashion-MNIST | [500, 500, 200] | -- | -- | 10 |
| MNIST | [500, 500, 200] | -- | -- | 10 |

All networks used **ReLU** activation functions. Adam optimizer hyperparameters tuned via cross-validation.

### 4.2 Comparisons

- **Neural persistence vs. clustering coefficient** (traditional graph measure): Clustering coefficient fails to distinguish trained networks from poorly trained ones; NP succeeds.
- **Neural persistence vs. random weight matrices:** Compared trained perceptrons against random Gaussian matrices, random uniform matrices, and diverging networks.
- **NP-based early stopping vs. validation-loss-based early stopping:** Exhaustive comparison over a G x G parameter grid of patience (g) and burn-in rate (b), evaluated on all datasets.
- **Different regularization techniques:** Compared unmodified networks vs. batch normalization vs. dropout (50%).

---

## 5. Key Results

### 5.1 Neural Persistence During Training

- **Trained networks** are clearly distinguished from random/diverging networks by their NP values.
- **Uniform random matrices** have lower NP than Gaussian ones (consistent with functional sparsity -- few neurons have large absolute weights).
- **Diverging networks** (bad learning rate) have NP similar to random Gaussian matrices.
- NP exhibits **clear change points** during training for Fashion-MNIST, which can be exploited for early stopping.
- For CIFAR-10 (where FCNs struggle), NP shows rather incremental growth with no clear maximum.

### 5.2 Regularization and Best Practices

- Networks with **dropout** yield the highest mean normalized NP (around 0.62-0.64 on MNIST).
- Networks with **batch normalization** yield intermediate NP (around 0.54-0.58).
- **Unmodified networks** yield the lowest NP (around 0.50-0.52).
- This ordering mirrors the test accuracy ordering, confirming NP captures something related to generalization quality.
- Dropout's effect is explained by analogy to ensemble learning: independent sub-network training increases per-layer redundancy.

### 5.3 Early Stopping Results

| Dataset | Barycentre (epoch diff, acc diff) | Final Test Accuracy |
|---------|-----------------------------------|---------------------|
| Fashion-MNIST | (-0.53, -0.08) | 86.72 +/- 0.43 |
| MNIST | (+0.17, -0.06) | 96.16 +/- 0.24 |
| CIFAR-10 | (-1.33, -1.13) | 52.19 +/- 3.40 |
| IMDB | (-1.68, +0.07) | 87.35 +/- 0.03 |

Key findings:
- On Fashion-MNIST, NP-based stopping averages **0.53 epochs earlier** with only 0.08% accuracy loss.
- On IMDB, NP-based stopping averages **1.68 epochs earlier** with 0.07% accuracy *gain*.
- NP-based stopping does **not require a validation set**, freeing that data for training.
- NP triggers reliably for more parameter combinations than validation loss.
- NP stops earlier when overfitting can occur, and later when longer training is beneficial.

### 5.4 Data Scarcity and Noisy Labels

- NP-based stopping trains **longer when fewer data samples are available**, yielding better accuracy than validation/training loss.
- With **permuted labels**, NP-based stopping remains stable, while validation loss stopping degrades (trains too long, overfits to noise).
- NP is robust to batch size variations (unlike training loss).

### 5.5 Depth Effects

- Adding layers initially increases NP variability (more valid training configurations).
- Beyond a certain depth, variability decreases: very deep architectures converge to similar NP regimes.
- Different datasets (MNIST vs. Fashion-MNIST) with same architecture lead to shifted NP distributions, confirming NP captures data-dependent structural differences.

### 5.6 NP vs. Validation Accuracy (No Direct Correlation)

- For deeper networks, **no correlation** between high NP and high accuracy was observed.
- Networks initialized with artificially high NP (via beta distribution weights) do not achieve better accuracy.
- This motivates using NP as a **change-based** criterion (detecting stagnation) rather than an absolute threshold.

---

## 6. Applications

### 6.1 Early Stopping Without Validation Data
The primary application: an early stopping criterion based on monitoring NP stagnation. Particularly valuable in **data-scarce regimes** where reserving a validation set is costly.

### 6.2 Identifying Effective Regularization
NP correctly captures the benefits of dropout and batch normalization, suggesting it could be used to evaluate regularization strategies.

### 6.3 Potential for Architecture Search (Conjectured)
The authors conjecture that NP could assess architecture suitability for a given task, though this is deferred to future work. NP appears unreliable when the architecture is fundamentally incapable of learning the data.

### 6.4 Network Comparison
Normalized NP enables comparing networks of different sizes and architectures on a common scale.

---

## 7. Limitations

1. **Fully-connected layers only (primarily):** The main theory and best results apply to fully-connected layers. Extension to CNNs requires a separate closed-form approximation that does **not** work well for early stopping (up to 4% accuracy loss, no configurations improve accuracy).

2. **0-dimensional homology only:** Only connected components (H_0) are tracked. Higher-dimensional topological features (tunnels, voids via cliques) are mentioned but not used.

3. **No direct correlation with generalization for deep networks:** High NP does not necessarily mean high accuracy. The measure is more useful for tracking *changes* during training.

4. **Sensitive to architecture-task fit:** On CIFAR-10 with FCNs (a poor fit), NP shows incremental growth with no clear stopping signal, making it unreliable for architectures that cannot learn the task.

5. **Does not model activation functions explicitly:** While activations influence weight evolution, they are not directly incorporated into the filtration.

6. **Limited to feedforward networks:** The stratified graph formulation requires edges only between adjacent layers. Recurrent connections, attention mechanisms, and skip connections are not addressed.

7. **Using p-norm of all weights as a proxy does not work:** The authors tested using the p-norm of all weights directly (without persistent homology) as an early stopping criterion; it was never triggered, confirming that PH captures information hidden in the raw weight statistics.

8. **Small-scale experiments:** Experiments use relatively small networks (up to [800, 300, 800]) and simple datasets (MNIST, Fashion-MNIST, CIFAR-10, IMDB). No experiments on modern large-scale architectures.

---

## 8. Code and Tools

- **GitHub repository:** https://github.com/BorgwardtLab/Neural-Persistence
- **Framework:** Keras (Chollet et al., 2015) with TensorFlow backend
- **Key dependencies:** Standard persistent homology computation via union-find data structures
- The implementation includes both exact computation and the closed-form CNN approximation (Algorithm 3)

---

## 9. Relevance to Our Project: Analyzing LLM Training Dynamics

### 9.1 Direct Applicability

**Neural persistence can be computed on the fully-connected (linear) layers of transformer-based LLMs.** Transformers consist primarily of:
- Attention layers (query/key/value projections + output projection) -- these are linear layers
- MLP/FFN blocks (two linear layers with activation)
- Layer normalization

Each linear layer maps naturally to a bipartite graph (stratified graph layer) where NP can be computed. This provides a **data-independent structural complexity measure** that can be tracked throughout LLM training.

### 9.2 Specific Use Cases for LLM Training Dynamics

1. **Training phase detection:** NP's change points during training (visible in Fashion-MNIST) could reveal distinct phases in LLM training -- initial rapid learning, refinement, and convergence/overfitting. This aligns with our project's goal of identifying topological signatures of training dynamics.

2. **Early stopping for LLM pretraining:** Since LLM pretraining is enormously expensive, an NP-based signal that does not require validation data could provide early warning of convergence, potentially saving millions of dollars in compute.

3. **Scaling law prediction:** By computing NP at multiple checkpoints during training of models at different scales, we could investigate whether NP follows predictable scaling laws -- potentially enabling predictions about larger models from smaller ones.

4. **Layer-wise analysis:** Computing NP per layer in a transformer reveals which layers develop complex structure early vs. late in training. This could illuminate the "depth-wise learning dynamics" of LLMs.

5. **Comparing training recipes:** Just as NP distinguishes dropout vs. batch normalization, it could compare different LLM training configurations (learning rate schedules, optimizers, regularization).

### 9.3 Adaptations Needed

1. **Scale challenge:** LLMs have billions of parameters. The O(n log n) sorting step per layer is feasible but needs efficient implementation. The union-find step is nearly linear, which helps.

2. **Attention mechanism:** The paper's stratified graph formulation does not capture attention patterns. We may need to extend the framework to handle attention weight matrices or treat multi-head attention as multiple parallel bipartite graphs.

3. **Beyond H_0:** For LLMs, higher-dimensional persistent homology (tracking cycles, voids) in the weight graph could capture richer structural information about learned representations.

4. **Normalization layers:** Batch/layer normalization parameters interact with weight magnitudes. The normalization step (dividing by w_max) partially addresses this, but care is needed when comparing across layers with different normalization schemes.

5. **Sparse attention / mixture-of-experts:** Modern LLM architectures with sparse components would require adapting the filtration to handle non-fully-connected layers.

### 9.4 Key Takeaways for Our Project

- **Neural persistence is the foundational paper** for applying persistent homology to neural network weight structures.
- The **per-layer decomposition** and **normalization scheme** are directly reusable.
- The **early stopping application** is the most mature practical contribution and most relevant to monitoring LLM training.
- The **limitation on CNNs** is a cautionary note: naive extension to new architectures may not work; careful design of the filtration is essential.
- The **lack of correlation between absolute NP and accuracy** for deep networks suggests we should focus on **NP dynamics (rate of change)** rather than absolute NP values when analyzing LLM training.

---

## Key Equations Summary

| Equation | Description |
|----------|-------------|
| NP(G_k) = \|\|D_k\|\|_p | Neural persistence of layer k (Eq. 1) |
| 0 <= NP(G_k) <= (max phi - min phi)(n-1)^{1/p} | Theoretical bounds (Theorem 1, Eq. 2) |
| \|\|1 - w_max\|\|_p <= NP(G_k) <= \|\|1 - w_min\|\|_p | Empirical bounds (Theorem 2, Eq. 5) |
| NP_tilde(G_k) = NP(G_k) / NP(G_k^+) | Normalized NP (Definition 3) |
| NP_bar(G) = (1/l) sum NP_tilde(G_k) | Mean normalized NP (Definition 4) |

---

*Notes compiled from all 9 chunks of the paper (25 pages including appendix).*
*Paper PDF chunks: papers/pages/rieck2019_neural_persistence_chunk_001.pdf through _009.pdf*
