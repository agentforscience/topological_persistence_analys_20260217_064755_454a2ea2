# Notes: "A Dynamical Model of Neural Scaling Laws" -- Bordelon, Atanasov, Pehlevan (2024)

**Citation:** Blake Bordelon, Alexander Atanasov, Cengiz Pehlevan. "A Dynamical Model of Neural Scaling Laws." Proceedings of the 41st International Conference on Machine Learning (ICML), Vienna, Austria. PMLR 235, 2024. arXiv:2402.01092v4.

**Affiliations:** SEAS & Kempner Institute, Harvard University; Department of Physics, Harvard University.

---

## 1. Problem Statement

The paper addresses a fundamental open question in deep learning: **why do neural networks exhibit power-law scaling of performance with training time, model size, and dataset size, and why are the scaling exponents different for each of these resources?**

Specifically, the authors aim to explain the following empirically observed phenomena:

1. **Power-law scaling of test loss:** The test loss of a model with N parameters trained for t steps follows L(t, N) ~ L_0 + a_t * t^{-r_t} + a_N * N^{-r_N}, where the exponents r_t and r_N depend on the dataset and architecture.

2. **Asymmetric compute-optimal exponents:** The exponents r_t (time) and r_N (model size) are generally different, leading to an asymmetric compute-optimal allocation rule where training time should be increased faster than model size as compute budget grows (i.e., t ~ C^{c_2} and N ~ C^{c_1} with c_2 > c_1). This observation was central to the Chinchilla scaling findings (Hoffmann et al., 2022).

3. **Wider models train faster:** Under mean-field/muP scaling, wider networks converge faster, obtaining lower test loss at equal number of iterations.

4. **Finite-width convergence rates change over time:** Early in training, finite-width corrections go as 1/width (universal), but at late time the exponent becomes task-dependent (width^{-c}).

5. **Ensembling is not the same as going wider:** Ensembling finite models cannot match the performance of a single wider model.

6. **Gradual buildup of overfitting:** The gap between training and test loss accumulates over time due to repeated reuse of data.

---

## 2. Theoretical Framework

### Model Architecture: Random Feature Model (Teacher-Student Setting)

The paper uses a **random feature model** as a solvable proxy for neural network training in the linearized (lazy/kernel) regime. The setup is:

**Teacher Model:**
- Input x in R^D drawn from distribution p(x).
- Target function: y(x) = (1/sqrt(M)) * w* . psi(x) + sigma * epsilon(x)
- Features psi(x) in R^M are the eigenfunctions of the infinite-width NTK.
- Feature covariance is diagonal in the eigenbasis: <psi_k(x) psi_l(x)> = delta_{kl} * lambda_k.

**Student Model:**
- Uses a random projection of the teacher features: f(x) = (1/sqrt(N)) * w . A * psi(x)
- A in R^{N x M} is a random matrix with iid entries (mean zero, variance one).
- N is interpreted as the model size; N -> infinity recovers the infinite-width kernel.
- This projection models the mismatch between finite-width and infinite-width NTK features.

**Training:**
- Gradient flow on MSE loss over a dataset D of P samples.
- Extensions to discrete-time SGD, momentum, and one-pass online SGD are also derived.

### Mathematical Machinery: Dynamical Mean Field Theory (DMFT)

The core analytical tool is **Dynamical Mean Field Theory (DMFT)**, derived from statistical physics (Martin-Siggia-Rose path integral formalism and dynamical cavity methods).

Key elements:
- The residual error vector v^0 = w* - (1/sqrt(N)) A^T w(t) tracks the discrepancy between target and learned weights.
- A chain of auxiliary vectors {v^1, v^2, v^3, v^4} propagates information through the data matrix Psi and projection matrix A.
- **Order parameters** are correlation functions C_i(t,s) and response functions R_i(t,s) that capture cross-time statistics.
- The response functions are **time-translation invariant (TTI)**, enabling exact analysis in the Fourier domain.
- The **transfer function** H_k(omega) = 1 / (i*omega + lambda_k * R_1(omega) * R_3(omega)) characterizes how each eigenmode of the kernel is learned over time.
- Self-consistent equations for R_1(omega) and R_3(omega) couple the effects of finite data (P) and finite model size (N).

The t -> infinity limit of the DMFT recovers known static results from random matrix theory / replica methods (Atanasov et al., 2023; Maloney et al., 2022).

---

## 3. Key Predictions

### 3.1 Asymmetric Compute-Optimal Scaling

For power-law features with eigenvalues lambda_k ~ k^{-b} and task-power (w*_k)^2 * lambda_k ~ k^{-a}, the compute-optimal allocation of compute C = N*t is:

- **Training time:** t ~ C^{bm / (a-1 + bm)}
- **Model size:** N ~ C^{(a-1) / (a-1 + bm)}
- **Optimal loss:** L*(C) ~ C^{-(a-1)m / (a-1 + bm)}

where m = min{a-1, 2b}.

For the typical regime where a-1 < 2b (difficult tasks), this simplifies to:
- **L*(C) ~ C^{-(a-1)/(1+b)}**
- **t ~ C^{b(a-1) / (a-1 + b(a-1))} = C^{b/(1+b)}**
- **N ~ C^{(a-1) / ((a-1)(1+b))} = C^{1/(1+b)}**

Since b > 1 typically, the exponent for t is larger than for N, meaning **training time should scale faster than model size** with increasing compute. This is consistent with the Chinchilla findings (Hoffmann et al., 2022) and larger asymmetries observed in vision MLPs (Bachmann et al., 2024).

In the limit b -> 1, time and parameter count scale linearly together.

### 3.2 Ensembling is Suboptimal

The theory explains why ensembling E models of size N is suboptimal compared to training a single model of size E*N:
- Ensembling reduces **variance** (due to random initialization) by a factor of 1/E.
- But it does not reduce **bias**, which depends on N and P.
- Increasing N reduces both bias and variance, making it strictly preferable.

### 3.3 Wider Models Train Faster but Need Sufficient Data

- In data-rich regimes (large P), wider models strictly improve test loss at all times.
- In data-limited regimes, wider models can overfit, and the "wider is better" phenomenon reverses.
- Optimal early stopping or regularization restores monotonic improvement with width.

---

## 4. Power Law Exponents

The paper derives bottleneck scaling exponents for power-law features with (w*_k)^2 * lambda_k ~ k^{-a} and lambda_k ~ k^{-b}:

| Resource Bottleneck | Scaling Exponent | Condition |
|---------------------|------------------|-----------|
| **Time** (P, N -> inf) | L ~ t^{-(a-1)/b} | Always |
| **Data** (t, N -> inf) | L ~ P^{-min{a-1, 2b}} | a-1 < 2b: L ~ P^{-(a-1)} |
| **Model** (t, P -> inf) | L ~ N^{-min{a-1, 2b}} | a-1 < 2b: L ~ N^{-(a-1)} |
| **Compute-optimal** | L*(C) ~ C^{-(a-1)/(1+b)} | For typical tasks where a-1 < 2b |

**Key asymmetry:** The time exponent r_t = (a-1)/b differs from the model/data exponent r_N = r_P = min{a-1, 2b}. For difficult tasks (a-1 < 2b), r_N = a-1 while r_t = (a-1)/b. Since b > 1 typically, r_t < r_N, meaning the model converges slower with time than with model size. This is the origin of the asymmetric compute-optimal strategy.

### Mechanism Behind the Asymmetry

The asymmetry arises from **rank constraints** in the effective dynamics:
- For model and data bottlenecks: the effective rank k* of the learned subspace scales as k* ~ N or k* ~ P directly.
- For the time bottleneck: the k-th eigenfeature is learned on a timescale tau_k ~ k^b, so at time t, approximately k* ~ t^{1/b} modes have been learned.
- The unlearned variance is L ~ k*^{-(a-1)}, giving L ~ t^{-(a-1)/b} for time vs L ~ N^{-(a-1)} for model size.

### Early-Time vs Late-Time Convergence

- **Early time:** Finite-width corrections scale universally as 1/N (and 1/P for data), independent of task structure. This is consistent with prior perturbative analyses (Bahri et al., 2021; Dyer & Gur-Ari, 2020).
- **Late time:** Corrections become task-dependent, scaling as N^{-(a-1)} or P^{-(a-1)}, which can be very different from 1/N.

### Concrete Empirical Values (from CIFAR-5M experiments)

For a Wide ResNet on CIFAR-5M (animate vs inanimate classification):
- NTK spectral decay: lambda_k ~ k^{-2.0} (so b = 2.0)
- Task-power decay: k^{-0.15} (so a ~ 1.15)
- Predicted compute-optimal exponent: L*(C) ~ C^{-(a-1)/(1+b)} = C^{-0.15/3.0} = C^{-0.05}
- This was verified experimentally for linearized (kernel-regime) networks.
- Feature-learning networks achieved substantially better scaling (discussed below).

---

## 5. Experimental Validation

### 5.1 Synthetic Power-Law Features

The primary validation uses synthetic datasets with power-law structured features:
- Eigenvalues lambda_k = k^{-b} and target weights (w*_k)^2 * lambda_k = k^{-a}.
- Systematic variation of a in {1.5, 2.0} and b in {1.0, 1.25, 1.5, 2.0}.
- All predicted scalings (time, model, data, compute-optimal) verified against simulations.
- Standard deviations across random draws of data and projections shown to be small (O(1/P + 1/N)).

### 5.2 Realistic Vision Tasks

- **CIFAR-5M dataset** (Nakkiran et al., 2021) with animate vs inanimate binary classification.
- **Architecture:** Wide ResNet (Zagoruyko & Komodakis, 2016) with varying channel widths.
- Extracted NTK spectra and task-power distributions from initial kernels across widths.
- Fitted power laws: lambda_k ~ k^{-2.0}, task-power ~ k^{-0.15}.
- Predicted compute-optimal scaling L*(C) ~ C^{-0.05} matched linearized networks excellently.

### 5.3 Language Models (Transformers)

- Trained transformers on Wikitext with 100M and 5M tokens.
- Observed power-law scaling of loss with training time and width: L ~ t^{-0.16} + N^{-0.76} (for 100M token regime).
- Demonstrated overfitting and "wider is not always better" reversal on the 5M-token dataset due to data reuse.

### 5.4 Online SGD

- Extended DMFT to one-pass SGD with minibatch size B.
- In online regime: no data bottleneck (no overfitting), loss limited only by time and model size.
- Batch size B introduces additive variance but no asymptotic plateau.
- Continuous-time limit of online SGD matches P -> infinity limit of batch gradient flow.

---

## 6. Connection to Training Dynamics

### 6.1 Temporal Evolution

The DMFT provides a complete description of the temporal trajectory of training:

1. **Transfer functions H_k(t):** Each eigenmode k of the kernel is learned according to H_k(t), which starts at 1 (unlearned) and decays toward 0 (fully learned). The mode-k timescale is tau_k ~ k^b / (R_1 * R_3), which depends on both the spectral structure and the finite-size effects.

2. **Early-time regime:** H_k(t) ~ exp(-lambda_k * t) -- pure exponential decay at infinite-width rate. Finite-size corrections are O(1/N + 1/P) and build up gradually over time (Equation 83 in the paper):
   H_k(t) ~ exp(-lambda_k * t) + c*(1/alpha + 1/nu)/lambda_k * [1 - exp(-lambda_k * t) - lambda_k * t * exp(-lambda_k * t)]

3. **Late-time regime:** Dynamics are frozen in the nullspace of the effective operator (1/N * A^T A)(1/P * Psi^T Psi). The unlearned components determine the final loss floor.

4. **Timescale density:** The transfer function can be decomposed as H_k(t) = integral du rho_k(u) exp(-u*t), where rho_k(u) is a density of timescales. For finite N, P, this density spreads out from a Dirac mass at lambda_k, introducing a distribution of learning rates. This recovers the Marchenko-Pastur law in the isotropic case.

### 6.2 Train-Test Gap Dynamics

The exact gap between train and test loss is expressed in terms of DMFT order parameters (Equation 18):

L(t) - L_hat(t) = -(2/P) * integral R_{0,2}(t,t') C_1(t,t') dt' + (1/P^2) * double integral R_{0,2}(t,t') R_{0,2}(t,s') C_1(t',s') dt' ds'

- At early time: gap is O(1/P) -- universal.
- At late time: gap grows with a task-dependent scaling with P.
- Larger datasets P delay the onset of overfitting.

### 6.3 Effect of Optimization Algorithms

The framework handles multiple optimizers:
- **Gradient flow** (continuous time): primary analysis.
- **Discrete-time GD:** DMFT equations defined on Z-transform instead of Fourier.
- **Momentum:** Replaces d/dt with beta * d^2/dt^2 + d/dt; modifies transfer function poles.
- **One-pass SGD:** Data matrix Psi(t) is redrawn each step; eliminates train-test gap.

---

## 7. Key Results

### Main Findings

1. **Asymmetric scaling is generic:** For power-law structured features, the time exponent r_t = (a-1)/b always differs from the model exponent r_N = a-1 (for typical tasks). Since b > 1, we have r_t < r_N, so models improve faster by increasing size than by training longer.

2. **Compute-optimal strategy depends on spectral structure:** The optimal allocation t ~ C^{b/(1+b)}, N ~ C^{1/(1+b)} depends on the spectral decay rate b. Faster spectral decay (larger b) means allocating more compute to training time rather than model size.

3. **Model and data bottleneck exponents are identical:** r_N = r_P = min{a-1, 2b}. This symmetry between data and model size is specific to the random feature setting.

4. **Ensembling is strictly suboptimal:** Increasing model size N is always preferable to ensembling E models of size N/E at fixed compute, because larger N reduces bias in addition to variance.

5. **Overfitting effects are gradual and predictable:** The train-test gap grows as an integral over the history of training, with the rate depending on P. The theory precisely predicts when overfitting kicks in.

6. **Feature learning breaks lazy-regime scaling:** Feature-learning networks on CIFAR-5M achieve substantially better compute-optimal scaling than predicted by the initial NTK. The after-kernel evolves throughout training (spectrum flattens from k^{-2.0} to k^{-1.4}), and its alignment with the task improves over time. A full theory of scaling laws requires understanding this kernel evolution.

### Summary of Scaling Relations

For power-law features with (w*_k)^2 lambda_k ~ k^{-a} and lambda_k ~ k^{-b}, and with m = min{a-1, 2b}:

| Quantity | Scaling |
|----------|---------|
| Test loss vs time | L ~ t^{-(a-1)/b} |
| Test loss vs model size | L ~ N^{-m} |
| Test loss vs data size | L ~ P^{-m} |
| Compute-optimal loss | L*(C) ~ C^{-(a-1)m / (a-1 + bm)} |
| Optimal N vs C | N* ~ C^{(a-1) / (a-1 + bm)} |
| Optimal t vs C | t* ~ C^{bm / (a-1 + bm)} |
| Early-time width correction | Delta L ~ 1/N |
| Late-time width correction | Delta L ~ N^{-(a-1)} |

---

## 8. Limitations

1. **Lazy/kernel regime only:** The model operates in the linearized regime and does not capture feature learning. The paper explicitly shows (Section 5.1, Appendix L) that feature-learning networks achieve much better scaling than predicted. The after-kernel continues evolving throughout training, suggesting a mechanistic theory of kernel evolution is needed.

2. **Random projection assumption:** The projection matrix A has iid entries, which is a simplification. In real neural networks, the relationship between finite-width and infinite-width NTK features is more structured.

3. **Power-law structure assumed:** The theory focuses on power-law eigenvalues and target coefficients. While this is empirically well-motivated, real data distributions may have more complex spectral structure (e.g., low-rank spikes observed in Figure 12b).

4. **MSE loss only:** The theoretical analysis uses mean squared error. While cross-entropy results are shown empirically, the theory does not directly analyze cross-entropy or other losses.

5. **No regularization analyzed explicitly:** The paper focuses on unregularized gradient descent/flow. Explicit regularization (weight decay, dropout) is not incorporated, though early stopping serves as implicit regularization.

6. **Single-layer model:** The random feature model has effectively one trainable layer. Multi-layer dynamics, depth effects, and residual connections are not captured.

7. **Task homogeneity:** The theory treats the task as a single spectral decomposition. It does not model discrete skill acquisition or multi-task settings (cf. Michaud et al., 2023; Arora & Goyal, 2023).

8. **Proportional limit:** The DMFT is exact in the proportional limit (N/M = nu, P/M = alpha, all going to infinity). Finite-size fluctuations decay as O(1/P + 1/N) but are not zero for practical sizes.

---

## 9. Relevance to Our Project: Topological Analysis of Training Trajectories

This paper is directly relevant to our project on topological persistence analysis of LLM training dynamics in several key ways:

### 9.1 Training Dynamics as Trajectories in Order-Parameter Space

The DMFT framework describes training as an evolution of order parameters {C_0(t,s), C_1(t,s), R_1(t,s), R_3(t,s)} in a high-dimensional space. The trajectory of these order parameters could be analyzed topologically:
- **Persistence diagrams** of the correlation functions C_i(t,t) (the loss curves) could capture topological features of the loss landscape traversed during training.
- **Phase transitions** between early-time (universal 1/N corrections) and late-time (task-dependent exponents) regimes could manifest as topological features (births/deaths in persistence diagrams).

### 9.2 Timescale Density and Multi-Scale Structure

The paper's timescale density rho_k(u) (Section F / Appendix F) provides a natural multi-scale decomposition of training dynamics. For finite N, P, each eigenmode's timescale spreads from a Dirac mass into a distribution. This multi-scale structure is exactly the kind of phenomenon that persistent homology is designed to detect:
- The spread of timescales at different model sizes could generate different topological signatures.
- The transition from concentrated (large N,P) to spread-out (small N,P) timescale densities could be captured by persistence barcodes.

### 9.3 Predicting Scaling from Topological Signatures

The key insight that **scaling exponents are determined by spectral structure (a, b)** suggests a connection to topological invariants:
- If topological features of the training trajectory (e.g., persistence diagram statistics) correlate with the spectral exponents a and b, then topological analysis could serve as an alternative method for predicting scaling behavior.
- The asymmetric compute-optimal strategy could potentially be inferred from topological features of early training, before the full scaling regime is reached.

### 9.4 Feature Learning and Kernel Evolution

The paper's most significant limitation -- the inability to capture feature learning -- is potentially addressable through topological methods:
- The after-kernel evolution (Figure 12) shows that the NTK spectrum flattens over training (b decreasing from 2.0 to 1.4). This spectral evolution changes the effective scaling exponents.
- Tracking topological features of the kernel trajectory (or weight-space trajectory) during feature learning could capture the dynamical improvement of scaling that feature learning provides.
- Persistent homology could detect when the kernel stabilizes (topological features stop changing), signaling convergence to a fixed spectral structure.

### 9.5 Overfitting Transition Detection

The gradual buildup of the train-test gap (Equation 18 / Section E) is a dynamical phenomenon that topological methods could detect:
- The onset of overfitting could correspond to the appearance of new topological features (loops, cavities) in the loss landscape or weight-space trajectory.
- The rate of topological feature creation/destruction might correlate with the overfitting rate 1/P predicted by the theory.

### 9.6 Practical Implications for Our Pipeline

1. **Synthetic test cases:** The power-law feature model provides an excellent controlled setting for testing our topological pipeline. We can generate training trajectories with known scaling exponents (a, b) and test whether our topological features can recover them.

2. **Spectral exponent extraction:** The paper's method of extracting a and b from NTK spectra (Figures 7a-b) could be complemented by topological features, potentially providing more robust estimates.

3. **Compute-optimal prediction:** If topological features of early training can predict a and b, we could potentially predict the compute-optimal strategy without running the full scaling experiment.

4. **Phase transition detection:** The transition from early-time (universal) to late-time (task-dependent) scaling could be a natural target for topological analysis -- detecting when the training dynamics leave the "universal" regime.

---

## Key Equations Reference

- **Test loss decomposition:** L(t, N) ~ L_0 + a_t * t^{-r_t} + a_N * N^{-r_N} (Eq. 1 in intro)
- **Power-law features:** (w*_k)^2 * lambda_k ~ k^{-a}, lambda_k ~ k^{-b} (Eq. 13)
- **Transfer function:** H_k(omega) = 1 / (i*omega + lambda_k * R_1(omega) * R_3(omega)) (Eq. 10)
- **Self-consistent equations:** R_1(omega), R_3(omega) (Eq. 11)
- **Bottleneck scalings:** L ~ t^{-(a-1)/b}, L ~ P^{-min{a-1,2b}}, L ~ N^{-min{a-1,2b}} (Eq. 14)
- **Compute-optimal:** t ~ C^{bm/(a-1+bm)}, N ~ C^{(a-1)/(a-1+bm)}, L* ~ C^{-(a-1)m/(a-1+bm)} (Eq. 17)
- **Train-test gap:** L(t) - L_hat(t) = DMFT integral expression (Eq. 18)
- **Bias of ensembled model:** B(t,N,P) = sum_k lambda_k (w*_k)^2 H_k(t,N,P)^2 (Eq. 19)
