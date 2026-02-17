# Resource Catalog: Topological Persistence Analysis of LLM Training Dynamics for Scaling Prediction

## Papers (22 PDFs in `papers/`)

### Category 1: TDA Applied to Loss Landscapes and Neural Network Structure

| # | File | Citation | Key Contribution | Deep Notes |
|---|------|----------|-----------------|------------|
| 1 | `rieck2019_neural_persistence.pdf` | Rieck et al., ICLR 2019 (arXiv:1812.09764) | Neural persistence: structural complexity measure for NNs based on H_0 persistent homology of weight graphs | `papers/notes_rieck2019.md` |
| 2 | `xie2024_evaluating_loss_landscapes_topology.pdf` | Xie et al., NeurIPS Workshop 2024 (arXiv:2411.09807) | Pipeline for loss landscape topology via merge trees and persistence diagrams; correlations with ML metrics | `papers/notes_xie2024.md` |
| 3 | `geniesse2024_visualizing_loss_topological_profiles.pdf` | Geniesse et al., 2024 (arXiv:2411.12136) | Higher-dimensional loss landscape visualization using topological landscape profiles from merge trees | `papers/notes_geniesse2024.md` |
| 4 | `ballarin2024_topological_loss_surfaces_betti.pdf` | Ballarin et al., Neural Networks 2024 (arXiv:2401.00358) | Theoretical bounds for Betti numbers of loss surfaces as functions of depth/width | `papers/notes_ballarin2024.md` (pending) |
| 5 | `ballester2024_tda_neural_network_survey.pdf` | Ballester et al., 2024 (arXiv:2312.05840) | Comprehensive survey: TDA for NN analysis in 4 categories (structure, decision regions, activations, training dynamics) | -- |
| 6 | `li2018_visualizing_loss_landscape.pdf` | Li et al., NeurIPS 2018 (arXiv:1712.09913) | Filter-normalized loss landscape visualization; skip connections smooth landscapes | -- |
| 7 | `horoi2021_geometry_topology_loss_landscapes.pdf` | Horoi et al., 2021 (arXiv:2102.00485) | Jump-and-retrain sampling + PHATE trajectory visualization + computational homology | -- |
| 8 | `draxler2018_loss_surface_mode_connectivity.pdf` | Draxler et al., ICML 2018 (arXiv:1803.00885) | Loss surface mode connectivity: essentially no barriers between optima on low-loss paths | -- |

### Category 2: Neural Scaling Laws and Training Dynamics

| # | File | Citation | Key Contribution | Deep Notes |
|---|------|----------|-----------------|------------|
| 9 | `kaplan2020_scaling_laws_neural_lm.pdf` | Kaplan et al. (OpenAI), 2020 (arXiv:2001.08361) | Foundational power-law scaling of LM performance with model size, data, and compute | -- |
| 10 | `hoffmann2022_chinchilla_training_compute_optimal.pdf` | Hoffmann et al. (DeepMind), 2022 (arXiv:2203.15556) | Compute-optimal training; model size and tokens should scale roughly equally | -- |
| 11 | `bordelon2024_dynamical_model_scaling_laws.pdf` | Bordelon et al., ICML 2024 (arXiv:2402.01092) | Dynamical mean field theory model explaining asymmetric scaling exponents for time vs model size | `papers/notes_bordelon2024.md` |
| 12 | `porian2024_scaling_laws_beyond_fixed_duration.pdf` | Porian et al., NeurIPS 2024 (arXiv:2405.18392) | Scaling laws with variable LR schedules and training durations; validated at 1B/8B scale | -- |
| 13 | `gadre2025_llms_on_line_loss_to_loss.pdf` | Gadre et al., 2025 (arXiv:2502.12120) | Loss-to-loss scaling across >6000 checkpoints; pretraining data dominates scaling | -- |
| 14 | `luo2025_multi_power_law_loss_curve.pdf` | Luo et al., 2025 (arXiv:2501.02751) | Multi-power law framework for loss curve prediction with LR decay effects | -- |
| 15 | `isik2024_scaling_laws_downstream_performance.pdf` | Isik et al., 2024 (arXiv:2402.04177) | Two-stage prediction: FLOPs → loss → downstream; 5-10% error for 7B/13B from ≤3B models | -- |
| 16 | `hai2024_scaling_law_lr_annealing.pdf` | Hai et al., 2024 | Scaling law with learning rate annealing effects | -- |
| 17 | `rende2024_scaling_laws_data_distribution.pdf` | Rende et al., 2024 | Scaling laws dependence on data distribution properties | -- |
| 18 | `huang2025_upscale_nn_scaling_law_survey.pdf` | Huang et al., 2025 | Comprehensive neural scaling law survey (UpScale) | -- |

### Category 3: TDA Tools and Methods

| # | File | Citation | Key Contribution |
|---|------|----------|-----------------|
| 19 | `tauzin2021_giotto_tda.pdf` | Tauzin et al., JMLR 2021 (arXiv:2004.02551) | giotto-tda: scikit-learn compatible Python TDA library with C++ backend |
| 20 | `horn2023_expressivity_persistent_homology_graphs.pdf` | Horn et al., 2023 (arXiv:2302.09826) | Theoretical expressivity results for PH in graph classification |
| 21 | `burella2021_giotto_ph.pdf` | Burella Pérez et al., 2021 | giotto-ph: high-performance persistent homology computation |
| 22 | `lee2023_universal_scaling_absorbing_phase.pdf` | Lee et al., 2023 | Universal scaling in absorbing phase transitions (mathematical tools) |

### Deep Reading Notes Available

| Notes File | Paper | Key Extractions |
|-----------|-------|----------------|
| `papers/notes_rieck2019.md` | Neural Persistence (Rieck et al.) | Algorithm details, per-layer filtration, early stopping method, O(n*alpha(n)) complexity |
| `papers/notes_xie2024.md` | Loss Landscape Topology (Xie et al.) | 6-stage pipeline, merge tree + PD metrics, inverse saddle-point/performance relationship, 4 hypotheses for LLM adaptation |
| `papers/notes_geniesse2024.md` | Topological Landscape Profiles (Geniesse et al.) | Hessian-based n-dim sampling, basin/valley representation, PINN and UNet experiments, phase transition detection |
| `papers/notes_bordelon2024.md` | Dynamical Scaling Laws (Bordelon et al.) | DMFT framework, power-law exponent derivations, asymmetric compute-optimal strategy, feature learning limitations |
| `papers/notes_ballarin2024.md` | Betti Number Bounds (Ballarin et al.) | *(Pending - being generated by background agent)* |

---

## Datasets (`datasets/`)

### 1. Pythia Model Checkpoints (Primary Dataset)

- **Source**: EleutherAI/pythia on HuggingFace
- **Location**: `datasets/pythia_checkpoints/pythia-14m/`
- **Status**: Config metadata downloaded for 6 checkpoints of pythia-14m
- **Checkpoints downloaded**: step0, step1000, step10000, step50000, step100000, step143000
- **Contents per checkpoint**: `config.json` (model architecture), `summary.json` (metadata + parameter count estimate)
- **Architecture**: GPTNeoXForCausalLM, hidden_size=128, 6 layers, 4 attention heads, ~14M parameters
- **Download script**: `datasets/download_pythia_checkpoints.py`
- **Full weight download**: Not yet performed (configs only). Run the download script with weight-download flags for full checkpoints.

**Available Pythia model sizes for scaling analysis:**

| Model | Parameters | Hidden | Layers | Heads | Checkpoints |
|-------|-----------|--------|--------|-------|-------------|
| pythia-14m | 14M | 128 | 6 | 4 | 154 |
| pythia-70m | 70M | 512 | 6 | 8 | 154 |
| pythia-160m | 160M | 768 | 12 | 12 | 154 |
| pythia-410m | 410M | 1024 | 24 | 16 | 154 |
| pythia-1b | 1B | 2048 | 16 | 8 | 154 |
| pythia-1.4b | 1.4B | 2048 | 24 | 16 | 154 |
| pythia-2.8b | 2.8B | 2560 | 32 | 32 | 154 |
| pythia-6.9b | 6.9B | 4096 | 32 | 32 | 154 |
| pythia-12b | 12B | 5120 | 36 | 40 | 154 |

All models trained on The Pile (deduplicated version available), same data order, same hyperparameters (except model size). This makes Pythia ideal for controlled scaling analysis.

### 2. Synthetic Training Data (for method development)

- **Status**: To be generated by `datasets/generate_sample_training_data.py`
- **Purpose**: Controlled test cases with known scaling exponents for validating TDA pipeline
- **Contents**: Synthetic loss curves with power-law scaling, configurable spectral parameters (a, b)

### 3. Recommended Additional Datasets (not yet downloaded)

| Dataset | Source | Use Case |
|---------|--------|----------|
| OLMo Checkpoints | AI2 (HuggingFace: allenai/OLMo-*) | Independent validation with different architecture |
| The Pile (deduped) | EleutherAI | Training data for reproducing Pythia experiments |
| Wikitext-103 | HuggingFace datasets | Lightweight LM evaluation |

---

## Code Repositories (`code/`)

### 1. Pythia (EleutherAI)

- **Location**: `code/pythia/`
- **Source**: https://github.com/EleutherAI/pythia
- **Purpose**: Model definitions, training configs, checkpoint management for the Pythia model suite
- **Key files**: Model cards, training scripts, evaluation configs

### 2. giotto-tda

- **Location**: `code/giotto-tda/`
- **Source**: https://github.com/giotto-ai/giotto-tda
- **Purpose**: Reference implementation for persistent homology, persistence diagrams, landscapes, and images
- **Key modules**: `VietorisRipsPersistence`, `CubicalPersistence`, persistence diagram vectorizations

### 3. ripser.py

- **Location**: `code/ripser-py/`
- **Source**: https://github.com/scikit-tda/ripser.py
- **Purpose**: Fast Vietoris-Rips persistent homology computation (Lean C++ implementation with Python bindings)
- **Key advantage**: Faster than giotto-tda for pure VR computation on point clouds

### 4. Recommended Additional Repositories (not yet cloned)

| Repository | URL | Purpose |
|-----------|-----|---------|
| Neural-Persistence | https://github.com/BorgwardtLab/Neural-Persistence | Original neural persistence implementation (Rieck et al.) |
| loss-landscape | https://github.com/tomgoldstein/loss-landscape | Filter-normalized loss landscape visualization (Li et al.) |
| PyHessian | https://github.com/amirgholami/PyHessian | Hessian computation for NN (top eigenvalues, trace, density) |
| Topology ToolKit (TTK) | https://topology-tool-kit.github.io/ | Merge tree computation, topological data analysis |

---

## Installed Python Packages

### Core ML/DL Stack
| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.10.0+cpu | Deep learning framework (CPU-only build) |
| transformers | 5.2.0 | HuggingFace model loading, tokenization |
| datasets | 4.5.0 | HuggingFace dataset loading |
| huggingface-hub | 1.4.1 | Model/dataset download from HuggingFace |
| safetensors | 0.7.0 | Fast tensor serialization |
| tokenizers | 0.22.2 | Fast tokenization |
| scikit-learn | 1.3.2 | ML utilities (preprocessing, metrics) |

### TDA Stack
| Package | Version | Purpose |
|---------|---------|---------|
| giotto-tda | 0.6.2 | Topological data analysis (persistence diagrams, landscapes, images) |
| giotto-ph | 0.2.4 | High-performance persistent homology (C++ backend) |
| ripser | 0.6.14 | Fast Vietoris-Rips persistent homology |
| persim | 0.3.8 | Persistence image/diagram comparison tools |
| pyflagser | 0.4.7 | Directed flag complex persistent homology |

### Scientific Computing
| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 1.26.4 | Numerical computing |
| scipy | 1.17.0 | Scientific computing, sparse matrices, optimization |
| pandas | 3.0.0 | Data manipulation and analysis |
| matplotlib | 3.10.8 | Plotting and visualization |
| plotly | 6.5.2 | Interactive visualization |
| networkx | 3.6.1 | Graph algorithms |
| igraph | 1.0.0 | High-performance graph algorithms |
| sympy | 1.14.0 | Symbolic mathematics |

### Utilities
| Package | Version | Purpose |
|---------|---------|---------|
| arxiv | 2.4.0 | arXiv API access for paper search |
| pypdf | 6.7.0 | PDF reading and chunking |
| requests | 2.32.5 | HTTP requests |
| httpx | 0.28.1 | Async HTTP client |
| tqdm | 4.67.3 | Progress bars |
| pyyaml | 6.0.3 | YAML parsing |
| cython | 3.2.4 | C extensions for Python |

---

## Environment Setup

- **Python**: 3.10+ (isolated virtual environment at `.venv/`)
- **Package manager**: uv (for fast dependency resolution)
- **PyTorch**: CPU-only build (GPU not available in this environment)
- **Build tools**: cmake 4.2.1, setuptools 70.2.0, cython 3.2.4

### Reproduction Steps
```bash
# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate

# Install core dependencies
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers datasets huggingface-hub safetensors
uv pip install giotto-tda ripser persim pyflagser
uv pip install numpy scipy pandas matplotlib plotly networkx igraph
uv pip install scikit-learn sympy
uv pip install arxiv pypdf requests httpx tqdm pyyaml

# Download Pythia checkpoint configs
python datasets/download_pythia_checkpoints.py
```

---

## Key Methodological Pipeline (from Literature)

### Neural Persistence Pipeline (per checkpoint)
1. Load model weights at checkpoint
2. For each linear layer, construct bipartite weight graph
3. Build descending filtration (sort edges by absolute weight, descending)
4. Compute H_0 persistent homology via union-find
5. Normalize by layer dimensions → neural persistence score
6. Track NP across checkpoints → temporal NP curve

### Loss Landscape Topology Pipeline (per checkpoint)
1. Compute top-n Hessian eigenvectors (via PyHessian)
2. Sample loss on grid along eigenvector directions (resolution r=21-41)
3. Construct k-NN graph (k=4n) on sampled points
4. Compute merge tree (via TTK or custom implementation)
5. Extract: saddle count, average persistence, total persistence, basin depths
6. Track metrics across checkpoints → topological time series

### Scaling Prediction Pipeline
1. Compute TDA metrics at each checkpoint for multiple model sizes
2. Fit temporal models to TDA metric time series
3. Extract features: rates of change, phase transitions, steady-state values
4. Train predictor: TDA features → scaling exponents (a, b)
5. Validate: predict loss at unseen model sizes or training steps

---

## Research Gaps Addressed

1. **No prior TDA on transformer/LLM training dynamics** (Ballester et al. 2024 survey)
2. **No connection between topological features and scaling laws** (novel contribution)
3. **Temporal topological analysis** largely unexplored (only Rieck 2019, Muller 2024)
4. **Higher-dimensional homology** (H_1, H_2) unexplored for loss landscapes
5. **Scalability** of PH to LLM-scale parameters (per-layer approach makes it tractable)
