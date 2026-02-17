# Code Repositories

This directory contains cloned repositories relevant to the research.

## Cloned Repositories

### 1. pythia (`code/pythia/`)
- **Source**: https://github.com/EleutherAI/pythia
- **Purpose**: Model definitions, training configurations, and evaluation scripts for the Pythia model suite
- **Usage**: Reference for model architecture details, checkpoint naming conventions, and training hyperparameters

### 2. giotto-tda (`code/giotto-tda/`)
- **Source**: https://github.com/giotto-ai/giotto-tda
- **Purpose**: Reference implementation for TDA computations (also installed as Python package)
- **Key modules**:
  - `gtda.homology` - Persistent homology computation
  - `gtda.diagrams` - Persistence diagram operations
  - `gtda.images` - Persistence images
  - `gtda.plotting` - Visualization utilities

### 3. ripser.py (`code/ripser-py/`)
- **Source**: https://github.com/scikit-tda/ripser.py
- **Purpose**: Fast Vietoris-Rips persistent homology (also installed as Python package)
- **Key advantage**: Lean C++ backend, faster than giotto-tda for pure VR computation on point clouds

## Recommended Additional Repositories

These repositories should be cloned for the full experimental pipeline:

| Repository | URL | Purpose |
|-----------|-----|---------|
| Neural-Persistence | https://github.com/BorgwardtLab/Neural-Persistence | Original neural persistence code (Rieck et al. ICLR 2019) |
| loss-landscape | https://github.com/tomgoldstein/loss-landscape | Filter-normalized landscape visualization (Li et al. NeurIPS 2018) |
| PyHessian | https://github.com/amirgholami/PyHessian | Hessian eigenvalue/trace computation for neural networks |
