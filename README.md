# Topological Persistence Analysis of LLM Training Dynamics for Scaling Prediction

This project investigates whether persistent topological features of transformer weight spaces can predict optimal scaling decisions during LLM training.

## Key Findings

- **H₀ total persistence strongly anticorrelates with training loss** (ρ = -0.59 to -0.91, all p < 10⁻⁵) — the correlation strengthens with model size
- **Combined TDA + loss extrapolation improves final loss prediction by 10–29%** compared to loss-only baselines (when using ≥30% of training)
- **Phase transitions in topological features** occur at different training fractions for different model sizes, with larger models showing later diminishing returns (30-36% vs 8-16%)
- **H₀ total persistence scales with model size** as a power law (exponent ≈ 0.204, ρ = 0.90, p = 0.037)
- **Results validated on real Pythia-14m** (14M parameter transformer) showing the same qualitative trends

## How to Reproduce

```bash
# Create and activate environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install transformers datasets giotto-tda ripser persim
uv pip install numpy scipy pandas matplotlib

# Train models (5 GPT-style transformers, ~4 min on GPU)
python src/train.py

# Extract TDA features (~16s)
python src/tda_features.py

# Run analysis and generate plots (~5s)
python src/analysis.py

# Pythia-14m validation (~15s)
python src/pythia_validation.py

# Supplementary analysis and figures
python src/supplementary_analysis.py
```

## File Structure

```
├── REPORT.md                    # Full research report with results
├── README.md                    # This file
├── planning.md                  # Research plan
├── literature_review.md         # Literature review
├── resources.md                 # Resource catalog
├── src/
│   ├── models.py                # GPT model definitions (5 sizes)
│   ├── train.py                 # Training script with checkpoint saving
│   ├── tda_features.py          # TDA feature extraction pipeline
│   ├── analysis.py              # Correlation, prediction, visualization
│   ├── pythia_validation.py     # Pythia-14m validation
│   └── supplementary_analysis.py # Bootstrap CIs, scaling plots
├── results/
│   ├── training_summary.json    # Training results
│   ├── tda_features.json        # All TDA features per checkpoint
│   ├── analysis_results.json    # Correlations, scaling, predictions
│   ├── pythia_validation.json   # Pythia-14m TDA results
│   ├── checkpoints/             # Model checkpoints (51 per model)
│   └── plots/                   # All generated figures
├── papers/                      # 22 reference papers (PDFs + notes)
├── datasets/                    # Synthetic data + Pythia configs
└── code/                        # Reference implementations
```

## Environment

- Python 3.12, PyTorch 2.6.0+cu124
- GPU: 2x NVIDIA RTX 3090 (24GB each)
- Total runtime: ~5 minutes (training + TDA + analysis)
- Seed: 42 for reproducibility

See [REPORT.md](REPORT.md) for the full research report.
