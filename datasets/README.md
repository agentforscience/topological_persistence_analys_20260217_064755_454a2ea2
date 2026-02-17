# Datasets

This directory contains dataset resources for the topological persistence analysis research.

## Contents

### Pythia Checkpoint Configs (`pythia_checkpoints/`)

Model configuration metadata for EleutherAI/pythia-14m at 6 training checkpoints:
- `step_000000/` - Initial (random) weights
- `step_001000/` - Very early training
- `step_010000/` - Early training
- `step_050000/` - Mid training
- `step_100000/` - Late training
- `step_143000/` - Final checkpoint

Each checkpoint directory contains:
- `config.json` - Full model architecture configuration
- `summary.json` - Extracted metadata and parameter count estimate

### Download Scripts

- `download_pythia_checkpoints.py` - Downloads Pythia model configs from HuggingFace
- `generate_sample_training_data.py` - Generates synthetic training curves with known scaling properties

## Downloading Full Model Weights

The current checkpoint data contains only configs (not weights). To download full weights for TDA analysis:

```python
from transformers import AutoModelForCausalLM

# Download a specific checkpoint
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-14m",
    revision="step10000"
)
```

## Recommended Datasets for Full Experiments

| Dataset | Size | Source | Purpose |
|---------|------|--------|---------|
| Pythia-14m weights (6 ckpts) | ~300MB | HuggingFace | Method development |
| Pythia-70m weights (6 ckpts) | ~1.4GB | HuggingFace | Small-scale validation |
| Pythia-160m weights (6 ckpts) | ~3.2GB | HuggingFace | Medium-scale analysis |
| Pythia-410m weights (6 ckpts) | ~8.2GB | HuggingFace | Scaling comparison |
| Synthetic curves | ~1MB | Generated | Pipeline testing |

## .gitignore

Large model weight files should not be committed to git. The `.gitignore` in this directory excludes:
- `*.bin`, `*.safetensors` (model weights)
- `pythia_checkpoints/*/step_*/pytorch_model.bin`
