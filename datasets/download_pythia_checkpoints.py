#!/usr/bin/env python3
"""
Download Pythia-14m model configs and summaries at key training checkpoints.

This script downloads a SMALL Pythia model (pythia-14m) at several key
checkpoints to demonstrate the approach of analyzing how model structure
evolves during training. To save disk space, we only download the model
config and generate a summary of the model architecture -- NOT the full
model weights.

Checkpoints downloaded:
  - step0:      Initial (random) weights
  - step1000:   Very early training
  - step10000:  Early training
  - step50000:  Mid training
  - step100000: Late training
  - step143000: Final checkpoint (end of training)
"""

import json
import os
import sys

from huggingface_hub import hf_hub_download, HfApi
from transformers import AutoConfig


# Configuration
MODEL_NAME = "EleutherAI/pythia-14m"
CHECKPOINTS = [0, 1000, 10000, 50000, 100000, 143000]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "pythia_checkpoints", "pythia-14m")


def get_checkpoint_revision(step: int) -> str:
    """Return the HuggingFace revision string for a given training step."""
    return f"step{step}"


def download_checkpoint_config(step: int) -> dict:
    """
    Download the model config for a given checkpoint step.

    Returns a dict containing the config and metadata summary.
    """
    revision = get_checkpoint_revision(step)
    step_dir = os.path.join(OUTPUT_DIR, f"step_{step:06d}")
    os.makedirs(step_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading config for {MODEL_NAME} at {revision}")
    print(f"{'='*60}")

    # Download the model config
    try:
        config = AutoConfig.from_pretrained(MODEL_NAME, revision=revision)
        config_dict = config.to_dict()

        # Save the config
        config_path = os.path.join(step_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        print(f"  Config saved to: {config_path}")

    except Exception as e:
        print(f"  Warning: Could not download config for {revision}: {e}")
        config_dict = {}

    # Build a summary of the checkpoint
    summary = {
        "model_name": MODEL_NAME,
        "checkpoint_step": step,
        "revision": revision,
        "architecture": config_dict.get("architectures", ["Unknown"]),
        "model_type": config_dict.get("model_type", "Unknown"),
        "hidden_size": config_dict.get("hidden_size", "Unknown"),
        "num_hidden_layers": config_dict.get("num_hidden_layers", "Unknown"),
        "num_attention_heads": config_dict.get("num_attention_heads", "Unknown"),
        "intermediate_size": config_dict.get("intermediate_size", "Unknown"),
        "vocab_size": config_dict.get("vocab_size", "Unknown"),
        "max_position_embeddings": config_dict.get("max_position_embeddings", "Unknown"),
    }

    # Estimate parameter count from config
    if all(isinstance(summary[k], int) for k in [
        "hidden_size", "num_hidden_layers", "num_attention_heads",
        "vocab_size"
    ]):
        h = summary["hidden_size"]
        L = summary["num_hidden_layers"]
        V = summary["vocab_size"]
        # Rough parameter estimate: embedding + L * (attention + FFN) + output
        intermediate = config_dict.get("intermediate_size", 4 * h)
        param_estimate = (
            V * h +                          # token embeddings
            L * (4 * h * h + 4 * h) +        # attention layers (Q, K, V, O)
            L * (h * intermediate + intermediate * h + intermediate + h) +  # FFN
            h + V * h                         # layer norm + output projection
        )
        summary["estimated_parameters"] = param_estimate
        summary["estimated_parameters_millions"] = round(param_estimate / 1e6, 2)

    # Save the summary
    summary_path = os.path.join(step_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to: {summary_path}")

    # Print key info
    print(f"  Architecture: {summary['architecture']}")
    print(f"  Hidden size: {summary['hidden_size']}")
    print(f"  Layers: {summary['num_hidden_layers']}")
    print(f"  Attention heads: {summary['num_attention_heads']}")
    if "estimated_parameters_millions" in summary:
        print(f"  Estimated parameters: {summary['estimated_parameters_millions']}M")

    return summary


def main():
    """Download configs and summaries for all specified checkpoints."""
    print(f"Pythia Checkpoint Config Downloader")
    print(f"Model: {MODEL_NAME}")
    print(f"Checkpoints: {CHECKPOINTS}")
    print(f"Output directory: {OUTPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_summaries = []
    for step in CHECKPOINTS:
        try:
            summary = download_checkpoint_config(step)
            all_summaries.append(summary)
        except Exception as e:
            print(f"\nERROR downloading step {step}: {e}")
            all_summaries.append({
                "model_name": MODEL_NAME,
                "checkpoint_step": step,
                "error": str(e),
            })

    # Save a combined manifest
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    manifest = {
        "model_name": MODEL_NAME,
        "num_checkpoints": len(CHECKPOINTS),
        "checkpoints": CHECKPOINTS,
        "summaries": all_summaries,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"Manifest saved to: {manifest_path}")
    print(f"Total checkpoints processed: {len(all_summaries)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
