"""
Validate topological analysis on real Pythia model checkpoints.

Downloads select checkpoints from Pythia-14m and computes neural persistence
to validate that the TDA pipeline works on real transformer weights and
that trends observed in small models generalize.
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from tda_features import compute_neural_persistence_layer, compute_weight_space_ph

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def download_and_analyze_pythia():
    """Download Pythia-14m checkpoints and compute TDA features."""
    from transformers import AutoModelForCausalLM

    model_name = "EleutherAI/pythia-14m"
    # Select a sparse set of checkpoints spanning training
    steps = [0, 1000, 5000, 10000, 50000, 100000, 143000]

    results = []
    print(f"Analyzing {model_name} across {len(steps)} checkpoints...")

    for step in steps:
        print(f"\n  Step {step}...")
        start = time.time()

        try:
            revision = f"step{step}"
            model = AutoModelForCausalLM.from_pretrained(
                model_name, revision=revision,
                torch_dtype=torch.float32
            )
        except Exception as e:
            print(f"    Failed to load step {step}: {e}")
            continue

        state_dict = model.state_dict()
        elapsed_load = time.time() - start
        print(f"    Loaded in {elapsed_load:.1f}s")

        # Compute neural persistence per layer
        layer_np = {}
        all_np_values = []

        for name, param in state_dict.items():
            if "weight" not in name or param.ndim != 2:
                continue
            if "layernorm" in name.lower() or "layer_norm" in name.lower():
                continue
            if "embed" in name.lower():
                continue

            w = param.cpu().numpy()
            # For large layers, subsample rows
            if w.shape[0] > 512:
                idx = np.random.choice(w.shape[0], 512, replace=False)
                w = w[idx]
            if w.shape[1] > 512:
                idx = np.random.choice(w.shape[1], 512, replace=False)
                w = w[:, idx]

            np_metrics = compute_neural_persistence_layer(w)
            layer_np[name] = np_metrics["neural_persistence"]
            all_np_values.append(np_metrics["neural_persistence"])

        # Compute weight-space PH on a sample
        weight_vecs = []
        for name, param in state_dict.items():
            if "weight" not in name or param.ndim != 2:
                continue
            if "layernorm" in name.lower() or "layer_norm" in name.lower() or "embed" in name.lower():
                continue
            w = param.cpu().numpy()
            # Take first 50 rows as point cloud
            for row in w[:50]:
                weight_vecs.append(row[:64].tolist())  # Truncate to 64 dims

        if weight_vecs:
            ph_features = compute_weight_space_ph(weight_vecs, max_points=200, max_dim=1)
        else:
            ph_features = {"h0_total_persistence": 0, "h1_total_persistence": 0}

        result = {
            "step": step,
            "np_mean": float(np.mean(all_np_values)) if all_np_values else 0,
            "np_std": float(np.std(all_np_values)) if all_np_values else 0,
            "np_min": float(np.min(all_np_values)) if all_np_values else 0,
            "np_max": float(np.max(all_np_values)) if all_np_values else 0,
            "n_layers_analyzed": len(all_np_values),
            "h0_total": ph_features.get("h0_total_persistence", 0),
            "h1_total": ph_features.get("h1_total_persistence", 0),
        }
        results.append(result)

        elapsed_total = time.time() - start
        print(f"    NP mean={result['np_mean']:.4f}, H0={result['h0_total']:.4f}, "
              f"H1={result['h1_total']:.4f} (total {elapsed_total:.1f}s)")

        # Clean up to save memory
        del model, state_dict
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save results
    output_path = os.path.join(RESULTS_DIR, "pythia_validation.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    download_and_analyze_pythia()
