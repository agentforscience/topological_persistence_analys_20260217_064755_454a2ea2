#!/usr/bin/env python3
"""
Generate synthetic training loss curve data for testing topological analysis.

This script creates realistic synthetic training loss curves for multiple
"model sizes" that simulate neural scaling law behavior. The generated data
can be used to test and develop topological persistence analysis methods
before applying them to real training data.

The loss curves follow a power-law scaling relationship:
    L(N, t) = a * N^(-alpha) * f(t) + L_inf

where:
    - N is the model size (number of parameters)
    - t is the training step
    - alpha is the scaling exponent (~0.076 for language models)
    - f(t) captures the training dynamics (exponential decay + noise)
    - L_inf is the irreducible loss

The script also generates:
    - Learning rate schedules (cosine with warmup)
    - Gradient norm trajectories
    - Loss landscape curvature estimates (synthetic)
"""

import json
import os
import sys

import numpy as np


# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "synthetic_training_curves.npz")
SUMMARY_FILE = os.path.join(BASE_DIR, "synthetic_training_curves_summary.json")

# Random seed for reproducibility
SEED = 42

# Model sizes (in millions of parameters)
MODEL_SIZES_M = [14, 70, 160, 410, 1000, 2800]

# Training configuration
NUM_STEPS = 143000
LOG_INTERVAL = 100  # Log every N steps
NUM_LOG_POINTS = NUM_STEPS // LOG_INTERVAL

# Scaling law parameters (inspired by Chinchilla / Kaplan et al.)
SCALING_EXPONENT_ALPHA = 0.076  # Loss scaling with model size
IRREDUCIBLE_LOSS = 1.69         # Minimum achievable loss (entropy of natural text)
BASE_LOSS_COEFFICIENT = 10.0    # Scaling coefficient


def cosine_schedule_with_warmup(num_steps, warmup_steps=2000, min_lr_ratio=0.1):
    """Generate a cosine learning rate schedule with linear warmup."""
    schedule = np.zeros(num_steps)
    # Warmup phase
    for t in range(min(warmup_steps, num_steps)):
        schedule[t] = t / warmup_steps
    # Cosine decay phase
    for t in range(warmup_steps, num_steps):
        progress = (t - warmup_steps) / max(1, num_steps - warmup_steps)
        schedule[t] = min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
            1.0 + np.cos(np.pi * progress)
        )
    return schedule


def generate_loss_curve(
    model_size_m,
    num_steps,
    log_interval,
    rng,
    alpha=SCALING_EXPONENT_ALPHA,
    L_inf=IRREDUCIBLE_LOSS,
    base_coeff=BASE_LOSS_COEFFICIENT,
):
    """
    Generate a realistic synthetic training loss curve for a given model size.

    The curve follows an approximate power-law decay with:
    - Initial high loss that decays following training dynamics
    - Noise that decreases as training progresses (loss stabilizes)
    - Occasional "spikes" simulating gradient instabilities
    - Scaling behavior: larger models achieve lower loss
    """
    steps = np.arange(0, num_steps, log_interval, dtype=np.float64)
    n_points = len(steps)

    # Base loss from scaling law: L = a * N^(-alpha)
    size_factor = base_coeff * (model_size_m ** (-alpha))

    # Training dynamics: exponential-like decay
    # Characteristic decay scale depends on model size
    decay_scale = 0.3 + 0.1 * np.log10(model_size_m)
    t_normalized = steps / num_steps

    # Multi-phase training dynamics
    # Phase 1: Rapid initial loss decrease
    rapid_decay = 3.0 * np.exp(-8.0 * t_normalized)
    # Phase 2: Slower power-law like improvement
    slow_decay = size_factor * (1.0 + t_normalized) ** (-decay_scale)
    # Phase 3: Very slow convergence near the end
    convergence = 0.1 * size_factor * np.exp(-2.0 * t_normalized)

    # Combine phases
    base_loss = L_inf + rapid_decay + slow_decay + convergence

    # Add realistic noise
    # Noise amplitude decreases with model size and training progress
    noise_base = 0.02 / np.sqrt(model_size_m / 14.0)
    noise_decay = 1.0 / (1.0 + 5.0 * t_normalized)
    noise = noise_base * noise_decay * rng.standard_normal(n_points)

    # Add occasional loss spikes (simulating gradient instabilities)
    # More common in larger models and early training
    num_spikes = rng.poisson(3)
    for _ in range(num_spikes):
        spike_location = rng.integers(0, n_points)
        spike_magnitude = rng.exponential(0.1) * (1.0 - 0.5 * t_normalized[spike_location])
        spike_width = rng.integers(1, 5)
        for j in range(max(0, spike_location - spike_width),
                       min(n_points, spike_location + spike_width)):
            distance = abs(j - spike_location)
            base_loss[j] += spike_magnitude * np.exp(-distance)

    loss = base_loss + noise

    return steps, loss


def generate_gradient_norms(steps, loss_curve, model_size_m, rng):
    """Generate synthetic gradient norm trajectory correlated with loss."""
    n_points = len(steps)
    t_normalized = steps / steps[-1]

    # Gradient norms roughly correlate with loss rate of change
    loss_diff = np.abs(np.gradient(loss_curve))
    base_grad_norm = loss_diff * 100.0

    # Add model size effect: larger models tend to have larger gradient norms
    size_effect = np.sqrt(model_size_m / 14.0)

    # Add noise
    noise = rng.lognormal(0, 0.3, n_points)

    grad_norms = base_grad_norm * size_effect * noise
    # Clip to reasonable range
    grad_norms = np.clip(grad_norms, 1e-4, 100.0)

    return grad_norms


def generate_loss_curvature(steps, loss_curve, rng):
    """
    Generate synthetic loss landscape curvature estimates.

    In practice, this would be computed from the Hessian of the loss.
    Here we simulate it as a metric that initially increases (loss landscape
    becomes sharper) then decreases (model finds flatter minima).
    """
    n_points = len(steps)
    t_normalized = steps / steps[-1]

    # Curvature typically increases then decreases during training
    # (sharp -> broad minima transition)
    curvature = (
        0.5 * np.exp(-0.5 * t_normalized) +
        2.0 * t_normalized * np.exp(-3.0 * t_normalized) +
        0.1
    )

    # Add noise
    noise = 1.0 + 0.1 * rng.standard_normal(n_points)
    curvature = curvature * noise

    return np.clip(curvature, 0.01, 10.0)


def main():
    """Generate all synthetic training data and save to disk."""
    print("Synthetic Training Data Generator")
    print(f"Model sizes: {MODEL_SIZES_M} (millions of parameters)")
    print(f"Training steps: {NUM_STEPS}")
    print(f"Log interval: {LOG_INTERVAL}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    rng = np.random.default_rng(SEED)

    all_data = {}
    summary_data = {
        "description": "Synthetic training loss curves for topological persistence analysis",
        "model_sizes_millions": MODEL_SIZES_M,
        "num_steps": NUM_STEPS,
        "log_interval": LOG_INTERVAL,
        "scaling_exponent": SCALING_EXPONENT_ALPHA,
        "irreducible_loss": IRREDUCIBLE_LOSS,
        "seed": SEED,
        "models": [],
    }

    for i, size_m in enumerate(MODEL_SIZES_M):
        print(f"Generating data for model size: {size_m}M parameters...")

        # Generate loss curve
        steps, loss = generate_loss_curve(size_m, NUM_STEPS, LOG_INTERVAL, rng)

        # Generate learning rate schedule
        lr_schedule = cosine_schedule_with_warmup(len(steps))

        # Generate gradient norms
        grad_norms = generate_gradient_norms(steps, loss, size_m, rng)

        # Generate curvature estimates
        curvature = generate_loss_curvature(steps, loss, rng)

        # Store arrays with descriptive keys
        prefix = f"model_{size_m}m"
        all_data[f"{prefix}_steps"] = steps
        all_data[f"{prefix}_loss"] = loss
        all_data[f"{prefix}_lr_schedule"] = lr_schedule
        all_data[f"{prefix}_grad_norms"] = grad_norms
        all_data[f"{prefix}_curvature"] = curvature

        # Summary statistics
        model_summary = {
            "size_millions": size_m,
            "array_prefix": prefix,
            "num_data_points": len(steps),
            "initial_loss": float(loss[0]),
            "final_loss": float(loss[-1]),
            "min_loss": float(np.min(loss)),
            "max_loss": float(np.max(loss)),
            "mean_loss": float(np.mean(loss)),
            "loss_reduction_percent": float(
                100.0 * (loss[0] - loss[-1]) / loss[0]
            ),
            "mean_grad_norm": float(np.mean(grad_norms)),
            "mean_curvature": float(np.mean(curvature)),
        }
        summary_data["models"].append(model_summary)

        print(f"  Initial loss: {loss[0]:.4f}")
        print(f"  Final loss:   {loss[-1]:.4f}")
        print(f"  Min loss:     {np.min(loss):.4f}")
        print(f"  Loss reduction: {model_summary['loss_reduction_percent']:.1f}%")
        print()

    # Also store the model sizes array for easy access
    all_data["model_sizes_millions"] = np.array(MODEL_SIZES_M)
    all_data["log_interval"] = np.array([LOG_INTERVAL])

    # Save the data
    print(f"Saving data to {OUTPUT_FILE}...")
    np.savez_compressed(OUTPUT_FILE, **all_data)
    file_size = os.path.getsize(OUTPUT_FILE)
    print(f"  File size: {file_size / 1024:.1f} KB")

    # Save the summary
    print(f"Saving summary to {SUMMARY_FILE}...")
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary_data, f, indent=2)

    # Print scaling law verification
    print("\nScaling Law Verification (final loss vs model size):")
    print(f"{'Model Size':>12s} {'Final Loss':>12s} {'Expected Scaling':>18s}")
    print("-" * 45)
    for model_info in summary_data["models"]:
        size = model_info["size_millions"]
        final_loss = model_info["final_loss"]
        expected = IRREDUCIBLE_LOSS + BASE_LOSS_COEFFICIENT * (size ** (-SCALING_EXPONENT_ALPHA))
        print(f"{size:>10d}M {final_loss:>12.4f} {expected:>18.4f}")

    print("\nDone! Data ready for topological persistence analysis.")


if __name__ == "__main__":
    main()
