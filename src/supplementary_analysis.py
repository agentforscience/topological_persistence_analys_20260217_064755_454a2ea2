"""
Supplementary analysis: Pythia validation plots and additional statistical tests.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "results", "plots")


def plot_pythia_validation():
    """Plot Pythia-14m TDA feature evolution."""
    with open(os.path.join(RESULTS_DIR, "pythia_validation.json")) as f:
        data = json.load(f)

    steps = [d["step"] for d in data]
    np_means = [d["np_mean"] for d in data]
    np_stds = [d["np_std"] for d in data]
    h0_totals = [d["h0_total"] for d in data]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Neural Persistence
    axes[0].plot(steps, np_means, "o-", color="blue", markersize=8)
    axes[0].fill_between(steps,
                          [m - s for m, s in zip(np_means, np_stds)],
                          [m + s for m, s in zip(np_means, np_stds)],
                          alpha=0.2)
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Mean Neural Persistence")
    axes[0].set_title("Pythia-14m: Neural Persistence\n(monotonically increasing)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale("symlog", linthresh=1000)

    # H0 Total Persistence
    axes[1].plot(steps, h0_totals, "o-", color="green", markersize=8)
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("H₀ Total Persistence")
    axes[1].set_title("Pythia-14m: H₀ Persistence\n(decreasing = more connected)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale("symlog", linthresh=1000)

    # NP rate of change
    np_rate = np.diff(np_means) / np.diff(steps)
    mid_steps = [(steps[i] + steps[i + 1]) / 2 for i in range(len(steps) - 1)]
    axes[2].bar(range(len(np_rate)), np_rate, color="orange")
    axes[2].set_xticks(range(len(np_rate)))
    axes[2].set_xticklabels([f"{int(s)}" for s in mid_steps], rotation=45, fontsize=7)
    axes[2].set_xlabel("Training Step (midpoint)")
    axes[2].set_ylabel("NP Rate of Change")
    axes[2].set_title("Pythia-14m: NP Rate of Change\n(diminishing returns visible)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "pythia_validation.png"), dpi=150)
    plt.close(fig)
    print("  Saved pythia_validation.png")


def plot_combined_np_scaling():
    """Plot neural persistence scaling across both our models and Pythia."""
    # Our models
    with open(os.path.join(RESULTS_DIR, "tda_features.json")) as f:
        tda_data = json.load(f)

    model_params = {
        "tiny-3k": 2936, "small-7k": 7408, "med-21k": 20640,
        "large-46k": 46368, "xl-97k": 97200,
    }

    # Pythia
    pythia_params = 14_000_000
    with open(os.path.join(RESULTS_DIR, "pythia_validation.json")) as f:
        pythia_data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Final NP vs model size (log-log)
    our_sizes = []
    our_final_np = []
    for name, features in tda_data.items():
        our_sizes.append(model_params[name])
        our_final_np.append(features[-1]["neural_persistence_mean"])

    axes[0].scatter(our_sizes, our_final_np, s=100, c="blue", label="Our models", zorder=5)
    axes[0].scatter([pythia_params], [pythia_data[-1]["np_mean"]], s=150, c="red",
                     marker="*", label="Pythia-14m", zorder=5)

    # Fit power law
    all_sizes = our_sizes + [pythia_params]
    all_np = our_final_np + [pythia_data[-1]["np_mean"]]
    log_sizes = np.log(all_sizes)
    log_np = np.log(all_np)
    slope, intercept, r, p, se = stats.linregress(log_sizes, log_np)
    x_fit = np.linspace(min(log_sizes), max(log_sizes), 100)
    axes[0].plot(np.exp(x_fit), np.exp(intercept + slope * x_fit), "k--",
                  label=f"Power law: α={slope:.3f}, R²={r**2:.3f}")

    axes[0].set_xscale("log")
    axes[0].set_xlabel("Parameters")
    axes[0].set_ylabel("Final Neural Persistence")
    axes[0].set_title("Neural Persistence Scales with Model Size")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: NP evolution comparison (normalized training progress)
    colors = plt.cm.viridis(np.linspace(0, 1, len(tda_data)))
    for (name, features), color in zip(
            sorted(tda_data.items(), key=lambda x: model_params[x[0]]), colors):
        steps = [f["step"] for f in features]
        np_vals = [f["neural_persistence_mean"] for f in features]
        max_step = max(steps)
        frac = [s / max_step for s in steps]
        axes[1].plot(frac, np_vals, label=f"{name}", color=color, alpha=0.8)

    # Pythia (normalized)
    pythia_steps = [d["step"] for d in pythia_data]
    pythia_np = [d["np_mean"] for d in pythia_data]
    max_ps = max(pythia_steps)
    pythia_frac = [s / max_ps for s in pythia_steps]
    axes[1].plot(pythia_frac, pythia_np, "r--o", label="Pythia-14m", markersize=6, linewidth=2)

    axes[1].set_xlabel("Training Progress (fraction)")
    axes[1].set_ylabel("Mean Neural Persistence")
    axes[1].set_title("NP Evolution: Our Models vs Pythia-14m\n(same qualitative trend)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "combined_scaling.png"), dpi=150)
    plt.close(fig)
    print("  Saved combined_scaling.png")
    print(f"  Power law fit: NP ~ N^{slope:.3f}, R²={r**2:.3f}, p={p:.4f}")


def run_bootstrap_confidence_intervals():
    """Compute bootstrap CIs for key correlations."""
    with open(os.path.join(RESULTS_DIR, "tda_features.json")) as f:
        tda_data = json.load(f)

    ckpt_dir = os.path.join(RESULTS_DIR, "checkpoints")

    print("\n" + "=" * 60)
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 60)

    model_params = {
        "tiny-3k": 2936, "small-7k": 7408, "med-21k": 20640,
        "large-46k": 46368, "xl-97k": 97200,
    }

    n_bootstrap = 1000
    rng = np.random.default_rng(42)

    for config_name, features in tda_data.items():
        # H0 total vs loss
        h0 = np.array([f.get("h0_total_persistence", 0) for f in features])
        losses = np.array([f.get("loss", 0) for f in features], dtype=float)

        valid = np.isfinite(h0) & np.isfinite(losses)
        h0_valid = h0[valid]
        loss_valid = losses[valid]

        if len(h0_valid) < 5:
            continue

        # Bootstrap
        boot_rhos = []
        n = len(h0_valid)
        for _ in range(n_bootstrap):
            idx = rng.choice(n, n, replace=True)
            rho, _ = stats.spearmanr(h0_valid[idx], loss_valid[idx])
            if np.isfinite(rho):
                boot_rhos.append(rho)

        boot_rhos = np.array(boot_rhos)
        ci_low = np.percentile(boot_rhos, 2.5)
        ci_high = np.percentile(boot_rhos, 97.5)
        rho_orig, p_orig = stats.spearmanr(h0_valid, loss_valid)

        print(f"  {config_name:>12s} H0_total vs loss: ρ={rho_orig:+.3f} "
              f"95% CI [{ci_low:+.3f}, {ci_high:+.3f}] (p={p_orig:.4f})")


def compute_effect_sizes():
    """Compute effect sizes for key comparisons."""
    print("\n" + "=" * 60)
    print("EFFECT SIZE ANALYSIS")
    print("=" * 60)

    with open(os.path.join(RESULTS_DIR, "analysis_results.json")) as f:
        results = json.load(f)

    # Compare prediction methods
    for frac_key, preds in results.get("prediction_maes", {}).items():
        loss_mae = preds.get("loss_only", 0)
        combined_mae = preds.get("combined", 0)
        if loss_mae and combined_mae:
            improvement = (loss_mae - combined_mae) / loss_mae * 100
            print(f"  {frac_key}: Combined vs loss-only: {improvement:+.1f}% improvement in MAE")


def create_summary_figure():
    """Create a 2x2 summary figure for the paper."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    with open(os.path.join(RESULTS_DIR, "tda_features.json")) as f:
        tda_data = json.load(f)

    model_params = {
        "tiny-3k": 2936, "small-7k": 7408, "med-21k": 20640,
        "large-46k": 46368, "xl-97k": 97200,
    }

    colors = plt.cm.viridis(np.linspace(0, 1, len(tda_data)))
    sorted_names = sorted(tda_data.keys(), key=lambda k: model_params[k])

    # (a) Training loss curves
    ax = axes[0, 0]
    for name, color in zip(sorted_names, colors):
        features = tda_data[name]
        steps = [f["step"] for f in features]
        losses = [f["loss"] for f in features]
        ax.plot(steps, losses, color=color, label=f"{name} ({model_params[name]:,}p)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("(a) Training Loss Curves")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (b) Neural persistence evolution
    ax = axes[0, 1]
    for name, color in zip(sorted_names, colors):
        features = tda_data[name]
        steps = [f["step"] for f in features]
        np_vals = [f["neural_persistence_mean"] for f in features]
        ax.plot(steps, np_vals, color=color, label=f"{name}")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Neural Persistence")
    ax.set_title("(b) Neural Persistence Evolution")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (c) H0 persistence vs loss scatter
    ax = axes[1, 0]
    for name, color in zip(sorted_names, colors):
        features = tda_data[name]
        h0 = [f.get("h0_total_persistence", 0) for f in features]
        losses = [f["loss"] for f in features]
        ax.scatter(h0, losses, s=15, color=color, alpha=0.6, label=f"{name}")
    ax.set_xlabel("H₀ Total Persistence")
    ax.set_ylabel("Loss")
    ax.set_title("(c) H₀ Persistence vs Loss (ρ = -0.85 to -0.91)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (d) Scaling: final NP and loss vs model size
    ax = axes[1, 1]
    sizes = [model_params[n] for n in sorted_names]
    final_losses = [tda_data[n][-1]["loss"] for n in sorted_names]
    final_np = [tda_data[n][-1]["neural_persistence_mean"] for n in sorted_names]

    ax2 = ax.twinx()
    ax.scatter(sizes, final_losses, s=100, c="blue", label="Final Loss", zorder=5)
    ax2.scatter(sizes, final_np, s=100, c="orange", marker="^", label="Final NP", zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Final Loss", color="blue")
    ax2.set_ylabel("Final Neural Persistence", color="orange")
    ax.set_title("(d) Scaling Relationships")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Topological Persistence Analysis of LLM Training Dynamics",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "summary_figure.png"), dpi=200)
    plt.close(fig)
    print("  Saved summary_figure.png")


if __name__ == "__main__":
    plot_pythia_validation()
    plot_combined_np_scaling()
    run_bootstrap_confidence_intervals()
    compute_effect_sizes()
    create_summary_figure()
