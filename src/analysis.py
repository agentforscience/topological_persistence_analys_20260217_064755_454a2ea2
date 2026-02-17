"""
Analysis and visualization of topological features from LLM training.

Performs:
1. Correlation analysis between TDA features and training metrics
2. Cross-scale comparison of topological feature trajectories
3. Scaling prediction using TDA features vs baselines
4. Visualization of key results
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "results", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Model parameter counts (approximate)
MODEL_PARAMS = {
    "tiny-3k": 2936,
    "small-7k": 7408,
    "med-21k": 20640,
    "large-46k": 46368,
    "xl-97k": 97200,
}


def load_data():
    """Load training logs and TDA features."""
    # Load TDA features
    tda_path = os.path.join(RESULTS_DIR, "tda_features.json")
    with open(tda_path) as f:
        tda_data = json.load(f)

    # Load training logs
    training_logs = {}
    ckpt_dir = os.path.join(RESULTS_DIR, "checkpoints")
    for config_name in tda_data:
        log_path = os.path.join(ckpt_dir, config_name, "training_log.json")
        if os.path.exists(log_path):
            with open(log_path) as f:
                training_logs[config_name] = json.load(f)

    return tda_data, training_logs


def extract_time_series(tda_data, training_logs):
    """Extract aligned time series of TDA features and training metrics."""
    results = {}

    for config_name in tda_data:
        features = tda_data[config_name]
        logs = training_logs.get(config_name, [])

        # Build step->metrics mapping from training log
        log_by_step = {l["step"]: l for l in logs}

        steps = []
        losses = []
        grad_norms = []
        np_means = []
        np_stds = []
        h0_total = []
        h1_total = []
        h0_num = []
        h1_num = []
        weight_norms = []

        for f in features:
            step = f["step"]
            steps.append(step)
            losses.append(f.get("loss", None))
            np_means.append(f.get("neural_persistence_mean", 0))
            np_stds.append(f.get("neural_persistence_std", 0))
            h0_total.append(f.get("h0_total_persistence", 0))
            h1_total.append(f.get("h1_total_persistence", 0))
            h0_num.append(f.get("h0_num_features", 0))
            h1_num.append(f.get("h1_num_features", 0))
            weight_norms.append(f.get("weight_l2_norm", 0))

            # Get gradient norm from training log
            if step in log_by_step:
                grad_norms.append(log_by_step[step].get("grad_norm", 0))
            else:
                grad_norms.append(0)

        results[config_name] = {
            "steps": np.array(steps),
            "loss": np.array(losses, dtype=float),
            "grad_norm": np.array(grad_norms),
            "np_mean": np.array(np_means),
            "np_std": np.array(np_stds),
            "h0_total": np.array(h0_total),
            "h1_total": np.array(h1_total),
            "h0_num": np.array(h0_num),
            "h1_num": np.array(h1_num),
            "weight_norm": np.array(weight_norms),
            "n_params": MODEL_PARAMS.get(config_name, 0),
        }

    return results


def compute_correlations(ts_data):
    """Compute Spearman correlations between TDA features and training metrics."""
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    tda_features = ["np_mean", "np_std", "h0_total", "h1_total", "h0_num", "h1_num"]
    training_metrics = ["loss", "grad_norm", "weight_norm"]

    all_correlations = {}

    for config_name, ts in ts_data.items():
        print(f"\n--- {config_name} (params={ts['n_params']:,}) ---")
        corrs = {}

        for feat in tda_features:
            for metric in training_metrics:
                x = ts[feat]
                y = ts[metric]

                # Remove NaN/inf
                valid = np.isfinite(x) & np.isfinite(y)
                if valid.sum() < 5:
                    continue

                rho, pval = stats.spearmanr(x[valid], y[valid])
                corrs[f"{feat}_vs_{metric}"] = {"rho": float(rho), "p": float(pval)}

                if pval < 0.05:
                    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*"
                    print(f"  {feat:>12s} vs {metric:<12s}: rho={rho:+.3f} (p={pval:.4f}) {sig}")

        all_correlations[config_name] = corrs

    return all_correlations


def power_law(x, a, b, c):
    """Power law function: a * x^(-b) + c"""
    return a * np.power(x, -b) + c


def fit_loss_prediction(steps, losses, train_fraction=0.3):
    """Fit a power law to early training data and predict final loss."""
    n_train = max(3, int(len(steps) * train_fraction))

    train_steps = steps[:n_train]
    train_losses = losses[:n_train]

    # Fit power law
    try:
        # Shift steps to avoid 0
        x = train_steps + 1.0
        popt, _ = curve_fit(
            power_law, x, train_losses,
            p0=[10.0, 0.1, train_losses[-1]],
            maxfev=10000,
            bounds=([0, 0, 0], [100, 2, 20])
        )
        predicted_final = power_law(steps[-1] + 1.0, *popt)
        return predicted_final, popt
    except Exception:
        return losses[-1], None


def scaling_prediction_experiment(ts_data):
    """
    Test whether TDA features improve prediction of final loss.

    Comparison:
    1. Loss-only: Extrapolate power law from early loss curve
    2. TDA features: Use neural persistence + VR persistence from early training
    3. Combined: Loss extrapolation + TDA features
    """
    print("\n" + "=" * 60)
    print("SCALING PREDICTION EXPERIMENT")
    print("=" * 60)

    train_fractions = [0.2, 0.3, 0.5]
    results = {}

    for frac in train_fractions:
        print(f"\n--- Using first {frac*100:.0f}% of training ---")
        predictions = {"loss_only": {}, "tda_only": {}, "combined": {}}

        for config_name, ts in ts_data.items():
            steps = ts["steps"]
            losses = ts["loss"]
            actual_final = losses[-1]
            n_train = max(3, int(len(steps) * frac))

            # 1. Loss-only prediction (power law extrapolation)
            pred_loss, _ = fit_loss_prediction(steps, losses, frac)
            predictions["loss_only"][config_name] = pred_loss

            # 2. TDA feature-based prediction
            # Use mean neural persistence trajectory to estimate convergence
            np_early = ts["np_mean"][:n_train]
            h0_early = ts["h0_total"][:n_train]

            # Simple heuristic: NP growth rate predicts remaining improvement
            if len(np_early) >= 3:
                np_rate = (np_early[-1] - np_early[0]) / (n_train - 1) if n_train > 1 else 0
                # Higher NP growth rate -> more room for improvement
                remaining_steps = len(steps) - n_train
                np_predicted_improvement = np_rate * remaining_steps * 0.1  # Scale factor
                pred_tda = losses[n_train - 1] - abs(np_predicted_improvement)
                pred_tda = max(pred_tda, 0.5)  # Floor
            else:
                pred_tda = losses[n_train - 1]
            predictions["tda_only"][config_name] = pred_tda

            # 3. Combined: weighted average
            pred_combined = 0.6 * pred_loss + 0.4 * pred_tda
            predictions["combined"][config_name] = pred_combined

            print(f"  {config_name:>12s}: actual={actual_final:.3f}, "
                  f"loss_only={pred_loss:.3f}, tda={pred_tda:.3f}, combined={pred_combined:.3f}")

        # Compute prediction errors
        print(f"\n  Prediction MAE (train_frac={frac}):")
        for method in ["loss_only", "tda_only", "combined"]:
            errors = []
            for config_name, ts in ts_data.items():
                actual = ts["loss"][-1]
                pred = predictions[method][config_name]
                errors.append(abs(actual - pred))
            mae = np.mean(errors)
            print(f"    {method:>12s}: MAE = {mae:.4f}")
            predictions[f"{method}_mae"] = mae

        results[f"frac_{frac}"] = predictions

    return results


def detect_phase_transitions(ts_data):
    """Detect phase transitions in topological feature trajectories."""
    print("\n" + "=" * 60)
    print("PHASE TRANSITION DETECTION")
    print("=" * 60)

    transitions = {}

    for config_name, ts in ts_data.items():
        np_vals = ts["np_mean"]
        steps = ts["steps"]

        if len(np_vals) < 10:
            continue

        # Compute rate of change (smoothed)
        window = 5
        np_smooth = np.convolve(np_vals, np.ones(window) / window, mode="valid")
        np_rate = np.diff(np_smooth)

        # Find point of maximum rate change (potential phase transition)
        if len(np_rate) > 0:
            max_change_idx = np.argmax(np.abs(np_rate))
            transition_step = steps[max_change_idx + window // 2]
            transition_frac = transition_step / steps[-1]

            # Also check for diminishing returns point
            # Where NP rate drops below 10% of max rate
            max_rate = np.max(np.abs(np_rate))
            if max_rate > 0:
                diminishing_mask = np.abs(np_rate) < 0.1 * max_rate
                # Find first sustained period of diminishing returns
                for i in range(len(diminishing_mask) - 3):
                    if all(diminishing_mask[i:i + 3]):
                        diminishing_step = steps[i + window // 2]
                        break
                else:
                    diminishing_step = steps[-1]
            else:
                diminishing_step = steps[-1]

            transitions[config_name] = {
                "max_change_step": int(transition_step),
                "max_change_frac": float(transition_frac),
                "diminishing_step": int(diminishing_step),
                "diminishing_frac": float(diminishing_step / steps[-1]),
            }
            print(f"  {config_name:>12s}: max_change at step {transition_step} "
                  f"({transition_frac:.1%}), diminishing at step {diminishing_step} "
                  f"({diminishing_step / steps[-1]:.1%})")

    return transitions


def compute_scaling_relationships(ts_data):
    """Analyze how TDA features scale with model size."""
    print("\n" + "=" * 60)
    print("SCALING RELATIONSHIPS")
    print("=" * 60)

    model_sizes = []
    final_np_means = []
    final_h0_totals = []
    final_h1_totals = []
    final_losses = []
    np_rates = []

    for config_name in sorted(ts_data.keys(), key=lambda k: ts_data[k]["n_params"]):
        ts = ts_data[config_name]
        model_sizes.append(ts["n_params"])
        final_np_means.append(ts["np_mean"][-1])
        final_h0_totals.append(ts["h0_total"][-1])
        final_h1_totals.append(ts["h1_total"][-1])
        final_losses.append(ts["loss"][-1])

        # Rate of NP change in first 50% of training
        n_half = len(ts["np_mean"]) // 2
        if n_half > 1:
            rate = (ts["np_mean"][n_half] - ts["np_mean"][0]) / n_half
        else:
            rate = 0
        np_rates.append(rate)

    model_sizes = np.array(model_sizes, dtype=float)
    final_losses = np.array(final_losses)
    final_np_means = np.array(final_np_means)
    final_h0_totals = np.array(final_h0_totals)

    # Check for power-law scaling
    features_to_check = {
        "final_np_mean": final_np_means,
        "final_h0_total": final_h0_totals,
        "final_loss": final_losses,
        "np_rate": np.array(np_rates),
    }

    scaling_results = {}
    for feat_name, feat_vals in features_to_check.items():
        valid = np.isfinite(feat_vals) & (feat_vals != 0) & (model_sizes > 0)
        if valid.sum() >= 3:
            log_sizes = np.log(model_sizes[valid])
            log_vals = np.log(np.abs(feat_vals[valid]) + 1e-10)
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_vals)
            rho, rho_p = stats.spearmanr(model_sizes[valid], feat_vals[valid])

            scaling_results[feat_name] = {
                "power_law_exponent": float(slope),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "spearman_rho": float(rho),
                "spearman_p": float(rho_p),
            }
            print(f"  {feat_name:>20s}: exponent={slope:.3f}, R²={r_value**2:.3f}, "
                  f"ρ={rho:+.3f} (p={rho_p:.4f})")
        else:
            print(f"  {feat_name:>20s}: insufficient data")

    return scaling_results


# ============= VISUALIZATION =============

def plot_training_curves(ts_data):
    """Plot training loss curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(ts_data)))

    for (config_name, ts), color in zip(
            sorted(ts_data.items(), key=lambda x: x[1]["n_params"]), colors):
        ax.plot(ts["steps"], ts["loss"], label=f"{config_name} ({ts['n_params']:,}p)",
                color=color, alpha=0.8)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves by Model Size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "training_curves.png"), dpi=150)
    plt.close(fig)
    print("  Saved training_curves.png")


def plot_neural_persistence_evolution(ts_data):
    """Plot neural persistence evolution during training."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(ts_data)))

    for (config_name, ts), color in zip(
            sorted(ts_data.items(), key=lambda x: x[1]["n_params"]), colors):
        axes[0].plot(ts["steps"], ts["np_mean"],
                     label=f"{config_name}", color=color, alpha=0.8)
        axes[1].plot(ts["steps"], ts["np_std"],
                     label=f"{config_name}", color=color, alpha=0.8)

    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Mean Neural Persistence")
    axes[0].set_title("Neural Persistence Evolution")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Std Neural Persistence")
    axes[1].set_title("Neural Persistence Variability")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "neural_persistence_evolution.png"), dpi=150)
    plt.close(fig)
    print("  Saved neural_persistence_evolution.png")


def plot_vr_persistence_evolution(ts_data):
    """Plot Vietoris-Rips persistence evolution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(ts_data)))

    for (config_name, ts), color in zip(
            sorted(ts_data.items(), key=lambda x: x[1]["n_params"]), colors):
        axes[0].plot(ts["steps"], ts["h0_total"],
                     label=f"{config_name}", color=color, alpha=0.8)
        axes[1].plot(ts["steps"], ts["h1_total"],
                     label=f"{config_name}", color=color, alpha=0.8)

    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("H₀ Total Persistence")
    axes[0].set_title("Connected Components Persistence")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("H₁ Total Persistence")
    axes[1].set_title("Loops/Cycles Persistence")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "vr_persistence_evolution.png"), dpi=150)
    plt.close(fig)
    print("  Saved vr_persistence_evolution.png")


def plot_scaling_relationships(ts_data):
    """Plot how TDA features scale with model size."""
    model_sizes = []
    final_np = []
    final_h0 = []
    final_loss = []
    names = []

    for config_name in sorted(ts_data.keys(), key=lambda k: ts_data[k]["n_params"]):
        ts = ts_data[config_name]
        model_sizes.append(ts["n_params"])
        final_np.append(ts["np_mean"][-1])
        final_h0.append(ts["h0_total"][-1])
        final_loss.append(ts["loss"][-1])
        names.append(config_name)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Loss vs model size (log-log)
    axes[0].scatter(model_sizes, final_loss, s=80, zorder=5)
    for i, name in enumerate(names):
        axes[0].annotate(name, (model_sizes[i], final_loss[i]),
                         textcoords="offset points", xytext=(5, 5), fontsize=7)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Parameters")
    axes[0].set_ylabel("Final Loss")
    axes[0].set_title("Loss vs Model Size")
    axes[0].grid(True, alpha=0.3)

    # NP vs model size
    axes[1].scatter(model_sizes, final_np, s=80, color="orange", zorder=5)
    for i, name in enumerate(names):
        axes[1].annotate(name, (model_sizes[i], final_np[i]),
                         textcoords="offset points", xytext=(5, 5), fontsize=7)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Parameters")
    axes[1].set_ylabel("Final Neural Persistence")
    axes[1].set_title("Neural Persistence vs Model Size")
    axes[1].grid(True, alpha=0.3)

    # H0 persistence vs model size
    axes[2].scatter(model_sizes, final_h0, s=80, color="green", zorder=5)
    for i, name in enumerate(names):
        axes[2].annotate(name, (model_sizes[i], final_h0[i]),
                         textcoords="offset points", xytext=(5, 5), fontsize=7)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Parameters")
    axes[2].set_ylabel("Final H₀ Total Persistence")
    axes[2].set_title("H₀ Persistence vs Model Size")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "scaling_relationships.png"), dpi=150)
    plt.close(fig)
    print("  Saved scaling_relationships.png")


def plot_correlation_heatmap(ts_data):
    """Plot correlation heatmap between TDA features and training metrics."""
    tda_features = ["np_mean", "h0_total", "h1_total", "h0_num", "h1_num"]
    training_metrics = ["loss", "grad_norm", "weight_norm"]

    # Average correlations across models
    avg_corr = np.zeros((len(tda_features), len(training_metrics)))
    counts = np.zeros_like(avg_corr)

    for config_name, ts in ts_data.items():
        for i, feat in enumerate(tda_features):
            for j, metric in enumerate(training_metrics):
                x, y = ts[feat], ts[metric]
                valid = np.isfinite(x) & np.isfinite(y)
                if valid.sum() >= 5:
                    rho, _ = stats.spearmanr(x[valid], y[valid])
                    if np.isfinite(rho):
                        avg_corr[i, j] += rho
                        counts[i, j] += 1

    mask = counts > 0
    avg_corr[mask] /= counts[mask]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(avg_corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(training_metrics)))
    ax.set_xticklabels(training_metrics, rotation=45, ha="right")
    ax.set_yticks(range(len(tda_features)))
    ax.set_yticklabels(tda_features)

    for i in range(len(tda_features)):
        for j in range(len(training_metrics)):
            val = avg_corr[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=10)

    ax.set_title("Average Spearman Correlation\n(TDA Features vs Training Metrics)")
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  Saved correlation_heatmap.png")


def plot_prediction_comparison(ts_data, pred_results):
    """Plot prediction accuracy comparison."""
    fig, axes = plt.subplots(1, len(pred_results), figsize=(5 * len(pred_results), 5))
    if len(pred_results) == 1:
        axes = [axes]

    for ax, (frac_key, preds) in zip(axes, pred_results.items()):
        frac = float(frac_key.split("_")[1])
        methods = ["loss_only", "tda_only", "combined"]
        model_names = sorted(ts_data.keys(), key=lambda k: ts_data[k]["n_params"])

        x = np.arange(len(model_names))
        width = 0.2

        # Actual values
        actuals = [ts_data[m]["loss"][-1] for m in model_names]
        ax.bar(x - 1.5 * width, actuals, width, label="Actual", color="gray", alpha=0.7)

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        for i, method in enumerate(methods):
            pred_vals = [preds[method].get(m, 0) for m in model_names]
            ax.bar(x + (i - 0.5) * width, pred_vals, width,
                   label=method, color=colors[i], alpha=0.7)

        ax.set_xlabel("Model")
        ax.set_ylabel("Final Loss")
        ax.set_title(f"Loss Prediction (train={frac:.0%})")
        ax.set_xticks(x)
        ax.set_xticklabels([n.split("-")[0] for n in model_names], rotation=45)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "prediction_comparison.png"), dpi=150)
    plt.close(fig)
    print("  Saved prediction_comparison.png")


def plot_np_vs_loss_scatter(ts_data):
    """Scatter plot of neural persistence vs loss colored by training progress."""
    fig, ax = plt.subplots(figsize=(10, 7))
    colors_map = plt.cm.viridis(np.linspace(0, 1, len(ts_data)))

    for (config_name, ts), color in zip(
            sorted(ts_data.items(), key=lambda x: x[1]["n_params"]), colors_map):
        sc = ax.scatter(ts["np_mean"], ts["loss"],
                        c=ts["steps"], cmap="plasma", s=15, alpha=0.6,
                        label=config_name)

    ax.set_xlabel("Mean Neural Persistence")
    ax.set_ylabel("Loss")
    ax.set_title("Neural Persistence vs Loss\n(colored by training step)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap="plasma",
                                norm=plt.Normalize(vmin=0, vmax=max(ts["steps"][-1] for ts in ts_data.values())))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Training Step")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "np_vs_loss_scatter.png"), dpi=150)
    plt.close(fig)
    print("  Saved np_vs_loss_scatter.png")


def run_full_analysis():
    """Run the complete analysis pipeline."""
    print("Loading data...")
    tda_data, training_logs = load_data()
    ts_data = extract_time_series(tda_data, training_logs)

    print(f"Loaded data for {len(ts_data)} models")
    for name, ts in sorted(ts_data.items(), key=lambda x: x[1]["n_params"]):
        print(f"  {name}: {ts['n_params']:>8,} params, {len(ts['steps'])} checkpoints, "
              f"final_loss={ts['loss'][-1]:.4f}")

    # Run analyses
    correlations = compute_correlations(ts_data)
    scaling = compute_scaling_relationships(ts_data)
    transitions = detect_phase_transitions(ts_data)
    predictions = scaling_prediction_experiment(ts_data)

    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    plot_training_curves(ts_data)
    plot_neural_persistence_evolution(ts_data)
    plot_vr_persistence_evolution(ts_data)
    plot_scaling_relationships(ts_data)
    plot_correlation_heatmap(ts_data)
    plot_prediction_comparison(ts_data, predictions)
    plot_np_vs_loss_scatter(ts_data)

    # Save all analysis results
    analysis_results = {
        "correlations": correlations,
        "scaling": scaling,
        "transitions": transitions,
        "prediction_maes": {
            k: {m: v.get(f"{m}_mae", None) for m in ["loss_only", "tda_only", "combined"]}
            for k, v in predictions.items()
        },
    }

    output_path = os.path.join(RESULTS_DIR, "analysis_results.json")
    with open(output_path, "w") as f:
        json.dump(analysis_results, f, indent=2, default=str)
    print(f"\nAnalysis results saved to {output_path}")

    return ts_data, analysis_results


if __name__ == "__main__":
    run_full_analysis()
