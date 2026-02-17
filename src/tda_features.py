"""
Topological Data Analysis feature extraction from model checkpoints.

Computes:
1. Neural Persistence (Rieck et al. 2019): H_0 persistent homology of weight graphs
2. Weight-space Vietoris-Rips persistence: PH of sampled weight vectors
3. Summary statistics: total persistence, average persistence, Betti numbers
"""

import json
import os
import sys
import time

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, os.path.dirname(__file__))
from models import create_model, MODEL_CONFIGS, VOCAB_SIZE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "results", "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def compute_neural_persistence_layer(weight_matrix):
    """
    Compute neural persistence for a single linear layer weight matrix.

    Following Rieck et al. (2019):
    1. Treat weight matrix as bipartite graph (input nodes -> output nodes)
    2. Build descending filtration by absolute weight magnitude
    3. Compute H_0 persistent homology via union-find
    4. Return normalized total persistence

    Args:
        weight_matrix: numpy array of shape (out_features, in_features)

    Returns:
        dict with neural persistence metrics
    """
    W = np.abs(weight_matrix)
    n_out, n_in = W.shape
    n_nodes = n_out + n_in

    # Create edge list: (weight, out_node, in_node)
    edges = []
    for i in range(n_out):
        for j in range(n_in):
            edges.append((W[i, j], i, n_out + j))

    # Sort edges in descending order (largest weight first)
    edges.sort(key=lambda e: -e[0])

    # Union-Find for H_0 persistence
    parent = list(range(n_nodes))
    rank = [0] * n_nodes

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True

    # Track births and deaths
    # All components born at filtration value = max_weight (the first edge)
    max_weight = edges[0][0] if edges else 1.0
    birth_times = {i: max_weight for i in range(n_nodes)}
    persistence_pairs = []

    for weight_val, u, v in edges:
        pu, pv = find(u), find(v)
        if pu != pv:
            # Merge: the younger component dies
            # Birth of dying component = max_weight, death = weight_val
            union(u, v)
            persistence_pairs.append((max_weight, weight_val))

    # Compute persistence values
    persistences = [b - d for b, d in persistence_pairs]

    if not persistences:
        return {
            "total_persistence": 0.0,
            "avg_persistence": 0.0,
            "max_persistence": 0.0,
            "num_features": 0,
            "neural_persistence": 0.0,
        }

    total_pers = sum(persistences)
    # Normalize by maximum possible persistence (Rieck et al.)
    max_possible = max_weight * (n_nodes - 1)
    neural_pers = total_pers / max_possible if max_possible > 0 else 0.0

    return {
        "total_persistence": float(total_pers),
        "avg_persistence": float(np.mean(persistences)),
        "max_persistence": float(max(persistences)),
        "num_features": len(persistences),
        "neural_persistence": float(neural_pers),
    }


def compute_weight_space_ph(weight_vectors, max_points=200, max_dim=1):
    """
    Compute Vietoris-Rips persistent homology on weight vectors.

    Samples weight vectors from different layers/filters and computes
    PH to capture the topological structure of the weight space.

    Args:
        weight_vectors: list of 1D numpy arrays (flattened weight tensors)
        max_points: maximum number of points to use (subsample if needed)
        max_dim: maximum homology dimension (0=components, 1=loops)

    Returns:
        dict with PH summary statistics
    """
    import ripser

    # Stack and subsample if needed
    points = np.array(weight_vectors)
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]

    if len(points) < 3:
        return {
            "h0_total_persistence": 0.0,
            "h0_avg_persistence": 0.0,
            "h0_num_features": 0,
            "h1_total_persistence": 0.0,
            "h1_avg_persistence": 0.0,
            "h1_num_features": 0,
        }

    # Compute pairwise distances
    try:
        result = ripser.ripser(points, maxdim=max_dim, thresh=2.0)
    except Exception:
        # Fallback if ripser fails
        return {
            "h0_total_persistence": 0.0,
            "h0_avg_persistence": 0.0,
            "h0_num_features": 0,
            "h1_total_persistence": 0.0,
            "h1_avg_persistence": 0.0,
            "h1_num_features": 0,
        }

    features = {}
    for dim in range(max_dim + 1):
        dgm = result["dgms"][dim]
        # Filter out infinite persistence
        finite_mask = np.isfinite(dgm[:, 1])
        dgm_finite = dgm[finite_mask]

        if len(dgm_finite) > 0:
            pers = dgm_finite[:, 1] - dgm_finite[:, 0]
            pers = pers[pers > 0]  # Only positive persistence
            features[f"h{dim}_total_persistence"] = float(np.sum(pers))
            features[f"h{dim}_avg_persistence"] = float(np.mean(pers)) if len(pers) > 0 else 0.0
            features[f"h{dim}_num_features"] = int(len(pers))
            features[f"h{dim}_max_persistence"] = float(np.max(pers)) if len(pers) > 0 else 0.0
        else:
            features[f"h{dim}_total_persistence"] = 0.0
            features[f"h{dim}_avg_persistence"] = 0.0
            features[f"h{dim}_num_features"] = 0
            features[f"h{dim}_max_persistence"] = 0.0

    return features


def compute_persistence_diagram_distance(dgm1, dgm2):
    """Compute Wasserstein distance between persistence diagrams using persim."""
    from persim import wasserstein

    if len(dgm1) == 0 or len(dgm2) == 0:
        return 0.0
    try:
        return float(wasserstein(dgm1, dgm2, order=1))
    except Exception:
        return 0.0


def extract_weight_vectors_for_ph(model_state_dict, min_dim=4):
    """
    Extract weight vectors from model state dict for VR persistence.

    For each weight matrix, we treat rows (or columns) as points in weight space.
    Small weight matrices are flattened and treated as single points.
    """
    vectors = []
    for name, param in model_state_dict.items():
        if "weight" not in name or "ln" in name or "emb" in name:
            continue
        w = param.cpu().numpy()
        if w.ndim == 2 and w.shape[0] >= min_dim:
            # Use rows as points (each output neuron is a point in input-space)
            vectors.extend(w.tolist())
        elif w.ndim == 1 and len(w) >= min_dim:
            vectors.append(w.tolist())

    # Pad to equal length if needed
    if not vectors:
        return []

    max_len = max(len(v) for v in vectors)
    padded = [v + [0.0] * (max_len - len(v)) for v in vectors]
    return padded


def extract_features_from_checkpoint(ckpt_path, config_name):
    """
    Extract all TDA features from a single checkpoint.

    Args:
        ckpt_path: path to checkpoint .pt file
        config_name: model configuration name

    Returns:
        dict with all TDA features
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["model_state_dict"]
    step = ckpt["step"]

    features = {"step": step, "config_name": config_name, "loss": ckpt.get("loss", None)}

    # 1. Neural Persistence per layer
    layer_np = {}
    all_np_values = []

    for name, param in state_dict.items():
        if "weight" not in name or param.ndim != 2:
            continue
        if "ln" in name or "emb" in name:
            continue

        w = param.cpu().numpy()
        np_metrics = compute_neural_persistence_layer(w)
        layer_np[name] = np_metrics
        all_np_values.append(np_metrics["neural_persistence"])

    if all_np_values:
        features["neural_persistence_mean"] = float(np.mean(all_np_values))
        features["neural_persistence_std"] = float(np.std(all_np_values))
        features["neural_persistence_max"] = float(np.max(all_np_values))
        features["neural_persistence_min"] = float(np.min(all_np_values))
    else:
        features["neural_persistence_mean"] = 0.0
        features["neural_persistence_std"] = 0.0
        features["neural_persistence_max"] = 0.0
        features["neural_persistence_min"] = 0.0

    features["layer_neural_persistence"] = layer_np

    # 2. Weight-space VR persistence
    weight_vecs = extract_weight_vectors_for_ph(state_dict)
    if weight_vecs:
        ph_features = compute_weight_space_ph(weight_vecs, max_points=200, max_dim=1)
        features.update(ph_features)
    else:
        features.update({
            "h0_total_persistence": 0.0, "h0_avg_persistence": 0.0, "h0_num_features": 0,
            "h1_total_persistence": 0.0, "h1_avg_persistence": 0.0, "h1_num_features": 0,
        })

    # 3. Weight statistics (non-topological baselines)
    all_weights = []
    for name, param in state_dict.items():
        if param.requires_grad if hasattr(param, 'requires_grad') else True:
            all_weights.append(param.cpu().numpy().flatten())
    if all_weights:
        all_w = np.concatenate(all_weights)
        features["weight_l2_norm"] = float(np.sqrt(np.sum(all_w ** 2)))
        features["weight_mean"] = float(np.mean(all_w))
        features["weight_std"] = float(np.std(all_w))
        features["weight_max_abs"] = float(np.max(np.abs(all_w)))

    return features


def extract_all_features(config_name):
    """Extract TDA features from all checkpoints of a model."""
    ckpt_dir = os.path.join(CHECKPOINT_DIR, config_name)
    if not os.path.exists(ckpt_dir):
        print(f"  No checkpoints found for {config_name}")
        return []

    ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
    print(f"  Processing {len(ckpt_files)} checkpoints for {config_name}...")

    all_features = []
    for i, fname in enumerate(ckpt_files):
        ckpt_path = os.path.join(ckpt_dir, fname)
        features = extract_features_from_checkpoint(ckpt_path, config_name)
        all_features.append(features)

        if (i + 1) % 10 == 0:
            print(f"    Processed {i + 1}/{len(ckpt_files)} checkpoints")

    return all_features


def run_all_extractions():
    """Extract TDA features from all model configurations."""
    print("=" * 60)
    print("TDA Feature Extraction")
    print("=" * 60)

    all_results = {}
    total_start = time.time()

    for config_name in MODEL_CONFIGS:
        print(f"\nModel: {config_name}")
        start = time.time()
        features = extract_all_features(config_name)
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s ({len(features)} checkpoints)")
        all_results[config_name] = features

    total_time = time.time() - total_start
    print(f"\nAll extractions complete in {total_time:.1f}s")

    # Save results
    output_path = os.path.join(RESULTS_DIR, "tda_features.json")

    # Convert layer-level features to serializable format
    serializable = {}
    for name, features_list in all_results.items():
        serializable[name] = []
        for f in features_list:
            sf = {k: v for k, v in f.items() if k != "layer_neural_persistence"}
            # Flatten layer persistence into summary
            if "layer_neural_persistence" in f:
                layer_np = f["layer_neural_persistence"]
                sf["n_analyzed_layers"] = len(layer_np)
                layer_names = sorted(layer_np.keys())
                for idx, ln in enumerate(layer_names):
                    sf[f"layer_{idx}_np"] = layer_np[ln]["neural_persistence"]
            serializable[name].append(sf)

    with open(output_path, "w") as fp:
        json.dump(serializable, fp, indent=2)
    print(f"Features saved to {output_path}")

    return all_results


if __name__ == "__main__":
    run_all_extractions()
