"""
Train a family of small GPT models and save frequent checkpoints.

Each model is trained on character-level text prediction from Wikitext-2.
Checkpoints are saved every N steps for topological analysis.
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, os.path.dirname(__file__))
from models import SmallGPT, MODEL_CONFIGS, VOCAB_SIZE, MAX_SEQ_LEN, create_model

# Configuration
SEED = 42
NUM_STEPS = 5000
CHECKPOINT_INTERVAL = 100  # Save every 100 steps -> 50 checkpoints
BATCH_SIZE = 64
SEQ_LEN = MAX_SEQ_LEN
LR = 3e-3
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "results", "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_text_data():
    """Load text data for character-level LM training.

    Uses a simple approach: download wikitext-2 via HuggingFace datasets,
    concatenate, and encode at character level.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
        text = "\n".join([t for t in ds["text"] if t.strip()])
    except Exception:
        # Fallback: generate synthetic text
        print("WARNING: Could not load wikitext-2, using synthetic data")
        rng = np.random.default_rng(SEED)
        text = "".join([chr(rng.integers(32, 127)) for _ in range(500000)])

    # Character-level encoding: map to [0, VOCAB_SIZE)
    # Use ASCII subset, clamp to vocab range
    encoded = []
    for ch in text:
        code = ord(ch)
        if code < VOCAB_SIZE:
            encoded.append(code)
        else:
            encoded.append(code % VOCAB_SIZE)

    data = torch.tensor(encoded, dtype=torch.long)
    print(f"Training data: {len(data):,} tokens ({len(data) / 1e6:.2f}M)")
    return data


def get_batch(data, batch_size, seq_len, device):
    """Get a random batch of sequences from the data."""
    max_start = len(data) - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s:s + seq_len] for s in starts]).to(device)
    y = torch.stack([data[s + 1:s + seq_len + 1] for s in starts]).to(device)
    return x, y


def cosine_lr_schedule(step, warmup_steps, total_steps, lr_max, lr_min=1e-5):
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))


def train_model(config_name, data, num_steps=NUM_STEPS, device=DEVICE):
    """Train a single model and save checkpoints."""
    set_seed(SEED)

    model = create_model(config_name).to(device)
    n_params = model.count_parameters()
    print(f"\nTraining {config_name}: {n_params:,} parameters on {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    # Create checkpoint directory
    ckpt_dir = os.path.join(CHECKPOINT_DIR, config_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training loop
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    training_log = []

    start_time = time.time()

    for step in range(num_steps + 1):
        # Update learning rate
        lr = cosine_lr_schedule(step, WARMUP_STEPS, num_steps, LR)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward pass
        x, y = get_batch(data, BATCH_SIZE, SEQ_LEN, device)
        logits = model(x)
        loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()

        optimizer.step()

        # Compute weight norms
        weight_norms = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                weight_norms[name] = param.data.norm(2).item()

        # Log
        step_log = {
            "step": step,
            "loss": loss.item(),
            "lr": lr,
            "grad_norm": grad_norm,
            "weight_norm_total": sum(weight_norms.values()),
        }
        training_log.append(step_log)

        # Save checkpoint
        if step % CHECKPOINT_INTERVAL == 0:
            ckpt_path = os.path.join(ckpt_dir, f"step_{step:06d}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "loss": loss.item(),
                "config_name": config_name,
                "n_params": n_params,
            }, ckpt_path)

            if step % (CHECKPOINT_INTERVAL * 5) == 0:
                elapsed = time.time() - start_time
                print(f"  Step {step:>5d}/{num_steps} | loss={loss.item():.4f} | "
                      f"grad_norm={grad_norm:.4f} | lr={lr:.6f} | time={elapsed:.1f}s")

    elapsed = time.time() - start_time
    print(f"  Training complete in {elapsed:.1f}s")

    # Save training log
    log_path = os.path.join(ckpt_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f)

    return training_log


def train_all_models():
    """Train all model configurations."""
    print("=" * 60)
    print("Training Small GPT Model Family")
    print(f"Device: {DEVICE}")
    print(f"Steps: {NUM_STEPS}, Checkpoint interval: {CHECKPOINT_INTERVAL}")
    print(f"Batch size: {BATCH_SIZE}, Seq len: {SEQ_LEN}")
    print("=" * 60)

    data = get_text_data()

    all_logs = {}
    total_start = time.time()

    for config_name in MODEL_CONFIGS:
        logs = train_model(config_name, data)
        all_logs[config_name] = logs

    total_time = time.time() - total_start
    print(f"\nAll models trained in {total_time:.1f}s")

    # Save combined results
    summary = {
        "seed": SEED,
        "num_steps": NUM_STEPS,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "device": DEVICE,
        "total_time_seconds": total_time,
        "models": {},
    }
    for name, logs in all_logs.items():
        summary["models"][name] = {
            "n_params": logs[0].get("n_params", 0) if logs else 0,
            "final_loss": logs[-1]["loss"],
            "min_loss": min(l["loss"] for l in logs),
            "num_checkpoints": NUM_STEPS // CHECKPOINT_INTERVAL + 1,
        }

    summary_path = os.path.join(RESULTS_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    return all_logs


if __name__ == "__main__":
    train_all_models()
