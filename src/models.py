"""
Small GPT-style transformer models for topological analysis experiments.

Defines a family of decoder-only transformer models with varying sizes
(~1K to ~50K parameters) for studying how topological features of weight
spaces evolve during training and scale across model sizes.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallGPT(nn.Module):
    """Minimal GPT-style decoder-only transformer for controlled experiments."""

    def __init__(self, vocab_size, hidden_dim, n_layers, n_heads, max_seq_len=128, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"

        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block."""

    def __init__(self, hidden_dim, n_heads, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, n_heads, dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, hidden_dim, n_heads, dropout=0.0):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, nh, T, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention with causal mask
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        y = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    """Standard transformer MLP with GELU activation."""

    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.fc2 = nn.Linear(4 * hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


# Model configurations: name -> (hidden_dim, n_layers, n_heads)
# Designed to span ~3K to ~97K parameters (with vocab_size=128)
MODEL_CONFIGS = {
    "tiny-3k":   {"hidden_dim": 8,   "n_layers": 1, "n_heads": 1},
    "small-7k":  {"hidden_dim": 16,  "n_layers": 1, "n_heads": 2},
    "med-21k":   {"hidden_dim": 24,  "n_layers": 2, "n_heads": 2},
    "large-46k": {"hidden_dim": 32,  "n_layers": 3, "n_heads": 4},
    "xl-97k":    {"hidden_dim": 48,  "n_layers": 3, "n_heads": 4},
}

VOCAB_SIZE = 128  # Character-level tokenization for simplicity
MAX_SEQ_LEN = 128


def create_model(config_name, vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN):
    """Create a model from a named configuration."""
    cfg = MODEL_CONFIGS[config_name]
    model = SmallGPT(
        vocab_size=vocab_size,
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        max_seq_len=max_seq_len,
    )
    return model


def list_model_configs(vocab_size=VOCAB_SIZE):
    """Print model configurations with parameter counts."""
    for name, cfg in MODEL_CONFIGS.items():
        model = create_model(name, vocab_size=vocab_size)
        n_params = model.count_parameters()
        print(f"{name:>12s}: hidden={cfg['hidden_dim']:>3d}, layers={cfg['n_layers']}, "
              f"heads={cfg['n_heads']}, params={n_params:>8,d}")
        del model


if __name__ == "__main__":
    list_model_configs()
