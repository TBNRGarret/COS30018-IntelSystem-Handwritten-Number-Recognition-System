"""
config.py – Shared configuration for all models.
All files import from here to ensure consistent settings
and a single MNIST download location.
"""

import os
import torch

# ── Paths ─────────────────────────────────────────────────
# Resolve absolute path relative to this config file.
# No matter which directory you run each script from,
# MNIST will always be stored/read from the same folder.
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")          # ./data (absolute)
CKPT_DIR  = os.path.join(BASE_DIR, "checkpoints")   # ./checkpoints
PLOT_DIR  = os.path.join(BASE_DIR, "plots")         # ./plots

# Create directories if they don't exist
for d in [DATA_DIR, CKPT_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Device ────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Training hyperparameters ──────────────────────────────
BATCH_SIZE = 64
NUM_WORKERS = 4   # Set to 0 if running on Windows and getting DataLoader errors

# Per-model epoch defaults (can be overridden at call site)
EPOCHS = {
    "perceptron": 20,
    "mlp":        30,
    "cnn":        30,
}

# ── MNIST normalization stats ─────────────────────────────
MNIST_MEAN = (0.1307,)
MNIST_STD  = (0.3081,)
