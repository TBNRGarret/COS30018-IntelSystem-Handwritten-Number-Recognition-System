"""
model_2_mlp.py – Multilayer Perceptron (MLP) for MNIST digit classification.

Three variants are defined for comparison:
    MLP_Small  : 784 → 256 → 128 → 10
    MLP_Medium : 784 → 512 → 256 → 128 → 10   (default)
    MLP_Large  : 784 → 1024 → 512 → 256 → 128 → 10

Regularization techniques applied:
    - Batch Normalization : stabilizes and accelerates training
    - Dropout             : reduces overfitting
    - Weight decay (L2)   : penalizes large weights in Adam optimizer
    - LR scheduler        : ReduceLROnPlateau halves LR when val_loss stagnates

Strengths over Single Perceptron:
    - Hidden layers + ReLU allow learning non-linear decision boundaries
    - Can distinguish visually similar digits better (3 vs 8, 5 vs 6)

Remaining weakness vs CNN:
    - Images are flattened to 1-D vectors before the first layer.
      This destroys the 2-D spatial structure: pixel (0,0) and pixel (27,27)
      are treated as completely independent features.
    - More parameters than CNN but generally lower accuracy.

Expected accuracy: ~97–98% on MNIST
"""

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from config import DEVICE, EPOCHS, CKPT_DIR, PLOT_DIR
from data_loader import get_loaders

SAVE_PATH = os.path.join(CKPT_DIR, "mlp_best.pth")
DROPOUT   = 0.3


# ── Model variants ────────────────────────────────────────
def _fc_block(in_features: int, out_features: int, dropout: float) -> nn.Sequential:
    """Reusable fully-connected block: Linear → BN → ReLU → Dropout."""
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
    )


class MLP_Small(nn.Module):
    """Two hidden layers: 784 → 256 → 128 → 10."""

    def __init__(self, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            _fc_block(784, 256, dropout),
            _fc_block(256, 128, dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x): return self.net(x)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLP_Medium(nn.Module):
    """Three hidden layers: 784 → 512 → 256 → 128 → 10. (default)"""

    def __init__(self, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            _fc_block(784, 512, dropout),
            _fc_block(512, 256, dropout),
            _fc_block(256, 128, dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x): return self.net(x)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLP_Large(nn.Module):
    """Four hidden layers: 784 → 1024 → 512 → 256 → 128 → 10."""

    def __init__(self, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            _fc_block(784,  1024, dropout),
            _fc_block(1024, 512,  dropout),
            _fc_block(512,  256,  dropout),
            _fc_block(256,  128,  dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x): return self.net(x)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Training step ─────────────────────────────────────────
def _train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


# ── Evaluation step ───────────────────────────────────────
@torch.no_grad()
def _evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        out   = model(images)
        preds = out.argmax(1)

        total_loss += criterion(out, labels).item() * images.size(0)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


# ── Generic training loop ─────────────────────────────────
def train(
    model=None,
    model_name: str = "MLP_Medium",
    epochs: int     = EPOCHS["mlp"],
    lr: float       = 1e-3,
    save_path: str  = SAVE_PATH,
) -> dict:
    """
    Train an MLP model. If no model is passed, defaults to MLP_Medium.

    Returns a result dict compatible with task3_compare_all.py.
    """
    if model is None:
        model = MLP_Medium().to(DEVICE)

    train_loader, test_loader = get_loaders(augment=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Halve the LR when validation loss stops improving for 3 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    print(f"\n{'='*55}")
    print(f"  Model  : {model_name}")
    print(f"  Params : {model.num_parameters:,}")
    print(f"  Device : {DEVICE}")
    print(f"{'='*55}")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    best_preds, best_labels = [], []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        t_loss, t_acc = _train_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc, preds, labels = _evaluate(model, test_loader, criterion)
        scheduler.step(v_loss)

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        if v_acc > best_acc:
            best_acc = v_acc
            best_preds, best_labels = preds, labels
            torch.save(model.state_dict(), save_path)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch [{epoch:2d}/{epochs}]  "
              f"Train {t_acc*100:.2f}% / {t_loss:.4f}  |  "
              f"Val   {v_acc*100:.2f}% / {v_loss:.4f}  |  LR {current_lr:.2e}")

    elapsed = time.time() - t0
    print(f"\n  Best val accuracy   : {best_acc*100:.2f}%")
    print(f"  Total training time : {elapsed:.1f}s\n")

    _plot_curves(history, model_name)
    _plot_confusion(best_labels, best_preds, model_name)
    print(classification_report(best_labels, best_preds,
                                target_names=[str(i) for i in range(10)]))

    return {
        "name":     model_name,
        "best_acc": best_acc,
        "history":  history,
        "preds":    best_preds,
        "labels":   best_labels,
        "time":     elapsed,
        "params":   model.num_parameters,
    }


def compare_variants(epochs: int = EPOCHS["mlp"]):
    """
    Train Small / Medium / Large and print a comparison table.
    Use this to justify the chosen variant in the project report.
    """
    configs = [
        ("MLP_Small",  MLP_Small(),  os.path.join(CKPT_DIR, "mlp_small.pth")),
        ("MLP_Medium", MLP_Medium(), os.path.join(CKPT_DIR, "mlp_medium.pth")),
        ("MLP_Large",  MLP_Large(),  os.path.join(CKPT_DIR, "mlp_large.pth")),
    ]
    results = {}
    for name, model, path in configs:
        model = model.to(DEVICE)
        results[name] = train(model, name, epochs, save_path=path)

    print(f"\n{'='*60}")
    print(f"  {'Variant':<14} {'Best Acc':>10} {'Time':>10} {'Params':>12}")
    print(f"{'='*60}")
    for name, res in results.items():
        print(f"  {name:<14} {res['best_acc']*100:>9.2f}% "
              f"{res['time']:>9.1f}s {res['params']:>12,}")
    print(f"{'='*60}")

    _plot_variant_comparison(results)
    return results


# ── Visualization helpers ─────────────────────────────────
def _plot_curves(history: dict, title: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(history["train_acc"]) + 1)

    ax1.plot(ep, [a * 100 for a in history["train_acc"]], label="Train")
    ax1.plot(ep, [a * 100 for a in history["val_acc"]],   label="Val")
    ax1.set(title=f"{title} – Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(ep, history["train_loss"], label="Train")
    ax2.plot(ep, history["val_loss"],   label="Val")
    ax2.set(title=f"{title} – Loss", xlabel="Epoch", ylabel="Loss")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = title.lower().replace(" ", "_")
    path = os.path.join(PLOT_DIR, f"{safe_name}_curves.png")
    plt.savefig(path, dpi=150); plt.show()
    print(f"  [Saved] {path}")


def _plot_confusion(labels, preds, title: str):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=range(10), yticklabels=range(10))
    plt.title(f"{title} – Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    safe_name = title.lower().replace(" ", "_")
    path = os.path.join(PLOT_DIR, f"{safe_name}_cm.png")
    plt.savefig(path, dpi=150); plt.show()
    print(f"  [Saved] {path}")


def _plot_variant_comparison(results: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for name, res in results.items():
        ep = range(1, len(res["history"]["val_acc"]) + 1)
        ax1.plot(ep, [a * 100 for a in res["history"]["val_acc"]], label=name)
        ax2.plot(ep, res["history"]["val_loss"], label=name)

    ax1.set(title="MLP Variants – Validation Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set(title="MLP Variants – Validation Loss", xlabel="Epoch", ylabel="Loss")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "mlp_variant_comparison.png")
    plt.savefig(path, dpi=150); plt.show()
    print(f"  [Saved] {path}")


# ── Inference helper ──────────────────────────────────────
@torch.no_grad()
def predict(model: nn.Module, image_tensor: torch.Tensor):
    """
    Predict a single 28x28 image.

    Args:
        model:        Trained MLP instance.
        image_tensor: Tensor of shape [1, 28, 28].

    Returns:
        predicted_digit (int), confidence (float), probabilities (np.ndarray)
    """
    model.eval()
    out   = model(image_tensor.unsqueeze(0).to(DEVICE))
    probs = torch.softmax(out, dim=1)[0]
    pred  = probs.argmax().item()
    return pred, probs[pred].item(), probs.cpu().numpy()


if __name__ == "__main__":
    # Train the default MLP_Medium
    #result = train()
    #print(f"[Done] Best accuracy: {result['best_acc'] * 100:.2f}%")

    # Uncomment to compare all three variants:
    compare_variants()
