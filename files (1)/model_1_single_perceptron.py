"""
model_1_single_perceptron.py – Single Perceptron for MNIST digit classification.

Architecture:
    Input (784) ──► Linear(784 → 10) ──► Logits

A single linear layer with no hidden units or non-linear activations.
Mathematically equivalent to multinomial logistic regression.

Strengths:
    - Extremely fast to train and infer
    - Minimal parameters (7,850 total)
    - Easy to interpret: weights are per-pixel class scores

Weaknesses:
    - Can only learn a linear decision boundary (hyperplane in 784-D space)
    - Cannot capture spatial structure of digits (curves, loops, etc.)
    - Hard accuracy ceiling around 92–93% on MNIST

Role in Task 3:
    Serves as the baseline. Its limitations motivate the need for
    deeper architectures (MLP, CNN).
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

SAVE_PATH = os.path.join(CKPT_DIR, "perceptron_best.pth")


# ── Model ─────────────────────────────────────────────────
class SinglePerceptron(nn.Module):
    """
    Single linear layer: 784 pixel inputs → 10 class logits.
    No hidden layer, no non-linear activation.
    CrossEntropyLoss applies log-softmax internally, so we output raw logits.
    """

    def __init__(self, input_size: int = 784, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear  = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, 1, 28, 28] → flatten → [B, 784] → linear → [B, 10]
        return self.linear(self.flatten(x))

    @property
    def num_parameters(self) -> int:
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


# ── Main training loop ────────────────────────────────────
def train(epochs: int = EPOCHS["perceptron"]) -> dict:
    # No augmentation – a linear model gains nothing from it
    train_loader, test_loader = get_loaders(augment=False)

    model     = SinglePerceptron().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print(f"\n{'='*55}")
    print(f"  Model  : Single Perceptron")
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

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        if v_acc > best_acc:
            best_acc = v_acc
            best_preds, best_labels = preds, labels
            torch.save(model.state_dict(), SAVE_PATH)

        print(f"  Epoch [{epoch:2d}/{epochs}]  "
              f"Train {t_acc*100:.2f}% / {t_loss:.4f}  |  "
              f"Val   {v_acc*100:.2f}% / {v_loss:.4f}")

    elapsed = time.time() - t0
    print(f"\n  Best val accuracy   : {best_acc*100:.2f}%")
    print(f"  Total training time : {elapsed:.1f}s\n")

    _plot_curves(history, "Single Perceptron")
    _plot_confusion(best_labels, best_preds, "Single Perceptron")
    print(classification_report(best_labels, best_preds,
                                target_names=[str(i) for i in range(10)]))

    return {
        "name":     "Single Perceptron",
        "best_acc": best_acc,
        "history":  history,
        "preds":    best_preds,
        "labels":   best_labels,
        "time":     elapsed,
        "params":   model.num_parameters,
    }


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
    path = os.path.join(PLOT_DIR, "perceptron_curves.png")
    plt.savefig(path, dpi=150); plt.show()
    print(f"  [Saved] {path}")


def _plot_confusion(labels, preds, title: str):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10))
    plt.title(f"{title} – Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "perceptron_cm.png")
    plt.savefig(path, dpi=150); plt.show()
    print(f"  [Saved] {path}")


# ── Inference helper ──────────────────────────────────────
@torch.no_grad()
def predict(model: SinglePerceptron, image_tensor: torch.Tensor):
    """
    Predict a single 28x28 image.

    Args:
        model:        Trained SinglePerceptron instance.
        image_tensor: Tensor of shape [1, 28, 28].

    Returns:
        predicted_digit (int), confidence (float), probabilities (np.ndarray)
    """
    model.eval()
    out   = model(image_tensor.unsqueeze(0).to(DEVICE))   # [1, 10]
    probs = torch.softmax(out, dim=1)[0]
    pred  = probs.argmax().item()
    return pred, probs[pred].item(), probs.cpu().numpy()


if __name__ == "__main__":
    result = train()
    print(f"[Done] Best accuracy: {result['best_acc'] * 100:.2f}%")
