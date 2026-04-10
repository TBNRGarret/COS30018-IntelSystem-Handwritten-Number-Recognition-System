"""
model_3_cnn.py – Convolutional Neural Network (CNN) for MNIST digit classification.

Why CNNs outperform MLP on images:
    1. Local connectivity  – each filter sees a small spatial patch, not all 784 pixels.
    2. Weight sharing      – the same filter slides across the entire image, so the
                             network learns edge detectors regardless of position.
    3. Hierarchical features:
         Layer 1 → edges and corners
         Layer 2 → curves and loops
         Layer 3 → digit-level shapes
    4. Translation invariance – MaxPooling makes features robust to small shifts.

Four variants are provided for experimentation and report comparison:

    ① CNN_Custom     – designed from scratch (3 conv blocks)        → recommended
    ② CNN_LeNet5     – classic LeNet-5 architecture (LeCun 1998)    → historical baseline
    ③ CNN_Pretrained – ResNet18 backbone frozen, only head trained   → Transfer Learning demo
    ④ CNN_Finetune   – ResNet18 fully unfrozen, layer-wise LR       → best possible accuracy

Expected accuracy: 99%+ on MNIST (CNN_Custom / CNN_Finetune)
"""

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models
from sklearn.metrics import confusion_matrix, classification_report

from config import DEVICE, EPOCHS, CKPT_DIR, PLOT_DIR
from data_loader import get_loaders

# ──────────────────────────────────────────────────────────
# ① CNN_Custom – designed from scratch
# ──────────────────────────────────────────────────────────
class CNN_Custom(nn.Module):
    """
    Three-block convolutional backbone followed by a two-layer classifier head.

    Architecture:
        Input  [B, 1, 28, 28]
        Block1: Conv(1→32, 3×3) + BN + ReLU + MaxPool(2) → [B, 32, 14, 14]
        Block2: Conv(32→64, 3×3) + BN + ReLU + MaxPool(2) → [B, 64,  7,  7]
        Block3: Conv(64→128, 3×3) + BN + ReLU              → [B, 128, 7,  7]
        Flatten → [B, 6272]
        FC(6272→256) + ReLU + Dropout
        FC(256→10)
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1 – detect low-level features (edges, corners)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 28×28 → 14×14

            # Block 2 – detect mid-level features (curves, arcs)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 14×14 → 7×7

            # Block 3 – detect high-level features (digit shapes)
            # No pooling here to preserve spatial resolution
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),               # 128 × 7 × 7 = 6272
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────
# ② CNN_LeNet5 – classic 1998 architecture
# ──────────────────────────────────────────────────────────
class CNN_LeNet5(nn.Module):
    """
    Modernised LeNet-5 (LeCun et al., 1998).
    Differences from original: ReLU instead of Tanh, MaxPool instead of AvgPool.

    Architecture:
        Input  [B, 1, 28, 28]
        C1: Conv(1→6, 5×5) + ReLU  → [B, 6, 24, 24]
        S2: MaxPool(2)              → [B, 6, 12, 12]
        C3: Conv(6→16, 5×5) + ReLU → [B, 16, 8, 8]
        S4: MaxPool(2)              → [B, 16, 4, 4]
        Flatten                    → [B, 256]
        F5: FC(256→120) + ReLU
        F6: FC(120→84) + ReLU
        Output: FC(84→10)
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),   # 28→24
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # 24→12
            nn.Conv2d(6, 16, kernel_size=5),  # 12→8
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # 8→4
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.fc(self.conv(x))

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────
# ③ CNN_Pretrained – Transfer Learning (frozen backbone)
# ──────────────────────────────────────────────────────────
class CNN_Pretrained(nn.Module):
    """
    ResNet18 pretrained on ImageNet; backbone is frozen.
    Only the new classification head is trained.

    Strategy: FREEZE all conv layers → only train the final FC head.

    Useful when:
        - Labelled data is scarce
        - Training compute is limited

    Note on MNIST:
        ImageNet features (animals, objects) differ greatly from MNIST (digits).
        Accuracy may not beat CNN_Custom – but this is an important discussion
        point for the project report (domain gap, transfer learning trade-offs).
    """

    def __init__(self, num_classes: int = 10, freeze_backbone: bool = True):
        super().__init__()

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # ResNet18 expects 3-channel input; MNIST is 1-channel.
        # Replace the first conv layer to accept grayscale images.
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                # Only keep conv1 (just re-initialised) and fc unfrozen
                if "fc" not in name and "conv1" not in name:
                    param.requires_grad = False

        in_features = self.backbone.fc.in_features  # 512 for ResNet18
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    @property
    def num_parameters(self):
        # Report only trainable parameters (backbone is largely frozen)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────
# ④ CNN_Finetune – Full fine-tuning with layer-wise LR
# ──────────────────────────────────────────────────────────
class CNN_Finetune(nn.Module):
    """
    ResNet18 with ALL layers unfrozen and layer-wise learning rates.

    Strategy:
        - Backbone layers: very small LR (1e-5) – pretrained weights are
          already good; large updates would destroy learned features.
        - Classification head: larger LR (1e-3) – trained from scratch.

    Use get_param_groups() to pass to the optimizer instead of model.parameters().
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def get_param_groups(self, lr_backbone: float = 1e-5, lr_head: float = 1e-3):
        """
        Return parameter groups with different learning rates.
        Pass the returned list directly to the optimizer.
        """
        backbone_params = [
            p for name, p in self.named_parameters() if "backbone.fc" not in name
        ]
        head_params = [
            p for name, p in self.named_parameters() if "backbone.fc" in name
        ]
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params,     "lr": lr_head},
        ]

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
    model_name: str = "CNN_Custom",
    epochs: int     = EPOCHS["cnn"],
    optimizer=None,
    save_path: str  = None,
) -> dict:
    """
    Train any CNN variant. Defaults to CNN_Custom.

    Returns a result dict compatible with task3_compare_all.py.
    """
    if model is None:
        model = CNN_Custom().to(DEVICE)
    if save_path is None:
        save_path = os.path.join(CKPT_DIR, f"{model_name.lower()}_best.pth")

    train_loader, test_loader = get_loaders(augment=True)
    criterion = nn.CrossEntropyLoss()

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Cosine annealing: smoothly decays LR from initial to near-zero
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        if v_acc > best_acc:
            best_acc = v_acc
            best_preds, best_labels = preds, labels
            torch.save(model.state_dict(), save_path)

        print(f"  Epoch [{epoch:2d}/{epochs}]  "
              f"Train {t_acc*100:.2f}% / {t_loss:.4f}  |  "
              f"Val   {v_acc*100:.2f}% / {v_loss:.4f}")

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


def compare_all_variants(epochs: int = EPOCHS["cnn"]):
    """
    Train all four CNN variants and print a side-by-side comparison.
    Use this for the project report CNN section.
    """
    results = {}

    # ① Custom CNN
    m = CNN_Custom().to(DEVICE)
    results["CNN_Custom"] = train(m, "CNN_Custom", epochs)

    # ② LeNet-5
    m = CNN_LeNet5().to(DEVICE)
    results["CNN_LeNet5"] = train(m, "CNN_LeNet5", epochs)

    # ③ Transfer Learning (frozen)
    m = CNN_Pretrained(freeze_backbone=True).to(DEVICE)
    results["CNN_Transfer"] = train(m, "CNN_Transfer", epochs)

    # ④ Fine-tuning (all layers, layer-wise LR)
    m   = CNN_Finetune().to(DEVICE)
    opt = optim.Adam(m.get_param_groups(lr_backbone=1e-5, lr_head=1e-3))
    results["CNN_Finetune"] = train(m, "CNN_Finetune", epochs, optimizer=opt)

    # Summary table
    print(f"\n{'='*65}")
    print(f"  {'Model':<18} {'Best Acc':>10} {'Time (s)':>10} {'Params':>12}")
    print(f"{'='*65}")
    for name, res in results.items():
        print(f"  {name:<18} {res['best_acc']*100:>9.2f}% "
              f"{res['time']:>9.1f}  {res['params']:>12,}")
    print(f"{'='*65}")

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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
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

    ax1.set(title="CNN Variants – Validation Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set(title="CNN Variants – Validation Loss", xlabel="Epoch", ylabel="Loss")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "cnn_variant_comparison.png")
    plt.savefig(path, dpi=150); plt.show()
    print(f"  [Saved] {path}")


# ── Visualization: what the conv filters learned ──────────
def visualize_filters(model: CNN_Custom, save: bool = True):
    """Display the learned filters of the first convolutional layer."""
    first_conv = model.features[0]   # nn.Conv2d(1, 32, 3)
    weights    = first_conv.weight.data.cpu()  # [32, 1, 3, 3]
    n          = min(weights.shape[0], 32)

    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axes.flatten()):
        if i < n:
            ax.imshow(weights[i, 0], cmap="viridis", interpolation="nearest")
        ax.axis("off")

    plt.suptitle("Learned filters – Conv Layer 1 (CNN_Custom)", fontsize=12)
    plt.tight_layout()
    if save:
        path = os.path.join(PLOT_DIR, "cnn_conv1_filters.png")
        plt.savefig(path, dpi=150)
        print(f"  [Saved] {path}")
    plt.show()


# ── Inference helper ──────────────────────────────────────
@torch.no_grad()
def predict(model: nn.Module, image_tensor: torch.Tensor):
    """
    Predict a single 28x28 image.

    Args:
        model:        Trained CNN instance (any variant).
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
    # Train CNN_Custom (recommended single-model run)
    #result = train()
    #print(f"[Done] Best accuracy: {result['best_acc'] * 100:.2f}%")
        
    # Uncomment to compare all four variants:
    compare_all_variants()
