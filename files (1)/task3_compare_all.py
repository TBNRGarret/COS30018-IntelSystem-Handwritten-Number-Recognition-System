"""
task3_compare_all.py – Task 3 master comparison script.

Trains Single Perceptron, MLP_Medium, and CNN_Custom in sequence,
then produces a unified comparison table and publication-ready plots.

Usage:
    python task3_compare_all.py

Output files (saved to ./plots/):
    task3_accuracy_comparison.png  – val accuracy curves + bar chart + scatter
    task3_confusion_matrices.png   – all three confusion matrices side by side
"""

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from config import DEVICE, EPOCHS, CKPT_DIR, PLOT_DIR
from data_loader import get_loaders

# Import model classes (not their train() functions – we run our own loop here
# to keep the comparison perfectly consistent: same data, same epochs).
from model_1_single_perceptron import SinglePerceptron
from model_2_mlp               import MLP_Medium
from model_3_cnn               import CNN_Custom


# ── Shared training/evaluation loop ──────────────────────
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


def run_experiment(model, model_name: str, epochs: int) -> dict:
    """
    Run a complete training + evaluation experiment for one model.
    Uses the same train/test loaders (MNIST downloaded once via data_loader.py).
    """
    # Perceptron benefits from no augmentation; MLP and CNN benefit from it.
    augment      = model_name != "Single Perceptron"
    train_loader, test_loader = get_loaders(augment=augment)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  ▶  {model_name}  ({n_params:,} params)")
    print(f"     {'Epoch':<8} {'Train Acc':>10} {'Val Acc':>10}")

    history  = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
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
            ckpt = os.path.join(CKPT_DIR, f"{model_name.lower().replace(' ', '_')}_compare.pth")
            torch.save(model.state_dict(), ckpt)

        print(f"     [{epoch:2d}/{epochs}]  {t_acc*100:>9.2f}%  {v_acc*100:>9.2f}%")

    elapsed = time.time() - t0
    return {
        "name":     model_name,
        "best_acc": best_acc,
        "history":  history,
        "preds":    best_preds,
        "labels":   best_labels,
        "time":     elapsed,
        "params":   n_params,
    }


# ── Main ──────────────────────────────────────────────────
def main():
    # Use a common epoch count so all models are compared fairly.
    # Perceptron converges fast, so fewer epochs is fine, but we keep
    # it uniform here – it just plateaus early.
    COMPARE_EPOCHS = 20

    experiments = [
        ("Single Perceptron", SinglePerceptron()),
        ("MLP Medium",        MLP_Medium()),
        ("CNN Custom",        CNN_Custom()),
    ]

    print("=" * 60)
    print("  TASK 3 – Model Comparison")
    print(f"  Epochs per model : {COMPARE_EPOCHS}")
    print(f"  Device           : {DEVICE}")
    print("=" * 60)

    all_results = {}
    for name, model in experiments:
        model = model.to(DEVICE)
        all_results[name] = run_experiment(model, name, COMPARE_EPOCHS)

    # ── Summary table ──────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  {'Model':<20} {'Best Acc':>10} {'Time (s)':>10} {'Params':>12}")
    print(f"{'='*65}")
    for name, res in all_results.items():
        print(f"  {name:<20} {res['best_acc']*100:>9.2f}%"
              f"  {res['time']:>9.1f}  {res['params']:>12,}")
    print(f"{'='*65}")

    # ── Detailed report for best model ────────────────────
    best = max(all_results.values(), key=lambda r: r["best_acc"])
    print(f"\n  Best model: {best['name']}  ({best['best_acc']*100:.2f}%)")
    print(classification_report(best["labels"], best["preds"],
                                target_names=[str(i) for i in range(10)]))

    # ── Plots ─────────────────────────────────────────────
    _plot_master_comparison(all_results)
    _plot_all_confusion_matrices(all_results)


# ── Visualization ─────────────────────────────────────────
def _plot_master_comparison(results: dict):
    """Three-panel figure: accuracy curves, bar chart, accuracy vs time scatter."""
    fig = plt.figure(figsize=(18, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    names  = list(results.keys())

    # Panel 1: Validation accuracy curves
    ax1 = fig.add_subplot(gs[0, 0])
    for (name, res), color in zip(results.items(), colors):
        ep = range(1, len(res["history"]["val_acc"]) + 1)
        ax1.plot(ep, [a * 100 for a in res["history"]["val_acc"]],
                 color=color, linewidth=2, label=name)
    ax1.set(title="Validation Accuracy Over Epochs",
            xlabel="Epoch", ylabel="Accuracy (%)")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    # Panel 2: Final accuracy bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    accs = [res["best_acc"] * 100 for res in results.values()]
    bars = ax2.bar(names, accs, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set(title="Best Validation Accuracy", ylabel="Accuracy (%)")
    ax2.set_ylim([85, 100])
    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.1,
                 f"{acc:.2f}%", ha="center", va="bottom", fontweight="bold")
    ax2.set_xticklabels(names, rotation=10, ha="right")
    ax2.grid(True, axis="y", alpha=0.3)

    # Panel 3: Accuracy vs training time (bubble size = param count)
    ax3 = fig.add_subplot(gs[0, 2])
    for (name, res), color in zip(results.items(), colors):
        ax3.scatter(
            res["time"], res["best_acc"] * 100,
            s=res["params"] / 500,       # scale bubble to visible range
            color=color, alpha=0.8, edgecolors="black", linewidth=1,
            label=f"{name}\n({res['params']:,} params)",
        )
    ax3.set(title="Accuracy vs Training Time\n(bubble = # params)",
            xlabel="Training Time (s)", ylabel="Accuracy (%)")
    ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3)

    plt.suptitle("Task 3 – Model Comparison: Perceptron vs MLP vs CNN",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "task3_accuracy_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [Saved] {path}")


def _plot_all_confusion_matrices(results: dict):
    """Confusion matrices for all three models in one figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmaps = ["Blues", "Greens", "Oranges"]

    for ax, (name, res), cmap in zip(axes, results.items(), cmaps):
        cm = confusion_matrix(res["labels"], res["preds"])
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                    xticklabels=range(10), yticklabels=range(10),
                    annot_kws={"size": 7})
        ax.set_title(f"{name}\n{res['best_acc']*100:.2f}%", fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")

    plt.suptitle("Confusion Matrices – All Models", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "task3_confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [Saved] {path}")


if __name__ == "__main__":
    main()
