"""
inference_pipeline.py – End-to-end inference: image → recognized number.

Ties together Task 1 (preprocessing), Task 2 (CCA segmentation),
and Task 3 (CNN digit recognition) into a single callable pipeline.

Usage:
    from inference_pipeline import recognize_number

    number_str, details = recognize_number("path/to/image.jpg")
    print(number_str)   # e.g. "2025"
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from config import DEVICE, CKPT_DIR, PLOT_DIR
from task1_preprocessing import to_grayscale
from task2  import segment_digits, visualise_patches
from model_3_cnn         import CNN_Custom


# ── Load trained model ────────────────────────────────────
def load_model(
    ckpt_path: str  = os.path.join(CKPT_DIR, "cnn_custom_best.pth"),
    device: torch.device = DEVICE,
) -> CNN_Custom:
    """Load CNN_Custom weights from checkpoint."""
    model = CNN_Custom().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"[Model] Loaded weights from {ckpt_path}")
    return model


# ── Single-digit inference ────────────────────────────────
@torch.no_grad()
def predict_digit(
    model:        CNN_Custom,
    patch:        np.ndarray,         # (28, 28) float32 [0, 1]
    device:       torch.device = DEVICE,
) -> tuple[int, float, np.ndarray]:
    """
    Run the model on a single preprocessed digit patch.

    Returns:
        predicted_digit   (int)
        confidence        (float)   softmax probability of predicted class
        all_probabilities (ndarray) shape (10,)
    """
    # (28, 28) → (1, 1, 28, 28) tensor
    tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)

    # Normalize to MNIST stats (model was trained with these)
    tensor = (tensor - 0.1307) / 0.3081

    logits = model(tensor)                   # (1, 10)
    probs  = torch.softmax(logits, dim=1)[0] # (10,)
    pred   = probs.argmax().item()
    return pred, probs[pred].item(), probs.cpu().numpy()


# ── Full pipeline ─────────────────────────────────────────
def recognize_number(
    image_source:  str | np.ndarray,
    model:         CNN_Custom = None,
    ckpt_path:     str = os.path.join(CKPT_DIR, "cnn_custom_best.pth"),
    visualize:     bool = True,
    save:          bool = False,
) -> tuple[str, list[dict]]:
    """
    End-to-end recognition of a handwritten multi-digit number.

    Args:
        image_source:  File path (str) or BGR numpy array.
        model:         Pre-loaded model (optional – avoids reloading on repeated calls).
        ckpt_path:     Path to model checkpoint file.
        visualize:     Display the annotated result.
        save:          Save visualization to PLOT_DIR.

    Returns:
        number_str : str           – recognized number, e.g. "2025"
        details    : list[dict]    – per-digit info (box, digit, confidence, probs)
    """
    # ── Load image ──
    if isinstance(image_source, str):
        image = cv2.imread(image_source)
        if image is None:
            raise FileNotFoundError(f"Cannot open image: {image_source}")
    else:
        image = image_source

    # ── Load model if not provided ──
    if model is None:
        model = load_model(ckpt_path)

    # ── Task 2: Segmentation (CCA) ──
    patches, boxes = segment_digits(image, visualize=False)

    if len(patches) == 0:
        print("[Warning] No digits detected. Check input image quality.")
        return "", []

    # ── Task 3: Digit recognition ──
    details = []
    for patch, box in zip(patches, boxes):
        digit, conf, probs = predict_digit(model, patch)
        details.append({
            "box":        box,
            "patch":      patch,
            "digit":      digit,
            "confidence": conf,
            "probs":      probs,
        })

    number_str = "".join(str(d["digit"]) for d in details)

    if visualize or save:
        _visualize_result(image, details, number_str, save)

    print(f"[Result] Recognized number: '{number_str}'")
    return number_str, details


# ── Result visualization ──────────────────────────────────
def _visualize_result(
    image:      np.ndarray,
    details:    list[dict],
    number_str: str,
    save:       bool = False,
):
    """
    Two-row figure:
        Row 1: Annotated original image with bounding boxes + predictions
        Row 2: Individual digit patches with digit label + confidence bar
    """
    n       = len(details)
    fig     = plt.figure(figsize=(max(12, 2.5 * n), 7))

    # ── Top: annotated image ──
    ax_main = fig.add_axes([0.05, 0.45, 0.90, 0.50])
    orig_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) \
               if image.ndim == 3 else image
    ax_main.imshow(orig_rgb, cmap="gray" if image.ndim == 2 else None)

    for i, d in enumerate(details):
        box  = d["box"]
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        pad  = 6
        rect = plt.Rectangle((x - pad, y - pad), w + 2*pad, h + 2*pad,
                              linewidth=2.5, edgecolor="lime", facecolor="none")
        ax_main.add_patch(rect)
        ax_main.text(box["cx"], y - pad - 4,
                     f"{d['digit']} ({d['confidence']*100:.0f}%)",
                     color="red", fontsize=10, fontweight="bold",
                     ha="center", va="bottom",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))

    ax_main.set_title(f"Recognized: \"{number_str}\"", fontsize=14, fontweight="bold")
    ax_main.axis("off")

    # ── Bottom: digit patches + confidence bars ──
    for i, d in enumerate(details):
        # Patch image
        ax_img = fig.add_axes([0.05 + i * (0.90/n), 0.08, 0.80/n, 0.30])
        ax_img.imshow(d["patch"], cmap="gray", vmin=0, vmax=1)
        ax_img.set_title(f"→ {d['digit']}\n{d['confidence']*100:.1f}%",
                         fontsize=9, fontweight="bold")
        ax_img.axis("off")

        # Confidence bar
        ax_bar = fig.add_axes([0.05 + i * (0.90/n), 0.03, 0.80/n, 0.06])
        colors = ["#3498db"] * 10
        colors[d["digit"]] = "#2ecc71"
        ax_bar.bar(range(10), d["probs"], color=colors)
        ax_bar.set_xticks(range(10))
        ax_bar.set_xticklabels(range(10), fontsize=6)
        ax_bar.set_ylim(0, 1)
        ax_bar.tick_params(axis="y", labelsize=5)

    plt.suptitle("HNRS – End-to-End Inference Result",
                 fontsize=12, fontweight="bold", y=1.01)

    if save:
        path = os.path.join(PLOT_DIR, f"inference_{number_str}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {path}")
    plt.show()


# ── Quick demo ────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        result, _ = recognize_number(img_path, visualize=True, save=True)
        print(f"Final answer: {result}")
    else:
        print("[INFO] Usage: python inference_pipeline.py path/to/image.jpg")
        print("[INFO] No image provided – running a quick sanity test on MNIST.")

        from torchvision import datasets, transforms
        from config import DATA_DIR

        dataset = datasets.MNIST(DATA_DIR, train=False, download=True,
                                 transform=transforms.ToTensor())

        # Pack 4 MNIST digits side-by-side into one synthetic number image
        chosen = [0, 2, 4, 9]    # sample indices
        strips = []
        for idx in chosen:
            t, _ = dataset[idx]
            arr  = (t.squeeze().numpy() * 255).astype(np.uint8)
            strips.append(arr)

        number_canvas = np.concatenate(strips, axis=1)   # (28, 112)
        number_bgr    = cv2.cvtColor(number_canvas, cv2.COLOR_GRAY2BGR)

        model  = load_model() if os.path.exists(
                     os.path.join(CKPT_DIR, "cnn_custom_best.pth")) else None
        if model:
            recognize_number(number_bgr, model=model, visualize=True, save=True)
        else:
            print("[Warning] No checkpoint found. Train a model first with model_3_cnn.py")
