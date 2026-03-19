"""
task1_preprocessing.py – Image preprocessing pipeline for HNRS (Task 1).

Responsibility of Task 1:
    Transform the raw input image into a clean binary representation
    that Task 2 (segmentation) can reliably run CCA on.

    Raw image  →  Grayscale  →  Binary mask  (Task 1 output)
                                     ↓
                              Task 2 uses this binary mask for CCA

Techniques investigated for pixel-value representation:
    A. Grayscale only            – keep soft intensity values, no thresholding
    B. Simple threshold          – fixed cutoff (e.g. pixel < 128 → foreground)
    C. Adaptive threshold        – local neighbourhood threshold (handles uneven lighting)
    D. Otsu's binarization [CHOSEN]
       – automatically finds the optimal global threshold by maximising
         inter-class variance between foreground (digit) and background pixels
       – paired with Gaussian blur to suppress noise before thresholding

Why Otsu's over the others:
    ┌────────────────────┬──────────────────────────────────────────────────┐
    │ Technique          │ Key limitation                                   │
    ├────────────────────┼──────────────────────────────────────────────────┤
    │ Grayscale only     │ Soft gradients confuse CCA boundary detection    │
    │ Simple threshold   │ Fixed cutoff fails when lighting/contrast varies │
    │ Adaptive threshold │ Sensitive to kernel size; noisy on smooth areas  │
    │ Otsu's          ✓ │ Data-driven; robust across different images with │
    │                    │ no manual tuning; single global threshold        │
    └────────────────────┴──────────────────────────────────────────────────┘

Note on Smart Resize:
    Smart Resize is NOT applied here. It is applied in Task 2 after CCA
    extracts each individual digit bounding box. Applying resize to the
    full image before segmentation would change the scale of every digit
    and break the CCA area-filtering thresholds.

Output of preprocess():
    (gray, binary)
      gray   – (H, W) uint8   grayscale image (kept for visualization / cropping)
      binary – (H, W) uint8   foreground=255, background=0  (fed to Task 2 CCA)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import PLOT_DIR

TARGET_SIZE = 28   # model input size; used only for the resize comparison demo


# ══════════════════════════════════════════════════════════
# Core preprocessing steps
# ══════════════════════════════════════════════════════════

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert any image to single-channel grayscale.

    Handles BGR (OpenCV default), BGRA (with alpha channel), and
    images already in grayscale format.

    Args:
        image: np.ndarray  shape (H, W, 3), (H, W, 4), or (H, W)  uint8

    Returns:
        np.ndarray  shape (H, W)  uint8
    """
    if image.ndim == 2:
        return image                                    # already grayscale
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) # drop alpha channel
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """
    Correct uneven illumination using CLAHE
    (Contrast Limited Adaptive Histogram Equalization).

    Problem it solves:
        Real-world photos are often unevenly lit — one corner bright,
        a shadow across part of the page, etc.  Otsu's method computes a
        single global threshold from the whole-image histogram.  When the
        image has a strong brightness gradient, that threshold is pulled
        toward the dominant region and digit strokes vanish entirely.

    How CLAHE works:
        The image is split into small tiles (tileGridSize).
        Each tile gets its own histogram equalisation, boosting local
        contrast.  clipLimit caps the amplification to avoid over-
        enhancing noise in flat regions.  Neighbouring tiles are blended
        with bilinear interpolation to remove tile boundary artefacts.

    Effect:
        After CLAHE, digit strokes are locally high-contrast regardless
        of the overall brightness gradient, so Otsu reliably separates
        ink from paper across the whole image.

    Args:
        gray: Grayscale image  (H, W)  uint8

    Returns:
        Illumination-normalised grayscale image  (H, W)  uint8
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def binarize_otsu(gray: np.ndarray, invert: bool = True) -> np.ndarray:
    """
    Produce a clean binary mask using:
        CLAHE  →  Gaussian blur  →  Otsu's global threshold

    Pipeline rationale:
        1. CLAHE (normalize_illumination)
               Equalise local contrast so digit strokes are visible even
               when the photo has uneven lighting or shadows.
        2. Gaussian blur (5×5)
               Suppress high-frequency noise so Otsu's threshold histogram
               is smooth and the split point is stable.
        3. Otsu's threshold
               Automatically find the optimal global cutoff that maximises
               inter-class variance between ink pixels and paper pixels.

    Why invert=True by default:
        Real-world images have dark ink on a light background.
        Inverting makes digits white (255) and background black (0),
        matching MNIST convention expected by the CNN model.

    Args:
        gray:   Grayscale image  (H, W)  uint8
        invert: If True output is digits=255 / background=0.

    Returns:
        Binary image  (H, W)  uint8  values in {0, 255}
    """
    equalised = normalize_illumination(gray)            # fix uneven lighting
    blurred   = cv2.GaussianBlur(equalised, (5, 5), 0) # suppress noise
    flags     = (cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) if invert \
                else (cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary = cv2.threshold(blurred, 0, 255, flags)
    return binary


# ══════════════════════════════════════════════════════════
# Comparison techniques (for report – not used in pipeline)
# ══════════════════════════════════════════════════════════

def _simple_threshold(gray: np.ndarray, cutoff: int = 127) -> np.ndarray:
    """Fixed global threshold – baseline comparison."""
    _, binary = cv2.threshold(gray, cutoff, 255, cv2.THRESH_BINARY_INV)
    return binary


def _adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """
    Local neighbourhood threshold – handles uneven illumination.
    Block size 11, C=2 are standard defaults.
    """
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=2
    )


# ══════════════════════════════════════════════════════════
# Task 1 output function  ← this is what Task 2 calls
# ══════════════════════════════════════════════════════════

def preprocess(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Task 1 preprocessing pipeline (chosen technique).

        BGR image
            → Grayscale
            → CLAHE  (normalize_illumination: fixes uneven lighting/shadows)
            → Gaussian blur  (noise suppression, inside binarize_otsu)
            → Otsu binary mask

    Returns both grayscale and binary so Task 2 can:
        • run CCA on `binary`  (digits=255 / background=0)
        • crop ROI patches from `binary` before smart resize

    Args:
        image: Raw input image (BGR, BGRA, or grayscale)  uint8

    Returns:
        gray   – (H, W) uint8  grayscale image (for visualisation)
        binary – (H, W) uint8  binary mask, digits=255 background=0
    """
    gray   = to_grayscale(image)
    binary = binarize_otsu(gray, invert=True)   # CLAHE + blur + Otsu inside
    return gray, binary


# ══════════════════════════════════════════════════════════
# Smart Resize  (defined here, applied in Task 2)
# ══════════════════════════════════════════════════════════

def smart_resize(
    patch:         np.ndarray,
    size:          int = TARGET_SIZE,
    padding_value: int = 0,
) -> np.ndarray:
    """
    Aspect-ratio-preserving resize with symmetric square padding.

    Applied in Task 2 to each individual digit crop AFTER CCA segmentation.
    Defined here because it is a preprocessing operation and imports are
    kept in task1 to avoid circular dependencies.

    Steps:
        1. Scale patch so its longest side = `size` pixels.
        2. Pad the shorter side symmetrically to produce (size × size).
        3. Normalize pixel values to [0.0, 1.0] (float32).

    Why not naive resize (cv2.resize directly to 28×28):
        Tall narrow digits like "1" would be squashed horizontally.
        Wide digits like "0" would be stretched vertically.
        Preserving the aspect ratio keeps the digit geometry intact.

    Args:
        patch:         Single-digit image crop  (H, W)  uint8
        size:          Target side length in pixels.
        padding_value: Fill value (0 = black background, matches MNIST).

    Returns:
        np.ndarray  (size, size)  float32  [0.0, 1.0]
    """
    h, w    = patch.shape[:2]
    scale   = size / max(h, w)
    new_h   = int(round(h * scale))
    new_w   = int(round(w * scale))
    resized = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas   = np.full((size, size), padding_value, dtype=np.uint8)
    pad_top  = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return canvas.astype(np.float32) / 255.0


def naive_resize(patch: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    """
    Direct resize to (size × size) – distorts aspect ratio.
    Included for comparison only; not used in the pipeline.
    """
    return cv2.resize(patch, (size, size),
                      interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


# ══════════════════════════════════════════════════════════
# Visualizations for project report
# ══════════════════════════════════════════════════════════

def compare_binarization_techniques(image_path: str, save: bool = True):
    """
    Six-panel comparison showing all investigated techniques plus the
    intermediate CLAHE step that makes Otsu work on real-world photos.

    Panels:
        Original | Grayscale | CLAHE (new) | Simple | Adaptive | Otsu [CHOSEN]

    Use for the Task 1 section of the project report.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    gray    = to_grayscale(image)
    clahe   = normalize_illumination(gray)   # intermediate step, shown for clarity

    steps = [
        ("Original",                           cv2.cvtColor(image, cv2.COLOR_BGR2RGB), False),
        ("A. Grayscale only",                  gray,                           True),
        ("B. Simple threshold\n(cutoff=127)", _simple_threshold(gray),        True),
        ("C. Adaptive threshold\n(local neighbourhood)", _adaptive_threshold(gray), True),
        ("D. CLAHE\n(illumination correction,\nintermediate step)", clahe,   True),
        ("E. CLAHE + Otsu\n[CHOSEN]",         binarize_otsu(gray),            True),
    ]

    fig, axes = plt.subplots(1, len(steps), figsize=(22, 4))
    for ax, (title, img, is_gray) in zip(axes, steps):
        ax.imshow(img, cmap="gray" if is_gray else None)
        color = "darkgreen" if "[CHOSEN]" in title else "black"
        ax.set_title(title, fontsize=8.5, fontweight="bold", color=color)
        ax.axis("off")

    plt.suptitle("Task 1 – Binarization Technique Comparison\n"
                 "(Output used as input to Task 2 CCA)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(PLOT_DIR, "task1_binarization_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {path}")
    plt.show()


def compare_resize_techniques(image_path: str, save: bool = True):
    """
    Side-by-side comparison of naive resize vs smart resize on a digit crop.
    Applied AFTER CCA in Task 2 – shown here for completeness in the report.

    Args:
        image_path: Path to an image containing a single digit.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    gray = to_grayscale(image)

    naive = naive_resize(gray, TARGET_SIZE)
    smart = smart_resize(gray, TARGET_SIZE)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title(f"Original crop\n({gray.shape[1]}×{gray.shape[0]})",
                      fontweight="bold")

    axes[1].imshow(naive, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"C. Naive resize → {TARGET_SIZE}×{TARGET_SIZE}\n"
                      f"(distorts aspect ratio)", fontweight="bold")

    axes[2].imshow(smart, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"D. Smart resize → {TARGET_SIZE}×{TARGET_SIZE}\n"
                      f"[CHOSEN] (preserves aspect ratio)", fontweight="bold",
                      color="darkgreen")

    for ax in axes:
        ax.axis("off")

    plt.suptitle("Resize Technique Comparison\n"
                 "(applied in Task 2 per-digit crop)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(PLOT_DIR, "task1_resize_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {path}")
    plt.show()


def show_full_pipeline(image_path: str, save: bool = True):
    """
    Visualise the complete Task 1 pipeline on a real image:
        Original → Grayscale → CLAHE → Otsu Binary (Task 1 output)

    Args:
        image_path: Path to a real handwritten number image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    gray, binary = preprocess(image)
    clahe        = normalize_illumination(gray)   # intermediate step for display

    steps = [
        ("Step 0 – Original input",      cv2.cvtColor(image, cv2.COLOR_BGR2RGB), False),
        ("Step 1 – Grayscale",           gray,   True),
        ("Step 2 – CLAHE\n(illumination normalisation)", clahe, True),
        ("Step 3 – Otsu binary\n[Task 1 output → Task 2]", binary, True),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (title, img, is_gray) in zip(axes, steps):
        ax.imshow(img, cmap="gray" if is_gray else None)
        color = "darkgreen" if "Task 1 output" in title else "black"
        ax.set_title(title, fontsize=9.5, fontweight="bold", color=color)
        ax.axis("off")

    # Arrow annotations between panels
    for xpos in [0.265, 0.51, 0.755]:
        fig.text(xpos, 0.5, "→", fontsize=24, ha="center", va="center")

    plt.suptitle("Task 1 – Preprocessing Pipeline (Chosen Technique)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(PLOT_DIR, "task1_pipeline.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {path}")
    plt.show()


def demo_on_mnist(save: bool = True):
    """
    Demonstrate Task 1 preprocessing on MNIST samples (no external image needed).
    Shows how binarization affects different digit shapes.
    """
    from torchvision import datasets, transforms
    from config import DATA_DIR

    dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True,
                             transform=transforms.ToTensor())

    sample_indices = [0, 1, 4, 7, 9]   # diverse digit shapes
    fig, axes = plt.subplots(3, len(sample_indices), figsize=(14, 8))
    col_titles = ["Grayscale only", "Simple threshold", "Adaptive threshold",
                  "Otsu [CHOSEN]"]

    for col, idx in enumerate(sample_indices):
        img_tensor, label = dataset[idx]
        # Tensor [1,28,28] → uint8 [28,28], then upscale to simulate real photo
        img_np    = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
        img_large = cv2.resize(img_np, (84, 112))        # non-square, as in real life
        # Add noise to simulate camera capture
        img_noisy = np.clip(
            img_large.astype(np.int16) + np.random.randint(-25, 25, img_large.shape),
            0, 255
        ).astype(np.uint8)

        gray     = img_noisy                              # already grayscale
        variants = [
            gray,
            _simple_threshold(gray),
            _adaptive_threshold(gray),
            binarize_otsu(gray),
        ]

        # Row 0: column headers (digit labels)
        axes[0][col].imshow(img_noisy, cmap="gray")
        axes[0][col].set_title(f"Digit: {label}", fontsize=9, fontweight="bold")
        axes[0][col].axis("off")

        for row, (variant, title) in enumerate(zip(variants, col_titles)):
            if col == 0:
                axes[row + 1][col].set_ylabel(title, fontsize=8, rotation=15,
                                              labelpad=55)
            # Only show 3 technique rows (skip grayscale row for space)
            pass

        for row in range(len(variants)):
            axes[row][col].axis("off")

    # Cleaner 2-row layout: grayscale input + otsu output
    fig2, axes2 = plt.subplots(2, len(sample_indices), figsize=(14, 5))
    for col, idx in enumerate(sample_indices):
        img_tensor, label = dataset[idx]
        img_np    = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
        img_large = cv2.resize(img_np, (84, 112))
        img_noisy = np.clip(
            img_large.astype(np.int16) + np.random.randint(-25, 25, img_large.shape),
            0, 255
        ).astype(np.uint8)

        gray, binary = preprocess(img_noisy[..., np.newaxis]
                                  if img_noisy.ndim == 2 else
                                  cv2.cvtColor(img_noisy, cv2.COLOR_GRAY2BGR))

        axes2[0][col].imshow(img_noisy, cmap="gray")
        axes2[0][col].set_title(f"Digit: {label}", fontsize=9)
        axes2[0][col].axis("off")

        axes2[1][col].imshow(binary, cmap="gray")
        axes2[1][col].axis("off")

    axes2[0][0].set_ylabel("Step 1\nGrayscale", fontsize=9)
    axes2[1][0].set_ylabel("Step 2\nOtsu Binary\n[CHOSEN]", fontsize=9, color="darkgreen")

    plt.suptitle("Task 1 – Chosen Pipeline: Grayscale → Otsu Binary",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    plt.close(fig)   # close the unused first figure

    if save:
        path = os.path.join(PLOT_DIR, "task1_mnist_demo.png")
        fig2.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {path}")
    plt.show()


# ══════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from file_picker import pick_image

    path = sys.argv[1] if len(sys.argv) > 1 else pick_image(
        "Select a handwritten number image for Task 1 demo"
    )

    if path:
        print(f"[Task 1] Processing: {path}")
        show_full_pipeline(path)
        compare_binarization_techniques(path)
    else:
        print("[INFO] No image selected – running MNIST demo.")
        demo_on_mnist()