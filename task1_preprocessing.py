"""

task1_preprocessing.py – Image preprocessing pipeline for HNRS (Task 1).

"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola

from config import PLOT_DIR, PREPROCESSING

TARGET_SIZE = PREPROCESSING["target_size"]


# ══════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════

def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image[:, :, :3]


# ══════════════════════════════════════════════════════════
# Paper region detection (crop border noise)
# ══════════════════════════════════════════════════════════

def crop_paper_from_bgr(bgr: np.ndarray, pad: int = 15) -> tuple[np.ndarray, tuple]:
    """
    Detect the brightest paper region and crop out dark borders

    """
    hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]

    blurred  = cv2.GaussianBlur(value, (31, 31), 0)
    _, coarse = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    
    k      = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 51))
    coarse = cv2.morphologyEx(coarse, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(coarse, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = bgr.shape[:2]
        return bgr, (0, 0, w, h)

    largest   = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    img_h, img_w = bgr.shape[:2]
    x1 = max(x - pad, 0);   y1 = max(y - pad, 0)
    x2 = min(x + w + pad, img_w);  y2 = min(y + h + pad, img_h)
    return bgr[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)

def remove_background_illumination(gray: np.ndarray) -> np.ndarray:
    """
    Divide each pixel by a blurred version of itself.
   
    """
    # Large blur 
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=51, sigmaY=51)

    
    norm = cv2.divide(gray.astype(np.float32), blur.astype(np.float32) + 1e-6)

    # Scale result back to 0–255
    norm = np.clip(norm * 128, 0, 255).astype(np.uint8)
    return norm



# ══════════════════════════════════════════════════════════
# Colour detection
# ══════════════════════════════════════════════════════════

MIN_MEAN_SATURATION = PREPROCESSING["min_mean_saturation"]
INK_SATURATION_THRESHOLD = PREPROCESSING["ink_saturation_threshold"]
INK_VALUE_THRESHOLD = PREPROCESSING["ink_value_threshold"]


def is_colorful_image(bgr: np.ndarray) -> bool:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean()) >= MIN_MEAN_SATURATION


def extract_colorful_number(bgr: np.ndarray) -> np.ndarray | None:
    """Isolate colored ink from paper using HSV saturation.
    Returns binary mask (0/255), or None if no colored ink found.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S   = hsv[:, :, 1]
    V   = hsv[:, :, 2]

    # Exclude white paper (low sat + high value) and dark shadows
    paper_mask  = (S < 30) & (V > 200)
    shadow_mask = V < 30

    ink_mask = (
        (S >= INK_SATURATION_THRESHOLD) &
        (V >= INK_VALUE_THRESHOLD) &
        ~paper_mask &
        ~shadow_mask
    ).astype(np.uint8) * 255

    if ink_mask.sum() == 0:
        return None 

    # Light cleanup
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN,  k_open,  iterations=1)
    mask = cv2.morphologyEx(mask,     cv2.MORPH_CLOSE, k_close, iterations=1)
    return mask


# ══════════════════════════════════════════════════════════
# Grayscale fallback pipeline (pencil/black pen)
# ══════════════════════════════════════════════════════════

CLAHE_CLIP_LIMIT = PREPROCESSING["clahe_clip_limit"]
CLAHE_TILE_GRID  = PREPROCESSING["clahe_tile_grid"]

def apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                             tileGridSize=CLAHE_TILE_GRID)
    return clahe.apply(gray)


def binarize_otsu(gray: np.ndarray, invert: bool = True) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    flags   = (cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) if invert \
              else (cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary = cv2.threshold(blurred, 0, 255, flags)
    return binary

def binarize_sauvola(gray: np.ndarray, window_size: int = 25, k: float = 0.2) -> np.ndarray:
    
    thresh_map = threshold_sauvola(gray, window_size=window_size, k=k)

    binary = (gray < thresh_map).astype(np.uint8) * 255
    return binary



def binarize_for_faint_pencil(gray: np.ndarray, use_sauvola: bool = False) -> np.ndarray:
    """
    Robust binarization for pencil/pen strokes.

    """
    otsu_mask = binarize_otsu(gray, invert=True)
    adaptive_cfg = PREPROCESSING["faint_pencil_adaptive"]
    adaptive_mask = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=adaptive_cfg["block_size"],
        C=adaptive_cfg["c"],
    )
    fused = cv2.bitwise_or(otsu_mask, adaptive_mask)
    if use_sauvola:
        sauvola_mask = binarize_sauvola(gray)
        fused = cv2.bitwise_or(fused, sauvola_mask)
    return fused


def detect_horizontal_lines(binary: np.ndarray) -> np.ndarray:
    """Detect horizontal lines using a wide rectangular kernel."""
    img_w = binary.shape[1]
    min_width = PREPROCESSING["line_detection"]["horizontal_min_width"]
    line_min_width = max(img_w // 3, min_width)
    kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (line_min_width, 1))
    line_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    dilate_k  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    return cv2.dilate(line_mask, dilate_k, iterations=1)


def detect_grid_lines(binary: np.ndarray) -> np.ndarray:
    """
    Detect both horizontal and vertical lines
    and merge them with bitwise OR.

    """
    img_h, img_w = binary.shape
    line_cfg = PREPROCESSING["line_detection"]

    # Horizontal lines
    kh        = cv2.getStructuringElement(
                    cv2.MORPH_RECT,
                    (max(img_w // 3, line_cfg["horizontal_min_width"]), 1),
                )
    h_lines   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kh, iterations=1)
    h_lines   = cv2.dilate(h_lines,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)),
                            iterations=1)

    # Vertical lines
    kv        = cv2.getStructuringElement(
                    cv2.MORPH_RECT,
                    (1, max(img_h // 3, line_cfg["vertical_min_height"])),
                )
    v_lines   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kv, iterations=1)
    v_lines   = cv2.dilate(v_lines,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)),
                            iterations=1)

    return cv2.bitwise_or(h_lines, v_lines)


def remove_noise_grayscale(binary: np.ndarray, has_grid: bool = False) -> np.ndarray:
    """Remove paper lines and small noise for the grayscale fallback pipeline."""
    line_mask = detect_grid_lines(binary) if has_grid \
                else detect_horizontal_lines(binary)
    no_lines  = cv2.subtract(binary, line_mask)

    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened  = cv2.morphologyEx(no_lines, cv2.MORPH_OPEN,  k_open,  iterations=1)
    closed  = cv2.morphologyEx(opened,   cv2.MORPH_CLOSE, k_close, iterations=1)
    return closed


# ══════════════════════════════════════════════════════════
# Task 1 output function  ← MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════

def preprocess(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    bgr = _ensure_bgr(image)

    if is_colorful_image(bgr):
        # ── COLOR PATH ─────────────────────────────────────────────
        color_pad = PREPROCESSING["crop_padding"]["color"]
        bgr_crop, _ = crop_paper_from_bgr(bgr, pad=color_pad)
        gray          = to_grayscale(bgr_crop)
        binary        = extract_colorful_number(bgr_crop)

        if binary is None or binary.sum() == 0:
            #fall back to grayscale
            gray_norm = remove_background_illumination(gray)
            enhanced  = apply_clahe(gray_norm)
            binary    = binarize_for_faint_pencil(enhanced)
            binary    = remove_noise_grayscale(binary)
    else:
        # ── GRAYSCALE FALLBACK ─────────────────────────────────────
        gray_full    = to_grayscale(bgr)
        gray_pad = PREPROCESSING["crop_padding"]["grayscale"]
        gray_crop, _ = crop_paper_from_bgr(
               cv2.cvtColor(gray_full, cv2.COLOR_GRAY2BGR), pad=gray_pad)
        gray         = to_grayscale(gray_crop)
        # Normalise illumination first, then enhance + binarize
        gray_norm    = remove_background_illumination(gray)
        enhanced     = apply_clahe(gray_norm)
        binary       = binarize_for_faint_pencil(enhanced)
        binary       = remove_noise_grayscale(binary)

    return gray, binary


# ══════════════════════════════════════════════════════════
# Smart Resize  (defined here, applied in Task 2)
# ══════════════════════════════════════════════════════════

def smart_resize(patch: np.ndarray, size: int = TARGET_SIZE,
                 padding_value: int = 0) -> np.ndarray:
    h, w = patch.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size), dtype=np.float32)
    scale    = size / max(h, w)
    new_h    = max(1, int(round(h * scale)))   
    new_w    = max(1, int(round(w * scale)))   
    resized  = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas   = np.full((size, size), padding_value, dtype=np.uint8)
    pad_top  = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas.astype(np.float32) / 255.0


def naive_resize(patch: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    return cv2.resize(patch, (size, size),
                      interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


# ══════════════════════════════════════════════════════════
# Visualisation helpers
# ══════════════════════════════════════════════════════════

def _simple_threshold(gray: np.ndarray, cutoff: int = 127) -> np.ndarray:
    _, binary = cv2.threshold(gray, cutoff, 255, cv2.THRESH_BINARY_INV)
    return binary

def _adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, blockSize=11, C=2)


def show_full_pipeline(image_path: str, save: bool = True):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    use_colour  = is_colorful_image(bgr)
    bgr_crop, _ = crop_paper_from_bgr(bgr, pad=15)
    gray        = to_grayscale(bgr_crop)

    if use_colour:
        sat_mask = extract_colorful_number(bgr_crop)
        binary   = sat_mask if (sat_mask is not None and sat_mask.sum() > 0) \
                   else remove_noise_grayscale(binarize_otsu(apply_clahe(gray)))

        # Visualise saturation channel
        hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
        sat_vis = hsv[:, :, 1]

        steps = [
            ("Step 0\nOriginal",                    cv2.cvtColor(bgr,      cv2.COLOR_BGR2RGB), False),
            ("Step 1\nPaper crop",                  cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB), False),
            ("Step 2\nSaturation channel (HSV)",    sat_vis,                                    True),
            ("Step 3\nSaturation mask\n[→ Task 2]", binary,                                     True),
        ]
        mode = "Colour / Saturation HSV"
    else:
        gray_norm = remove_background_illumination(gray)
        enhanced = apply_clahe(gray)
        binary_raw = binarize_for_faint_pencil(enhanced)
        lines    = detect_horizontal_lines(binary_raw)
        binary   = remove_noise_grayscale(binary_raw)
        steps = [
            ("Step 0\nOriginal",               cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), False),
            ("Step 1\nPaper crop",             gray,                                  True),
            ("Step 2\nCLAHE",                  enhanced,                              True),
            ("Step 3\nOtsu+Adaptive binary",   binary_raw,                            True),
            ("Step 4\nLine mask",              lines,                                 True),
            ("Step 5\nClean\n[→ Task 2]",      binary,                                True),
        ]
        mode = "Grayscale / Otsu"

    fig, axes = plt.subplots(1, len(steps), figsize=(5 * len(steps), 4))
    for ax, (title, img, is_gray) in zip(axes, steps):
        ax.imshow(img, cmap="gray" if is_gray else None)
        ax.set_title(title, fontsize=9, fontweight="bold",
                     color="darkgreen" if "Task 2" in title else "black")
        ax.axis("off")
    plt.suptitle(f"Task 1 – Full Pipeline  [{mode}]", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save:
        path = os.path.join(PLOT_DIR, "task1_pipeline.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {path}")
    plt.show()


def compare_binarization_techniques(image_path: str, save: bool = True):
    bgr  = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")
    gray     = to_grayscale(bgr)
    enhanced = apply_clahe(gray)
    _, sat_bin = preprocess(bgr)

    steps = [
        ("Original",                          cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), False),
        ("A. Grayscale only",                 gray,                                  True),
        ("B. Simple threshold\n(cutoff=127)", _simple_threshold(gray),               True),
        ("C. Adaptive threshold",             _adaptive_threshold(gray),             True),
        ("D. CLAHE + Otsu",                   remove_noise_grayscale(
                                                  binarize_otsu(enhanced)),           True),
        ("E. Otsu+Adaptive / Saturation\n[CHOSEN]", sat_bin,                         True),
    ]
    fig, axes = plt.subplots(1, len(steps), figsize=(22, 4))
    for ax, (title, img, is_gray) in zip(axes, steps):
        ax.imshow(img, cmap="gray" if is_gray else None)
        ax.set_title(title, fontsize=9, fontweight="bold",
                     color="darkgreen" if "CHOSEN" in title else "black")
        ax.axis("off")
    plt.suptitle("Task 1 – Binarization Technique Comparison", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save:
        path = os.path.join(PLOT_DIR, "task1_binarization_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {path}")
    plt.show()


def compare_resize_techniques(image_path: str, save: bool = True):
    bgr  = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")
    gray  = to_grayscale(bgr)
    naive = naive_resize(gray, TARGET_SIZE)
    smart = smart_resize(gray, TARGET_SIZE)
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title(f"Original ({gray.shape[1]}×{gray.shape[0]})", fontweight="bold")
    axes[1].imshow(naive, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Naive resize → {TARGET_SIZE}×{TARGET_SIZE}", fontweight="bold")
    axes[2].imshow(smart, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Smart resize → {TARGET_SIZE}×{TARGET_SIZE} [CHOSEN]",
                      fontweight="bold", color="darkgreen")
    for ax in axes: ax.axis("off")
    plt.suptitle("Resize Technique Comparison", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save:
        path = os.path.join(PLOT_DIR, "task1_resize_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {path}")
    plt.show()


def demo_on_mnist(save: bool = True):
    from torchvision import datasets, transforms
    from config import DATA_DIR
    dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True,
                             transform=transforms.ToTensor())
    sample_indices = [0, 1, 4, 7, 9]
    fig, axes = plt.subplots(2, len(sample_indices), figsize=(14, 5))
    for col, idx in enumerate(sample_indices):
        img_tensor, label = dataset[idx]
        img_np    = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
        img_large = cv2.resize(img_np, (84, 112))
        img_noisy = np.clip(img_large.astype(np.int16) +
                            np.random.randint(-25, 25, img_large.shape),
                            0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(img_noisy, cv2.COLOR_GRAY2BGR)
        gray, binary = preprocess(bgr)
        axes[0][col].imshow(img_noisy, cmap="gray")
        axes[0][col].set_title(f"Digit: {label}", fontsize=9)
        axes[0][col].axis("off")
        axes[1][col].imshow(binary, cmap="gray")
        axes[1][col].axis("off")
    axes[0][0].set_ylabel("Input", fontsize=9)
    axes[1][0].set_ylabel("Pipeline output", fontsize=9, color="darkgreen")
    plt.suptitle("Task 1 – Pipeline on MNIST Samples", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save:
        path = os.path.join(PLOT_DIR, "task1_mnist_demo.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {path}")
    plt.show()


if __name__ == "__main__":
    import sys
    from file_picker import pick_image
    path = sys.argv[1] if len(sys.argv) > 1 else pick_image(
        "Select a handwritten number image for Task 1 demo")
    if path:
        print(f"[Task 1] Processing: {path}")
        show_full_pipeline(path)
        compare_binarization_techniques(path)
    else:
        print("[INFO] No image selected - running MNIST demo.")
        demo_on_mnist()