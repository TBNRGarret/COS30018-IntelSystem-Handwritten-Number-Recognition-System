"""
task2.py – Digit segmentation via Connected Component Analysis (CCA).

Pipeline position:
    Task 1 (preprocess)  →  Task 2 (segment)  →  Task 3 (model predict)

    Task 1 output:  gray   (H, W) uint8   – grayscale image
                    binary (H, W) uint8   – digits=255, background=0
    Task 2 input:   binary  ← passed directly in memory (no disk I/O)
    Task 2 output:  list of (mnist_patch, x, y, w, h) sorted left→right
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ── Link to Task 1 ─────────────────────────────────────────
from task1_preprocessing import preprocess, smart_resize

# ── Filtering constants──────────────────────────────────────
ASPECT_RATIO_MIN   = 0.15   
ASPECT_RATIO_MAX   = 6.0   
FILL_RATIO_MAX     = 0.92   
FILL_RATIO_H_TH    = 12     
MAX_DIGIT_W_RATIO  = 0.28 


# ══════════════════════════════════════════════════════════
# Wide-component splitter (vertical projection valleys)
# ══════════════════════════════════════════════════════════

def _try_split_wide_component(
    binary_image: np.ndarray,
    x: int, y: int, w: int, h: int,
    min_width: int = 6,
    min_area:  int = 200,
    debug: bool = False,
) -> list[tuple]:
    """
    Split a wide connected component by finding vertical projection valleys.

    Returns list of (sx, sy, sw, sh) segments, or [] if cannot split cleanly.
    """
    crop = binary_image[y:y + h, x:x + w].copy()
    if crop.size == 0:
        return []

    # Erode horizontally to break thin bridges between touching digits
    h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    eroded = cv2.erode(crop, h_kern, iterations=1)

    # Vertical projection: how many white pixels per column
    col_sum = np.sum(eroded == 255, axis=0).astype(np.float32)
    if col_sum.max() <= 0:
        return []

    # Smooth to reduce noise in the projection curve
    ksize = min(11, max(3, (w // 15) | 1)) 
    col_blur = cv2.GaussianBlur(col_sum.reshape(1, -1), (1, ksize), 0).flatten()
    norm = col_blur / (col_blur.max() + 1e-9)

    # Adaptive valley threshold based on estimated number of merged digits
    num_expected = max(1, int(round(w / max(h * 0.65, 1))))
    valley_thresh = max(0.12, min(0.50, 0.25 + 0.07 * (num_expected - 1)))

    # Find all local minima below threshold
    candidates = [
        i for i in range(1, w - 1)
        if norm[i] < valley_thresh
        and norm[i] <= norm[i - 1]
        and norm[i] <= norm[i + 1]
    ]

    # Last resort: if no valley found, try the deepest point in the middle half
    if not candidates:
        mid_start = w // 4
        mid_end   = 3 * w // 4
        if mid_end > mid_start:
            mid_vals = [(i, norm[i]) for i in range(mid_start, mid_end)]
            best_i, best_v = min(mid_vals, key=lambda t: t[1])
            if best_v < 0.55:
                candidates = [best_i]

    if not candidates:
        return []

    # Merge nearby candidates into one cut point
    merged = []
    for idx in candidates:
        if not merged or idx - merged[-1] > max(2, min_width // 2):
            merged.append(idx)

    # Build segments from cut points
    boundaries = [0] + merged + [w]
    segments = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        sw = b - a
        if sw < min_width:
            continue
        seg_crop = crop[:, a:b]
        seg_area = int(np.sum(seg_crop == 255))
        if seg_area < min_area:
            continue
        # Reject if the resulting segment has an impossible digit aspect ratio
        seg_ratio = h / sw if sw > 0 else 99
        if seg_ratio > ASPECT_RATIO_MAX or seg_ratio < ASPECT_RATIO_MIN:
            continue
        segments.append((x + a, y, sw, h))

    if len(segments) <= 1:
        return []

    if debug:
        print(f"  [Split] @({x},{y}) w={w} h={h} → {len(segments)} parts")
    return segments


# ══════════════════════════════════════════════════════════
# Watershed fallback splitter
# ══════════════════════════════════════════════════════════

def _watershed_split_component(
    binary_image: np.ndarray,
    x: int, y: int, w: int, h: int,
    min_area: int = 200,
    debug: bool = False,
) -> list[tuple]:
    """
    Fallback: split a wide blob using distance transform + watershed.
    Returns list of (sx, sy, sw, sh) or [] if cannot split.
    """
    crop = binary_image[y:y + h, x:x + w].copy()
    if crop.size == 0:
        return []

    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg = (cv2.morphologyEx(crop, cv2.MORPH_OPEN, k) == 255).astype(np.uint8)
    if fg.sum() == 0:
        return []

    dt = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    if dt.max() <= 0:
        return []
    dt_norm = dt / dt.max()
    _, peaks = cv2.threshold((dt_norm * 255).astype(np.uint8), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    peaks = cv2.morphologyEx(peaks, cv2.MORPH_OPEN, k)

    num_markers, markers = cv2.connectedComponents(peaks)
    if num_markers <= 1:
        return []

    markers = markers.astype(np.int32)
    mask3   = cv2.cvtColor((fg * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    try:
        cv2.watershed(mask3, markers)
    except Exception:
        return []

    segments = []
    for lab in range(1, int(markers.max()) + 1):
        seg = (markers == lab).astype(np.uint8)
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            sx, sy, sw, sh = cv2.boundingRect(c)
            area = int(cv2.countNonZero(seg[sy:sy+sh, sx:sx+sw]))
            seg_ratio = sh / sw if sw > 0 else 99
            if (area >= min_area and sw >= 4
                    and ASPECT_RATIO_MIN <= seg_ratio <= ASPECT_RATIO_MAX):
                segments.append((x + sx, y + sy, sw, sh))

    if len(segments) <= 1:
        return []
    if debug:
        print(f"  [Watershed] @({x},{y}) w={w} h={h} → {len(segments)} parts")
    return segments


# ══════════════════════════════════════════════════════════
# Reading-order sort (top-to-bottom, left-to-right)
# ══════════════════════════════════════════════════════════

def _sort_reading_order(digits: list, row_tolerance: float = 0.55) -> list:
    """
    Group digits into rows by vertical centre proximity, then sort each
    row left-to-right. Returns a flat list in reading order.
    """
    if not digits:
        return digits

    with_cy = [(p, x, y, w, h, y + h // 2) for p, x, y, w, h in digits]
    avg_h   = max(1, sum(e[4] for e in with_cy) / len(with_cy))
    thresh  = avg_h * row_tolerance

    with_cy.sort(key=lambda e: e[5])

    rows, cur, cy_ref = [], [with_cy[0]], with_cy[0][5]
    for e in with_cy[1:]:
        if abs(e[5] - cy_ref) <= thresh:
            cur.append(e)
        else:
            rows.append(cur)
            cur, cy_ref = [e], e[5]
    rows.append(cur)

    result = []
    for row in rows:
        for p, x, y, w, h, _ in sorted(row, key=lambda e: e[1]):
            result.append((p, x, y, w, h))
    return result


# ══════════════════════════════════════════════════════════
# Main segmentation function-CCA
# ══════════════════════════════════════════════════════════

def segment_digits(binary_image: np.ndarray, debug: bool = False) -> list[tuple]:
    """
    Run CCA on binary_image and return a list of
    (mnist_patch, x, y, w, h) tuples sorted in reading order.

    """
    img_h, img_w = binary_image.shape
    img_area     = img_h * img_w

    min_area    = max(120,  min(600,  int(img_area * 0.00020)))
    min_h       = max(6,    min(35,   int(img_h    * 0.015)))
    min_w       = max(4,    min(18,   int(img_w    * 0.006)))
    max_digit_w = max(60,   int(img_w * MAX_DIGIT_W_RATIO))
    edge_margin = max(3,    min(15,   int(min(img_h, img_w) * 0.008)))

    # ── Stroke thickening────────────────────────
    dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    detect_binary = cv2.dilate(binary_image, dk, iterations=1)

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        detect_binary, connectivity=8
    )

    if debug:
        print(f"[Seg] image={img_w}x{img_h}  "
              f"min_area={min_area}  min_h={min_h}  min_w={min_w}  "
              f"max_digit_w={max_digit_w}  edge={edge_margin}")

    digits = []

    for i in range(1, num_labels):          # label 0 = background
        x, y, w, h, area = (int(v) for v in stats[i])

        # 1 ── Area ───────────────────────────────────────────────────
        if area < min_area:
            continue

        # 2 ── Size ───────────────────────────────────────────────────
        if h < min_h or w < min_w:
            continue

        # 3 ── Aspect ratio───────────────────────────────────────────
        ratio = h / w
        if ratio < ASPECT_RATIO_MIN or ratio > ASPECT_RATIO_MAX:
            continue

        # 4 ── Wide component: try to split ───────────────────────────
        if w > max_digit_w:
            splits = _try_split_wide_component(
                binary_image, x, y, w, h,
                min_width=min_w, min_area=min_area, debug=debug)

            if not splits:
                splits = _watershed_split_component(
                    binary_image, x, y, w, h,
                    min_area=min_area, debug=debug)

            if splits:
                for sx, sy, sw, sh in splits:
                    crop = binary_image[sy:sy + sh, sx:sx + sw]
                    if crop.size > 0:
                        digits.append((smart_resize(crop, 28), sx, sy, sw, sh))
            else:
                if debug:
                    print(f"  [Dropped] unsplittable wide blob w={w} h={h}")
            continue

        # 5 ── Fill ratio (flat ruled-line rejection) ──────────────────
        fill = area / (w * h)
        if h < FILL_RATIO_H_TH and fill > FILL_RATIO_MAX:
            continue

        # 6 ── Edge margin (ignore components touching the border) ─────
        if (x < edge_margin or y < edge_margin
                or (x + w) > (img_w - edge_margin)
                or (y + h) > (img_h - edge_margin)):
            continue

        # 7 ── Noise blob rejection (irregular smears / scratches) ─────
        roi = binary_image[y:y + h, x:x + w]
        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            largest_cnt = max(cnts, key=cv2.contourArea)
            hull_area   = cv2.contourArea(cv2.convexHull(largest_cnt))
            cnt_area    = cv2.contourArea(largest_cnt)
            if hull_area > 0 and (cnt_area / hull_area) < 0.15:
                if debug:
                    print(f"  [Dropped] noise blob @({x},{y}) "
                          f"solidity={cnt_area/hull_area:.2f}")
                continue

        # ── Accepted digit ────────────────────────────────────────────
        crop = binary_image[y:y + h, x:x + w]
        digits.append((smart_resize(crop, 28), x, y, w, h))

    if debug:
        print(f"[Seg] {len(digits)} digit(s) accepted")

    return _sort_reading_order(digits)


# ══════════════════════════════════════════════════════════
# Full pipeline helper
# ══════════════════════════════════════════════════════════

def run_pipeline(image: np.ndarray) -> tuple[list, np.ndarray, np.ndarray]:
    gray, binary = preprocess(image)
    digits       = segment_digits(binary)
    return digits, gray, binary


# ══════════════════════════════════════════════════════════
# Visualisation helpers
# ══════════════════════════════════════════════════════════

def visualise_bounding_boxes(
    gray:   np.ndarray,
    binary: np.ndarray,
    digits: list,
    title:  str = "CCA Segmentation Result",
) -> None:
    display = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for i, (_, x, y, w, h) in enumerate(digits):
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            display, str(i + 1),
            (x, max(y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    plt.title(f"{title}  –  {len(digits)} digit(s) detected")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualise_patches(digits: list) -> None:
    """Display each normalised 28×28 patch side by side."""
    if not digits:
        print("[Task 2] No digits detected.")
        return
    n = len(digits)
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 3))
    if n == 1:
        axes = [axes]
    for ax, (patch, x, y, w, h) in zip(axes, digits):
        ax.imshow(patch, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"x={x}", fontsize=9)
        ax.axis("off")
    plt.suptitle("Task 2 Output – Normalised 28×28 Patches",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from file_picker import pick_image

    path = sys.argv[1] if len(sys.argv) > 1 else pick_image(
        "Select a handwritten number image for Task 2 demo"
    )

    if not path:
        print("[INFO] No image selected.")
    else:
        print(f"[Task 2] Processing: {path}")
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {path}")

        digits, gray, binary = run_pipeline(image)
        print(f"[Task 2] {len(digits)} digit(s) detected.")
        visualise_bounding_boxes(gray, binary, digits)
        visualise_patches(digits)