"""
file_picker.py – GUI file/folder picker utilities.

Uses tkinter.filedialog (Python standard library – no extra install needed).
Import these helpers anywhere in the project instead of hardcoding file paths.

Quick usage:
    from file_picker import pick_image, pick_images, pick_folder

    path   = pick_image()            # pick one image
    paths  = pick_images()           # pick multiple images at once
    folder = pick_folder()           # pick a folder (returns all images inside)
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import cv2
import numpy as np

# Supported image extensions
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")


def _get_root() -> tk.Tk:
    """Create a hidden Tk root window (required before any dialog)."""
    root = tk.Tk()
    root.withdraw()          # hide the empty Tk window
    root.attributes("-topmost", True)   # bring dialog to front
    return root


# ── Single image ──────────────────────────────────────────
def pick_image(
    title:      str = "Select an image",
    start_dir:  str = "",
) -> str | None:
    """
    Open a file-picker dialog and return the selected image path.

    Returns:
        Absolute path string, or None if the user cancelled.

    Example:
        path = pick_image()
        if path:
            img = cv2.imread(path)
    """
    root = _get_root()
    path = filedialog.askopenfilename(
        title       = title,
        initialdir  = start_dir or os.path.expanduser("~"),
        filetypes   = [
            ("Image files", " ".join(f"*{e}" for e in IMAGE_EXTENSIONS)),
            ("JPEG",        "*.jpg *.jpeg"),
            ("PNG",         "*.png"),
            ("All files",   "*.*"),
        ],
    )
    root.destroy()
    return path if path else None


# ── Multiple images ───────────────────────────────────────
def pick_images(
    title:      str = "Select one or more images",
    start_dir:  str = "",
) -> list[str]:
    """
    Open a multi-select file-picker and return a list of image paths.

    Returns:
        List of absolute path strings (empty list if cancelled).

    Example:
        paths = pick_images()
        for p in paths:
            img = cv2.imread(p)
    """
    root = _get_root()
    paths = filedialog.askopenfilenames(
        title       = title,
        initialdir  = start_dir or os.path.expanduser("~"),
        filetypes   = [
            ("Image files", " ".join(f"*{e}" for e in IMAGE_EXTENSIONS)),
            ("All files",   "*.*"),
        ],
    )
    root.destroy()
    return list(paths)


# ── Folder of images ──────────────────────────────────────
def pick_folder(
    title:      str = "Select a folder containing images",
    start_dir:  str = "",
) -> list[str]:
    """
    Open a folder-picker dialog and return all image paths inside that folder.
    Non-recursive (top-level files only).

    Returns:
        Sorted list of absolute image paths (empty list if cancelled or empty folder).

    Example:
        paths = pick_folder()
        for p in paths:
            img = cv2.imread(p)
    """
    root = _get_root()
    folder = filedialog.askdirectory(
        title      = title,
        initialdir = start_dir or os.path.expanduser("~"),
    )
    root.destroy()

    if not folder:
        return []

    paths = sorted(
        str(p) for p in Path(folder).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    print(f"[Picker] Found {len(paths)} image(s) in: {folder}")
    return paths


# ── Convenience: load directly ────────────────────────────
def load_image(
    title: str = "Select an image",
) -> tuple[np.ndarray, str] | tuple[None, None]:
    """
    Pick one image and load it with cv2.imread in a single call.

    Returns:
        (bgr_image, path)  or  (None, None) if cancelled.

    Example:
        img, path = load_image()
        if img is not None:
            cv2.imshow("preview", img)
    """
    path = pick_image(title)
    if path is None:
        print("[Picker] No file selected.")
        return None, None

    img = cv2.imread(path)
    if img is None:
        print(f"[Picker] Could not read image: {path}")
        return None, None

    print(f"[Picker] Loaded: {path}  shape={img.shape}")
    return img, path


def load_images(
    title: str = "Select images",
) -> list[tuple[np.ndarray, str]]:
    """
    Pick multiple images and load them all with cv2.imread.

    Returns:
        List of (bgr_image, path) tuples for successfully loaded files.
    """
    paths = pick_images(title)
    result = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            result.append((img, p))
            print(f"[Picker] Loaded: {p}  shape={img.shape}")
        else:
            print(f"[Picker] Warning – could not read: {p}")
    return result


# ── Demo ──────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Opening file picker... (select any image on your computer)")
    img, path = load_image("Pick any test image")

    if img is not None:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(6, 5))
        plt.imshow(rgb)
        plt.title(f"Loaded: {Path(path).name}\nShape: {img.shape}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    else:
        print("No image selected – demo cancelled.")
