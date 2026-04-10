import sys
import cv2
import matplotlib.pyplot as plt
from task1_preprocessing import preprocess, show_full_pipeline, compare_binarization_techniques
from task2 import segment_digits, visualise_bounding_boxes, visualise_patches
from file_picker import pick_image

path = pick_image("Select a handwritten number image")

if not path:
    print("No image selected.")
else:
    image = cv2.imread(path)
    if image is None:
        print(f"Error: Cannot load {path}")
        sys.exit(1)

    # ── Task 1 ──────────────────────────────────
    gray, binary = preprocess(image)

    # ── Task 2 ──────────────────────────────────
    digits = segment_digits(binary)

    visualise_bounding_boxes(gray, binary, digits)
    visualise_patches(digits)