import cv2
import numpy as np
import os

# ==============================
# PARAMETERS (tunable)
# ==============================

INPUT_IMAGES = [
    "input_digits_1.png",
    "input_digits_2.png",
    "input_digits_3.png",
    "input_digits_4.jpg",
    "input_digits_5.jpg"
]

OUTPUT_FOLDER = "segmented_digits"

MIN_COMPONENT_AREA = 200
DIGIT_SIZE = (28, 28)

USE_ADAPTIVE_THRESHOLD = False


# ==============================
# CREATE OUTPUT FOLDER
# ==============================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==============================
# PROCESS EACH IMAGE
# ==============================

for img_index, image_path in enumerate(INPUT_IMAGES):

    image = cv2.imread(image_path)

    if image is None:
        print(f"Skipping {image_path} (not found)")
        continue

    print(f"\nProcessing {image_path}")

    # ==============================
    # STEP 1 – GRAYSCALE
    # ==============================

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ==============================
    # STEP 2 – BLUR
    # ==============================

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # ==============================
    # STEP 3 – BINARIZATION
    # ==============================

    if USE_ADAPTIVE_THRESHOLD:

        binary = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )

    else:

        _, binary = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

    # ==============================
    # STEP 4 – ENSURE DIGITS WHITE
    # ==============================

    white_pixels = np.sum(binary == 255)
    black_pixels = np.sum(binary == 0)

    if white_pixels > black_pixels:
        binary = cv2.bitwise_not(binary)

    # ==============================
    # STEP 5 – MORPHOLOGICAL CLEANUP
    # ==============================

    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # ==============================
    # STEP 6 – CONNECTED COMPONENTS
    # ==============================

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    digits = []

    # ==============================
    # STEP 7 – FILTER COMPONENTS
    # ==============================

    for i in range(1, num_labels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < MIN_COMPONENT_AREA:
            continue

        digits.append((x, y, w, h))

    # ==============================
    # STEP 8 – SORT LEFT → RIGHT
    # ==============================

    digits = sorted(digits, key=lambda d: d[0])

    # ==============================
    # STEP 9 – CROP DIGITS
    # ==============================

    digit_images = []

    for i, (x, y, w, h) in enumerate(digits):

        digit = binary[y:y+h, x:x+w]

        digit_resized = cv2.resize(digit, DIGIT_SIZE)

        digit_images.append(digit_resized)

        output_path = f"{OUTPUT_FOLDER}/img{img_index}_digit_{i}.png"
        cv2.imwrite(output_path, digit_resized)

    # ==============================
    # STEP 10 – VISUALIZATION
    # ==============================

    display = image.copy()

    for (x, y, w, h) in digits:
        cv2.rectangle(display, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow(f"Original Image {img_index}", image)
    cv2.imshow(f"Binary Image {img_index}", binary)
    cv2.imshow(f"Detected Digits {img_index}", display)

    print(f"{len(digit_images)} digits detected in {image_path}")

# ==============================
# FINAL WAIT
# ==============================

cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nSegmentation complete.")
