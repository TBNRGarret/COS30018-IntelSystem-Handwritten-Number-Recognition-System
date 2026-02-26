import cv2
import matplotlib.pyplot as plt

img_path = "/Users/vev2024/downloads/IMG_8851.JPG"
img = cv2.imread(img_path, 0)

#Convert to gray image
#digit(White), background(Black)
_, binary = cv2.threshold(
    img, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# ==================== Connected Components Analysis (CCA) ====================
def segment_digits(binary_image):
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )

    digits = []

    for i in range(1, num_labels):  #remove vackground
        x, y, w, h, area = stats[i]

        # ---------- Filtering rules ----------
        area_min = 1500
        ratio_min = 0.7
        ratio_max = 5.0

        if area < area_min:
            continue
        if h < 12 or w < 8:
            continue
        ratio = h / w
        if ratio < ratio_min or ratio > ratio_max:
            continue

        digit = binary_image[y:y+h, x:x+w]
        digit = cv2.resize(digit, (28, 28))
        digits.append((digit, x, y, w, h))

    digits.sort(key=lambda d: d[1])
    return digits


digits = segment_digits(binary)

# ==================== VISUALIZATION ====================

img_bbox = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

# Draw bounding boxes and index numbers on detected digits
for i, (_, x, y, w, h) in enumerate(digits):
    cv2.rectangle(img_bbox, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img_bbox, str(i+1), (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
# Display segmentation result
plt.figure(figsize=(10,4))
plt.imshow(cv2.cvtColor(img_bbox, cv2.COLOR_BGR2RGB))
plt.title(f"Connected Components Analysis (CCA) ({len(digits)} digits)")
plt.axis("off")
plt.show()

# Display each extracted digit (28×28)
plt.figure(figsize=(12,3))
for i, (d, _, _, _, _) in enumerate(digits):
    plt.subplot(1, len(digits), i+1)
    plt.imshow(d, cmap="gray")
    plt.axis("off")

plt.suptitle("Extracted Digits (28×28)")
plt.show()