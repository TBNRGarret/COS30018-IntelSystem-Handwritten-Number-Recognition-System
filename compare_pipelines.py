import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image

def get_mnist_samples(num_samples=5):
    """Load raw MNIST dataset and return some sample images and labels."""
    data_path = './COS30018-IntelSystem-Handwritten-Number-Recognition-System/data'
    dataset = datasets.MNIST(root=data_path, train=True, download=True)
    samples = []
    for i in range(num_samples):
        img, label = dataset[i]
        samples.append((np.array(img), label))
    return samples

def deskew(img):
    """Deskew the image using moments."""
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * 28 * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (28, 28), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def apply_pipelines(img):
    """Apply various preprocessing pipelines to an image."""
    results = {}
    
    # 1. Raw
    results['Raw'] = img
    
    # 2. Baseline (Normalized 0-1)
    results['Baseline'] = img.astype(np.float32) / 255.0
    
    # 3. Otsu Thresholding
    _, thresh_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results['Otsu Thresh'] = thresh_otsu
    
    # 4. Blur + Otsu
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    _, blur_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results['Blur + Otsu'] = blur_otsu
    
    # 5. Morphological Ops (Opening to remove noise)
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(thresh_otsu, cv2.MORPH_OPEN, kernel)
    results['Morph Open'] = opening
    
    # 6. Dilation (Thickening)
    dilation = cv2.dilate(thresh_otsu, kernel, iterations=1)
    results['Dilation'] = dilation

    # 7. Deskewing
    deskewed = deskew(img)
    results['Deskewed'] = deskewed
    
    # 8. Deskewed + Otsu
    _, deskew_otsu = cv2.threshold(deskewed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results['Deskewed + Otsu'] = deskew_otsu

    return results

def visualize_comparison(num_samples=5):
    samples = get_mnist_samples(num_samples)
    
    # Apply pipelines to the first sample to get the list of pipelines
    first_img, _ = samples[0]
    pipeline_results = apply_pipelines(first_img)
    pipeline_names = list(pipeline_results.keys())
    
    num_pipelines = len(pipeline_names)
    fig, axes = plt.subplots(num_samples, num_pipelines, figsize=(num_pipelines * 2, num_samples * 2))
    
    if num_samples == 1:
        axes = [axes]

    for i, (img, label) in enumerate(samples):
        results = apply_pipelines(img)
        for j, name in enumerate(pipeline_names):
            ax = axes[i][j]
            ax.imshow(results[name], cmap='gray')
            if i == 0:
                ax.set_title(name, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"Label: {label}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.tight_layout()
    plt.savefig('preprocessing_comparison.png', dpi=150)
    print("Comparison visualization saved as 'preprocessing_comparison.png'")
    plt.show()

if __name__ == "__main__":
    visualize_comparison(6)
