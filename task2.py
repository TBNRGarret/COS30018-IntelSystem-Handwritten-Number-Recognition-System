import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# TASK 1 (PHASE 2): SMART RESIZE & PADDING
# ==========================================
def normalize_to_mnist(digit_crop):
    """Giữ tỷ lệ ảnh, ép về 20px, sau đó thêm viền đen để ra đúng 28x28"""
    h, w = digit_crop.shape
    if h == 0 or w == 0:
        return np.zeros((28, 28), dtype=np.uint8)
        
    target_size = 28
    digit_size = 20
    scale = digit_size / max(w, h)
    
    new_w, new_h = int(w * scale), int(h * scale)
    resized_digit = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    final_28x28 = np.zeros((target_size, target_size), dtype=np.uint8)
    start_x = (target_size - new_w) // 2
    start_y = (target_size - new_h) // 2
    
    final_28x28[start_y:start_y+new_h, start_x:start_x+new_w] = resized_digit
    return final_28x28

# ==========================================
# TASK 2: CONNECTED COMPONENTS ANALYSIS (CCA)
# ==========================================
def segment_digits(binary_image):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )

    digits = []
    # Bắt đầu từ 1 để bỏ qua background (label 0)
    for i in range(1, num_labels):  
        x, y, w, h, area = stats[i]

        # ---------- Filtering rules (Giữ nguyên logic cực tốt của nhóm bạn) ----------
        area_min = 1000 # Điều chỉnh nhẹ area_min vì nét chữ có thể nhỏ
        ratio_min = 0.2 # Điều chỉnh lại ratio vì số '1' có thể có ratio h/w rất lớn
        ratio_max = 5.0
        
        img_h, img_w = binary_image.shape
        margin = 100 # Bỏ qua mọi vật thể nằm cách mép ảnh 20 pixels
        if x < margin or y < margin or (x + w) > (img_w - margin) or (y + h) > (img_h - margin):
            continue

        if area < area_min:
            continue
        if h < 12 or w < 4:
            continue
        ratio = h / w
        if ratio < ratio_min or ratio > ratio_max:
            continue

        # Cắt chữ số từ ảnh nhị phân
        digit_crop = binary_image[y:y+h, x:x+w]
        
        # SỬ DỤNG SMART RESIZE THAY VÌ NAIVE RESIZE
        mnist_ready_digit = normalize_to_mnist(digit_crop)
        
        digits.append((mnist_ready_digit, x, y, w, h))

    # Tạm thời sort theo X (từ trái qua phải)
    digits.sort(key=lambda d: d[1])
    return digits

# ==========================================
# THỰC THI (EXECUTION)
# ==========================================
if __name__ == "__main__":
    # KHÔNG CẦN CHẠY LẠI THRESHOLD NỮA! 
    # Load thẳng tấm ảnh nhị phân nền đen chữ trắng đã được Adaptive Threshold xử lý cực đẹp từ Phase 1
    # Thay đường dẫn này thành đường dẫn trỏ tới file '2_adaptive_binary.jpg' của bạn
    img_path = r"2_adaptive_binary.jpg" 
    
    binary_map = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if binary_map is None:
        print("Lỗi: Không tìm thấy file ảnh nhị phân!")
    else:
        # Chạy thuật toán cắt
        digits = segment_digits(binary_map)

        # Trực quan hóa kết quả cắt (Vẽ Bounding Box)
        img_bbox = cv2.cvtColor(binary_map, cv2.COLOR_GRAY2BGR)
        for i, (_, x, y, w, h) in enumerate(digits):
            cv2.rectangle(img_bbox, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_bbox, str(i+1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        plt.figure(figsize=(10,6))
        plt.imshow(cv2.cvtColor(img_bbox, cv2.COLOR_BGR2RGB))
        plt.title(f"CCA Detected {len(digits)} digits")
        plt.axis("off")
        plt.show()

        # Hiển thị từng chữ số ĐÃ ĐƯỢC CHUẨN HÓA 28x28 (Sẵn sàng đưa cho AI đoán)
        if len(digits) > 0:
            plt.figure(figsize=(15, 3))
            for i, (d, _, _, _, _) in enumerate(digits):
                plt.subplot(1, len(digits), i+1)
                plt.imshow(d, cmap="gray")
                plt.title(f"Digit {i+1}")
                plt.axis("off")
            plt.tight_layout()
            plt.show()