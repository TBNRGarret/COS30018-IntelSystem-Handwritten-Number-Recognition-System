import cv2
import numpy as np

def robust_global_preprocessing(image_bgr):
    """
    Task 1 (Phase 1): Xử lý ảnh có độ chênh lệch ánh sáng cực gắt.
    Sử dụng Adaptive Thresholding thay cho Global Otsu.
    """
    # 1. Grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Khử nhiễu mạnh tay (Gaussian Blur với kernel lớn)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # 3. ADAPTIVE THRESHOLDING (Vũ khí bí mật trị bóng râm/chói sáng)
    binary_map = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        blockSize=51, 
        C=15
    )
    
    # 4. Morphological Operations (Dọn dẹp tàn dư)
    kernel = np.ones((3,3), np.uint8)
    
    # Opening: Xóa các chấm trắng li ti còn sót lại
    cleaned_map = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Dilation: Làm nét chữ đậm lên một chút
    final_map = cv2.dilate(cleaned_map, kernel, iterations=1)
    
    return final_map, gray

def resize_for_display(img, max_height=800):
    """Hàm phụ trợ: Thu nhỏ ảnh trên màn hình để dễ xem"""
    h, w = img.shape[:2]
    if h > max_height:
        scale = max_height / h
        return cv2.resize(img, (int(w * scale), max_height), interpolation=cv2.INTER_AREA)
    return img

# ==========================================
# PHẦN MAIN ĐỂ CHẠY TRÊN VS CODE
# ==========================================
if __name__ == "__main__":
    # ---> ĐIỀN TÊN FILE ẢNH CỦA BẠN VÀO ĐÂY <---
    # Ví dụ: "image_e8ff42.jpg"
    image_path = r"D:\Swinburne\Sem 8 - 2026\COS30018 - Intelligent Systems\Project MNIST\abc.jpg"
    
    # 1. Đọc ảnh từ file
    img_bgr = cv2.imread(image_path)
    
    if img_bgr is None:
        print(f"[LỖI] Không tìm thấy ảnh tại: '{image_path}'")
        print("Hãy chắc chắn ảnh đang nằm cùng thư mục với file code Python này.")
    else:
        print(f"[THÀNH CÔNG] Đã load ảnh: {img_bgr.shape}")
        
        # 2. Chạy hàm tiền xử lý Adaptive
        final_binary, original_gray = robust_global_preprocessing(img_bgr)
        
        # 3. Lưu kết quả ra file
        cv2.imwrite("1_gray.jpg", original_gray)
        cv2.imwrite("2_adaptive_binary.jpg", final_binary)
        print(">> Đã lưu 2 ảnh kết quả ra thư mục.")

        # 4. Hiển thị lên màn hình
        cv2.imshow("1. Original Gray", resize_for_display(original_gray))
        cv2.imshow("2. Robust Adaptive Binary", resize_for_display(final_binary))
        
        print(">> Bấm phím BẤT KỲ trên bàn phím để đóng cửa sổ ảnh.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()