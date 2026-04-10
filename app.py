import cv2
import numpy as np
import base64
import uvicorn
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from task1_preprocessing import preprocess as run_preprocess
from task2 import segment_digits
from inference_pipeline import load_model, predict_digit

# ==============================
# APP SETUP
# ==============================

# Build paths relative to this file (backend/app.py)
# so the app works regardless of which directory you run it from
BASE_DIR     = Path(__file__).parent         # .../backend/
FRONTEND_DIR = BASE_DIR                      # frontend files are in root directory too

app = FastAPI(title="HNRS – Handwritten Number Recognition System")

# Serve index.html and style.css from the root folder
app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")


# ==============================
# PARAMETERS & GLOBAL MODEL
# ==============================

DEFAULT_MIN_COMPONENT_AREA = 100
DEFAULT_DIGIT_SIZE         = (28, 28)
DEFAULT_USE_ADAPTIVE       = False

model = None

@app.on_event("startup")
def startup_event():
    global model
    ckpt_path = BASE_DIR / "checkpoints" / "cnn_custom_best.pth"
    if ckpt_path.exists():
        import torch
        model = load_model(str(ckpt_path), device=torch.device("cpu"))
        print(f"[App] Model loaded successfully from {ckpt_path}")
    else:
        print("[App] Warning: Model checkpoint not found. Inference will fail until a model is trained.")


# ==============================
# SERVE FRONTEND
# ==============================

@app.get("/")
def root():
    return FileResponse(FRONTEND_DIR / "index.html")


# ==============================
# HELPER – encode cv2 image → base64 string
# so it can be sent as JSON and displayed in HTML
# ==============================

def encode_image(cv2_img: np.ndarray) -> str:
    _, buffer = cv2.imencode(".png", cv2_img)
    return base64.b64encode(buffer).decode("utf-8")


# ==============================
# HELPER – decode uploaded bytes → cv2 image
# ==============================

def decode_upload(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return image


# ==============================
# TASK 2 – SEGMENTATION LOGIC
# (adapted from ImageSegmentation.py)
# Changes from original:
#   - removed cv2.imshow() calls (no display on server)
#   - removed file I/O (no disk writes; results returned as JSON)
#   - image comes from HTTP upload instead of hardcoded path list
#   - parameters accepted as function arguments instead of globals
# ==============================

def run_segmentation(
    image:                np.ndarray,
    min_component_area:   int  = DEFAULT_MIN_COMPONENT_AREA,
    digit_size:           tuple = DEFAULT_DIGIT_SIZE,
    use_adaptive:         bool = DEFAULT_USE_ADAPTIVE,
) -> dict:
    """
    Run the full segmentation pipeline on a single image.
    Returns a dict with intermediate images (base64) and detected digit crops.
    """

    # ── STEP 1 – GRAYSCALE ──────────────────────────────────────────────────
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ── STEP 2 – BLUR ───────────────────────────────────────────────────────
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # ── STEP 3 – BINARIZATION ───────────────────────────────────────────────
    if use_adaptive:
        binary = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )
    else:
        _, binary = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

    # ── STEP 4 – ENSURE DIGITS ARE WHITE ON BLACK ───────────────────────────
    white_pixels = np.sum(binary == 255)
    black_pixels = np.sum(binary == 0)
    if white_pixels > black_pixels:
        binary = cv2.bitwise_not(binary)

    # ── STEP 5 – MORPHOLOGICAL CLEANUP ──────────────────────────────────────
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # ── STEP 6 – CONNECTED COMPONENTS ───────────────────────────────────────
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    # ── STEP 7 – FILTER COMPONENTS ──────────────────────────────────────────
    digits = []
    for i in range(1, num_labels):
        x    = stats[i, cv2.CC_STAT_LEFT]
        y    = stats[i, cv2.CC_STAT_TOP]
        w    = stats[i, cv2.CC_STAT_WIDTH]
        h    = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_component_area:
            continue
        digits.append((x, y, w, h))

    # ── STEP 8 – SORT LEFT → RIGHT ──────────────────────────────────────────
    digits = sorted(digits, key=lambda d: d[0])

    # ── STEP 9 – CROP AND RESIZE EACH DIGIT ─────────────────────────────────
    digit_crops_b64 = []
    for (x, y, w, h) in digits:
        crop         = binary[y : y + h, x : x + w]
        crop_resized = cv2.resize(crop, digit_size)
        digit_crops_b64.append(encode_image(crop_resized))

    # ── STEP 10 – DRAW BOUNDING BOXES ON ORIGINAL ───────────────────────────
    display = image.copy()
    for (x, y, w, h) in digits:
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # ── RETURN ALL RESULTS ───────────────────────────────────────────────────
    return {
        "digit_count":      len(digits),
        "bounding_boxes":   [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in digits],
        "original_b64":     encode_image(image),
        "binary_b64":       encode_image(binary),
        "annotated_b64":    encode_image(display),
        "digit_crops_b64":  digit_crops_b64,
    }


# ==============================
# ROUTE – POST /segment
# Accepts: image file upload + optional params
# Returns: JSON with base64 images and bounding boxes
# ==============================

@app.post("/segment")
async def segment(
    file: UploadFile = File(...)
):
    file_bytes = await file.read()
    image      = decode_upload(file_bytes)
    
    gray, binary = run_preprocess(image)
    digits = segment_digits(binary)
    
    digit_crops_b64 = []
    bboxes = []
    display = image.copy()
    
    for patch, x, y, w, h in digits:
        patch_b64 = encode_image((patch * 255).astype(np.uint8))
        digit_crops_b64.append(patch_b64)
        bboxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    return JSONResponse(content={
        "digit_count": len(digits),
        "bounding_boxes": bboxes,
        "original_b64": encode_image(image),
        "binary_b64": encode_image(binary),
        "annotated_b64": encode_image(display),
        "digit_crops_b64": digit_crops_b64,
    })


# ==============================
# ROUTE – POST /preprocess   (Task 1)
# ==============================

@app.post("/preprocess")
async def preprocess(file: UploadFile = File(...)):
    file_bytes = await file.read()
    image = decode_upload(file_bytes)
    gray, binary = run_preprocess(image)
    return JSONResponse(content={
        "preprocessed_b64": encode_image(binary)
    })


# ==============================
# ROUTE – POST /predict       (Task 3 stub)
# ==============================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Currently unused by UI directly since /recognize handles the e2e mapping.
    raise HTTPException(status_code=501, detail="Task 3 – single digit endpoint not used.")


# ==============================
# ROUTE – POST /recognize     (Task 4)
# Orchestrates preprocess → segment → predict for a full number image
# ==============================

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    global model
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded. Please ensure cnn_custom_best.pth exists.")
        
    file_bytes = await file.read()
    image = decode_upload(file_bytes)
    
    gray, binary = run_preprocess(image)
    digits = segment_digits(binary)
    
    details = []
    for patch, x, y, w, h in digits:
        digit, conf, probs = predict_digit(model, patch)
        details.append({
            "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "digit": int(digit),
            "confidence": float(conf),
            "probs": [float(p) for p in probs],
            "patch_b64": encode_image((patch * 255).astype(np.uint8))
        })
        
    number_str = "".join(str(d["digit"]) for d in details)
    avg_conf = sum(d["confidence"] for d in details) / len(details) if details else 0.0
    
    return JSONResponse(content={
        "number": number_str,
        "digits": details,
        "avg_confidence": avg_conf,
        "binary_b64": encode_image(binary)
    })


# ==============================
# ROUTE – POST /evaluate      (Task 4 stub)
# Runs model against MNIST test set
# ==============================

@app.post("/evaluate")
async def evaluate():
    # TODO (Task 4): load MNIST test set, run model, return metrics + confusion matrix
    # Expected return: { "accuracy": float, "precision": float, "recall": float,
    #                    "f1": float, "confusion_matrix": [[int × 10] × 10] }
    raise HTTPException(status_code=501, detail="Task 4 – evaluation not yet implemented.")


# ==============================
# ENTRY POINT
# Run with: python app.py
# Or:       uvicorn app:app --reload
# ==============================

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
