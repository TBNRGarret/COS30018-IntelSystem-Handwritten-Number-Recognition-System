import cv2
import numpy as np
import base64
import uvicorn
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# ==============================
# APP SETUP
# ==============================

# Build paths relative to this file (backend/app.py)
# so the app works regardless of which directory you run it from
BASE_DIR     = Path(__file__).parent         # .../backend/
FRONTEND_DIR = BASE_DIR.parent / "frontend"  # .../frontend/

app = FastAPI(title="HNRS – Handwritten Number Recognition System")

# Serve index.html and style.css from the sibling frontend/ folder
app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")


# ==============================
# PARAMETERS
# These were previously hardcoded globals in ImageSegmentation.py.
# Now exposed as endpoint arguments so the GUI can control them.
# ==============================

DEFAULT_MIN_COMPONENT_AREA = 100
DEFAULT_DIGIT_SIZE         = (28, 28)
DEFAULT_USE_ADAPTIVE       = False


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
    file:               UploadFile = File(...),
    min_component_area: int  = DEFAULT_MIN_COMPONENT_AREA,
    use_adaptive:       bool = DEFAULT_USE_ADAPTIVE,
):
    file_bytes = await file.read()
    image      = decode_upload(file_bytes)
    result     = run_segmentation(
        image,
        min_component_area = min_component_area,
        use_adaptive       = use_adaptive,
    )
    return JSONResponse(content=result)


# ==============================
# ROUTE – POST /preprocess   (Task 1 stub)
# Your teammate fills this in
# ==============================

@app.post("/preprocess")
async def preprocess(file: UploadFile = File(...)):
    # TODO (Task 1): implement preprocessing pipeline
    # Expected return: { "preprocessed_b64": "<base64 png>" }
    raise HTTPException(status_code=501, detail="Task 1 – preprocessing not yet implemented.")


# ==============================
# ROUTE – POST /predict       (Task 3 stub)
# Team member handling ML fills this in
# ==============================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # TODO (Task 3): run trained model on a single 28×28 digit image
    # Expected return: { "digit": int, "confidence": float, "all_probs": [float × 10] }
    raise HTTPException(status_code=501, detail="Task 3 – ML model not yet implemented.")


# ==============================
# ROUTE – POST /recognize     (Task 4 stub)
# Orchestrates preprocess → segment → predict for a full number image
# ==============================

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    # TODO (Task 4): chain preprocess → segment → predict → assemble number string
    # Expected return: { "number": str, "digits": [...], "avg_confidence": float }
    raise HTTPException(status_code=501, detail="Task 4 – full recognition pipeline not yet implemented.")


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
