# backend.py
import os
import re
import io
import csv
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

# === CONFIG ===
MODEL_PATH = os.path.join("outputs", "efficientnet_b0_pcb_final.pth")  # relative to app.py run location
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 128

# Replace these with your final class names (same order as training)
CLASS_NAMES = [
    "Missing Hole",
    "Mouse Bite",
    "Open Circuit",
    "Short",
    "Spur",
    "Spurious Copper",
    "Pin Hole"
]

# === MODEL LOADER (cached) ===
_model = None
_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def load_model(model_path=MODEL_PATH):
    global _model
    if _model is not None:
        return _model
    # create model skeleton
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    ckpt = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt)
    model.to(DEVICE)
    model.eval()
    _model = model
    return _model

# === HELPERS ===
def extract_number(filename):
    m = re.search(r'\d+', filename)
    return m.group() if m else None

def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv_img):
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

# === CORE: predict crop (PIL) -> (label, confidence) ===
def predict_crop(pil_crop, model=None):
    if model is None:
        model = load_model()
    t = _transform(pil_crop).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(t)
        probs = torch.nn.functional.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)
        label = CLASS_NAMES[pred.item()]
        return label, float(conf.item())

# === IMAGE PROCESS: compute diff mask, find contours ===
def diff_mask_from_pair(img_test_cv2, img_temp_cv2, threshold=10, min_area=100):
    # ensure same size
    if img_test_cv2.shape != img_temp_cv2.shape:
        img_temp_cv2 = cv2.resize(img_temp_cv2, (img_test_cv2.shape[1], img_test_cv2.shape[0]))
    diff = cv2.absdiff(img_temp_cv2, img_test_cv2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # filter small
    good_cnts = [c for c in contours if cv2.contourArea(c) >= min_area]
    return diff, mask, good_cnts

# === MAIN: process a pair of PIL images -> annotated PIL and CSV bytes ===
def process_pair_and_predict(pil_test, pil_temp,
                            model=None,
                            threshold=10,
                            min_area=100,
                            save_crops=False,
                            crop_prefix=""):
    """
    pil_test, pil_temp: PIL images (RGB)
    returns: annotated_pil (PIL), csv_bytes (io.BytesIO)
    CSV columns: crop_filename, x, y, w, h, predicted_label, confidence
    """
    if model is None:
        model = load_model()

    img_test = pil_to_cv2(pil_test)
    img_temp = pil_to_cv2(pil_temp)

    diff, mask, contours = diff_mask_from_pair(img_test, img_temp, threshold=threshold, min_area=min_area)

    annotated = img_test.copy()
    csv_buffer = io.StringIO()
    csv_writer = csv.writer(csv_buffer)
    csv_writer.writerow(["crop_name","x","y","w","h","pred_label","confidence"])

    crop_files = []
    for i, c in enumerate(contours):
        x,y,w,h = cv2.boundingRect(c)
        if w*h < min_area:
            continue
        crop_bgr = img_test[y:y+h, x:x+w]
        pil_crop = cv2_to_pil(crop_bgr)
        pred_label, conf = predict_crop(pil_crop, model=model)
        conf_pct = conf*100.0
        crop_name = f"{crop_prefix}crop_{i}.jpg"
        csv_writer.writerow([crop_name, x, y, w, h, pred_label, f"{conf_pct:.2f}"])
        crop_files.append((crop_name, pil_crop, pred_label, conf_pct))

        # annotate
        color = (0,0,255) if conf_pct>60 else (0,255,255)
        cv2.rectangle(annotated, (x,y), (x+w,y+h), color, 2)
        label_text = f"{pred_label} {conf_pct:.1f}%"
        cv2.putText(annotated, label_text, (x, max(15,y-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    annotated_pil = cv2_to_pil(annotated)
    csv_bytes = io.BytesIO(csv_buffer.getvalue().encode("utf-8"))

    return annotated_pil, csv_bytes, crop_files, mask, diff

