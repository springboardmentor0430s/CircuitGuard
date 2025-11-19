import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from collections import defaultdict
from src.model import build_efficientnet_b4

# ==========================================================
# CONFIGURATION
# ==========================================================
MODEL_PATH = "models/efficientnet_b4_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128

# Filtering thresholds
MIN_AREA = 100        # ignore tiny regions
MAX_AREA = 4000       # ignore huge regions
CONF_THRESH = 0.75    # skip low-confidence predictions

# Colors for visualization
COLOR_MAP = {
    1: (0, 0, 255),    # Open - red
    2: (0, 255, 0),    # Short - green
    3: (255, 0, 0),    # Spur - blue
    4: (255, 255, 0),  # Mousebite - cyan
    5: (255, 0, 255),  # Pin-hole - magenta
    6: (0, 255, 255)   # Spurious Cu - yellow
}

CLASS_NAMES = {
    1: "Open",
    2: "Short",
    3: "Spur",
    4: "Mousebite",
    5: "Pin-hole",
    6: "Spurious Cu"
}

# ==========================================================
# LOAD MODEL
# ==========================================================
model = build_efficientnet_b4(num_classes=6, pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ==========================================================
# IMAGE DIFFERENCE & MASK CREATION
# ==========================================================
def generate_mask(template, test):
    diff = cv2.absdiff(template, test)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)

    return mask

# ==========================================================
# PREDICTION FUNCTION
# ==========================================================
def predict_roi(roi):
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    from PIL import Image
    img = Image.fromarray(roi)
    t = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        o = model(t)
        p = torch.softmax(o, 1)[0]
        idx = p.argmax().item()
        cls_id = idx + 1
        return cls_id, float(p[idx])

# ==========================================================
# MAIN DEFECT DETECTION FUNCTION
# ==========================================================
def detect_and_overlay(template_path, test_path, save_path="overlay_result.jpg"):
    template = cv2.imread(template_path)
    test = cv2.imread(test_path)
    if template is None or test is None:
        print(" Error reading input images.")
        return

    mask = generate_mask(template, test)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_draw = test.copy()

    defect_counts = defaultdict(int)
    detected_boxes = 0

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < MIN_AREA or w * h > MAX_AREA:
            continue

        roi = test[y:y + h, x:x + w]
        if roi.size == 0:
            continue

        cls_id, prob = predict_roi(roi)
        if prob < CONF_THRESH:
            continue

        color = COLOR_MAP.get(cls_id, (255, 255, 255))
        name = CLASS_NAMES.get(cls_id, f"Class{cls_id}")

        cv2.rectangle(img_draw, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img_draw, f"{name} ({prob * 100:.1f}%)", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        defect_counts[name] += 1
        detected_boxes += 1

    # Blended overlay
    blended = cv2.addWeighted(template, 0.4, img_draw, 0.6, 0)
    cv2.imwrite(save_path, blended)

    print(f"\n Overlay saved to: {save_path}")
    print(f" Total detected boxes (after filtering): {detected_boxes}")
    print(" Detected Defects Summary:")
    for k, v in defect_counts.items():
        print(f"  - {k}: {v}")
    print()

# ==========================================================
# RUN EXAMPLE
# ==========================================================
if __name__ == "__main__":
    group = "group00041"
    sub = "00041"
    base = "00041000"

    template = f"data/DeepPCB/PCBData/{group}/{sub}/{base}_temp.jpg"
    test = f"data/DeepPCB/PCBData/{group}/{sub}/{base}_test.jpg"

    detect_and_overlay(template, test, "overlay_filtered.jpg")
