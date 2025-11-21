import os
import cv2
import time
import numpy as np
from collections import Counter
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import EfficientNet_B4_Weights

DEFAULT_CLASS_NAMES = [
    "missing_hole",
    "mousebite",
    "open",
    "short",
    "spur",
    "spurious_copper",
    "nondefect"
]

CLASS_COLORS = {
    "missing_hole": (0, 0, 255),        
    "mousebite": (0, 165, 255),         
    "open": (0, 255, 0),                
    "short": (255, 0, 0),               
    "spur": (255, 255, 0),              
    "spurious_copper": (255, 0, 255),  
    "nondefect": (180, 180, 180)        
}

ROI_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def load_model(model_path, device):
    model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, len(DEFAULT_CLASS_NAMES))
    )
    state = torch.load(model_path, map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def preprocess_difference(template, test):
    """Compute absolute difference mask between template and test images."""
    gray_t = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray_s = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_t, gray_s)

    blur = cv2.GaussianBlur(diff, (5, 5), 0)
    _, binary = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(blur, 40, 150)
    combined = cv2.bitwise_or(binary, edges)

    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.dilate(combined, kernel, iterations=2)
    combined = cv2.erode(combined, kernel, iterations=1)
    return combined

def iou(boxA, boxB):
    """Intersection-over-Union for merging overlapping bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW, interH = max(0, xB - xA), max(0, yB - yA)
    interArea = interW * interH
    areaA, areaB = boxA[2]*boxA[3], boxB[2]*boxB[3]
    return interArea / float(areaA + areaB - interArea + 1e-6)

def extract_rois(mask, min_size=20, merge_thresh=0.4):
    """Find bounding boxes and merge overlapping ones."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [(x, y, w, h) for (x, y, w, h) in [cv2.boundingRect(c) for c in contours]
             if w > min_size and h > min_size]

    merged = []
    while boxes:
        base = boxes.pop(0)
        bx, by, bw, bh = base
        to_merge = []
        for b in boxes:
            if iou(base, b) > merge_thresh:
                to_merge.append(b)
        for m in to_merge:
            boxes.remove(m)
            x1 = min(bx, m[0])
            y1 = min(by, m[1])
            x2 = max(bx + bw, m[0] + m[2])
            y2 = max(by + bh, m[1] + m[3])
            bx, by, bw, bh = x1, y1, x2 - x1, y2 - y1
        merged.append((bx, by, bw, bh))
    return merged

def classify_and_annotate(model, test, boxes, device):
    detections = []
    annotated = test.copy()
    height, width = test.shape[:2]

    for (x, y, w, h) in boxes:
        roi = test[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(roi_rgb)
        tensor = ROI_TRANSFORM(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(tensor)
            probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]
            idx = np.argmax(probs)
            label = DEFAULT_CLASS_NAMES[idx]
            score = probs[idx] * 100

        if label != "nondefect":
            detections.append({
                "box": (
                    round(x / width * 100, 1),
                    round(y / height * 100, 1),
                    round(w / width * 100, 1),
                    round(h / height * 100, 1)
                ), 
                "label": label,
                "score": float(score)
            })

            color = CLASS_COLORS.get(label, (0, 255, 0))
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated, f"{label} {score:.1f}%", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    counts = Counter([d["label"] for d in detections])
    return annotated, counts, detections

def run_inference(template, test, model, device):
    start = time.time()

    mask = preprocess_difference(template, test)
    boxes = extract_rois(mask, min_size=15, merge_thresh=0.4)
    annotated, counts, detections = classify_and_annotate(model, test, boxes, device)
    duration = round(time.time() - start, 3)

    return {
        "annotated": annotated,
        "counts": dict(counts),
        "time": duration,
        "detections": detections 
    }
