
import cv2, numpy as np, io, os, torch
from PIL import Image, ImageDraw, ImageFont
from config import IMAGE_SIZE, MODELS_DIR, DEVICE
from torchvision import transforms
import torch.nn.functional as F

# Simple label mapping (adjust to project's labels if needed)
LABELS = ["open", "short", "missing", "mousebite", "spur", "other"]

# TinyCNN model class (same as the saved checkpoint)
import torch.nn as nn
class TinyCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*16*16, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

_model = None

def load_model():
    global _model
    if _model is not None:
        return _model
    model_path = os.path.join(MODELS_DIR, "dummy_tinycnn.pt")
    device = DEVICE
    model = TinyCNN(num_classes=len(LABELS))
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        try:
            model.load_state_dict(state)
        except Exception:
            pass
    model.eval()
    _model = model.to(device)
    return _model

# reuse previous alignment/subtraction logic but operate on bytes
def align_and_subtract_bytes(template_bytes, test_bytes):
    template_np = np.frombuffer(template_bytes, dtype=np.uint8)
    test_np = np.frombuffer(test_bytes, dtype=np.uint8)
    template_img = cv2.imdecode(template_np, cv2.IMREAD_COLOR)
    test_img = cv2.imdecode(test_np, cv2.IMREAD_COLOR)
    t = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    s = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(t,s)
    _, thresh = cv2.threshold(diff,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    return cleaned, template_img, test_img

# Preprocessing for model
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
])

def run_inference_on_pair(template_bytes, test_bytes, conf_thresh=0.2):
    mask, template_img, test_img = align_and_subtract_bytes(template_bytes, test_bytes)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pil = Image.fromarray(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    logs = []
    model = load_model()
    device = DEVICE
    for i, cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cnt)
        if w<5 or h<5:
            continue
        pad = 2
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(test_img.shape[1], x+w+pad), min(test_img.shape[0], y+h+pad)
        roi = test_img[y1:y2, x1:x2]
        try:
            inp = preprocess(roi)
            inp = inp.unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(inp)
                probs = F.softmax(out, dim=1).cpu().numpy()[0]
                idx = int(probs.argmax())
                conf = float(probs[idx])
                label = LABELS[idx]
        except Exception as e:
            label = "error"
            conf = 0.0
        logs.append(f"ROI {i}: {label} ({conf:.2f}) at [{x1},{y1},{x2-x1},{y2-y1}]".format(i=i,label=label,conf=conf,x1=x1,y1=y1))
        draw.rectangle([x1,y1,x2,y2], outline="red", width=2)
        txt = "{} {:.2f}".format(label, conf)
        draw.text((x1, y1-12), txt, fill="red")
    if not contours:
        logs.append("No defects detected.")
    return pil, logs
