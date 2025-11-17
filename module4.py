import os
import re
import cv2
import torch
import numpy as np
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image

# ===================================================
# PATHS
# ===================================================
BASE_DIR = r"C:\Users\prati\Desktop\PCB_DATASET"
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "efficientnet_b0_pcb.pth")
IMAGE_BASE_DIR = os.path.join(BASE_DIR, "pcbimages")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "inference_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================================================
# CLASS NAMES
# ===================================================
CLASS_NAMES = [
    "Missing Hole",
    "Mouse Bite",
    "Open Circuit",
    "Short",
    "Spur",
    "Spurious Copper",
    "Pin Hole"
]

# ===================================================
# MODEL LOADING
# ===================================================
print("🔹 Loading trained model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print("✅ Model loaded successfully.")

# ===================================================
# TRANSFORM
# ===================================================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ===================================================
# HELPER FUNCTIONS
# ===================================================
def extract_number(filename):
    """Extracts the numeric part from a filename, e.g., '12100001_temp.jpg' -> '12100001'"""
    match = re.search(r'\d+', filename)
    return match.group() if match else None

def predict_defect(image_crop):
    img_tensor = transform(image_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        return CLASS_NAMES[pred.item()], conf.item() * 100

# ===================================================
# PROCESS TEST IMAGES
# ===================================================
for folder in sorted(os.listdir(IMAGE_BASE_DIR)):
    folder_path = os.path.join(IMAGE_BASE_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    print(f"\n📂 Processing folder: {folder}")
    image_files = os.listdir(folder_path)
    temp_files = [f for f in image_files if "_temp" in f]
    test_files = [f for f in image_files if "_test" in f]

    if not temp_files or not test_files:
        print("⚠️  No _temp/_test image pairs found here.")
        continue

    for temp_file in temp_files:
        temp_num = extract_number(temp_file)
        if not temp_num:
            continue

        # Find matching test image by number
        match = None
        for tf in test_files:
            if temp_num in tf:
                match = tf
                break

        if not match:
            print(f"⚠️  No matching test image found for {temp_file}")
            continue

        temp_path = os.path.join(folder_path, temp_file)
        test_path = os.path.join(folder_path, match)

        img_temp = cv2.imread(temp_path, cv2.IMREAD_COLOR)
        img_test = cv2.imread(test_path, cv2.IMREAD_COLOR)

        if img_temp is None or img_test is None:
            print(f"⚠️  Could not load images for {temp_file}")
            continue

        # ===================================================
        # DIFFERENCE DETECTION
        # ===================================================
        diff = cv2.absdiff(img_temp, img_test)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"   🔍 Found {len(contours)} contours in {match}")

        img_name = os.path.splitext(temp_file)[0]
        save_dir = os.path.join(OUTPUT_DIR, folder)
        os.makedirs(save_dir, exist_ok=True)

        saved_count = 0
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            if w*h < 100:
                continue

            defect_crop = img_test[y:y+h, x:x+w]
            crop_path = os.path.join(save_dir, f"{img_name}_defect{i}.jpg")
            cv2.imwrite(crop_path, defect_crop)
            saved_count += 1

            pil_crop = Image.fromarray(cv2.cvtColor(defect_crop, cv2.COLOR_BGR2RGB))
            pred_class, confidence = predict_defect(pil_crop)

            label_text = f"{pred_class} ({confidence:.1f}%)"
            cv2.rectangle(img_test, (x, y), (x+w, y+h), (0,0,255), 2)
            cv2.putText(img_test, label_text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        if saved_count == 0:
            print(f"⚠️  No valid ROI saved for {match}")
        else:
            annotated_path = os.path.join(save_dir, f"{img_name}_annotated.jpg")
            cv2.imwrite(annotated_path, img_test)
            print(f"✅ Saved {saved_count} ROI crops and annotation → {annotated_path}")

print("\n🎯 All folders processed. Check your outputs folder.")
