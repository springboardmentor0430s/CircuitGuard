import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
import os

# ==========================
# CONFIGURATION
# ==========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "efficientnet_b4_best.pth"
REFERENCE_IMAGE = r"PCBData\group12100\12100\12100000_temp.jpg"
TEST_IMAGE = r"PCBData\group12100\12100\12100000_test.jpg"
OUTPUT_IMAGE = "annotated_full_detection.jpg"
COMBINED_OUTPUT = "combined_output_with_heatmap.jpg"

CLASS_NAMES = ['copper', 'mousebite', 'noise', 'open', 'pin-hole', 'short', 'spur']

BOX_PADDING = 15
DETECTED_BOX_X_OFFSET = 3
MIN_DEFECT_AREA = 8
MAX_ASPECT_RATIO = 7.0
VISUALIZE = False


# ==========================
# LOAD MODEL
# ==========================
def load_model(model_path):
    print("📦 Loading EfficientNet-B4 (torchvision) model...")

    model = efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))

    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state, strict=False)

    model.to(DEVICE)
    model.eval()
    return model


# ==========================
# IMAGE PREPROCESSING
# ==========================
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)


# ==========================
# BOUNDING BOX ADJUSTER
# ==========================
def adjust_bbox(x, y, w, h, img_w, img_h, pad=BOX_PADDING, x_offset=DETECTED_BOX_X_OFFSET):
    x_start = max(0, x - pad - x_offset)
    y_start = max(0, y - pad)
    x_end = min(img_w, x + w + pad - x_offset)
    y_end = min(img_h, y + h + pad)
    if x_end <= x_start: x_end = x_start + 1
    if y_end <= y_start: y_end = y_start + 1
    return x_start, y_start, x_end, y_end


# ==========================
# DETECT + CLASSIFY + COMBINE
# ==========================
def detect_and_classify(model, ref_path, test_path, output_path):
    print("🚀 Starting full defect detection + classification pipeline...")

    ref = cv2.imread(ref_path)
    test = cv2.imread(test_path)
    if ref is None or test is None:
        raise ValueError("❌ Error: Could not read reference or test image.")

    H, W, _ = test.shape

    # --- Step 1: Image Subtraction ---
    diff = cv2.absdiff(ref, test)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Step 2: Morphological filtering ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    # --- Step 3: Find contours ---
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"🔍 Found {len(contours)} potential defect regions.")

    annotated = test.copy()
    heatmap_mask = np.zeros((H, W), dtype=np.float32)
    roi_counter = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        if area < MIN_DEFECT_AREA:
            continue
        if w > 0 and h > 0 and max(w/h, h/w) > MAX_ASPECT_RATIO:
            continue

        x1, y1, x2, y2 = adjust_bbox(x, y, w, h, W, H)
        roi = test[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        roi_tensor = preprocess_image(roi).to(DEVICE)
        with torch.no_grad():
            outputs = model(roi_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)

        label = CLASS_NAMES[pred_idx.item()]
        conf = confidence.item() * 100

        color = (0, 255, 0) if conf > 90 else (0, 165, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{label} ({conf:.1f}%)", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        roi_counter += 1

        # Add heat intensity proportional to confidence
        heatmap_mask[y1:y2, x1:x2] += conf / 100.0

    # --- Normalize heatmap ---
    heatmap_mask = np.clip(heatmap_mask, 0, 1)
    heatmap = cv2.applyColorMap((heatmap_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_overlay = cv2.addWeighted(annotated, 0.7, heatmap, 0.6, 0)

    # --- Step 4: Save annotated ---
    cv2.imwrite(output_path, annotated)
    print(f"✅ Annotated image saved to {output_path}")
    print(f"📊 Total classified ROIs: {roi_counter}")

    # --- Step 5: Combine all 5 images ---
        # --- Step 5: Combine all 5 images ---
        # --- Step 5: Combine all 5 images with borders & labels ---
    def resize_for_display(img):
        return cv2.resize(img, (400, 400))

    def add_label(image, text):
        """Add a label with black text on yellow background."""
        label_img = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_w, text_h = text_size

        # Yellow background rectangle behind text
        cv2.rectangle(label_img, (5, 5),
                      (10 + text_w, 10 + text_h + 5),
                      (0, 255, 255), -1)
        # Black text
        cv2.putText(label_img, text, (10, 10 + text_h),
                    font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        return label_img

    # Resize and label each image
    temp_disp = add_label(resize_for_display(ref), "Temp Image")
    test_disp = add_label(resize_for_display(test), "Test Image")
    clean_disp = add_label(cv2.cvtColor(resize_for_display(cleaned), cv2.COLOR_GRAY2BGR), "Threshold Image")
    annotated_disp = add_label(resize_for_display(annotated), "Annotated Image")
    heatmap_disp = add_label(resize_for_display(heatmap_overlay), "Defect Heatmap")

    # Define border size and color
    BORDER_SIZE = 10
    BORDER_COLOR = (50, 50, 50)

    # Add border around each image
    def add_border(img):
        return cv2.copyMakeBorder(img, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE,
                                  cv2.BORDER_CONSTANT, value=BORDER_COLOR)

    temp_disp = add_border(temp_disp)
    test_disp = add_border(test_disp)
    clean_disp = add_border(clean_disp)
    annotated_disp = add_border(annotated_disp)
    heatmap_disp = add_border(heatmap_disp)

    # Combine rows
    top_row = np.hstack((temp_disp, test_disp, clean_disp))
    bottom_row = np.hstack((annotated_disp, heatmap_disp))

    # Ensure both rows have the same width
    top_h, top_w, _ = top_row.shape
    bottom_h, bottom_w, _ = bottom_row.shape

    if top_w != bottom_w:
        pad_width = abs(top_w - bottom_w)
        if top_w > bottom_w:
            bottom_row = cv2.copyMakeBorder(bottom_row, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=BORDER_COLOR)
        else:
            top_row = cv2.copyMakeBorder(top_row, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=BORDER_COLOR)

    # Stack vertically
    combined = np.vstack((top_row, bottom_row))

    # Add outer border to final image
    combined = cv2.copyMakeBorder(combined, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE,
                                  cv2.BORDER_CONSTANT, value=BORDER_COLOR)

    cv2.imwrite(COMBINED_OUTPUT, combined)
    print(f"🖼️ Combined output with labels and borders saved as {COMBINED_OUTPUT}")




# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    detect_and_classify(model, REFERENCE_IMAGE, TEST_IMAGE, OUTPUT_IMAGE)
    print("🎯 Full detection + classification + heatmap visualization complete!")
