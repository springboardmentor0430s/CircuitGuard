import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from torchvision import transforms
from torchvision.ops import nms
from collections import Counter
from pathlib import Path
from datetime import datetime
from src.model import build_efficientnet_b4
from src.report_generator import generate_fancy_report

# ---------------- CONFIG ----------------
MODEL_PATH = "models/efficientnet_b4_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128
os.makedirs("outputs", exist_ok=True)
os.makedirs("logs", exist_ok=True)

LOG_FILE = "logs/analysis_log.csv"

COLOR_MAP = {
    1: (0, 0, 255), 2: (0, 128, 0), 3: (255, 0, 0),
    4: (0, 180, 180), 5: (180, 0, 180), 6: (0, 255, 255)
}
CLASS_NAMES = {
    1: "Open", 2: "Short", 3: "Spur",
    4: "Mousebite", 5: "Pin-hole", 6: "Spurious Cu"
}

# ---------------- THEME CSS ----------------
def load_css():
    css_path = Path(__file__).parent / "theme.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.set_page_config(page_title="CircuitGuard", layout="wide")
st.title(" CircuitGuard — Smart PCB Defect Detection Dashboard")
st.write("Upload a **Template Image** (reference) and a **Defective Image** to detect and analyze PCB defects.")

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    model = build_efficientnet_b4(num_classes=6, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

model = load_model()
tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])


# ---------------- LOGGING FUNCTION ----------------
def log_analysis_event(template_name, test_name, summary, conf_thresh, min_area, max_area, blur_strength, pdf_path):
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("timestamp,template,test,summary,confidence,min_area,max_area,blur,pdf_path\n")

    with open(LOG_FILE, "a") as f:
        f.write(
            f"{datetime.now()},{template_name},{test_name},{summary},"
            f"{conf_thresh},{min_area},{max_area},{blur_strength},{pdf_path}\n"
        )


# ---------------- PREDICT ----------------
def predict_roi(roi):
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(roi)
    t = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(t)
        p = torch.softmax(out, 1)[0]
        idx = p.argmax().item() + 1
        conf = float(p.max().item())
        return idx, conf


# ---------------- PREPROCESS ----------------
def preprocess_diff(template, test, blur_strength=5):
    diff = cv2.absdiff(template, test)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur_strength = max(1, int(blur_strength) // 2 * 2 + 1)
    blur = cv2.GaussianBlur(gray, (blur_strength, blur_strength), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


# ---------------- BAR + PIE CHARTS ----------------
def generate_bar_chart(summary):
    labels = list(summary.keys())
    values = list(summary.values())

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color="#6A5ACD")
    plt.title("Defect Count per Class")
    plt.xticks(rotation=30)
    plt.tight_layout()

    bar_path = "outputs/bar_chart.png"
    plt.savefig(bar_path)
    plt.close()
    return bar_path


def generate_pie_chart(summary):
    labels = list(summary.keys())
    values = list(summary.values())

    plt.figure(figsize=(5, 5))
    if sum(values) == 0:
        plt.text(0.5, 0.5, "No defects", ha="center", va="center", fontsize=14)
        plt.axis("off")
    else:
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)

    plt.title("Defect Distribution")
    plt.tight_layout()

    pie_path = "outputs/pie_chart.png"
    plt.savefig(pie_path)
    plt.close()
    return pie_path


# ---------------- ANALYSIS ----------------
def analyze_pcb(template, test, conf_thresh, min_area, max_area, blur_strength):
    mask = preprocess_diff(template, test, blur_strength)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < min_area or area > max_area:
            continue
        roi = test[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        cls_id, conf = predict_roi(roi)
        if conf >= conf_thresh:
            candidates.append((x, y, w, h, conf, cls_id))

    if candidates:
        boxes = torch.tensor([[x, y, x + w, y + h] for (x, y, w, h, _, _) in candidates], dtype=torch.float32)
        scores = torch.tensor([conf for (_, _, _, _, conf, _) in candidates])
        keep = nms(boxes, scores, iou_threshold=0.5)
        candidates = [candidates[i] for i in keep]

    img_draw = test.copy()
    heatmap = np.zeros(test.shape[:2], dtype=np.float32)
    summary = Counter()
    confidences, detections = [], []

    for (x, y, w, h, conf, cls_id) in candidates:
        name = CLASS_NAMES[cls_id]
        color = COLOR_MAP[cls_id]

        cv2.rectangle(img_draw, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img_draw, f"{name} ({conf*100:.1f}%)", (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        heatmap[y:y + h, x:x + w] += conf
        summary[name] += 1
        confidences.append(conf)
        detections.append({
            "class": name,
            "conf": round(conf * 100, 1),
            "coords": (x, y, w, h)
        })

    heatmap_norm = heatmap / (heatmap.max() + 1e-8)
    heat_color = cv2.applyColorMap((heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blended_heat = cv2.addWeighted(test, 0.6, heat_color, 0.4, 0)

    blended_overlay = cv2.addWeighted(template, 0.45, img_draw, 0.55, 0)

    annotated_path = "outputs/overlay_output.jpg"
    heatmap_path = "outputs/heatmap_output.jpg"
    cv2.imwrite(annotated_path, blended_overlay)
    cv2.imwrite(heatmap_path, blended_heat)

    return annotated_path, heatmap_path, dict(summary), confidences, detections


# ---------------- STREAMLIT UI ----------------
with st.sidebar:
    st.header("🛠 Manual Controls")
    conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    min_area = st.number_input("Min Defect Area (px)", 10, 10000, 40, 10)
    max_area = st.number_input("Max Defect Area (px)", 100, 50000, 3000, 50)
    blur_strength = st.slider("Blur Strength", 1, 15, 5, 2)
    view_mode = st.selectbox("View Mode", ["Overlay", "Heatmap", "Both"])

col1, col2 = st.columns(2)
template_file = col1.file_uploader(" Upload Template Image", type=["jpg", "jpeg", "png"])
test_file = col2.file_uploader(" Upload Defective Image", type=["jpg", "jpeg", "png"])

if st.button(" Run Analysis"):
    if not template_file or not test_file:
        st.warning("Please upload both images first.")
    else:
        template = cv2.imdecode(np.frombuffer(template_file.read(), np.uint8), 1)
        test = cv2.imdecode(np.frombuffer(test_file.read(), np.uint8), 1)

        with st.spinner("Analyzing defects..."):
            annotated_path, heatmap_path, summary, confidences, detections = analyze_pcb(
                template, test, conf_thresh, int(min_area), int(max_area), int(blur_strength)
            )

        avg_conf = float(np.mean(confidences) * 100) if confidences else 0.0
        total_defects = sum(summary.values())

        bar_path = generate_bar_chart(summary)
        pie_path = generate_pie_chart(summary)

        pdf_path = generate_fancy_report(
            summary, annotated_path, heatmap_path, detections,
            conf_thresh, min_area, max_area, avg_conf
        )

        log_analysis_event(
            template_file.name, test_file.name, summary,
            conf_thresh, min_area, max_area, blur_strength, pdf_path
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Defects", total_defects)
        c2.metric("Unique Types", len(summary))
        c3.metric("Avg Confidence", f"{avg_conf:.1f}%")
        top_def = max(summary.items(), key=lambda x: x[1])[0] if summary else "N/A"
        c4.metric("Top Defect", top_def)

        st.subheader("Results")
        if view_mode in ["Overlay", "Both"]:
            st.image(annotated_path, caption="Annotated Overlay", use_container_width=True)
        if view_mode in ["Heatmap", "Both"]:
            st.image(heatmap_path, caption="Confidence Heatmap", use_container_width=True)

        st.subheader("Detailed Defect Data")
        st.table([{**d} for d in detections])

        st.subheader("Charts")
        st.image(bar_path, caption="Bar Chart", use_container_width=True)
        st.image(pie_path, caption="Pie Chart", use_container_width=True)

        with open(pdf_path, "rb") as f:
            st.download_button(" Generate PDF", f, file_name="CircuitGuard_Report.pdf", mime="application/pdf")
