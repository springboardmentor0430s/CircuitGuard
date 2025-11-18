# ===== CircuitGuard AI - PCB Defect Detection Backend (FastAPI) =====
# File: pcb_project/backend/main.py

import os
import io
import cv2
import time
import uuid
import json
import csv
import base64
import traceback
import numpy as np
from typing import Dict, Any, List
from PIL import Image
from datetime import datetime

# ----------------- Torch / Model (optional) -----------------
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    models = None
    transforms = None

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

# ---------------------------
# PATHS & SETUP
# ---------------------------
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "resnet50_pcb_best.pth")
OUTPUTS_DIR = os.path.join(BACKEND_DIR, "outputs")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(FRONTEND_DIR, exist_ok=True)

# ---------------------------
# PARAMETERS
# ---------------------------
IMAGE_SIZE = 128
DEVICE = "cpu"
if TORCH_AVAILABLE:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "mousebite",
    "open",
    "pinhole",
    "short",
    "spur",
    "spurious copper",
]
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------
# MODEL LOADING
# ---------------------------
def _clean_state_dict_keys(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        elif k.startswith("model."):
            cleaned[k[len("model."):]] = v
        else:
            cleaned[k] = v
    return cleaned


CLASSIFIER_MODEL = None
classifier_transform = None

if TORCH_AVAILABLE:
    def load_model(path, num_classes, device):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")

        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )

        state = torch.load(path, map_location=device)

        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        if isinstance(state, dict):
            state = _clean_state_dict_keys(state)

        model.load_state_dict(state)
        model = model.to(device)
        model.eval()
        return model

    print("Loading classifier model...")
    try:
        CLASSIFIER_MODEL = load_model(MODEL_PATH, NUM_CLASSES, DEVICE)
        print("Model loaded on", DEVICE)
    except Exception as e:
        print("WARNING: Failed to load model:", e)
        CLASSIFIER_MODEL = None

    classifier_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
else:
    print("Torch not available – classifier disabled.")
    CLASSIFIER_MODEL = None

# ---------------------------
# IMAGE HELPERS
# ---------------------------
def to_b64_from_cv2(img):
    if img is None:
        return ""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ok, buff = cv2.imencode(".png", img)
    if not ok:
        return ""
    return base64.b64encode(buff).decode("utf-8")


def save_cv2_image(img, filename):
    path = os.path.join(OUTPUTS_DIR, filename)
    cv2.imwrite(path, img)
    return path

# ---------------------------
# DEFECT DETECTION
# ---------------------------
def align_images_orb(template_gray, test_gray, max_features=2000):
    try:
        orb = cv2.ORB_create(max_features)
        kp1, des1 = orb.detectAndCompute(template_gray, None)
        kp2, des2 = orb.detectAndCompute(test_gray, None)

        if des1 is None or des2 is None:
            return template_gray

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) < 8:
            return template_gray

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)

        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            return template_gray

        h, w = test_gray.shape[:2]
        warped = cv2.warpPerspective(template_gray, M, (w, h))
        return warped
    except Exception:
        return template_gray


def perform_defect_detection(test_img, template_img, diff_threshold=30, min_area=50):
    h, w = test_img.shape[:2]
    if template_img.shape[:2] != (h, w):
        template_img = cv2.resize(template_img, (w, h))

    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

    template_aligned = align_images_orb(template_gray, test_gray)
    diff = cv2.absdiff(test_gray, template_aligned)

    _, mask1 = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    _, mask2 = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    final_mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    final_mask = cv2.dilate(final_mask, kernel, iterations=2)

    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    defects = []
    idx_counter = 1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, wb, hb = cv2.boundingRect(cnt)
        padding = 6
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(test_img.shape[1], x + wb + padding)
        y1 = min(test_img.shape[0], y + hb + padding)
        roi = test_img[y0:y1, x0:x1]

        defects.append({
            "defect_id": idx_counter,
            "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
            "roi_image": roi,
            "position_xy": [int((x0 + x1) // 2), int((y0 + y1) // 2)],
            "size_wh": f"{x1 - x0}x{y1 - y0}",
            "area": float(area)
        })
        idx_counter += 1

    return defects, diff, final_mask


def classify_roi(roi_img):
    # If model not loaded, don't try to classify – avoid fake "unknown 0.0%"
    if CLASSIFIER_MODEL is None or not TORCH_AVAILABLE:
        return "unknown", 0.0
    try:
        roi_pil = Image.fromarray(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
        inp = classifier_transform(roi_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = CLASSIFIER_MODEL(inp)
            probs = torch.nn.functional.softmax(out, dim=1)[0]
            idx = int(torch.argmax(probs).item())
            return CLASS_NAMES[idx], float(probs[idx].item())
    except Exception:
        return "unknown", 0.0


def annotate_image(img, predictions):
    out = img.copy()
    for p in predictions:
        x, y, wb, hb = [int(v) for v in p["bbox"]]
        conf = p["confidence"]
        label = p["class"]
        # Colors like your good screenshot: green (high), orange (mid), red (low)
        color = (0, 255, 0) if conf >= 0.8 else ((0, 165, 255) if conf >= 0.6 else (0, 0, 255))
        thickness = 3 if conf >= 0.8 else 2
        cv2.rectangle(out, (x, y), (x + wb, y + hb), color, thickness)
        txt = f"{label} {conf*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = y - th - 8 if y - th - 8 > 0 else y + th + 8
        cv2.rectangle(out, (x, y_text - th - 6), (x + tw + 8, y_text + 4), color, -1)
        cv2.putText(out, txt, (x + 4, y_text - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out

# ---------------------------
# PLOTS / CHARTS
# ---------------------------
def _fig_to_cv2():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)


def plot_bar(predictions):
    df = pd.DataFrame(predictions) if predictions else pd.DataFrame(columns=["class"])
    plt.figure(figsize=(6, 4))
    if not df.empty:
        df["class"].value_counts().plot(kind="bar")
        plt.ylabel("Count")
        plt.xlabel("Defect Class")
    else:
        plt.text(0.5, 0.5, "No Defects Detected", ha="center")
        plt.axis("off")
    return _fig_to_cv2()


def plot_pie(predictions):
    df = pd.DataFrame(predictions) if predictions else pd.DataFrame(columns=["class"])
    plt.figure(figsize=(5, 5))
    if not df.empty:
        counts = df["class"].value_counts()
        plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
        plt.axis("equal")
    else:
        plt.text(0.5, 0.5, "No Defects", ha="center")
        plt.axis("off")
    return _fig_to_cv2()


def plot_scatter(predictions):
    plt.figure(figsize=(6, 4))
    if not predictions:
        plt.text(0.5, 0.5, "No Defects to plot", ha="center")
        plt.axis("off")
        return _fig_to_cv2()

    df = pd.DataFrame(predictions)
    classes = df["class"].unique()
    colors_map = sns.color_palette("tab10", n_colors=len(classes))

    for i, cls in enumerate(classes):
        subset = df[df["class"] == cls]
        xs = [p[0] for p in subset["position_xy"]]
        ys = [p[1] for p in subset["position_xy"]]
        sizes = (subset["confidence"].values * 120) + 20
        plt.scatter(xs, ys, label=cls, s=sizes, alpha=0.75, c=[colors_map[i]])

    plt.xlabel("X (px)")
    plt.ylabel("Y (px)")
    plt.gca().invert_yaxis()
    plt.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.title("Defect spatial distribution (size ~ confidence)")
    return _fig_to_cv2()


def plot_confidence_hist(predictions):
    plt.figure(figsize=(6, 4))
    if not predictions:
        plt.text(0.5, 0.5, "No Confidence Data", ha="center")
        plt.axis("off")
    else:
        df = pd.DataFrame(predictions)
        plt.hist(df["confidence"], bins=8, edgecolor="black")
        plt.xlabel("Confidence")
        plt.ylabel("Frequency")
    return _fig_to_cv2()

# ---------------------------
# HISTORY / CSV HELPERS
# ---------------------------
def _run_meta_path(report_id: str) -> str:
    return os.path.join(OUTPUTS_DIR, f"run_{report_id}.json")


def store_run_metadata(meta: Dict[str, Any]) -> None:
    with open(_run_meta_path(meta["report_id"]), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def list_runs(limit: int = 50) -> List[Dict[str, Any]]:
    metas = []
    for fname in os.listdir(OUTPUTS_DIR):
        if fname.startswith("run_") and fname.endswith(".json"):
            try:
                with open(os.path.join(OUTPUTS_DIR, fname), "r", encoding="utf-8") as f:
                    metas.append(json.load(f))
            except Exception:
                continue
    metas.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return metas[:limit]


def kind_to_filename(report_id: str, kind: str) -> str:
    mapping = {
        "annotated": f"annotated_{report_id}.png",
        "diff":      f"diff_{report_id}.png",
        "mask":      f"mask_{report_id}.png",
        "bar":       f"bar_{report_id}.png",
        "pie":       f"pie_{report_id}.png",
        "scatter":   f"scatter_{report_id}.png",
        "hist":      f"hist_{report_id}.png",
        "template":  f"template_{report_id}.png",
        "test":      f"test_{report_id}.png",
    }
    name = mapping.get(kind, "")
    return os.path.join(OUTPUTS_DIR, name) if name else ""


def _write_csv(report_id: str, predictions: List[Dict[str, Any]]) -> str:
    csv_path = os.path.join(OUTPUTS_DIR, f"defects_{report_id}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["defect_id", "class", "confidence", "x", "y", "w", "h", "area"])
        for p in predictions:
            x, y, w, h = p["bbox"]
            writer.writerow([
                p["defect_id"], p["class"], f"{p['confidence']:.4f}",
                x, y, w, h, int(p.get("area", 0))
            ])
    return csv_path

# ---------------------------
# PDF REPORT
# ---------------------------
def save_plot_temp(img_cv2, name):
    path = os.path.join(OUTPUTS_DIR, f"{name}.png")
    cv2.imwrite(path, img_cv2)
    return path


def generate_pdf_report(report_id, params, stats,
                        annotated_np, diff_np, mask_np, predictions):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_name = f"PCB_Report_{report_id}_{timestamp}.pdf"
    pdf_path = os.path.join(OUTPUTS_DIR, pdf_name)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch
    )
    styles = getSampleStyleSheet()
    elems = []

    title_style = ParagraphStyle(
        'title', parent=styles['Heading1'],
        alignment=TA_CENTER, fontSize=24, spaceAfter=12
    )
    elems.append(Spacer(1, 0.8 * inch))
    elems.append(Paragraph("CircuitGuard AI", title_style))
    elems.append(Paragraph("PCB Defect Detection Report",
                           ParagraphStyle('sub', parent=styles['Heading2'],
                                          alignment=TA_CENTER)))
    elems.append(Spacer(1, 0.3 * inch))

    meta_table_data = [
        ["Report ID:", report_id],
        ["Generated On:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Model:", "ResNet50 (custom)"],
        ["Device:", str(DEVICE)],
    ]
    meta_table = Table(meta_table_data,
                       colWidths=[2.2 * inch, 3.6 * inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#2c3e50")),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ]))
    elems.append(meta_table)
    elems.append(PageBreak())

    elems.append(Paragraph("Executive Summary", styles['Heading2']))
    summary_text = params.get(
        "summary_text",
        f"Detected {stats['total_defects']} defects."
    )
    elems.append(Paragraph(summary_text, styles['BodyText']))
    elems.append(Spacer(1, 0.2 * inch))

    stats_table_data = [
        ["Metric", "Value"],
        ["Total Defects", str(stats['total_defects'])],
        ["High Confidence (>=80%)", str(stats['high_count'])],
        ["Medium Confidence (60–80%)", str(stats['med_count'])],
        ["Low Confidence (<60%)", str(stats['low_count'])],
        ["Average Confidence", f"{stats['avg_conf']*100:.2f}%"],
    ]
    st = Table(stats_table_data,
               colWidths=[3 * inch, 2 * inch])
    st.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor("#3498db")),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elems.append(st)
    elems.append(PageBreak())

    # difference + mask
    diff_path = save_plot_temp(diff_np, f"diff_{report_id}")
    mask_path = save_plot_temp(mask_np, f"mask_{report_id}")
    ann_path = save_plot_temp(annotated_np, f"annotated_{report_id}")

    elems.append(Paragraph("Input Images & Maps", styles['Heading2']))
    elems.append(Paragraph("Difference Map", styles['Heading3']))
    elems.append(RLImage(diff_path, width=6 * inch, height=3 * inch))
    elems.append(Spacer(1, 0.15 * inch))
    elems.append(Paragraph("Binary Mask", styles['Heading3']))
    elems.append(RLImage(mask_path, width=6 * inch, height=3 * inch))
    elems.append(PageBreak())

    # charts
    bar_img = plot_bar(predictions)
    pie_img = plot_pie(predictions)
    scatter_img = plot_scatter(predictions)
    hist_img = plot_confidence_hist(predictions)

    bpath = save_plot_temp(bar_img, f"bar_{report_id}")
    ppath = save_plot_temp(pie_img, f"pie_{report_id}")
    spath = save_plot_temp(scatter_img, f"scatter_{report_id}")
    hpath = save_plot_temp(hist_img, f"hist_{report_id}")

    elems.append(Paragraph("Statistical Charts", styles['Heading2']))
    elems.append(Paragraph("Defect Count by Class", styles['Heading3']))
    elems.append(RLImage(bpath, width=6 * inch, height=3 * inch))
    elems.append(Spacer(1, 0.15 * inch))
    elems.append(Paragraph("Defect Distribution", styles['Heading3']))
    elems.append(RLImage(ppath, width=5 * inch, height=4 * inch))
    elems.append(PageBreak())

    elems.append(Paragraph("Spatial Distribution & Confidence",
                           styles['Heading2']))
    elems.append(RLImage(spath, width=6 * inch, height=3 * inch))
    elems.append(Spacer(1, 0.15 * inch))
    elems.append(RLImage(hpath, width=6 * inch, height=3 * inch))
    elems.append(PageBreak())

    elems.append(Paragraph("Annotated PCB & Defect Table",
                           styles['Heading2']))
    elems.append(RLImage(ann_path, width=6 * inch, height=4 * inch))
    elems.append(Spacer(1, 0.2 * inch))

    if predictions:
        table_data = [["ID", "Type", "Confidence",
                       "Position (X,Y)", "Size (W×H)"]]
        for p in predictions:
            table_data.append([
                f"#{p['defect_id']}",
                p['class'],
                f"{p['confidence']*100:.1f}%",
                f"({p['position_xy'][0]},{p['position_xy'][1]})",
                p['size_wh'],
            ])
        tbl = Table(table_data,
                    colWidths=[0.7 * inch, 1.5 * inch,
                               1.2 * inch, 1.6 * inch, 1.2 * inch])
        tbl.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0),
             colors.HexColor("#e74c3c")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ]))
        elems.append(tbl)
    else:
        elems.append(Paragraph("No high-confidence defects detected.",
                               styles['BodyText']))
    elems.append(PageBreak())

    elems.append(Paragraph("Recommendations", styles['Heading2']))
    elems.append(Paragraph(params.get(
        "recommendations",
        "If no defects detected, lower diff threshold or min area. "
        "Inspect high-confidence defects first."
    ), styles['BodyText']))
    elems.append(Spacer(1, 0.2 * inch))
    elems.append(Paragraph("Conclusion", styles['Heading2']))
    elems.append(Paragraph(params.get(
        "conclusion_text",
        "Automated inspection completed."
    ), styles['BodyText']))

    doc.build(elems)
    return pdf_path

# ---------------------------
# FASTAPI APP
# ---------------------------
app = FastAPI(title="CircuitGuard PCB Detection API - Advanced",
              version="1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# --------- Frontend pages ----------
@app.get("/", response_class=HTMLResponse)
def homepage():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h3>index.html not found in /frontend</h3>", status_code=404)


@app.get("/history", response_class=HTMLResponse)
def history_page():
    hist_path = os.path.join(FRONTEND_DIR, "history.html")
    if os.path.exists(hist_path):
        with open(hist_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h3>history.html not found in /frontend</h3>", status_code=404)

# --------- APIs ----------
@app.get("/api/history")
def history_api(limit: int = 50):
    try:
        runs = list_runs(limit=limit)
        return {"runs": runs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_file_exists": os.path.exists(MODEL_PATH),
        "model_loaded": CLASSIFIER_MODEL is not None,
        "torch_available": TORCH_AVAILABLE,
    }

# --------- PREDICT ----------
@app.post("/predict")
async def predict(
    template_file: UploadFile = File(...),
    test_file: UploadFile = File(...),
    difference_threshold: int = Form(30),
    min_roi_area: int = Form(50),
    confidence_threshold: float = Form(0.6),
    generate_report: bool = Form(False)
):
    report_id = f"RPT-{uuid.uuid4().hex[:8].upper()}"
    try:
        start = time.time()

        t_bytes = await template_file.read()
        s_bytes = await test_file.read()
        t_np = cv2.imdecode(np.frombuffer(t_bytes, np.uint8), cv2.IMREAD_COLOR)
        s_np = cv2.imdecode(np.frombuffer(s_bytes, np.uint8), cv2.IMREAD_COLOR)
        if t_np is None or s_np is None:
            raise ValueError("Invalid image files uploaded.")

        defects, diff_img, mask_img = perform_defect_detection(
            s_np, t_np,
            diff_threshold=difference_threshold,
            min_area=min_roi_area
        )

        predictions = []
        for d in defects:
            cl, conf = classify_roi(d["roi_image"])
            # 🔑 IMPORTANT: ignore unknown OR low confidence
            if cl != "unknown" and conf >= confidence_threshold:
                predictions.append({
                    "defect_id": d["defect_id"],
                    "class": cl,
                    "confidence": float(conf),
                    "bbox": d["bbox"],
                    "position_xy": d["position_xy"],
                    "size_wh": d["size_wh"],
                    "area": d.get("area", 0)
                })

        annotated = annotate_image(s_np.copy(), predictions)

        # plots
        bar_img = plot_bar(predictions)
        pie_img = plot_pie(predictions)
        scatter_img = plot_scatter(predictions)
        hist_img = plot_confidence_hist(predictions)

        processing_time = round(time.time() - start, 2)
        total = len(predictions)
        high = sum(1 for p in predictions if p['confidence'] >= 0.8)
        med = sum(1 for p in predictions if 0.6 <= p['confidence'] < 0.8)
        low = sum(1 for p in predictions if p['confidence'] < 0.6)
        avg_conf = float(np.mean([p['confidence'] for p in predictions]) if predictions else 0.0)

        # Save originals and analysis images (for history/downloads)
        save_cv2_image(t_np, f"template_{report_id}.png")
        save_cv2_image(s_np, f"test_{report_id}.png")
        save_cv2_image(annotated, f"annotated_{report_id}.png")
        save_cv2_image(diff_img, f"diff_{report_id}.png")
        save_cv2_image(mask_img, f"mask_{report_id}.png")
        save_plot_temp(bar_img, f"bar_{report_id}")
        save_plot_temp(pie_img, f"pie_{report_id}")
        save_plot_temp(scatter_img, f"scatter_{report_id}")
        save_plot_temp(hist_img, f"hist_{report_id}")

        # CSV
        csv_path = _write_csv(report_id, predictions)
        csv_name = os.path.basename(csv_path)

        pdf_name = None
        if generate_report:
            params = {
                "summary_text": (
                    f"Detected {total} defect(s). "
                    f"High:{high}, Medium:{med}, Low:{low}. "
                    f"Avg conf: {avg_conf*100:.2f}%."
                ),
                "parameters": {
                    "difference_threshold": difference_threshold,
                    "min_roi_area": min_roi_area,
                    "confidence_threshold": confidence_threshold,
                },
                "recommendations": (
                    "Inspect high-confidence regions first. "
                    "If too many false positives, increase min ROI area "
                    "or confidence threshold."
                ),
                "conclusion_text": (
                    f"Automated inspection completed in "
                    f"{processing_time:.2f} seconds."
                ),
            }
            pdf_path = generate_pdf_report(
                report_id,
                params,
                {
                    "total_defects": total,
                    "high_count": high,
                    "med_count": med,
                    "low_count": low,
                    "avg_conf": avg_conf,
                },
                annotated, diff_img, mask_img, predictions
            )
            pdf_name = os.path.basename(pdf_path)

        # store metadata for history
        run_meta = {
            "report_id": report_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": {
                "difference_threshold": difference_threshold,
                "min_roi_area": min_roi_area,
                "confidence_threshold": confidence_threshold,
            },
            "stats": {
                "total": total,
                "high": high,
                "med": med,
                "low": low,
                "avg_conf": avg_conf,
            },
            "files": {
                "pdf": pdf_name,
                "csv": csv_name,
                "template": f"template_{report_id}.png",
                "test": f"test_{report_id}.png",
                "annotated": f"annotated_{report_id}.png",
                "diff": f"diff_{report_id}.png",
                "mask": f"mask_{report_id}.png",
                "bar": f"bar_{report_id}.png",
                "pie": f"pie_{report_id}.png",
                "scatter": f"scatter_{report_id}.png",
                "hist": f"hist_{report_id}.png",
            },
        }
        store_run_metadata(run_meta)

        response = {
            "status": "success",
            "report_id": report_id,
            "processing_time_seconds": processing_time,
            "total_defects_detected": total,
            "statistics": {
                "high_count": high,
                "med_count": med,
                "low_count": low,
                "avg_conf": avg_conf,
            },
            "predictions": predictions,
            "overview_images": {
                "annotated_results": to_b64_from_cv2(annotated),
                "difference_image": to_b64_from_cv2(diff_img),
                "binary_mask": to_b64_from_cv2(mask_img),
                "bar_plot": to_b64_from_cv2(bar_img),
                "pie_plot": to_b64_from_cv2(pie_img),
                "scatter_plot": to_b64_from_cv2(scatter_img),
                "confidence_hist": to_b64_from_cv2(hist_img),
            },
            "csv_path": csv_name,
            "pdf_path": pdf_name,
        }

        return JSONResponse(response)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# --------- IMAGE / CSV / PDF DOWNLOADS ----------
@app.get("/view-image/{report_id}/{kind}")
def view_image(report_id: str, kind: str):
    try:
        path = kind_to_filename(report_id, kind)
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"No file for kind='{kind}' and report_id='{report_id}'."
            )
        headers = {"Content-Disposition":
                   f"inline; filename={os.path.basename(path)}"}
        return FileResponse(path=path, media_type="image/png", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/download-image/{report_id}/{kind}")
def download_image(report_id: str, kind: str):
    try:
        path = kind_to_filename(report_id, kind)
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"No file for kind='{kind}' and report_id='{report_id}'."
            )
        return FileResponse(path=path,
                            filename=os.path.basename(path),
                            media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/download-csv/{report_id}")
def download_csv(report_id: str):
    try:
        csv_path = os.path.join(OUTPUTS_DIR, f"defects_{report_id}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError("CSV not found for this report.")
        return FileResponse(path=csv_path,
                            filename=os.path.basename(csv_path),
                            media_type="text/csv")
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/download-pdf/{filename}")
def download_pdf(filename: str):
    try:
        if filename.lower() == "last":
            pdfs = [f for f in os.listdir(OUTPUTS_DIR)
                    if f.lower().endswith(".pdf")]
            if not pdfs:
                raise FileNotFoundError(
                    "No PDFs found. Run analysis with generate_report=True first."
                )
            pdfs.sort(key=lambda x: os.path.getmtime(
                os.path.join(OUTPUTS_DIR, x)),
                reverse=True)
            filepath = os.path.join(OUTPUTS_DIR, pdfs[0])
        else:
            filepath = os.path.join(OUTPUTS_DIR, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"PDF not found: {filename}")

        return FileResponse(path=filepath,
                            filename=os.path.basename(filepath),
                            media_type="application/pdf")
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

# ---------------------------
# RUNNER
# ---------------------------
if __name__ == "__main__":
    print("Starting CircuitGuard backend → http://127.0.0.1:8000/")
    uvicorn.run(app, host="127.0.0.1", port=8000)
