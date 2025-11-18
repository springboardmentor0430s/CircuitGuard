import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torchvision import transforms
from torchvision.models import efficientnet_b4
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle,KeepTogether,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
import tempfile
import datetime
import os

# ==========================
# CONFIGURATION
# ==========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['copper', 'mousebite', 'noise', 'open', 'pin-hole', 'short', 'spur']
MODEL_PATH = "efficientnet_b4_best.pth"

DEFECT_COLORS = {
    "copper": (0, 255, 255),
    "mousebite": (255, 0, 0),
    "noise": (255, 255, 0),
    "open": (0, 128, 255),
    "pin-hole": (255, 0, 255),
    "short": (0, 255, 0),
    "spur": (128, 0, 255),
}

# ==========================
# MODEL LOADER
# ==========================
@st.cache_resource
def load_model():
    model = efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    state = torch.load(MODEL_PATH, map_location=DEVICE)
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
# DEFECT DETECTION
# ==========================
def detect_defects(model, ref_img, test_img, kernel_size, min_defect_area, conf_thresh):
    diff = cv2.absdiff(ref_img, test_img)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated = test_img.copy()
    heatmap = np.zeros_like(gray, dtype=np.float32)
    defect_records = []
    H, W, _ = test_img.shape

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_defect_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if x + w <= 40:
            continue
        if x < 40:
            offset = 40 - x
            x += offset
            w -= offset
            if w <= 0:
                continue

        x1, y1 = max(0, x - 10), max(0, y - 10)
        x2, y2 = min(W, x + w + 10), min(H, y + h + 10)
        roi = test_img[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        roi_tensor = preprocess_image(roi).to(DEVICE)
        with torch.no_grad():
            outputs = model(roi_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)

        conf = confidence.item() * 100
        if conf < conf_thresh * 100:
            continue

        label = CLASS_NAMES[pred_idx.item()]
        color = DEFECT_COLORS.get(label, (0, 255, 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{label} ({conf:.1f}%)", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(annotated, f"{label} ({conf:.1f}%)", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        heatmap[y1:y2, x1:x2] += conf / 100.0
        defect_records.append({"Label": label, "Confidence": f"{conf:.2f}%", "X": x1, "Y": y1, "Width": w, "Height": h})

    heatmap = np.uint8(255 * (heatmap / heatmap.max())) if np.max(heatmap) > 0 else heatmap
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(test_img, 0.6, heatmap_color, 0.4, 0)
    return annotated, cleaned, overlay, defect_records

# ==========================
# STREAMLIT UI
# ==========================
st.set_page_config(page_title="PCB Defect Detector", layout="wide")
st.title("🔍 Automated PCB Defect Detection & Classification")

col_upload1, col_upload2 = st.columns(2)
with col_upload1:
    uploaded_temp = st.file_uploader("Upload Template Image", type=["jpg", "png", "jpeg"], key="temp")
with col_upload2:
    uploaded_test = st.file_uploader("Upload Test (Defective) Image", type=["jpg", "png", "jpeg"], key="test")

st.sidebar.header("⚙️ Settings Panel")
kernel_choice = st.sidebar.selectbox("Blur Radius", [3, 5, 7], index=1)
min_area = st.sidebar.slider("Minimum Defect Area", 5, 100, 8)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.8, 0.05)
include_heatmap = st.sidebar.radio("Include Heatmap?", ["No", "Yes"], index=0)

# Initialize session state for PDF persistence
if "pdf_ready" not in st.session_state:
    st.session_state["pdf_ready"] = None

if st.button("🚀 Run Defect Detection"):
    if uploaded_temp and uploaded_test:
        ref_img = cv2.imdecode(np.frombuffer(uploaded_temp.read(), np.uint8), cv2.IMREAD_COLOR)
        test_img_full = cv2.imdecode(np.frombuffer(uploaded_test.read(), np.uint8), cv2.IMREAD_COLOR)

        kernel_map = {3: (3, 3), 5: (5, 5), 7: (7, 7)}
        kernel_choice_tuple = kernel_map.get(kernel_choice, (3, 3))

        model = load_model()
        annotated, cleaned, heatmap, defect_data = detect_defects(
            model, ref_img, test_img_full, kernel_choice_tuple, min_area, confidence_threshold
        )

        # Store in session
        st.session_state["results_ready"] = True
        st.session_state["ref_img"] = ref_img
        st.session_state["test_img_full"] = test_img_full
        st.session_state["cleaned"] = cleaned
        st.session_state["annotated"] = annotated
        st.session_state["heatmap"] = heatmap
        st.session_state["df"] = pd.DataFrame(defect_data)
        st.session_state["include_heatmap"] = include_heatmap
        st.session_state["kernel_choice_tuple"] = kernel_choice_tuple
        st.session_state["min_area"] = min_area
        st.session_state["confidence_threshold"] = confidence_threshold
    else:
        st.error("Please upload both Template and Test images before running detection.")

# ==========================
# DISPLAY RESULTS
# ==========================
if st.session_state.get("results_ready"):
    ref_img = st.session_state["ref_img"]
    test_img_full = st.session_state["test_img_full"]
    cleaned = st.session_state["cleaned"]
    annotated = st.session_state["annotated"]
    heatmap = st.session_state["heatmap"]
    df = st.session_state["df"]
    include_heatmap = st.session_state["include_heatmap"]

    # --------------------------
    # INPUT IMAGES
    # --------------------------
    st.subheader("🖼️ Input Images")
    col1, col2 = st.columns(2)
    with col1:
        st.image(ref_img, caption="Template Image", use_container_width=True)
    with col2:
        st.image(test_img_full, caption="Test Image", use_container_width=True)

    # --------------------------
    # OUTPUT IMAGES
    # --------------------------
    st.subheader("🖼️ Output Images")
    if include_heatmap == "Yes":
        col3, col4, col5 = st.columns(3)
        with col3:
            st.image(cleaned, caption="Thresholded (Cleaned)", use_container_width=True)
        with col4:
            st.image(annotated, caption="Annotated Output", use_container_width=True)
        with col5:
            st.image(heatmap, caption="Defect Heatmap", use_container_width=True)
    else:
        col3, col4 = st.columns(2)
        with col3:
            st.image(cleaned, caption="Thresholded (Cleaned)", use_container_width=True)
        with col4:
            st.image(annotated, caption="Annotated Output", use_container_width=True)

    # --------------------------
    # DEFECT LOG + ANALYTICS
    # --------------------------
    if not df.empty:
        # Tabs for switching between Analysis and Charts
        tab1, tab2 = st.tabs(["🖼 Image Analysis", "📊 Charts Dashboard"])

        # ============================
        # 🖼 TAB 1 — IMAGE ANALYSIS
        # ============================
        with tab1:
            st.subheader("📋 Detected Defect Log")
            st.dataframe(df, use_container_width=True)

            #st.subheader("🧠 Image & Detection Results")
            # You can keep your detection result image display here
            # st.image(processed_image, caption="Detected Defects", use_container_width=True)

        # ============================
        # 📊 TAB 2 — CHARTS DASHBOARD
        # ============================
        with tab2:
            st.subheader("📊 Defect Distribution & Scatterplot Analysis")

            # Convert Confidence to numeric (e.g., "95%" -> 95)
            df["Confidence_Value"] = df["Confidence"].str.replace("%", "").astype(float)

            # ✅ Auto-create 'Area' if missing
            if "Area" not in df.columns and {"Width", "Height"}.issubset(df.columns):
                df["Area"] = df["Width"].astype(float) * df["Height"].astype(float)

            if "Area" in df.columns:
                df["Area"] = pd.to_numeric(df["Area"], errors="coerce").fillna(0)

            # Define color mapping for defects (BGR → RGB)
            DEFECT_COLORS = {
                "copper": (0, 255, 255),
                "mousebite": (255, 0, 0),
                "noise": (255, 255, 0),
                "open": (0, 128, 255),
                "pin-hole": (255, 0, 255),
                "short": (0, 255, 0),
                "spur": (128, 0, 255),
            }

            # Convert BGR → Matplotlib RGB (0–1 range)
            DEFECT_COLORS_RGB = {
                k: tuple(np.array(v[::-1]) / 255.0) for k, v in DEFECT_COLORS.items()
            }

            # Common defect counts
            defect_counts = df["Label"].value_counts()

            # --------------------------
            # 📊 TOP ROW: PIE + BAR + SCATTER
            # --------------------------
            col1, col2, col3 = st.columns(3)

            # --- Pie Chart ---
            with col1:
                fig1, ax1 = plt.subplots(figsize=(3.8, 3.8))
                ax1.pie(defect_counts.values, labels=defect_counts.index,
                        autopct="%1.1f%%", startangle=90)
                ax1.set_title("Defect Percentage")
                st.pyplot(fig1)

            # --- Bar Chart ---
            with col2:
                fig2, ax2 = plt.subplots(figsize=(3.8, 3.8))
                ax2.bar(defect_counts.index, defect_counts.values)
                ax2.set_title("Defect Count by Type")
                ax2.set_xlabel("Defect Type")
                ax2.set_ylabel("Count")
                plt.xticks(rotation=45)
                st.pyplot(fig2)

            # --- Scatter Plot (Confidence vs Area) ---
            with col3:
                if "Area" in df.columns and df["Area"].sum()>0:
                    fig3, ax3 = plt.subplots(figsize=(3.8, 3.8))

                    for label in df["Label"].unique():
                        subset = df[df["Label"] == label]
                        color = DEFECT_COLORS_RGB.get(label, (0.5, 0.5, 0.5))  # default gray
                        ax3.scatter(
                            subset["Area"], subset["Confidence_Value"],
                            color=[color],
                            label=label,
                            s=60,
                            alpha=0.8,
                            edgecolors='k'
                        )
                    ax3.set_title("Confidence vs Defect Area",fontsize=10)
                    ax3.set_xlabel("Defect Area",fontsize=8)
                    ax3.set_ylabel("Confidence (%)",fontsize=8)
                    ax3.legend(title="Defect Type",loc="lower right",title_fontsize=6,fontsize=6,framealpha=0.6)
                    fig3.tight_layout(rect=[0, 0, 1, 0.95])
                    st.pyplot(fig3)

        # --------------------------
        # EXPORT RESULTS
        # --------------------------
        st.subheader("📤 Export Results")

        timestamp = datetime.datetime.now().strftime("%d/%m/%Y at %I:%M %p")
        friendly_intro = f"Hi there! Welcome — this is a Printed Circuit Board (PCB) Defect Detection Report generated on {timestamp}."

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_pdf_path = f"{tmpdir}/PCB_Report.pdf"

            # Save temporary images
            paths = {
                "ref": f"{tmpdir}/template.jpg",
                "test": f"{tmpdir}/test.jpg",
                "th": f"{tmpdir}/thresholded.jpg",
                "ann": f"{tmpdir}/annotated.jpg",
                "hm": f"{tmpdir}/heatmap.jpg",
                "pie": f"{tmpdir}/pie_chart.jpg",
                "bar": f"{tmpdir}/bar_chart.jpg",
                "scatter": f"{tmpdir}/scatter_chart.jpg"
            }
            cv2.imwrite(paths["ref"], ref_img)
            cv2.imwrite(paths["test"], test_img_full)
            cv2.imwrite(paths["th"], cleaned)
            cv2.imwrite(paths["ann"], annotated)
            if include_heatmap == "Yes":
                cv2.imwrite(paths["hm"], heatmap)

            fig1.tight_layout(pad=2.0)
            fig2.tight_layout(pad=2.0)
            fig1.savefig(paths["pie"], bbox_inches="tight", dpi=200)
            fig2.savefig(paths["bar"], bbox_inches="tight", dpi=200)
            fig3.tight_layout(pad=2.0)
            fig3.savefig(paths["scatter"], bbox_inches="tight", dpi=200)

            # PDF setup
            doc = SimpleDocTemplate(temp_pdf_path, pagesize=A4)
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(name="SectionHeader", fontSize=17, leading=16, spaceAfter=8,
                                      textColor="#003366", fontName="Times-BoldItalic"))
            styles.add(ParagraphStyle(name="SummaryBox", backColor="#F5F5F5", borderColor="#CCCCCC",
                                      borderWidth=0.5, borderPadding=6, fontSize=11, leading=14))
            caption_style = ParagraphStyle(name="Caption", fontSize=9, alignment=1,
                                           textColor=colors.black, spaceBefore=3)

            story = []

            def add_border(canvas, doc):
                canvas.saveState()
                canvas.setLineWidth(2)
                canvas.setStrokeColor(colors.HexColor("#003366"))
                margin = 20
                canvas.rect(margin, margin, A4[0] - 2 * margin, A4[1] - 2 * margin)
                canvas.restoreState()

            # -------- PAGE 1: HEADER + INTRO --------
            story.append(Paragraph("<b>PCB Defect Detection Report</b>",
                                   ParagraphStyle("Title", fontSize=24, leading=22, alignment=1,
                                                  textColor="#002147")))
            story.append(Spacer(1, 14))

            now = datetime.datetime.now()
            current_date = now.strftime("%B %d, %Y")
            time_str = now.strftime("%I:%M %p")

            friendly_intro = (
                f"Hi there! Welcome — this is an automatically generated <b>PCB Defect Detection Report</b> "
                f"created on <b>{current_date}</b> at <b>{time_str}</b>. "
                f"This report compares a <b>Reference PCB image</b> with a <b>Test PCB image</b> using advanced "
                f"deep learning techniques (EfficientNet-B4) and classical image processing methods. "
                f"The goal is to accurately detect, localize, and classify possible defects, ensuring high board reliability "
                f"and production quality."
            )
            story.append(Paragraph(friendly_intro, ParagraphStyle("Intro", fontSize=11, leading=14, alignment=0)))
            story.append(Spacer(1, 15))

            # Objective
            story.append(Paragraph("<u><b>Objective</b></u>", styles["SectionHeader"]))
            story.append(Paragraph(
                "The objective of this report is to automatically identify and highlight defects such as open circuits, "
                "shorts, mouse bites, and pinholes on PCB surfaces. It provides visual insights, statistical analysis, "
                "and confidence-based defect evaluations to assist in automated quality inspection and decision-making.",
                ParagraphStyle("Body", fontSize=11, leading=14, alignment=0)
            ))
            story.append(Spacer(1, 10))

            # Confidence Note
            story.append(Paragraph("<u><b>Confidence Note</b></u>", styles["SectionHeader"]))
            story.append(Paragraph(
                "Each detected defect is assigned a confidence score ranging from 0% to 100%, representing "
                "the model’s certainty that a specific region contains the defect type. "
                "A higher confidence value indicates stronger model assurance and lower likelihood of misclassification.",
                ParagraphStyle("Body", fontSize=11, leading=14, alignment=0)
            ))
            story.append(Spacer(1, 50))

            # Input Images
            story.append(Paragraph("<u><b>Input Images</b></u>", styles["SectionHeader"]))
            story.append(Spacer(1, 6))
            story.append(Table([
                [RLImage(paths["ref"], width=240, height=180),
                 RLImage(paths["test"], width=240, height=180)]
            ], hAlign="CENTER"))
            story.append(Table([
                [Paragraph("Reference Image", caption_style),
                 Paragraph("Defected Image", caption_style)]
            ], hAlign="CENTER"))
            story.append(PageBreak())

            # -------- PAGE 2: OUTPUT IMAGES --------
            story.append(Paragraph("<u><b>Processed Outputs</b></u>", styles["SectionHeader"]))
            story.append(Spacer(1, 8))
            story.append(Table([
                [RLImage(paths["th"], width=240, height=180),
                 RLImage(paths["ann"], width=240, height=180)]
            ], hAlign="CENTER"))
            story.append(Table([
                [Paragraph("Thresholded Image", caption_style),
                 Paragraph("Annotated Image", caption_style)]
            ], hAlign="CENTER"))

            if include_heatmap == "Yes":
                story.append(Spacer(1, 15))
                story.append(Table([
                    [RLImage(paths["hm"], width=240, height=180)]
                ], hAlign="CENTER"))
                story.append(Table([
                    [Paragraph("Heatmap", caption_style)]
                ], hAlign="CENTER"))

            # -------- PAGE 3: DEFECT LOGS --------
            story.append(PageBreak())
            story.append(Paragraph("<u><b>Detected Defects Log</b></u>", styles["SectionHeader"]))
            story.append(Spacer(1, 4))
            story.append(Paragraph("List of all detected defects with confidence and area:", styles["Normal"]))
            story.append(Spacer(1, 8))

            # ✅ Include only specific columns in the defect log table
            required_columns = ["Label","Confidence", "X", "Y", "Width", "Height"]

            # Keep only those that exist in the DataFrame
            filtered_columns = [col for col in required_columns if col in df.columns]

            # Build table data (header + rows)
            table_data = [filtered_columns] + df[filtered_columns].values.tolist()

            # Create the defect log table
            table = Table(table_data, colWidths=[80, 60, 60, 60, 60], hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ]))
            story.append(table)
            story.append(Spacer(1, 90))


            # -------- PAGE 4: ANALYTICS + SUMMARY --------
            # 🟣 Scatter Plot
            story.append(Paragraph("<u><b>Defect Distribution</b></u>", styles["SectionHeader"]))
            story.append(Spacer(1, 8))
            story.append(Table([
                [RLImage(paths["scatter"], width=320, height=270)]
            ], hAlign="LEFT"))
            story.append(Spacer(1, 8))

            story.append(PageBreak())
            story.append(Paragraph("<u><b>Visual Analytics</b></u>", styles["SectionHeader"]))
            story.append(Spacer(1, 8))

            # 🟣 Pie and Bar Chart Row
            story.append(Table([
                [RLImage(paths["pie"], width=250, height=200),
                RLImage(paths["bar"], width=250, height=200)]
            ], hAlign="CENTER"))
            story.append(Spacer(1, 100))

            # Summary Section
            story.append(Paragraph("<u><b>Inspection Summary</b></u>", styles["SectionHeader"]))
            story.append(Spacer(1, 8))

            total_defects = len(df)
            avg_conf = df["Confidence"].apply(lambda x: float(str(x).replace("%", ""))).mean()
            top_defect = df["Label"].mode()[0]

            summary_text = f"""
            <b>Total Detected Defects:</b> {total_defects}<br/>
            <b>Average Confidence:</b> {avg_conf:.2f}%<br/>
            <b>Most Common Defect:</b> {top_defect}<br/><br/>
            <b>Inspection Parameters:</b><br/>
            • Kernel size used: {st.session_state['kernel_choice_tuple']}<br/>
            • Minimum defect area: {st.session_state['min_area']}<br/>
            • Confidence threshold: {st.session_state['confidence_threshold']:.2f}<br/>
            """
            story.append(Paragraph(summary_text, styles["SummaryBox"]))
            story.append(Spacer(1, 25))

            # Build PDF
            doc.build(story, onFirstPage=add_border, onLaterPages=add_border)

            # Streamlit download button
            with open(temp_pdf_path, "rb") as pdf_file:
                st.session_state["pdf_ready"] = pdf_file.read()

        # --------------------------
        # DOWNLOAD PDF (No Refresh)
        # --------------------------
        if st.session_state["pdf_ready"] is not None:
            st.download_button(
                label="📄 Download Detailed PDF Report",
                data=st.session_state["pdf_ready"],
                file_name=f"PCB_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                key="download_pdf_report",
                use_container_width=True
            )

        # --------------------------
        # DOWNLOAD COMBINED IMAGE
        # --------------------------
        import io

        def ensure_color(img):
            if img is None:
                return np.zeros((400, 400, 3), dtype=np.uint8)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img

        if "combined_image" not in st.session_state:
            st.session_state["combined_image"] = None

        cleaned_fixed = ensure_color(cleaned)
        annotated_fixed = ensure_color(annotated)
        heatmap_fixed = ensure_color(heatmap) if include_heatmap == "Yes" else None

        target_size = (400, 400)
        cleaned_resized = cv2.resize(cleaned_fixed, target_size)
        annotated_resized = cv2.resize(annotated_fixed, target_size)

        if include_heatmap == "Yes" and heatmap_fixed is not None:
            heatmap_resized = cv2.resize(heatmap_fixed, target_size)
            combined = cv2.hconcat([cleaned_resized, annotated_resized, heatmap_resized])
        else:
            combined = cv2.hconcat([cleaned_resized, annotated_resized])

        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        is_success, buffer = cv2.imencode(".jpg", combined_rgb)
        image_bytes = io.BytesIO(buffer)
        st.session_state["combined_image"] = image_bytes

        st.download_button(
            label="🖼️ Download Annotated Image",
            data=st.session_state["combined_image"],
            file_name="PCB_Annotated_Combined.jpg",
            mime="image/jpeg",
            key="download_combined",
            use_container_width=True
        )

    else:
        st.warning("✅ No defects detected above the selected confidence threshold.")
else:
    st.info("👆 Upload images and click 'Run Defect Detection' to begin.")
