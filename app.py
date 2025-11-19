import os
import io
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from fpdf import FPDF
from PIL import Image
from backend.backend import load_model, run_inference
import tempfile


MODEL_PATH = r"C:\Users\laksh\OneDrive\Desktop\coding\Circuitguard_Project\training\best_model.pth"
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

st.set_page_config(page_title="CircuitGuard - PCB Defect Detection", layout="wide")
st.title("🛠 CircuitGuard — PCB Defect Detection")


col1, col2 = st.columns(2)
with col1:
    template_file = st.file_uploader("📄 Upload Template Image", type=["jpg", "jpeg", "png"], key="template")
with col2:
    test_file = st.file_uploader("⚙ Upload Test Image", type=["jpg", "jpeg", "png"], key="test")


if template_file and test_file:
    template_image = np.array(Image.open(template_file).convert("RGB"))
    test_image = np.array(Image.open(test_file).convert("RGB"))
    preview_cols = st.columns(2)
    preview_cols[0].image(template_image, caption="Template Image", use_container_width=True)
    preview_cols[1].image(test_image, caption="Test Image", use_container_width=True)


    @st.cache_resource
    def _load_model(path, device):
        return load_model(path, device=device)

    try:
        model = _load_model(MODEL_PATH, DEVICE)
    except Exception as e:
        st.error(f"Failed to load model from {MODEL_PATH}: {e}")
        st.stop()

    if st.button("🔍 Analyze and Generate Report"):
        with st.spinner("Detecting defects and preparing report..."):
            result = run_inference(template_image, test_image, model, DEVICE)
            counts = result.get("counts", {})
            time_taken = result.get("time", 0.0)
            annotated = result.get("annotated")
            detections = result.get("detections", [])
            total_defects = sum(counts.values())

        
            temp_annotated = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            if annotated is None:
                st.error("No annotated image returned by backend.")
                st.stop()
            pil_annot = Image.fromarray(annotated[..., ::-1])
            pil_annot.save(temp_annotated.name)

        
            labels = list(counts.keys())
            values = list(counts.values())

        
            bar_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
            fig, ax = plt.subplots(figsize=(5, 3))
            if values:
                cmap = plt.cm.Blues(np.linspace(0.4, 0.8, max(1, len(labels))))
                ax.bar(labels, values, color=cmap, edgecolor="black")
            ax.set_title("Defect Count Distribution", color="#002e5b", fontweight="bold", fontsize=10)
            ax.tick_params(axis='x', labelrotation=30, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            plt.tight_layout()
            plt.savefig(bar_path, dpi=150)
            plt.close()

    
            pie_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
            fig, ax = plt.subplots(figsize=(4, 3))
            if values:
                ax.pie(values, labels=None, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab10.colors)
                ax.legend(labels, loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=7)
            ax.set_title("Defect Type Percentage", color="#002e5b", fontweight="bold", fontsize=10)
            plt.tight_layout()
            plt.savefig(pie_path, dpi=150, bbox_inches="tight")
            plt.close()

            
            pdf = FPDF(orientation="P", unit="mm", format="A4")
            pdf.set_auto_page_break(auto=True, margin=12)
            pdf.add_page()

    
            pdf.set_fill_color(0, 102, 204)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 12, " CIRCUITGUARD - PCB DEFECT ANALYSIS REPORT", ln=1, align="L", fill=True)
            pdf.ln(4)

        
            pdf.set_draw_color(0, 0, 0)
            pdf.set_line_width(0.5)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)

    
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", "BU", 13)
            pdf.cell(0, 8, "1. Summary", ln=1, align="L")
            pdf.set_font("Arial", "", 11)
            pdf.cell(0, 6, f"Total defects detected: {total_defects}", ln=1, align="L")
            pdf.cell(0, 6, f"Processing time (sec): {time_taken:.3f}", ln=1, align="L")
            pdf.ln(5)

            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)

    
            pdf.set_font("Arial", "BU", 13)
            pdf.cell(0, 8, "2. Defect Count Summary", ln=1, align="L")
            pdf.set_font("Arial", "B", 11)
            pdf.set_fill_color(0, 102, 204)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(110, 9, "Defect Type", border=1, align="C", fill=True)
            pdf.cell(60, 9, "Count", border=1, align="C", fill=True)
            pdf.ln(9)

            pdf.set_font("Arial", "", 11)
            row_fill = True
            for label, count in counts.items():
                pdf.set_fill_color(235, 242, 254) if row_fill else pdf.set_fill_color(255, 255, 255)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(110, 8, str(label), border=1, align="C", fill=True)
                pdf.cell(60, 8, str(count), border=1, align="C", fill=True)
                pdf.ln(8)
                row_fill = not row_fill
            pdf.ln(5)

            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)

            
            pdf.set_font("Arial", "BU", 13)
            pdf.cell(0, 8, "3. Statistical Visualizations", ln=1, align="L")

            
            pdf.image(bar_path, x=35, w=120)
            pdf.ln(30)  


            pdf.image(pie_path, x=45, w=110)
            pdf.ln(40)  

            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)

    
            pdf.set_font("Arial", "BU", 13)
            pdf.cell(0, 8, "4. Annotated PCB (Defects Highlighted)", ln=1, align="L")
            img_w = 120
            x_center = (210 - img_w) / 2
            pdf.image(temp_annotated.name, x=x_center, w=img_w)
            pdf.ln(10)

            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)

        
            if detections:
                pdf.set_font("Arial", "BU", 13)
                pdf.cell(0, 8, "5. Defect Position Summary", ln=1, align="L")
                pdf.set_font("Arial", "B", 11)
                pdf.set_fill_color(0, 102, 204)
                pdf.set_text_color(255, 255, 255)
                headers = ["Type", "X (%)", "Y (%)", "Width (%)", "Height (%)", "Confidence"]
                col_widths = [35, 25, 25, 25, 25, 30]
                for h, w in zip(headers, col_widths):
                    pdf.cell(w, 8, h, border=1, align="C", fill=True)
                pdf.ln(8)

                pdf.set_font("Arial", "", 10)
                pdf.set_text_color(0, 0, 0)
                row_fill = True
                for d in detections:
                    pdf.set_fill_color(240, 247, 255) if row_fill else pdf.set_fill_color(255, 255, 255)
                    row_fill = not row_fill
                    label = d["label"]
                    x, y, w, h = d["box"]
                    conf = d["score"]
                    vals = [label, f"{x:.1f}", f"{y:.1f}", f"{w:.1f}", f"{h:.1f}", f"{conf:.1f}%"]
                    for v, cw in zip(vals, col_widths):
                        pdf.cell(cw, 8, v, border=1, align="C", fill=True)
                    pdf.ln(8)
                pdf.ln(5)

            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)

    
            pdf.set_font("Arial", "BU", 13)
            pdf.cell(0, 8, "6. Model Insights & Observations", ln=1, align="L")
            pdf.set_font("Arial", "", 11)
            if counts:
                most_common = max(counts, key=counts.get)
                pdf.multi_cell(0, 8,
                               f"- The most frequently detected defect is '{most_common}'.\n"
                               f"- The detection was completed in {time_taken:.2f}s.\n"
                               f"- Defects are clearly highlighted for inspection.\n")
            else:
                pdf.multi_cell(0, 8, "No major defects detected. PCB quality appears optimal.")
            pdf.ln(8)

            

            
            pdf_output = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
            pdf.output(pdf_output)

        with open(pdf_output, "rb") as f:
            st.download_button(
                label="📥 Download Final PCB Defect Report (PDF)",
                data=f,
                file_name="CircuitGuard_Final_Report.pdf",
                mime="application/pdf"
            )

else:
    st.info("📸 Upload both Template and Test images to begin defect analysis.")
