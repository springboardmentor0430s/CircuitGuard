import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms
from fpdf import FPDF
import pandas as pd

from src.dataset import PCBDataset
from src.model import build_efficientnet_b4

# ---------------- CONFIG ----------------
DATA_ROOT = "./src/data/processed_rois"
MODEL_PATH = "models/efficientnet_b4_best.pth"
IMG_SIZE = 128
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "outputs/reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

dataset = PCBDataset(DATA_ROOT, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = sorted(dataset.class_to_idx.keys())

# ---------------- LOAD MODEL ----------------
model = build_efficientnet_b4(len(class_names), pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

# ---------------- EVALUATION ----------------
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# ---------------- METRICS ----------------
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print("\n✅ Evaluation Summary:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")

# ---------------- CLASSIFICATION REPORT ----------------
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).transpose()
report_path = os.path.join(OUTPUT_DIR, "classification_report.csv")
report_df.to_csv(report_path)
print(f"📄 Classification report saved to: {report_path}")

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.tight_layout()
plt.savefig(cm_path)
plt.close()

# ---------------- PER-CLASS METRIC CHARTS ----------------
metrics = ["precision", "recall", "f1-score"]
chart_paths = []
for metric in metrics:
    values = [report_dict[c][metric] for c in class_names]
    plt.figure(figsize=(6,4))
    plt.bar(class_names, values, color="teal")
    plt.title(f"{metric.title()} per Class")
    plt.xticks(rotation=30)
    plt.ylim(0,1)
    chart_path = os.path.join(OUTPUT_DIR, f"{metric}_chart.png")
    plt.tight_layout()
    plt.savefig(chart_path)
    chart_paths.append(chart_path)
    plt.close()

# ---------------- PDF REPORT ----------------
def generate_pdf_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "CircuitGuard - Model Evaluation Report", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, "", ln=True)

    # Summary metrics
    pdf.cell(0, 10, "Overall Performance:", ln=True)
    pdf.cell(0, 8, f"Accuracy  : {acc:.4f}", ln=True)
    pdf.cell(0, 8, f"Precision : {prec:.4f}", ln=True)
    pdf.cell(0, 8, f"Recall    : {rec:.4f}", ln=True)
    pdf.cell(0, 8, f"F1 Score  : {f1:.4f}", ln=True)
    pdf.cell(0, 10, "", ln=True)

    # Confusion matrix
    pdf.cell(0, 10, "Confusion Matrix:", ln=True)
    pdf.image(cm_path, x=25, w=160)
    pdf.cell(0, 10, "", ln=True)

    # Charts
    for path in chart_paths:
        pdf.image(path, x=25, w=160)
        pdf.cell(0, 10, "", ln=True)

    # Classification Report Table
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Detailed Classification Report", ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, "", ln=True)

    # Write report_df content (limit to 20 rows per page)
    for i, (cls, row) in enumerate(report_df.iterrows()):
        text = f"{cls:15s}  |  Prec: {row['precision']:.3f}  |  Rec: {row['recall']:.3f}  |  F1: {row['f1-score']:.3f}  |  Support: {int(row['support'])}"
        pdf.cell(0, 6, text, ln=True)
        if (i+1) % 25 == 0 and i < len(report_df) - 1:
            pdf.add_page()
            pdf.set_font("Arial", "", 10)

    save_path = os.path.join(OUTPUT_DIR, "evaluation_report.pdf")
    pdf.output(save_path)
    print(f"📘 Full PDF report saved to: {save_path}")

generate_pdf_report()
