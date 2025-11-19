# ==============================================================
# src/eval.py  --  Evaluate trained EfficientNet-B4 model + Save metrics
# ==============================================================

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import PCBDataset
from src.model import build_efficientnet_b4

# ---------------- CONFIG ----------------
DATA_ROOT = "./src/data/processed_rois"
MODEL_PATH = "models/efficientnet_b4_best.pth"
IMG_SIZE = 128
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ==============================================================
# ✅ MAIN BLOCK (needed for Windows multiprocessing)
# ==============================================================
if __name__ == "__main__":
    print("[INFO] Starting model evaluation...")

    # ---------------- DATA ----------------
    dataset = PCBDataset(DATA_ROOT, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    class_names = sorted(dataset.class_to_idx.keys())

    print(f"[INFO] Loaded {len(dataset)} ROI images across {len(class_names)} classes.")
    print(f"[INFO] Classes: {class_names}")

    # ---------------- MODEL ----------------
    model = build_efficientnet_b4(len(class_names), pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    # ---------------- EVALUATION ----------------
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    # ---------------- METRICS ----------------
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    overall_acc = accuracy_score(y_true, y_pred)

    # Convert report to DataFrame
    df_report = pd.DataFrame(report_dict).transpose()
    df_report["overall_accuracy"] = overall_acc

    # Save to CSV
    os.makedirs("models", exist_ok=True)
    csv_path = os.path.join("models", "evaluation_metrics.csv")
    df_report.to_csv(csv_path, index=True)

    print(f"\n✅ Detailed metrics saved to: {csv_path}")

    # ---------------- CONFUSION MATRIX ----------------
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - CircuitGuard (EfficientNet-B4)")
    plt.tight_layout()

    cm_path = os.path.join("models", "confusion_matrix.png")
    plt.savefig(cm_path)

    # ---------------- PRINT SUMMARY ----------------
    print("\nClassification Report:\n")
    print(df_report.round(4))
    print(f"\nOverall Accuracy: {overall_acc:.4f}")
    print(f"✅ Confusion matrix saved at: {cm_path}")
    print("\n[INFO] Evaluation complete.")
