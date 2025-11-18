import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import seaborn as sns
import pandas as pd
from tqdm import tqdm

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters (match your Module 3)
NUM_CLASSES = 6
IMAGE_SIZE = 128
CLASS_NAMES = ['mousebite', 'open', 'pinhole', 'short', 'spur', 'spurious copper']

# Paths (change if needed)
DATA_DIR = r"C:\Users\Dell\Downloads\newpcb-20251027T064831Z-1-001-20251027T105042Z-1-001\newpcb-20251027T064831Z-1-001\newpcb\PCBData\Dataset_split"
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_PATH = "resnet50_pcb_best.pth"  # Trained model path
OUTPUT_DIR = "annotated_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load trained model (ResNet50 from your training)
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, NUM_CLASSES)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Classification transform (same as training)
classify_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Dataset class
class PCBDefectDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}

        classes = sorted([
            cls for cls in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, cls))
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        for cls in self.class_to_idx:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.images.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, img_path  # Include path for annotation

# Load test dataset
test_dataset = PCBDefectDataset(TEST_DIR, transform=classify_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
print(f"Test samples: {len(test_dataset)}")

# Annotate and save output images
def annotate_image(image_tensor, predicted_label):
    image_pil = transforms.ToPILImage()(image_tensor.cpu())
    annotated = np.array(image_pil)
    label = CLASS_NAMES[predicted_label]
    cv2.putText(annotated, f"Pred: {label}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
    return Image.fromarray(annotated)

# Inference
print("Running inference on test set...")
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels, img_paths in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        for i, (img, pred, path) in enumerate(zip(images, predicted, img_paths)):
            if i < 10:  # Save first 10 annotated outputs
                annotated_img = annotate_image(img, pred.item())
                output_path = os.path.join(OUTPUT_DIR, f"annotated_{os.path.basename(path)}")
                annotated_img.save(output_path)

# Metrics
accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
cm = confusion_matrix(all_labels, all_preds)

print("\n--- Final Evaluation Report ---")
print(f"Test Accuracy: {accuracy * 100:.2f}% (Target: ≥97%)")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# Save report
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Value": [accuracy * 100, precision, recall, f1]
})
metrics_df.to_csv("final_evaluation_report.csv", index=False)

print("\n✅ Evaluation report saved to 'final_evaluation_report.csv'")
print(f"✅ Annotated sample images saved to '{OUTPUT_DIR}'")

# Additional validation: match rate
matches = sum(1 for pred, true in zip(all_preds, all_labels) if pred == true)
match_rate = matches / len(all_labels) * 100
print(f"Prediction Match Rate: {match_rate:.2f}%")
