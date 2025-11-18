import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
DATA_DIR = r"C:\Users\Dell\Downloads\newpcb-20251027T064831Z-1-001-20251027T105042Z-1-001\newpcb-20251027T064831Z-1-001\newpcb\PCBData\Dataset_split"
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_PATH = "resnet50_pcb_best.pth"

# Constants
IMAGE_SIZE = 128
NUM_CLASSES = 6

# Dataset
class PCBDefectDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images, self.labels = [], []
        self.transform = transform
        classes = sorted([cls for cls in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, cls))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        for cls in classes:
            for img in os.listdir(os.path.join(root_dir, cls)):
                if img.lower().endswith(('.jpg', '.png')):
                    self.images.append(os.path.join(root_dir, cls, img))
                    self.labels.append(self.class_to_idx[cls])
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# Transform
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load Test Data
test_dataset = PCBDefectDataset(TEST_DIR, transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_features, NUM_CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Evaluate
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Evaluating"):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Metrics
cm = confusion_matrix(all_labels, all_preds)
acc = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
print(f"✅ Test Accuracy: {acc:.2f}%")

# Confusion Matrix Plot
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(test_dataset.class_to_idx.keys()),
            yticklabels=list(test_dataset.class_to_idx.keys()))
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# Classification Report
report = classification_report(all_labels, all_preds, target_names=list(test_dataset.class_to_idx.keys()), output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("classification_report.csv", index=True)

# Print brief report
print(classification_report(all_labels, all_preds, target_names=list(test_dataset.class_to_idx.keys())))
print("\n📊 Confusion matrix and metrics saved successfully!")

# Accuracy Summary CSV
summary = pd.DataFrame({"Metric": ["Accuracy"], "Value": [acc]})
summary.to_csv("test_accuracy_summary.csv", index=False)
