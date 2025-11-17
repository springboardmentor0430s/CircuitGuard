import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms, models, datasets
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===================================================
# PATHS AND SETTINGS
# ===================================================
ROI_DIR = r"C:\Users\prati\Desktop\PCB_DATASET\outputs\roi_images"
MODEL_SAVE_PATH = r"C:\Users\prati\Desktop\PCB_DATASET\outputs\efficientnet_b0_pcb.pth"
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
INPUT_SIZE = 128  # smaller input to speed up training

# ===================================================
# CUSTOM DATASET
# ===================================================
class ROIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        self.class_names = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}

        for cls in self.class_names:
            cls_path = os.path.join(root_dir, cls)
            for file in os.listdir(cls_path):
                if file.lower().endswith(('.jpg', '.png')):
                    self.images.append(os.path.join(cls_path, file))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = datasets.folder.default_loader(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label

# ===================================================
# DATA TRANSFORMS
# ===================================================
transform_train = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ===================================================
# LOAD DATASET
# ===================================================
dataset = ROIDataset(ROI_DIR, transform=None)
labels = dataset.labels
print(f"Total dataset size: {len(dataset)}")
print(f"Class distribution: {Counter(labels)}")
num_classes = len(dataset.class_names)

# ===================================================
# STRATIFIED TRAIN/VAL SPLIT
# ===================================================
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(split.split(dataset.images, labels))

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)

train_dataset.dataset.transform = transform_train
val_dataset.dataset.transform = transform_val

# ===================================================
# WEIGHTED RANDOM SAMPLER FOR CLASS BALANCE
# ===================================================
train_labels = [dataset.labels[i] for i in train_idx]
class_counts = np.array([Counter(train_labels)[i] for i in range(num_classes)])
class_weights = 1. / class_counts
samples_weights = [class_weights[label] for label in train_labels]

sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

# ===================================================
# DATALOADERS
# ===================================================
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===================================================
# MODEL SETUP
# ===================================================
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)

# ===================================================
# TRAINING LOOP
# ===================================================
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in tqdm(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    scheduler.step(val_loss)

# ===================================================
# EVALUATION AFTER TRAINING
# ===================================================
print("\nGenerating classification report and confusion matrix...")

all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Classification Report
print("\n=== Classification Report ===")
from sklearn.metrics import classification_report
print(classification_report(all_labels, all_preds, target_names=dataset.class_names))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=dataset.class_names, yticklabels=dataset.class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.show()

# ===================================================
# SAVE MODEL
# ===================================================
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\n✅ Model trained and saved at: {MODEL_SAVE_PATH}")
