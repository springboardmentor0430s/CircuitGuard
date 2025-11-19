import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.dataset import PCBDataset
from src.model import build_efficientnet_b4

# ============================================================
# Config
# ============================================================
DATA_ROOT = "./src/data/processed_rois"
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "models/efficientnet_b4_best.pth"

# ============================================================
# Data Transforms
# ============================================================
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# ============================================================
# Training Function
# ============================================================
def train():

    print("\n[INFO] Loading dataset...")
    dataset = PCBDataset(DATA_ROOT, transform=train_transform)

    # Split train/validation
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    num_classes = len(dataset.class_to_idx)
    print(f"[INFO] Training on {num_classes} classes using {DEVICE}")

    # Model + Loss + Optimizer
    model = build_efficientnet_b4(num_classes=num_classes, pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_acc = 0

    # ============================================================
    # Epoch Loop
    # ============================================================
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # ============================================================
        # Validation Pass
        # ============================================================
        val_acc, val_loss = evaluate(model, val_loader, criterion)

        scheduler.step(val_loss)

        print(
            f"\nEpoch {epoch}: "
            f"Train Loss={train_loss/len(train_loader):.4f}, "
            f"Train Acc={train_acc*100:.2f}%, "
            f"Val Loss={val_loss:.4f}, "
            f"Val Acc={val_acc*100:.2f}%"
        )

        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"🔥 Saved new best model to {SAVE_PATH}")

    print("\n[INFO] Training Complete.")
    print(f"[INFO] Best Validation Accuracy: {best_val_acc*100:.2f}%")

    print("[INFO] Generating Confusion Matrix...")
    generate_confusion_matrix(model, val_loader, num_classes)


# ============================================================
# Validation Function
# ============================================================
def evaluate(model, loader, criterion):
    model.eval()
    total = 0
    correct = 0
    val_loss = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total, val_loss / len(loader)


# ============================================================
# Confusion Matrix
# ============================================================
def generate_confusion_matrix(model, loader, num_classes):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.show()

    print(cm)


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    train()
