from builtins import FileNotFoundError, float, int, len, list, print, range, round, sorted
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from model import build_efficientnet_b4

# ==========================================================
# Paths
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data" / "processed_rois"     # <-- your structure
MODEL_SAVE_PATH = BASE_DIR.parent / "models" / "efficientnet_b4_best.pth"

os.makedirs(MODEL_SAVE_PATH.parent, exist_ok=True)

# ==========================================================
# Custom Dataset
# ==========================================================
class ROIDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ])

        print("\n📌 Loading dataset...")

        for class_dir in sorted(root.iterdir()):
            if class_dir.is_dir() and class_dir.name.startswith("class_"):
                class_id = int(class_dir.name.split("_")[1])
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((img_path, class_id - 1))  # class 1-6 -> 0-5

        print(f"✅ Total images loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label


# ==========================================================
# Check class counts for debugging
# ==========================================================
def check_class_distribution(root):
    print("\n🔍 Class Distribution:")
    counts = {}

    for class_dir in sorted(root.iterdir()):
        if class_dir.is_dir() and class_dir.name.startswith("class_"):
            class_id = class_dir.name
            count = len(list(class_dir.glob("*.jpg")))
            counts[class_id] = count
            print(f"   {class_id}: {count}")

    return counts


# ==========================================================
# Training Loop
# ==========================================================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Training on: {device}")

    class_counts = check_class_distribution(DATA_ROOT)

    dataset = ROIDataset(DATA_ROOT)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    model = build_efficientnet_b4(num_classes=6, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    scaler = torch.cuda.amp.GradScaler()

    best_loss = float("inf")
    EPOCHS = 20

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0

        loop = tqdm(loader, total=len(loader), desc=f"Epoch {epoch}/{EPOCHS}")

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            loop.set_postfix(loss=round(loss.item(), 4))

        scheduler.step()

        avg_loss = epoch_loss / len(loader)
        print(f"📉 Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"💾 Model improved. Saved to {MODEL_SAVE_PATH}")

    print("\n🎉 Training complete!")
    print(f"🏆 Best model saved at {MODEL_SAVE_PATH}")


# ==========================================================
# Run Training
# ==========================================================
if __name__ == "__main__":
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"❌ Folder not found: {DATA_ROOT}")
    train_model()
