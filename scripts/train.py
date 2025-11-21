
import argparse, os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from pathlib import Path

class SimpleROIDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.files = list(Path(folder).glob("*.png"))
        self.transform = transform
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.files[idx]).convert("RGB")
        label = 0  # placeholder; user should replace reading labels
        if self.transform:
            img = self.transform(img)
        return img, label

def build_model(num_classes, pretrained=True):
    model = timm.create_model("efficientnet_b4", pretrained=pretrained, num_classes=num_classes)
    return model

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    dataset = SimpleROIDataset(args.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = build_model(args.num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} - loss: {running_loss/len(loader):.4f}")
        # save checkpoint
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"effb4_epoch{epoch+1}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=6)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
