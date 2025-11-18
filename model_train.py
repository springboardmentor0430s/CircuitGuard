import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image, ImageDraw, ImageFont


#CONFIGURATION

DATA_ROOT = 'Final_PCB_Split'  # Folder containing train/validation/test
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
VAL_DIR = os.path.join(DATA_ROOT, 'validation')
TEST_DIR = os.path.join(DATA_ROOT, 'test')

CLASS_NAMES = ['copper', 'mousebite', 'noise', 'open', 'pin-hole', 'short', 'spur']

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


#DATA TRANSFORMS
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}


#DATA LOADERS
def create_dataloaders():
    image_datasets = {
        'train': datasets.ImageFolder(TRAIN_DIR, transform=data_transforms['train']),
        'val': datasets.ImageFolder(VAL_DIR, transform=data_transforms['val']),
        'test': datasets.ImageFolder(TEST_DIR, transform=data_transforms['test'])
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        for x in ['train', 'val']
    }
    test_loader = DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False)
    return dataloaders, test_loader, image_datasets['train'].classes


#MODEL SETUP
def build_efficientnet(num_classes):
    print("🔧 Loading EfficientNet-B4 pre-trained model...")
    model = models.efficientnet_b4(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model.to(DEVICE)


#TRAINING FUNCTION
def train_model(model, dataloaders, num_epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        print(f"\n📘 Epoch {epoch+1}/{num_epochs}")
        print('-'*30)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects, total = 0.0, 0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects.double() / total

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), "efficientnet_b4_best(1).pth")

    print(f"\n✅ Best Validation Accuracy: {best_acc*100:.2f}%")

    #Plot Accuracy & Loss
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()


#EVALUATION ON TEST SET
def evaluate_model(model, test_loader, class_names):
    model.eval()
    preds, labels_list = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    cm = confusion_matrix(labels_list, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()

    print("\n--- Classification Report ---")
    print(classification_report(labels_list, preds, target_names=class_names))

    #Overall metrics
    accuracy = accuracy_score(labels_list, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_list, preds, average='macro')

    print("\n================= OVERALL METRICS =================")
    print(f"{'Metric':<20}{'Value'}")
    print(f"{'-'*40}")
    print(f"{'Test Accuracy':<20}{accuracy*100:.2f}%")
    print(f"{'Precision':<20}{precision:.3f}")
    print(f"{'Recall':<20}{recall:.3f}")
    print(f"{'F1-Score':<20}{f1:.3f}")

    #Per-class detailed performance
    class_prec, class_rec, class_f1, class_support = precision_recall_fscore_support(labels_list, preds)

    print("\n================= PER-CLASS PERFORMANCE =================")
    print(f"{'Defect Type':<15}{'Precision':<12}{'Recall':<10}{'F1-Score':<10}{'Support'}")
    print("-"*65)

    for cls, p, r, f, s in zip(class_names, class_prec, class_rec, class_f1, class_support):
        print(f"{cls:<15}{p:.3f}{' ' * 8}{r:.3f}{' ' * 8}{f:.3f}{' ' * 8}{s}")

    print("\n==========================================================")


#PREDICTION VISUALIZATION
def annotate_prediction(image_path, model, class_names):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    transform = data_transforms['test']
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        predicted_class = class_names[pred.item()]
        confidence = conf.item() * 100

    print(f"🧠 Predicted: {predicted_class} ({confidence:.2f}%)")

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((10, 10), f"{predicted_class}: {confidence:.1f}%", fill="red", font=font)
    output_path = "annotated_prediction.jpg"
    img.save(output_path)
    print(f"✅ Annotated image saved as: {output_path}")


#MAIN
if __name__ == "__main__":
    print("🚀 Starting EfficientNet-B4 PCB Defect Classification...")

    dataloaders, test_loader, class_names = create_dataloaders()
    model = build_efficientnet(len(class_names))

    train_model(model, dataloaders, num_epochs=EPOCHS)

    print("\n--- Evaluating Best Model ---")
    model.load_state_dict(torch.load("efficientnet_b4_best(1).pth"))
    evaluate_model(model, test_loader, class_names)

    #Example: test one image
    test_img = os.path.join(TEST_DIR, class_names[0], os.listdir(os.path.join(TEST_DIR, class_names[0]))[0])
    annotate_prediction(test_img, model, class_names)
