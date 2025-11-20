import os
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import time
from tqdm import tqdm
import json
from torch.optim import lr_scheduler # <-- NEW: Import the learning rate scheduler

# --- 1. SETUP ---
# (This section is unchanged)
data_dir = 'data/processed'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Using {device.upper()} device for training ---")

# --- 2. DATA TRANSFORMATION AND LOADING (UPDATED) ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((128, 128)),
        # --- NEW: More aggressive augmentation ---
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        # --- End of new augmentation ---
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([ # The test set should NOT have augmentation
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
print("\nLoading datasets with advanced augmentation...")
# ... (rest of the data loading is the same)
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
}
batch_size = 32
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
    'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False)
}
class_names = image_datasets['train'].classes
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
print("Datasets loaded successfully.")


# --- 3. MODEL DEFINITION ---
# (This section is unchanged)
print("\nLoading the pre-trained EfficientNet-B4 model...")
model = timm.create_model('efficientnet_b4', pretrained=True)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, len(class_names))
model = model.to(device)
print("Model defined and moved to the correct device.")


# --- 4. LOSS FUNCTION, OPTIMIZER, AND SCHEDULER (UPDATED) ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Train all parameters, not just the classifier

# --- NEW: Define the Learning Rate Scheduler ---
# This will decrease the learning rate by a factor of 0.1 every 7 epochs.
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
print("\nLoss function, optimizer, and LR scheduler have been defined.")


# --- 5. TRAINING LOOP (UPDATED) ---
def train_model(model, criterion, optimizer, scheduler, num_epochs=10): # Added scheduler
    start_time = time.time()
    print("\n--- Starting Advanced Model Training ---")
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"  {phase.capitalize()} Phase"):
                inputs = inputs.to(device)
                labels = labels.to(device)
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
            
            # --- NEW: Step the scheduler after each training epoch ---
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'  {phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'models/best_model.pth')
                print(f'  New best model saved with accuracy: {best_acc:.4f}')

    time_elapsed = time.time() - start_time
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Test Accuracy: {best_acc:.4f}')
    
    with open('outputs/training_history.json', 'w') as f:
        json.dump(history, f)
    print("Training history saved to 'outputs/training_history.json'")

    return model

# --- START THE TRAINING ---
os.makedirs('outputs', exist_ok=True)
model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=30) # Pass the scheduler to the function

# Save the final model
torch.save(model.state_dict(), 'models/last_epoch_model.pth')
print("Final model from last epoch saved.")