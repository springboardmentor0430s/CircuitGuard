import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

sys.path.append('.')

from model.efficientnet import PCBDefectClassifier, save_checkpoint
from src.data.dataset_loader import PCBDefectDataset


def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if dataset exists
    dataset_path = 'data/defect_dataset'
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please run: python data/prepare_dataset.py first")
        return
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = PCBDefectDataset(os.path.join(dataset_path, 'train'))
    val_dataset = PCBDefectDataset(os.path.join(dataset_path, 'val'))
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    
    train_dist = train_dataset.get_class_distribution()
    print("📊 Training class distribution:")
    for defect_type, count in train_dist.items():
        print(f"   {defect_type}: {count} samples")
    
    # Calculate optimal batch size to avoid last single-sample batch
    total_train = len(train_dataset)
    batch_size = 16
    
    if total_train % batch_size == 1:
        batch_size = 15  
        print(f"Adjusted batch size to {batch_size} to avoid single-sample batches")
    else:
        print(f"Using batch size: {batch_size}")
    
    # data loaders to avoid single-sample batches
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
        
    model = PCBDefectClassifier(num_classes=6).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model initialized with {total_params:,} total parameters")
    print(f"✅ Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_accuracy = 0
    current_lr = 0.001

    print("Starting training...")
    print("=" * 60)
    
    # Early stopping parameters
    patience = 3
    patience_counter = 0
    best_val_loss = float('inf')
    
    for epoch in range(30):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"\n📈 Epoch {epoch+1}/30")
        print("─" * 40)
        
        # Training progress bar
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, (data, targets) in enumerate(train_pbar):
            try:
                data, targets = data.to(device), targets.to(device)
                
                # Skip if batch has only 1 sample (safety check)
                if data.size(0) <= 1:
                    continue
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                current_acc = 100. * train_correct / train_total
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%',
                    'Batch': f'{batch_idx+1}/{len(train_loader)}'
                })
                
                    
            except Exception as e:
                print(f"❌ Error in batch {batch_idx}: {e}")
                continue
        
        train_pbar.close()
        
        if train_total > 0:
            train_accuracy = 100. * train_correct / train_total
            train_loss = train_loss / len(train_loader)
        else:
            print("⚠️  No training data processed this epoch")
            continue
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        print("📊 Running validation...")
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validating")
            for data, targets in val_pbar:
                try:
                    data, targets = data.to(device), targets.to(device)
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
                    
                    current_val_acc = 100. * val_correct / val_total
                    val_pbar.set_postfix({'Acc': f'{current_val_acc:.2f}%'})
                except Exception as e:
                    print(f"❌ Validation error: {e}")
                    continue
            
            val_pbar.close()
        
        if val_total > 0:
            val_accuracy = 100. * val_correct / val_total
            val_loss = val_loss / len(val_loader)
        else:
            print("⚠️  No validation data processed")
            continue
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"✅ Epoch {epoch+1} Summary:")
        print(f"   ⏱️  Time: {epoch_time:.2f}s")
        print(f"   📚 Learning Rate: {current_lr:.6f}")
        print(f"   🏋️  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"   🧪 Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"   ⚠️  Early stopping counter: {patience_counter}/{patience}")
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(
                model, optimizer, epoch, val_accuracy,
                'model/best_model.pth'
            )
            print(f"   💾 NEW BEST MODEL! Accuracy: {val_accuracy:.2f}%")
        
        # Check early stopping
        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered after {epoch+1} epochs")
            break
            
        # Check if target accuracy reached
        if val_accuracy >= 95 and epoch >= 2:
            print(f"🎯 TARGET REACHED! {val_accuracy:.2f}% accuracy - Consider stopping")
        
        print("─" * 40)
    
    
    print(f"\n💡 Total training time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    start_time = time.time()
    main()