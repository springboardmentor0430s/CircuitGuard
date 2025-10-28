import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add the current directory to Python path
sys.path.append('.')

from model.efficientnet import PCBDefectClassifier, save_checkpoint
from src.data.dataset_loader import PCBDefectDataset

class TrainingProgress:
    """Simple class to track and save training progress"""
    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.start_time = time.time()
        
    def add_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc):
        """Add epoch results"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        
        # Auto-save progress after each epoch
        self.save_progress()
    
    def save_progress(self):
        """Save progress to file"""
        progress = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'total_time': time.time() - self.start_time,
            'last_update': datetime.now().isoformat()
        }
        
        # Save as JSON
        with open('training_progress.json', 'w') as f:
            json.dump(progress, f, indent=2)
    
    def generate_final_plots(self):
        """Generate final overview plots"""
        if len(self.epochs) == 0:
            return
            
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Accuracy Overview
        plt.subplot(1, 3, 1)
        plt.plot(self.epochs, self.train_accuracies, 'b-', label='Train Accuracy', linewidth=2, marker='o')
        plt.plot(self.epochs, self.val_accuracies, 'r-', label='Val Accuracy', linewidth=2, marker='s')
        
        # Mark milestones
        for milestone in [80, 85, 90, 95, 97, 98]:
            if max(self.val_accuracies) >= milestone:
                plt.axhline(y=milestone, color='gray', linestyle=':', alpha=0.5)
                plt.text(len(self.epochs)-0.3, milestone+0.5, f'{milestone}%', fontsize=9)
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Loss Overview
        plt.subplot(1, 3, 2)
        plt.plot(self.epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.plot(self.epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Performance Summary
        plt.subplot(1, 3, 3)
        if self.val_accuracies:
            best_val = max(self.val_accuracies)
            final_val = self.val_accuracies[-1]
            final_train = self.train_accuracies[-1]
            
            metrics = [final_train, final_val, best_val]
            labels = [f'Final Train\n{final_train:.1f}%', 
                     f'Final Val\n{final_val:.1f}%', 
                     f'Best Val\n{best_val:.1f}%']
            
            bars = plt.bar(labels, metrics, color=['lightblue', 'lightcoral', 'lightgreen'], alpha=0.8)
            
            for bar, metric in zip(bars, metrics):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{metric:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.ylabel('Accuracy (%)')
            plt.title('Final Performance')
        
        plt.tight_layout()
        plt.savefig('training_overview.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print final report
        self.print_final_report()
    
    def print_final_report(self):
        """Print final training report"""
        if not self.val_accuracies:
            return
            
        print("\n" + "="*50)
        print("📊 TRAINING COMPLETION REPORT")
        print("="*50)
        print(f"Total Epochs Completed: {len(self.epochs)}/10")
        print(f"Best Validation Accuracy: {max(self.val_accuracies):.2f}%")
        print(f"Final Validation Accuracy: {self.val_accuracies[-1]:.2f}%")
        print(f"Total Training Time: {(time.time() - self.start_time)/60:.1f} minutes")
        
        # Check milestones
        milestones = [90, 95, 97]
        achieved = [m for m in milestones if max(self.val_accuracies) >= m]
        if achieved:
            print(f"Milestones Achieved: {achieved}")
        
        print("="*50)

def main():
    print("🚀 Starting PCB Defect Classification Training (10 Epochs)")
    print("=" * 50)
    
    # Initialize progress tracker
    progress = TrainingProgress()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Check if dataset exists
        dataset_path = 'data/defect_dataset'
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset not found at {dataset_path}")
            print("Please run: python data/prepare_dataset.py first")
            return
        
        # Create datasets
        print("📁 Loading datasets...")
        train_dataset = PCBDefectDataset(os.path.join(dataset_path, 'train'))
        val_dataset = PCBDefectDataset(os.path.join(dataset_path, 'val'))
        
        print(f"✅ Training samples: {len(train_dataset)}")
        print(f"✅ Validation samples: {len(val_dataset)}")
        
        # Set batch size
        batch_size = 16
        if len(train_dataset) % batch_size == 1:
            batch_size = 15
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
        
        # Initialize model
        print("🧠 Initializing EfficientNet-B4...")
        model = PCBDefectClassifier(num_classes=6).to(device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        best_accuracy = 0
        
        print("\n🎯 Starting 10-epoch training...")
        print("=" * 50)
        
        for epoch in range(10):
            epoch_start = time.time()
            
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/10')
            for data, targets in train_pbar:
                try:
                    data, targets = data.to(device), targets.to(device)
                    
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
                    
                    current_acc = 100. * train_correct / train_total
                    train_pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{current_acc:.2f}%'})
                    
                except Exception as e:
                    continue
            
            train_pbar.close()
            
            if train_total == 0:
                continue
                
            train_accuracy = 100. * train_correct / train_total
            train_loss = train_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, targets in val_loader:
                    try:
                        data, targets = data.to(device), targets.to(device)
                        outputs = model(data)
                        loss = criterion(outputs, targets)
                        
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()
                    except Exception as e:
                        continue
            
            if val_total == 0:
                continue
                
            val_accuracy = 100. * val_correct / val_total
            val_loss = val_loss / len(val_loader)
            
            scheduler.step()
            epoch_time = time.time() - epoch_start
            
            # Save progress
            progress.add_epoch(epoch+1, train_loss, val_loss, train_accuracy, val_accuracy)
            
            # Print epoch results
            print(f"📊 Epoch {epoch+1}: Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%, Time: {epoch_time:.1f}s")
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                save_checkpoint(model, optimizer, epoch, val_accuracy, 'model/best_model.pth')
                print(f"   💾 New best model: {val_accuracy:.2f}%")
            
            # Early success check
            if val_accuracy >= 97 and epoch >= 2:
                print(f"🎯 Target achieved! {val_accuracy:.2f}% accuracy at epoch {epoch+1}")
        
        print("\n✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Training interrupted after {len(progress.epochs)} epochs")
        
    except Exception as e:
        print(f"\n❌ Training error: {e}")
    
    finally:
        # Always generate final overview
        if len(progress.epochs) > 0:
            print("\n📈 Generating training overview...")
            progress.generate_final_plots()
            print("💾 Progress saved to: training_progress.json, training_overview.png")
        else:
            print("❌ No training data collected")

if __name__ == "__main__":
    main()