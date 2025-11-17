"""
Script to organize dataset into train/val/test splits
"""

from src.data_preparation.dataset_organizer import DatasetOrganizer

if __name__ == "__main__":
    # Create organizer
    organizer = DatasetOrganizer()
    
    # Organize dataset (85% train, 15% val from trainval set)
    split_info = organizer.organize_dataset(val_ratio=0.15)
    
    print("\n✓ Dataset organized successfully!")
    print("\nYou can now find organized data in:")
    print("  - data/raw/splits/train/")
    print("  - data/raw/splits/val/")
    print("  - data/raw/splits/test/")