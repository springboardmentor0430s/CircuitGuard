"""
Dataset Organizer for DeepPCB
Organizes image pairs into train/val/test splits
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.utils.file_operations import load_config, save_json, create_directory


class DatasetOrganizer:
    """
    Organizes DeepPCB dataset into train/val/test splits
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize Dataset Organizer
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.pcb_data_path = self.config['data']['raw_pcb_path']
        self.splits_path = self.config['data']['splits_path']
        self.metadata_path = self.config['data']['metadata_path']
        
        # Load image pairs CSV created by inspector
        pairs_csv_path = os.path.join(self.metadata_path, 'image_pairs.csv')
        self.pairs_df = pd.read_csv(pairs_csv_path)
        
        print(f"Loaded {len(self.pairs_df)} image pairs")
    
    def load_split_ids_from_files(self) -> Dict[str, List[str]]:
        """
        Try to load image IDs from split files if they exist
        
        Returns:
            Dictionary with 'trainval' and 'test' lists, or empty if not found
        """
        splits = {'trainval': [], 'test': []}
        
        for split_name in ['trainval', 'test']:
            split_file = os.path.join(self.pcb_data_path, split_name)
            
            if os.path.exists(split_file):
                with open(split_file, 'r') as f:
                    lines = f.readlines()
                
                # Extract image IDs
                ids = []
                for line in lines:
                    line = line.strip()
                    if line:
                        # Format: group12000/12000065
                        if '/' in line:
                            ids.append(line.split('/')[-1])
                        else:
                            ids.append(line)
                
                splits[split_name] = ids
                print(f"✓ Loaded {len(ids)} IDs from {split_name}")
            else:
                print(f"✗ {split_name} file not found")
        
        return splits
    
    def create_splits_from_scratch(self, train_ratio: float = 0.7, 
                                   val_ratio: float = 0.15,
                                   test_ratio: float = 0.15) -> Dict[str, List[str]]:
        """
        Create train/val/test splits from all available image pairs
        
        Args:
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            
        Returns:
            Dictionary with 'train', 'val', and 'test' lists
        """
        print("\n" + "=" * 60)
        print("CREATING SPLITS FROM ALL IMAGE PAIRS")
        print("=" * 60)
        
        # Get all image IDs
        all_ids = self.pairs_df['image_id'].tolist()
        
        # First split: separate test set
        train_val_ids, test_ids = train_test_split(
            all_ids,
            test_size=test_ratio,
            random_state=42
        )
        
        # Second split: separate train and val from remaining
        val_size = val_ratio / (train_ratio + val_ratio)
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=val_size,
            random_state=42
        )
        
        print(f"\nCreated splits:")
        print(f"  Train: {len(train_ids)} samples ({len(train_ids)/len(all_ids)*100:.1f}%)")
        print(f"  Val:   {len(val_ids)} samples ({len(val_ids)/len(all_ids)*100:.1f}%)")
        print(f"  Test:  {len(test_ids)} samples ({len(test_ids)/len(all_ids)*100:.1f}%)")
        
        return {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }
    
    def organize_split(self, split_name: str, image_ids: List[str]):
        """
        Organize images for a specific split

        Args:
            split_name: 'train', 'val', or 'test'
            image_ids: List of image IDs for this split
        """
        print(f"\n{'='*60}")
        print(f"ORGANIZING {split_name.upper()} SPLIT")
        print(f"{'='*60}")

        # Create directories
        split_base = os.path.join(self.splits_path, split_name)
        templates_dir = os.path.join(split_base, 'templates')
        test_images_dir = os.path.join(split_base, 'test_images')
        labels_dir = os.path.join(split_base, 'labels')

        create_directory(templates_dir)
        create_directory(test_images_dir)
        create_directory(labels_dir)

        # Filter pairs for this split
        split_pairs = self.pairs_df[self.pairs_df['image_id'].isin(image_ids)]

        print(f"Found {len(split_pairs)} pairs for {split_name}")

        # Copy files
        success_count = 0
        error_count = 0
        labels_copied = 0

        for idx, row in tqdm(split_pairs.iterrows(),
                            total=len(split_pairs),
                            desc=f"Copying {split_name} files"):
            image_id = row['image_id']

            try:
                # Copy template
                template_src = row['template_path']
                template_dst = os.path.join(templates_dir, f"{image_id}_temp.jpg")
                if os.path.exists(template_src):
                    shutil.copy2(template_src, template_dst)

                # Copy test image
                test_src = row['test_path']
                test_dst = os.path.join(test_images_dir, f"{image_id}_test.jpg")
                if os.path.exists(test_src):
                    shutil.copy2(test_src, test_dst)

                # Copy label if exists - FIX: Use correct extension
                if row['has_label'] and pd.notna(row['label_path']):
                    label_src = row['label_path']
                    # Add .txt extension for destination
                    label_dst = os.path.join(labels_dir, f"{image_id}.txt")
                    if os.path.exists(label_src):
                        shutil.copy2(label_src, label_dst)
                        labels_copied += 1

                success_count += 1

            except Exception as e:
                error_count += 1
                print(f"\nError processing {image_id}: {e}")

        print(f"✓ Successfully organized {success_count} pairs")
        print(f"✓ Labels copied: {labels_copied}")
        if error_count > 0:
            print(f"✗ Errors: {error_count}")
    
    def organize_dataset(self, train_ratio: float = 0.7,
                        val_ratio: float = 0.15,
                        test_ratio: float = 0.15):
        """
        Organize complete dataset into splits
        
        Args:
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set  
            test_ratio: Ratio for test set
        """
        print("=" * 60)
        print("DATASET ORGANIZATION")
        print("=" * 60)
        
        # Try to load splits from files first
        file_splits = self.load_split_ids_from_files()
        
        # If split files don't exist or are empty, create splits from scratch
        if not file_splits['trainval'] and not file_splits['test']:
            print("\n⚠ Split files not found or empty.")
            print("Creating splits from all available image pairs...")
            splits = self.create_splits_from_scratch(train_ratio, val_ratio, test_ratio)
        else:
            # Use existing splits and further split trainval
            print("\n✓ Using existing split files")
            trainval_ids = file_splits['trainval']
            test_ids = file_splits['test']
            
            # Split trainval into train and val
            val_size = val_ratio / (train_ratio + val_ratio)
            train_ids, val_ids = train_test_split(
                trainval_ids,
                test_size=val_size,
                random_state=42
            )
            
            splits = {
                'train': train_ids,
                'val': val_ids,
                'test': test_ids
            }
            
            print(f"\nSplit counts:")
            print(f"  Train: {len(train_ids)}")
            print(f"  Val: {len(val_ids)}")
            print(f"  Test: {len(test_ids)}")
        
        # Organize each split
        self.organize_split('train', splits['train'])
        self.organize_split('val', splits['val'])
        self.organize_split('test', splits['test'])
        
        # Save split information
        split_info = {
            'train': {
                'count': len(splits['train']),
                'ids': splits['train'][:100]  # Save first 100 IDs only to reduce file size
            },
            'val': {
                'count': len(splits['val']),
                'ids': splits['val'][:100]
            },
            'test': {
                'count': len(splits['test']),
                'ids': splits['test'][:100]
            },
            'total': len(self.pairs_df)
        }
        
        split_info_path = os.path.join(self.metadata_path, 'split_info.json')
        save_json(split_info, split_info_path)
        
        print("\n" + "=" * 60)
        print("✓ ORGANIZATION COMPLETE!")
        print("=" * 60)
        print(f"\nFinal split counts:")
        print(f"  Train: {split_info['train']['count']}")
        print(f"  Val:   {split_info['val']['count']}")
        print(f"  Test:  {split_info['test']['count']}")
        print(f"  Total: {split_info['total']}")
        print(f"\nSplit info saved to: {split_info_path}")
        
        return split_info


def main():
    """
    Main function to organize dataset
    """
    organizer = DatasetOrganizer()
    
    # Organize with 70% train, 15% val, 15% test
    split_info = organizer.organize_dataset(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )


if __name__ == "__main__":
    main()