"""
Dataset Inspector for DeepPCB dataset
Validates structure, finds image pairs, and generates statistics
"""

import os
import cv2
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.utils.file_operations import load_config, save_json, get_all_groups


class DatasetInspector:
    """
    Inspects and validates the DeepPCB dataset structure
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize Dataset Inspector
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.pcb_data_path = self.config['data']['raw_pcb_path']
        self.metadata_path = self.config['data']['metadata_path']
        
        self.template_suffix = self.config['dataset']['template_suffix']
        self.test_suffix = self.config['dataset']['test_suffix']
        self.label_folder_suffix = self.config['dataset']['label_folder_suffix']
        self.image_ext = self.config['dataset']['image_ext']
        
        self.groups = get_all_groups(self.pcb_data_path)
        
    def inspect_dataset_structure(self) -> Dict:
        """
        Inspect the overall dataset structure
        
        Returns:
            Dictionary with dataset statistics
        """
        print("=" * 60)
        print("INSPECTING DATASET STRUCTURE")
        print("=" * 60)
        
        stats = {
            'total_groups': len(self.groups),
            'groups': self.groups,
            'group_details': {}
        }
        
        print(f"\nFound {len(self.groups)} groups:")
        for group in self.groups:
            print(f"  - {group}")
            
        return stats
    
    def find_image_pairs(self, group_name: str) -> List[Tuple[str, str, str]]:
        """
        Find all template-test image pairs in a group
        
        Args:
            group_name: Name of the group folder
            
        Returns:
            List of tuples (image_id, template_path, test_path)
        """
        group_path = os.path.join(self.pcb_data_path, group_name, group_name.replace('group', ''))
        
        if not os.path.exists(group_path):
            return []
        
        pairs = []
        template_files = {}
        test_files = {}
        
        # Get all files in the group folder
        for filename in os.listdir(group_path):
            if filename.endswith(self.image_ext):
                if self.template_suffix in filename:
                    image_id = filename.replace(self.template_suffix + self.image_ext, '')
                    template_files[image_id] = os.path.join(group_path, filename)
                elif self.test_suffix in filename:
                    image_id = filename.replace(self.test_suffix + self.image_ext, '')
                    test_files[image_id] = os.path.join(group_path, filename)
        
        # Match template and test files
        for image_id in template_files:
            if image_id in test_files:
                pairs.append((
                    image_id,
                    template_files[image_id],
                    test_files[image_id]
                ))
        
        return pairs
    
    def inspect_group(self, group_name: str) -> Dict:
        """
        Inspect a specific group folder
        
        Args:
            group_name: Name of the group folder
            
        Returns:
            Dictionary with group statistics
        """
        group_image_folder = os.path.join(
            self.pcb_data_path, 
            group_name, 
            group_name.replace('group', '')
        )
        group_label_folder = os.path.join(
            self.pcb_data_path, 
            group_name, 
            group_name.replace('group', '') + self.label_folder_suffix
        )
        
        stats = {
            'name': group_name,
            'image_folder': group_image_folder,
            'label_folder': group_label_folder,
            'image_folder_exists': os.path.exists(group_image_folder),
            'label_folder_exists': os.path.exists(group_label_folder),
            'num_pairs': 0,
            'num_labels': 0,
            'sample_image_size': None
        }
        
        if stats['image_folder_exists']:
            pairs = self.find_image_pairs(group_name)
            stats['num_pairs'] = len(pairs)
            
            # Get sample image size
            if pairs:
                sample_template = cv2.imread(pairs[0][1])
                if sample_template is not None:
                    stats['sample_image_size'] = sample_template.shape
        
        if stats['label_folder_exists']:
            label_files = [f for f in os.listdir(group_label_folder) 
                          if os.path.isfile(os.path.join(group_label_folder, f))]
            stats['num_labels'] = len(label_files)
        
        return stats
    
    def inspect_all_groups(self) -> Dict:
        """
        Inspect all groups in the dataset
        
        Returns:
            Dictionary with complete dataset statistics
        """
        print("\n" + "=" * 60)
        print("INSPECTING ALL GROUPS")
        print("=" * 60 + "\n")
        
        all_stats = {
            'total_groups': len(self.groups),
            'total_pairs': 0,
            'total_labels': 0,
            'groups': {}
        }
        
        for group in tqdm(self.groups, desc="Inspecting groups"):
            group_stats = self.inspect_group(group)
            all_stats['groups'][group] = group_stats
            all_stats['total_pairs'] += group_stats['num_pairs']
            all_stats['total_labels'] += group_stats['num_labels']
            
            print(f"\n{group}:")
            print(f"  Image pairs: {group_stats['num_pairs']}")
            print(f"  Label files: {group_stats['num_labels']}")
            print(f"  Image size: {group_stats['sample_image_size']}")
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total groups: {all_stats['total_groups']}")
        print(f"Total image pairs: {all_stats['total_pairs']}")
        print(f"Total label files: {all_stats['total_labels']}")
        
        return all_stats
    
    def create_image_pairs_csv(self) -> pd.DataFrame:
        """
        Create a CSV file with all image pairs

        Returns:
            DataFrame with image pair information
        """
        print("\n" + "=" * 60)
        print("CREATING IMAGE PAIRS CSV")
        print("=" * 60 + "\n")

        all_pairs = []

        for group in tqdm(self.groups, desc="Processing groups"):
            pairs = self.find_image_pairs(group)

            for image_id, template_path, test_path in pairs:
                # Find corresponding label file
                label_folder = os.path.join(
                    self.pcb_data_path,
                    group,
                    group.replace('group', '') + self.label_folder_suffix
                )
                # FIX: Add .txt extension
                label_path = os.path.join(label_folder, image_id + '.txt')

                all_pairs.append({
                    'group': group,
                    'image_id': image_id,
                    'template_path': template_path,
                    'test_path': test_path,
                    'label_path': label_path if os.path.exists(label_path) else None,
                    'has_label': os.path.exists(label_path)
                })

        df = pd.DataFrame(all_pairs)

        # Save to CSV
        os.makedirs(self.metadata_path, exist_ok=True)
        csv_path = os.path.join(self.metadata_path, 'image_pairs.csv')
        df.to_csv(csv_path, index=False)

        print(f"Saved image pairs CSV to: {csv_path}")
        print(f"Total pairs: {len(df)}")
        print(f"Pairs with labels: {df['has_label'].sum()}")

        return df
    
    def load_split_file(self, split_name: str) -> List[str]:
        """
        Load train/val/test split file
        
        Args:
            split_name: 'trainval' or 'test'
            
        Returns:
            List of image IDs in the split
        """
        split_file = os.path.join(self.pcb_data_path, split_name)
        
        if not os.path.exists(split_file):
            print(f"Warning: Split file not found: {split_file}")
            return []
        
        with open(split_file, 'r') as f:
            lines = f.readlines()
        
        # Extract image IDs (remove group prefix if present)
        image_ids = []
        for line in lines:
            line = line.strip()
            if line:
                # Format is usually: group12000/12000065
                if '/' in line:
                    image_ids.append(line.split('/')[-1])
                else:
                    image_ids.append(line)
        
        return image_ids
    
    def run_full_inspection(self):
        """
        Run complete dataset inspection
        """
        # Inspect structure
        structure_stats = self.inspect_dataset_structure()
        
        # Inspect all groups
        group_stats = self.inspect_all_groups()
        
        # Create image pairs CSV
        pairs_df = self.create_image_pairs_csv()
        
        # Load split files
        print("\n" + "=" * 60)
        print("LOADING SPLIT FILES")
        print("=" * 60 + "\n")
        
        trainval_ids = self.load_split_file('trainval')
        test_ids = self.load_split_file('test')
        
        print(f"Trainval samples: {len(trainval_ids)}")
        print(f"Test samples: {len(test_ids)}")
        
        # Save complete statistics
        complete_stats = {
            'structure': structure_stats,
            'groups': group_stats,
            'splits': {
                'trainval': len(trainval_ids),
                'test': len(test_ids)
            }
        }
        
        stats_path = os.path.join(self.metadata_path, 'dataset_stats.json')
        save_json(complete_stats, stats_path)
        print(f"\nSaved dataset statistics to: {stats_path}")
        
        return complete_stats, pairs_df


def main():
    """
    Main function to run dataset inspection
    """
    inspector = DatasetInspector()
    stats, pairs_df = inspector.run_full_inspection()
    
    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - data/metadata/image_pairs.csv")
    print("  - data/metadata/dataset_stats.json")


if __name__ == "__main__":
    main()