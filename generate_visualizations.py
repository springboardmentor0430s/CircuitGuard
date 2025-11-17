"""
Generate comprehensive visualizations and analysis
"""

import os
from src.utils.visualization import DatasetVisualizer, ProcessingVisualizer
from src.utils.file_operations import create_directory, load_config

def main():
    print("="*60)
    print("GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*60)
    
    # Create output directory
    vis_output = "outputs/analysis_charts"
    create_directory(vis_output)
    
    # Initialize visualizers
    dataset_viz = DatasetVisualizer()
    processing_viz = ProcessingVisualizer()
    
    # 1. Dataset visualizations
    print("\n1. Creating dataset visualizations...")
    
    print("  - Group distribution chart")
    dataset_viz.plot_group_distribution(
        save_path=os.path.join(vis_output, '01_group_distribution.png')
    )
    
    print("  - Split distribution chart")
    dataset_viz.plot_split_distribution(
        save_path=os.path.join(vis_output, '02_split_distribution.png')
    )
    
    print("  - Label distribution chart")
    dataset_viz.plot_label_distribution(
        save_path=os.path.join(vis_output, '03_label_distribution.png')
    )
    
    print("  - Comprehensive dataset summary")
    dataset_viz.plot_dataset_summary(
        save_path=os.path.join(vis_output, '04_dataset_summary.png')
    )
    
    # 2. Image statistics
    print("\n2. Analyzing image statistics...")
    config = load_config()
    splits_path = config['data']['splits_path']
    
    for split in ['train', 'val', 'test']:
        print(f"  - {split.capitalize()} split statistics")
        image_dir = os.path.join(splits_path, split, 'templates')
        processing_viz.analyze_image_statistics(
            image_dir, 
            split=split,
            sample_size=100,
            save_path=os.path.join(vis_output, f'05_{split}_image_stats.png')
        )
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print(f"\nAll charts saved to: {vis_output}")
    print("\nGenerated visualizations:")
    print("  1. Group distribution")
    print("  2. Train/Val/Test split distribution")
    print("  3. Label availability distribution")
    print("  4. Comprehensive dataset summary")
    print("  5-7. Image statistics (train/val/test)")


if __name__ == "__main__":
    main()