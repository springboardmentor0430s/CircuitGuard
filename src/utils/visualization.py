"""
Visualization utilities for dataset and results analysis
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import json
import plotly.graph_objects as go
import plotly.express as px

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class DatasetVisualizer:
    """
    Creates visualizations for dataset statistics and analysis
    """
    
    def __init__(self, metadata_path: str = "data/metadata"):
        """
        Initialize visualizer
        
        Args:
            metadata_path: Path to metadata folder
        """
        self.metadata_path = metadata_path
        self.pairs_df = None
        self.split_info = None
        self.dataset_stats = None
        
        self._load_metadata()
    
    def _load_metadata(self):
        """Load all metadata files"""
        # Load image pairs CSV
        pairs_csv = os.path.join(self.metadata_path, 'image_pairs.csv')
        if os.path.exists(pairs_csv):
            self.pairs_df = pd.read_csv(pairs_csv)
        
        # Load split info
        split_json = os.path.join(self.metadata_path, 'split_info.json')
        if os.path.exists(split_json):
            with open(split_json, 'r') as f:
                self.split_info = json.load(f)
        
        # Load dataset stats
        stats_json = os.path.join(self.metadata_path, 'dataset_stats.json')
        if os.path.exists(stats_json):
            with open(stats_json, 'r') as f:
                self.dataset_stats = json.load(f)
    
    def plot_group_distribution(self, save_path: str = None):
        """Plot distribution of image pairs across groups"""
        if self.pairs_df is None:
            print("No pairs data available")
            return
        
        group_counts = self.pairs_df['group'].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        bars = ax.bar(range(len(group_counts)), group_counts.values, color='steelblue', alpha=0.8)
        ax.set_xticks(range(len(group_counts)))
        ax.set_xticklabels(group_counts.index, rotation=45, ha='right')
        ax.set_xlabel('Group', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Image Pairs', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Image Pairs Across Groups', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_split_distribution(self, save_path: str = None):
        """Plot train/val/test split distribution"""
        if self.split_info is None:
            print("No split info available")
            return
        
        splits = ['train', 'val', 'test']
        counts = [self.split_info[s]['count'] for s in splits]
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart
        bars = ax1.bar(splits, counts, color=colors, alpha=0.8)
        ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax1.set_title('Train/Val/Test Split Distribution', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = (count / sum(counts)) * 100
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({percentage:.1f}%)', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Pie chart
        ax2.pie(counts, labels=splits, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax2.set_title('Split Proportion', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_label_distribution(self, save_path: str = None):
        """Plot distribution of labeled vs unlabeled samples"""
        if self.pairs_df is None:
            print("No pairs data available")
            return
        
        has_label = self.pairs_df['has_label'].sum()
        no_label = len(self.pairs_df) - has_label
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart
        categories = ['With Labels', 'Without Labels']
        values = [has_label, no_label]
        colors = ['#27ae60', '#95a5a6']
        
        bars = ax1.bar(categories, values, color=colors, alpha=0.8)
        ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax1.set_title('Label Availability', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            percentage = (val / sum(values)) * 100
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val}\n({percentage:.1f}%)', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Pie chart
        ax2.pie(values, labels=categories, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax2.set_title('Label Coverage', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_dataset_summary(self, save_path: str = None):
        """Create comprehensive dataset summary visualization"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('CircuitGuard-PCB Dataset Summary', fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Total statistics
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        total_pairs = len(self.pairs_df) if self.pairs_df is not None else 0
        total_groups = len(self.pairs_df['group'].unique()) if self.pairs_df is not None else 0
        train_count = self.split_info['train']['count'] if self.split_info else 0
        val_count = self.split_info['val']['count'] if self.split_info else 0
        test_count = self.split_info['test']['count'] if self.split_info else 0
        
        summary_text = f"""
        DATASET OVERVIEW
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Total Image Pairs: {total_pairs}  |  Total Groups: {total_groups}
        Train: {train_count} ({train_count/total_pairs*100:.1f}%)  |  Val: {val_count} ({val_count/total_pairs*100:.1f}%)  |  Test: {test_count} ({test_count/total_pairs*100:.1f}%)
        """
        ax1.text(0.5, 0.5, summary_text, ha='center', va='center', 
                fontsize=12, family='monospace', fontweight='bold')
        
        # 2. Group distribution
        if self.pairs_df is not None:
            ax2 = fig.add_subplot(gs[1, :])
            group_counts = self.pairs_df['group'].value_counts().sort_index()
            ax2.bar(range(len(group_counts)), group_counts.values, color='steelblue', alpha=0.8)
            ax2.set_xticks(range(len(group_counts)))
            ax2.set_xticklabels(group_counts.index, rotation=45, ha='right', fontsize=9)
            ax2.set_ylabel('Count', fontsize=10, fontweight='bold')
            ax2.set_title('Image Pairs per Group', fontsize=12, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
        
        # 3. Split distribution
        if self.split_info:
            ax3 = fig.add_subplot(gs[2, 0])
            splits = ['Train', 'Val', 'Test']
            counts = [train_count, val_count, test_count]
            colors = ['#3498db', '#2ecc71', '#e74c3c']
            ax3.pie(counts, labels=splits, colors=colors, autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
            ax3.set_title('Split Distribution', fontsize=11, fontweight='bold')
        
        # 4. Label availability
        if self.pairs_df is not None:
            ax4 = fig.add_subplot(gs[2, 1])
            has_label = self.pairs_df['has_label'].sum()
            no_label = len(self.pairs_df) - has_label
            colors = ['#27ae60', '#95a5a6']
            ax4.pie([has_label, no_label], labels=['Labeled', 'Unlabeled'], 
                   colors=colors, autopct='%1.1f%%', startangle=90,
                   textprops={'fontsize': 10, 'fontweight': 'bold'})
            ax4.set_title('Label Availability', fontsize=11, fontweight='bold')
        
        # 5. Statistics box
        ax5 = fig.add_subplot(gs[2, 2])
        ax5.axis('off')
        stats_text = f"""KEY METRICS
        
Avg pairs/group: {total_pairs/total_groups:.1f}
Label coverage: {has_label/total_pairs*100:.1f}%
Total groups: {total_groups}
        """
        ax5.text(0.1, 0.5, stats_text, va='center', fontsize=10, 
                family='monospace', fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class ProcessingVisualizer:
    """
    Visualizes processing pipeline results
    """
    
    def __init__(self):
        pass
    
    def analyze_image_statistics(self, image_dir: str, split: str = 'train', 
                                 sample_size: int = 100, save_path: str = None):
        """Analyze and visualize image statistics"""
        image_paths = list(Path(image_dir).glob('*.jpg'))[:sample_size]
        
        if not image_paths:
            print(f"No images found in {image_dir}")
            return
        
        # Collect statistics
        sizes = []
        mean_intensities = []
        std_intensities = []
        
        for img_path in image_paths:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                sizes.append(img.shape)
                mean_intensities.append(np.mean(img))
                std_intensities.append(np.std(img))
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Image Statistics Analysis - {split.upper()} Split', 
                    fontsize=14, fontweight='bold')
        
        # 1. Image size distribution
        heights = [s[0] for s in sizes]
        widths = [s[1] for s in sizes]
        axes[0, 0].scatter(widths, heights, alpha=0.6, color='steelblue')
        axes[0, 0].set_xlabel('Width (pixels)', fontweight='bold')
        axes[0, 0].set_ylabel('Height (pixels)', fontweight='bold')
        axes[0, 0].set_title('Image Dimensions Distribution')
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Mean intensity distribution
        axes[0, 1].hist(mean_intensities, bins=30, color='coral', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Mean Pixel Intensity', fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontweight='bold')
        axes[0, 1].set_title('Mean Intensity Distribution')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Std deviation distribution
        axes[1, 0].hist(std_intensities, bins=30, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Standard Deviation', fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontweight='bold')
        axes[1, 0].set_title('Intensity Std Dev Distribution')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Statistics summary
        axes[1, 1].axis('off')
        summary = f"""
STATISTICS SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━
Samples analyzed: {len(sizes)}

Image Dimensions:
  Mean: {np.mean(heights):.1f} × {np.mean(widths):.1f}
  Mode: {max(set(sizes), key=sizes.count)}

Pixel Intensities:
  Mean: {np.mean(mean_intensities):.2f}
  Std: {np.mean(std_intensities):.2f}
  Range: [{np.min(mean_intensities):.1f}, {np.max(mean_intensities):.1f}]
        """
        axes[1, 1].text(0.1, 0.5, summary, va='center', fontsize=10,
                       family='monospace', fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_defect_statistics(self, results_list: List[Dict], save_path: str = None):
        """Plot statistics from defect detection results"""
        if not results_list:
            print("No results to visualize")
            return
        
        # Extract data
        num_defects = [r['num_defects'] for r in results_list]
        all_areas = []
        all_aspect_ratios = []
        
        for r in results_list:
            for prop in r['properties']:
                all_areas.append(prop['area'])
                all_aspect_ratios.append(prop['aspect_ratio'])
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Defect Detection Statistics', fontsize=14, fontweight='bold')
        
        # 1. Defects per image
        axes[0, 0].bar(range(len(num_defects)), num_defects, color='crimson', alpha=0.7)
        axes[0, 0].set_xlabel('Image Index', fontweight='bold')
        axes[0, 0].set_ylabel('Number of Defects', fontweight='bold')
        axes[0, 0].set_title('Defects Detected per Image')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Defect area distribution
        axes[0, 1].hist(all_areas, bins=30, color='orange', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Defect Area (pixels²)', fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontweight='bold')
        axes[0, 1].set_title('Defect Area Distribution')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Aspect ratio distribution
        axes[1, 0].hist(all_aspect_ratios, bins=30, color='purple', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Aspect Ratio (W/H)', fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontweight='bold')
        axes[1, 0].set_title('Defect Aspect Ratio Distribution')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Summary statistics
        axes[1, 1].axis('off')
        summary = f"""
DETECTION SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━
Images processed: {len(results_list)}
Total defects: {sum(num_defects)}
Avg defects/image: {np.mean(num_defects):.2f}

Defect Areas:
  Mean: {np.mean(all_areas):.1f} px²
  Median: {np.median(all_areas):.1f} px²
  Range: [{np.min(all_areas):.0f}, {np.max(all_areas):.0f}]

Aspect Ratios:
  Mean: {np.mean(all_aspect_ratios):.2f}
  Median: {np.median(all_aspect_ratios):.2f}
        """
        axes[1, 1].text(0.1, 0.5, summary, va='center', fontsize=10,
                       family='monospace', fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def plot_defect_scatter(defect_details: List[Dict]):
    """Return a Plotly scatter plot (area vs confidence) for interactive exploration

    Args:
        defect_details: List of defect dicts with keys 'area (px²)', 'confidence (%)', 'type', 'id'

    Returns:
        plotly.graph_objects.Figure
    """
    if not defect_details:
        fig = go.Figure()
        fig.update_layout(title='No defects to plot')
        return fig

    areas = [d.get('area (px²)') for d in defect_details]
    confidences = [d.get('confidence (%)') for d in defect_details]
    labels = [d.get('type') for d in defect_details]
    ids = [d.get('id') for d in defect_details]

    df = pd.DataFrame({
        'area': areas,
        'confidence': confidences,
        'type': labels,
        'id': ids
    })

    fig = px.scatter(
        df,
        x='area',
        y='confidence',
        color='type',
        hover_data=['id'],
        labels={'area': 'Area (px²)', 'confidence': 'Confidence (%)'},
        title='Defect Size vs Confidence'
    )

    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(height=500, legend_title_text='Defect Type')
    return fig


def plot_defect_area_hist(defect_details: List[Dict]):
    """Return an interactive histogram of defect areas"""
    if not defect_details:
        fig = go.Figure()
        fig.update_layout(title='No defects to plot')
        return fig

    areas = [d.get('area (px²)') for d in defect_details]
    fig = px.histogram(areas, nbins=25, title='Defect Area Distribution', labels={'value': 'Area (px²)', 'count': 'Frequency'})
    fig.update_layout(height=400)
    return fig