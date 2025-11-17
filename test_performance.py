"""
Performance testing and benchmarking
"""

import time
import torch
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.web_app.backend import get_backend
from src.utils.file_operations import load_config


def test_single_pair_performance(backend, template_path: str, test_path: str, iterations: int = 5):
    """Test performance on single image pair"""
    
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    
    times = []
    
    for _ in range(iterations):
        start = time.time()
        result = backend.process_image_pair(template, test_img)
        elapsed = time.time() - start
        
        if result['success']:
            times.append(elapsed)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times)
    }


def benchmark_system():
    """Run comprehensive system benchmarks"""
    
    print("="*60)
    print("PERFORMANCE BENCHMARKING")
    print("="*60)
    
    # Load backend
    print("\nInitializing backend...")
    backend = get_backend()
    
    # Load config
    config = load_config()
    
    # Get test images
    test_dir = Path(config['data']['splits_path']) / 'test'
    template_dir = test_dir / 'templates'
    test_img_dir = test_dir / 'test_images'
    
    # Get first 10 test pairs
    template_files = sorted(list(template_dir.glob('*_temp.jpg')))[:10]
    
    print(f"\nTesting on {len(template_files)} image pairs...")
    
    results = []
    
    for template_file in tqdm(template_files):
        image_id = template_file.stem.replace('_temp', '')
        test_file = test_img_dir / f'{image_id}_test.jpg'
        
        if test_file.exists():
            perf = test_single_pair_performance(
                backend,
                str(template_file),
                str(test_file),
                iterations=3
            )
            
            results.append({
                'image_id': image_id,
                'mean_time': perf['mean_time'],
                'std_time': perf['std_time'],
                'min_time': perf['min_time'],
                'max_time': perf['max_time']
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Summary statistics
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"\nAverage Processing Time: {df['mean_time'].mean():.3f}s ± {df['mean_time'].std():.3f}s")
    print(f"Fastest Processing: {df['min_time'].min():.3f}s")
    print(f"Slowest Processing: {df['max_time'].max():.3f}s")
    print(f"Throughput: {1/df['mean_time'].mean():.2f} images/second")
    
    # Device info
    print(f"\nDevice: {backend.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Save results
    output_path = 'outputs/performance_benchmark.csv'
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    return df


def test_batch_performance():
    """Test batch processing performance"""
    
    print("\n" + "="*60)
    print("BATCH PROCESSING TEST")
    print("="*60)
    
    backend = get_backend()
    config = load_config()
    
    # Get test images
    test_dir = Path(config['data']['splits_path']) / 'test'
    template_dir = test_dir / 'templates'
    test_img_dir = test_dir / 'test_images'
    
    # Test with 20 images
    template_files = sorted(list(template_dir.glob('*_temp.jpg')))[:20]
    
    start_time = time.time()
    
    for template_file in tqdm(template_files, desc="Batch processing"):
        image_id = template_file.stem.replace('_temp', '')
        test_file = test_img_dir / f'{image_id}_test.jpg'
        
        if test_file.exists():
            template = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
            test_img = cv2.imread(str(test_file), cv2.IMREAD_GRAYSCALE)
            
            result = backend.process_image_pair(template, test_img)
    
    total_time = time.time() - start_time
    
    print(f"\nBatch Results:")
    print(f"  Images Processed: {len(template_files)}")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Average Time/Image: {total_time/len(template_files):.3f}s")
    print(f"  Throughput: {len(template_files)/total_time:.2f} images/second")


def main():
    """Run all performance tests"""
    
    # Single pair benchmarks
    df = benchmark_system()
    
    # Batch processing test
    test_batch_performance()
    
    print("\n" + "="*60)
    print("BENCHMARKING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()