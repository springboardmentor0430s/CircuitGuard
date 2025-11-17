"""
Test script for dataset inspector
"""

from src.data_preparation.dataset_inspector import DatasetInspector

if __name__ == "__main__":
    # Create inspector
    inspector = DatasetInspector()
    
    # Run full inspection
    stats, pairs_df = inspector.run_full_inspection()
    
    # Show first few rows of pairs
    print("\n" + "=" * 60)
    print("SAMPLE IMAGE PAIRS")
    print("=" * 60)
    print(pairs_df.head(10))