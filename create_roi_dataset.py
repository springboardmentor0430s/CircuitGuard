"""
Create ROI dataset for defect classification
"""

from src.data_preparation.roi_extractor import ROIExtractor

if __name__ == "__main__":
    print("Creating ROI dataset from labeled PCB images...")
    
    extractor = ROIExtractor()
    stats = extractor.create_dataset()
    
    print("\n✓ ROI dataset created successfully!")
    print("\nYou can now proceed with model training.")