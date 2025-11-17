from pathlib import Path
import os
import sys

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.web_app.pdf_report import create_pdf_report


def main():
    out_dir = PROJECT_ROOT / "outputs" / "sample_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "sample_report.pdf"

    # Example class names
    class_names = ["no_defect", "solder_bridging", "component_missing", "misalignment"]

    # Synthetic result
    result = {
        'num_defects': 5,
        'processing_time': 1.23,
        'alignment_info': {
            'num_matches': 150,
            'num_inliers': 120
        },
        'classifications': [
            {
                'predicted_label': 'solder_bridging',
                'confidence': 0.92,
                'centroid': (123, 456),
                'area': 230,
                'bbox': (110, 440, 26, 32)
            },
            {
                'predicted_label': 'component_missing',
                'confidence': 0.88,
                'centroid': (50, 80),
                'area': 180,
                'bbox': (40, 70, 20, 30)
            },
            {
                'predicted_label': 'misalignment',
                'confidence': 0.75,
                'centroid': (200, 120),
                'area': 95,
                'bbox': (195, 110, 18, 16)
            },
            {
                'predicted_label': 'solder_bridging',
                'confidence': 0.67,
                'centroid': (300, 240),
                'area': 120,
                'bbox': (295, 235, 12, 18)
            },
            {
                'predicted_label': 'solder_bridging',
                'confidence': 0.55,
                'centroid': (400, 320),
                'area': 60,
                'bbox': (395, 315, 10, 12)
            }
        ]
    }

    try:
        create_pdf_report(result, str(output_path), class_names)
        print(f"PDF generated at: {output_path}")
    except Exception as e:
        print("Error while generating PDF:", e)


if __name__ == '__main__':
    main()
