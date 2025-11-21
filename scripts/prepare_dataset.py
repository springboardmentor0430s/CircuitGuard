
import os, cv2, argparse
import numpy as np
from pathlib import Path

def align_and_subtract(template_img, test_img):
    # For DeepPCB the images are expected pre-aligned. This function provides a placeholder
    # alignment step (identity) and returns absolute difference grayscale image.
    t = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    s = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(t, s)
    # Apply Otsu threshold
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Some morphological cleaning
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    return cleaned

def extract_rois_and_save(diff_mask, test_img, out_dir, base_name):
    contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    i = 0
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w < 5 or h < 5: 
            continue
        roi = test_img[y:y+h, x:x+w]
        out_path = out_dir / f"{base_name}_roi_{i}.png"
        cv2.imwrite(str(out_path), roi)
        i += 1
    return i

def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Expect pairs of images: template_*.png and test_*.png or the folder structure of DeepPCB
    for root, dirs, files in os.walk(data_dir):
        pngs = [f for f in files if f.lower().endswith((".png",".jpg",".jpeg"))]
        for fn in pngs:
            if "template" in fn.lower() or "ref" in fn.lower():
                # try to find corresponding test by naming convention
                base = fn.split("template")[0]
                # naive matching: pick any other image in same dir that isn't the template
                template_path = os.path.join(root, fn)
                candidates = [os.path.join(root, x) for x in pngs if x!=fn]
                if not candidates:
                    continue
                test_path = candidates[0]
                template_img = cv2.imread(template_path)
                test_img = cv2.imread(test_path)
                diff_mask = align_and_subtract(template_img, test_img)
                count = extract_rois_and_save(diff_mask, test_img, out_dir, Path(test_path).stem)
                print(f"Processed {template_path} + {test_path} -> {count} ROIs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    main(args)
