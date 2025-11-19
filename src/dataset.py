import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class PCBDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        PCB Dataset Loader
        -------------------
        - Reads folders like class_1, class_2 ...
        - Loads images using OpenCV (fast)
        - Converts BGR → RGB
        - Returns ndarray (required for transforms.ToPILImage())
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        # Scan class folders
        for cls_name in sorted(os.listdir(root_dir)):
            cls_path = os.path.join(root_dir, cls_name)

            # Skip files (roi_labels.csv, distribution.json, etc.)
            if not os.path.isdir(cls_path):
                continue

            if not cls_name.startswith("class_"):
                continue

            # Convert "class_1" → 0
            cls_idx = int(cls_name.split("_")[1]) - 1
            self.class_to_idx[cls_name] = cls_idx

            # Collect all image filenames
            for fname in os.listdir(cls_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_path, fname), cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"❌ Failed to read image: {img_path}")

        # Convert BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to numpy (required by transforms.ToPILImage)
        img = np.array(img)

        if self.transform:
            img = self.transform(img)

        return img, label
