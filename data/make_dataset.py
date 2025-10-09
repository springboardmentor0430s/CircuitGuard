import shutil
from pathlib import Path

# Define base path
BASE_DIR = Path(__file__).parents[2] / "data" / "raw"

# Define output folders
test_dir = "../../data/interim/test"
template_dir = "../../data/interim/template"
annot_dir = "../../data/interim/annotation"

for folder in [test_dir, template_dir, annot_dir]:
    folder.mkdir(parents=True, exist_ok=True)

# Collect all group folders (e.g., group00041, group00042...)
group_dirs = sorted([g for g in BASE_DIR.iterdir() if g.is_dir() and g.name.startswith("group")])

counter = 1
for group in group_dirs:
    test_files = sorted((group / "test").glob("*.jpg"))
    template_files = sorted((group / "template").glob("*.jpg"))
    annot_files = sorted((group / "annotations").glob("*.txt"))

    # Sanity check: groups should have equal counts
    if not (len(test_files) == len(template_files) == len(annot_files)):
        print(f"⚠️ Skipping {group.name} (unequal files: "
              f"test={len(test_files)}, temp={len(template_files)}, ann={len(annot_files)})")
        continue

    # Iterate through all files in the group
    for t_file, temp_file, ann_file in zip(test_files, template_files, annot_files):
        code = str(counter).zfill(4)  

        new_test = test_dir / f"{code}_test.jpg"
        new_template = template_dir / f"{code}_temp.jpg"
        new_annot = annot_dir / f"{code}.txt"

        shutil.copy(t_file, new_test)
        shutil.copy(temp_file, new_template)
        shutil.copy(ann_file, new_annot)

        print(f"✅ {group.name}: {t_file.name}, {temp_file.name}, {ann_file.name} -> {code}")
        counter += 1

print(f"Dataset reorganized successfully! Total files: {counter-1}")
