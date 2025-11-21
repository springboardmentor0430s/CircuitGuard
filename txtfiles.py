import os
import shutil


txt_folder = r"C:\Users\laksh\OneDrive\Desktop\coding\Circuitguard_Project\Dataset\PCBData\group92000\92000_not"
paired_folder = r"C:\Users\laksh\OneDrive\Desktop\coding\Circuitguard_Project\Dataset\PCBData_Paired"


for txt_file in os.listdir(txt_folder):
    if not txt_file.lower().endswith(".txt"):
        continue
    
    base_name = txt_file.replace(".txt", "")
    

    target_folder = os.path.join(paired_folder, base_name)
    if os.path.exists(target_folder):
        shutil.move(os.path.join(txt_folder, txt_file),
                    os.path.join(target_folder, txt_file))
        print(f"Moved {txt_file} to {target_folder}")
    else:
        print(f"No folder found for {txt_file}, skipping.")
