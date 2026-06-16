import os
import zipfile
import shutil
from pathlib import Path
import subprocess
import sys

def main():
    base_dataset_dir = Path(r"D:\Project\Datasets\NWPU-Crowd")
    zip_dir = base_dataset_dir / "NWPU-Crowd"
    
    print("="*60)
    print("  NWPU-Crowd Training Pipeline")
    print("="*60)
    
    # 1. Extraction
    print("\n[1/3] Extracting Dataset...")
    images_dir = base_dataset_dir / "images"
    jsons_dir = base_dataset_dir / "jsons"
    mats_dir = base_dataset_dir / "mats"
    
    images_dir.mkdir(exist_ok=True)
    jsons_dir.mkdir(exist_ok=True)
    mats_dir.mkdir(exist_ok=True)
    
    zip_files = sorted(zip_dir.glob("*.zip"))
    if not zip_files:
        print("Warning: No zip files found in", zip_dir)
    else:
        for zip_file in zip_files:
            print(f"  Extracting {zip_file.name}...")
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    for member in zip_ref.namelist():
                        filename = os.path.basename(member)
                        # Skip directories
                        if not filename:
                            continue
                        
                        source = zip_ref.open(member)
                        
                        if filename.endswith('.jpg'):
                            target_path = images_dir / filename
                        elif filename.endswith('.json'):
                            target_path = jsons_dir / filename
                        elif filename.endswith('.mat'):
                            target_path = mats_dir / filename
                        else:
                            target_path = base_dataset_dir / filename
                            
                        with open(target_path, "wb") as target:
                            shutil.copyfileobj(source, target)
            except Exception as e:
                print(f"  Error extracting {zip_file.name}: {e}")
                
        # Copy txt files
        for txt_file in zip_dir.glob("*.txt"):
            shutil.copy(txt_file, base_dataset_dir / txt_file.name)
            
    print("  Extraction complete.")
    
    # 2. Data Preparation
    print("\n[2/3] Converting annotations to YOLO format...")
    prep_script = Path(__file__).resolve().parent / "prepare_nwpu_yolo.py"
    subprocess.run([sys.executable, str(prep_script)], check=True)
    
    # 3. Training
    print("\n[3/3] Starting YOLOv8 Fine-tuning...")
    train_script = Path(__file__).resolve().parent / "train_yolo.py"
    subprocess.run([sys.executable, str(train_script)], check=True)

if __name__ == "__main__":
    main()
