import zipfile
import os
from pathlib import Path
import sys

def extract_dataset(source_dir, dest_dir):
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    zip_files = sorted(source_dir.glob("*.zip"))
    if not zip_files:
        print("No zip files found in", source_dir)
        return

    for zip_file in zip_files:
        print(f"Extracting {zip_file.name}...")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
        except Exception as e:
            print(f"Error extracting {zip_file.name}: {e}")
            
    # Move txt files if they exist in the root of NWPU-Crowd but not inside the zips
    for txt_file in source_dir.glob("*.txt"):
        print(f"Copying {txt_file.name}...")
        dest_path = dest_dir / txt_file.name
        dest_path.write_text(txt_file.read_text())
        
    print("Dataset extraction complete.")

if __name__ == "__main__":
    src = r"C:\Users\Dell\Datasets\NWPU-Crowd\NWPU-Crowd"
    dst = r"C:\Users\Dell\Datasets\NWPU-Crowd"
    
    # We redefine the extract_dataset logic here just in case it was lost, but the function extract_dataset should be untouched above.
    extract_dataset(src, dst)
