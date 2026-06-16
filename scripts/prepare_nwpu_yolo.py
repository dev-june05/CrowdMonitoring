import os
import json
import shutil
from pathlib import Path
import cv2

def process_nwpu_to_yolo(source_dir, output_dir, box_size=30):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    images_dir = source_dir / "images"
    jsons_dir = source_dir / "jsons"
    
    if not images_dir.exists() or not jsons_dir.exists():
        print(f"Error: {images_dir} or {jsons_dir} does not exist. Ensure extraction is complete.")
        return
        
    print("Setting up YOLO dataset structure...")
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    splits = {}
    for split in ["train", "val"]:
        split_file = source_dir / f"{split}.txt"
        if split_file.exists():
            with open(split_file, "r") as f:
                # Remove empty strings and newline characters, and strip .jpg if present
                splits[split] = [line.strip().split()[0].replace('.jpg', '') for line in f.readlines() if line.strip()]
        else:
            print(f"Warning: {split_file} not found!")
            splits[split] = []
            
    # Process images and labels
    for split, img_ids in splits.items():
        print(f"\nProcessing {split} split ({len(img_ids)} images)...")
        processed = 0
        for idx, img_id in enumerate(img_ids):
            img_path = images_dir / f"{img_id}.jpg"
            json_path = jsons_dir / f"{img_id}.json"
            
            if not img_path.exists() or not json_path.exists():
                continue
                
            out_img_path = output_dir / "images" / split / f"{img_id}.jpg"
            out_label_path = output_dir / "labels" / split / f"{img_id}.txt"
            
            # Skip if already processed
            if out_img_path.exists() and out_label_path.exists():
                processed += 1
                continue
                
            shutil.copy(img_path, out_img_path)
            
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            
            with open(json_path, "r") as f:
                data = json.load(f)
                
            points = data.get("points", [])
            
            lines = []
            for pt in points:
                px, py = pt[0], pt[1]
                # Create pseudo-box
                bw = box_size / w
                bh = box_size / h
                cx = px / w
                cy = py / h
                
                # Clip
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                
            with open(out_label_path, "w") as f:
                f.writelines(lines)
            
            processed += 1
            if processed % 100 == 0:
                print(f"  Processed {processed}/{len(img_ids)} images...")
                
        print(f"Finished {split} split. Total processed: {processed}")
                
    print(f"\nYOLO dataset successfully created at {output_dir}")

if __name__ == "__main__":
    src = r"D:\Project\Datasets\NWPU-Crowd"
    dst = r"D:\Project\Datasets\NWPU-YOLO"
    process_nwpu_to_yolo(src, dst)
