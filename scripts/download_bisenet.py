import urllib.request
import os
from pathlib import Path

def download_file(url, dest_path):
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Saved to {dest_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    config_dir = base_dir / "config"
    model_dir = base_dir / "models"
    
    config_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    
    config_url = "https://raw.githubusercontent.com/open-mmlab/mmsegmentation/main/configs/bisenetv2/bisenetv2_fcn_4xb4-160k_cityscapes-1024x1024.py"
    config_path = config_dir / "bisenetv2.py"
    
    checkpoint_url = "https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes_20210902_015551-bcf10f09.pth"
    checkpoint_path = model_dir / "bisenetv2.pth"
    
    print("Setting up BiSeNetV2...")
    download_file(config_url, config_path)
    download_file(checkpoint_url, checkpoint_path)
    print("Done!")
