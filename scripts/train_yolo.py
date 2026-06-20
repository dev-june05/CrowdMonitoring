import os
from pathlib import Path
from ultralytics import YOLO

def main():
    # Define absolute paths
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "nwpu.yaml"
    model_path = r"C:\Users\Dell\CrowdMonitoring\models\training\nwpu_finetune\weights\last.pt"
    save_dir = project_root / "models" / "training"
    
    print("=" * 60)
    print("  NWPU-Crowd YOLOv8 Fine-Tuning")
    print("=" * 60)
    
    print(f"\nInitializing YOLOv8 Nano from {model_path}...")
    model = YOLO(str(model_path))
    
    print(f"Starting fine-tuning using config: {config_path}")
    print("Settings: imgsz=640, batch=16, epochs=250, device=0 (RTX 4090), workers=16 (i9 14900K)")
    print("NOTE: Training at 640p for stability. SAHI will handle high-res tiled inference.\n")
    
    try:
        results = model.train(
            data=str(config_path),
            epochs=200,       
            imgsz=640,        # Train at 640. SAHI tiles input into 640x640 patches at inference for high-res detection.
            batch=4,         # Sweet spot for 24GB RTX 4090 at 640p
            device=0,         
            project=str(save_dir),
            name="nwpu_finetune",
            exist_ok=True,
            workers=16,   
            resume=True   
        )
        print("\nTraining complete!")
        print(f"Best weights saved to: {save_dir / 'nwpu_finetune' / 'weights' / 'best.pt'}")
    except Exception as e:
        print(f"\nError during training: {e}")

if __name__ == "__main__":
    main()
