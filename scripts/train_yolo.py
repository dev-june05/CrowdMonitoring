import os
from pathlib import Path
from ultralytics import YOLO

def main():
    # Define absolute paths
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "nwpu.yaml"
    model_path = project_root / "models" / "yolov8n.pt"
    save_dir = project_root / "models" / "training"
    
    print("=" * 60)
    print("  NWPU-Crowd YOLOv8 Fine-Tuning")
    print("=" * 60)
    
    print(f"\nInitializing YOLOv8 Nano from {model_path}...")
    model = YOLO(str(model_path))
    
    print(f"Starting fine-tuning using config: {config_path}")
    print("Settings: imgsz=640, batch=4, epochs=50, device=0 (RTX 3050)")
    
    try:
        results = model.train(
            data=str(config_path),
            epochs=50,
            imgsz=640,
            batch=2,          # Reduced batch size to prevent CUDA OOM on 4GB VRAM
            device=0,         # Use GPU 0
            project=str(save_dir),
            name="nwpu_finetune",
            exist_ok=True,
            workers=0         # Set workers=0 to fix Windows shared file mapping (error 1455)
        )
        print("\nTraining complete!")
        print(f"Best weights saved to: {save_dir / 'nwpu_finetune' / 'weights' / 'best.pt'}")
    except Exception as e:
        print(f"\nError during training: {e}")

if __name__ == "__main__":
    main()
