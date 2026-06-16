# Intelligent Crowd Monitoring and Density Estimation System

An advanced spatial crowd understanding system that uses computer vision to track individuals, estimate crowd density, detect congestion, and flag anomalous flow behavior without requiring camera calibration.

## Features
- **Person Tracking**: YOLOv8 + ByteTrack
- **Perspective-Aware KDE**: Computes crowd density heatmaps using bounding box height as a perspective proxy.
- **Congestion Alerting**: Grid-based multi-level (WARNING/CRITICAL) density alerts.
- **Flow Analysis**: Velocity vectors, counter-flow detection, and motion entropy analysis.
- **Auto-Segmentation**: BiSeNetV2 walkable area detection.

## Setup
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
pip install mmengine "mmcv>=2.0.0" mmsegmentation  # For auto-segmentation
```

## Running the Pipeline
```bash
# Webcam (Default is camera 0)
python main.py --camera 0

# Select specific webcam and reduce resolution for better FPS
python main.py --camera 0 --width 640 --height 480 --yolo-imgsz 320 --skip-frames 1

# Video File
python main.py --source path/to/video.mp4

# Advanced settings
python main.py --source video.mp4 --grid 12 12 --warn-threshold 3 --roi-mode auto
```

### Keyboard Controls
- `H` : Toggle Heatmap
- `G` : Toggle Grid
- `F` : Toggle Foot Points
- `V` : Toggle Velocity Arrows
- `M` : Switch ROI Mode (Manual ↔ Auto)
- `r` : Toggle ROI Visibility
- `R` : Redefine Manual ROI
- `D` : Toggle Stats Panel
- `S` : Save Screenshot
- `Q` : Quit

## Training on NWPU-Crowd
To fine-tune the model for dense, overhead surveillance:
1. Download the NWPU-Crowd dataset to `D:\Project\Datasets\NWPU-Crowd`
2. Extract the dataset: `python scripts\extract_nwpu.py`
3. Prepare the YOLO labels: `python scripts\prepare_nwpu_yolo.py`
4. Train the model: `python scripts\train_yolo.py`
