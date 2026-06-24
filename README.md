# Intelligent Crowd Monitoring and Density Estimation System

An advanced spatial crowd understanding system that uses computer vision to track individuals, estimate crowd density, detect congestion, and flag anomalous flow behavior without requiring camera calibration.

## Features
- **Person Tracking**: YOLOv8 + ByteTrack
- **Perspective Projection**: Converts pixel coordinates to real-world floor meters using Ground-Plane Homography.
- **Physical Density Calculation**: Groups people using DBSCAN and calculates precise footprint areas (people/m²) using Alpha Shapes.
- **Dynamic Context Risk**: Evaluates crowd safety thresholds based on physical density (Polus LOS standards) or falls back to place-specific count limits.
- **Alerting & Audit**: Auto-records video clips of sustained critical risk events and logs per-frame statistics to CSV.
- **Flow Analysis**: Velocity vectors, counter-flow detection, and motion entropy analysis.

## Setup
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Pipeline
```bash
# 1. Run Calibration Wizard (One-time setup per camera view)
python main.py --camera 0 --calibrate --perspective homography

# 2. Run Live Monitoring (with auto video clip recording on critical risk)
python main.py --camera 0 --perspective homography --record-alerts

# 3. Test on Video File (e.g. testing context risk fallback for a school)
python main.py --source path/to/video.mp4 --perspective proxy --place school

# Advanced settings
python main.py --source video.mp4 --grid 12 12 --warn-threshold 3
```

### Keyboard Controls
- `H` : Toggle Heatmap
- `G` : Toggle Grid
- `F` : Toggle Foot/Head Points
- `V` : Toggle Velocity Arrows
- `R` : Define/Redefine ROI (Interactive Polygon Tool)
- `D` : Toggle Stats Panel
- `S` : Save Screenshot
- `Q` : Quit

## Training on NWPU-Crowd
To fine-tune the model for dense, overhead surveillance:
1. Download the NWPU-Crowd dataset to `D:\Project\Datasets\NWPU-Crowd`
2. Extract the dataset: `python scripts\extract_nwpu.py`
3. Prepare the YOLO labels: `python scripts\prepare_nwpu_yolo.py`
4. Train the model: `python scripts\train_yolo.py`
