# Project Context: Intelligent Crowd Monitoring and Density Estimation System

## 1. Project Origins and Scope Pivot
This project began as a summer research internship. The initial inherited codebase attempted to perform crowd monitoring, face identification, and weapon detection. 

**The Pivot:**
After technical analysis, we decided to scrap the inherited codebase and start entirely from scratch. 
- *Why?* Face recognition in large crowds from surveillance angles is impractical and compute-heavy. Weapon detection suffers from extreme false positives and is outside the realistic scope of crowd flow analysis.
- *New Direction:* We pivoted strictly to **Crowd Density Estimation, Congestion Prediction, and Flow Analysis**. This is a highly relevant, research-oriented goal that is mathematically and computationally achievable without massive server racks.

---

## 2. Research & Methodology Decisions

To build a robust system without requiring complex manual camera calibration for every new room, we researched and implemented several cutting-edge techniques:

### A. Perspective Correction without Calibration
In a surveillance camera setup, people further away appear smaller (perspective distortion). Standard density algorithms fail here because 10 people far away take up fewer pixels than 2 people close to the camera.
- *Rejected Approaches:* Ground-Plane Homography (requires manual calibration per camera) and Monocular Depth Networks like MiDaS (too slow for real-time).
- *Chosen Approach: Bounding Box Proxy.* We use the height of the YOLO bounding box as a direct proxy for depth. A smaller box height = further away.

### B. Adaptive Kernel Density Estimation (KDE)
To generate the density heatmap, we don't just count bounding boxes. We project a 2D Gaussian "blob" onto the floor.
- *Foot Localization:* Instead of using the center of the bounding box (which floats in mid-air and distorts distance), we extract the **Foot Point** (bottom-center of the box) to map the person exactly to the 2D floor plane.
- *Adaptive Sigma:* The size (sigma) of the Gaussian blob is scaled dynamically based on the bounding box height (`sigma = gamma * bbox_height`). People close to the camera get wide blobs; people far away get tight blobs. This mathematically corrects perspective distortion in the final heatmap.

### C. Walkable Area Extraction (ROI)
We only care about people in "walkable" or relevant areas.
- *Phase 1 (Implemented):* Manual polygon ROI drawing via keyboard/mouse.
- *Phase 2 (Experimental):* Researched using BiSeNetV2 for real-time automatic semantic segmentation of the ground plane to automatically generate the ROI without human input.

### D. Flow and Congestion Analytics
- **Congestion:** We overlay a customizable grid on the ROI. We integrate the KDE heatmap values inside each cell to calculate true density, triggering WARNING or CRITICAL alerts.
- **Flow Analysis:** Using ByteTrack, we extract velocity vectors for each person. We aggregate this to calculate motion entropy (chaos) and counter-flow ratios.

---

## 3. Current State of the Codebase

We have successfully built a real-time inference pipeline that orchestrates all of the above methodologies.

**Tech Stack:** `Python 3.10+`, `ultralytics` (YOLOv8), `opencv-python`, `scipy`, `numpy`.

**Core Modules (`src/`):**
- `pipeline.py`: The orchestrator that manages video capture, frame resizing, and the execution loop.
- `detector.py`: Wraps YOLOv8 + ByteTrack.
- `foot_localizer.py`: Extracts foot points and handles ROI filtering.
- `density_estimator.py`: Computes the adaptive-bandwidth KDE heatmaps (highly optimized using sparse matrices and quarter-resolution scaling).
- `congestion.py` & `flow_analyzer.py`: Calculates alerts and physics/flow metrics.
- `visualizer.py`: A unified, z-ordered compositing engine that draws the heatmap, grid, vectors, and UI stats panel.

**Performance Optimization:**
We encountered extreme FPS drops (~3 FPS) when testing with high-res cameras (like a 4K DroidCam). We implemented extensive CLI flags to control performance:
- `--camera 0`: Explicit indexing for multi-camera setups.
- `--width 640 --height 480`: Caps the pipeline resolution.
- `--yolo-imgsz 320`: Drops YOLO inference resolution for speed.
- `--skip-frames 1`: Runs YOLO every other frame.
*Result: Boosted performance from 3 FPS to 26+ FPS on a laptop RTX 3050.*

---

## 4. The Critical Bottleneck (Why we stopped)

While the pipeline math is perfect, the actual person identification in our target environment (offices/classrooms with desks) is currently failing. 

**The Symptoms:**
1. YOLO misses people sitting in the back of the room.
2. For seated people, YOLO draws the bounding box around their upper torso (since legs are hidden). Our pipeline takes the bottom-center of this box, resulting in "floating foot points" that hover in the air above the desk, ruining the floor-plane density mapping.

**The Root Cause:**
We are currently using the standard `yolov8n.pt` weights. This model was trained on the COCO dataset, which is heavily biased towards **full-body, standing pedestrians** seen from horizontal angles. It fundamentally does not understand overhead perspectives or heavily occluded bodies (e.g., just a head and shoulders visible behind a monitor).

---

## 5. Future Goals & Next Steps (On the New PC)

To solve the occlusion and perspective failures, we **must train a custom YOLO model on the NWPU-Crowd dataset**. 
*NWPU-Crowd* is designed specifically for dense, overhead, heavily occluded surveillance scenarios where often only human heads are visible.

*(Note: The previous laptop overheated and crashed during Epoch 3 of training, necessitating the move to the new PC).*

### Immediate Action Plan on the New PC:
1. **Setup**: Clone the GitHub repo (`https://github.com/dev-june05/CrowdMonitoring`) and install dependencies (`pip install -r requirements.txt`).
2. **Dataset Prep**: We already wrote the extraction scripts. 
   - Run `python scripts/extract_nwpu.py` to unpack the data.
   - Run `python scripts/prepare_nwpu_yolo.py` to convert the NWPU point annotations into YOLO-compatible bounding boxes (specifically, we will generate fixed-size pseudo-boxes around the heads).
3. **Training**: Run `python scripts/train_yolo.py` to fine-tune YOLOv8 on the prepared NWPU dataset. With a better GPU, we should run this for 50-100 epochs.
4. **Pipeline Adjustment (Head Tracking)**: Once the model is trained to detect *heads* instead of *full bodies*, we need to update `foot_localizer.py` to become a `head_localizer.py`. We will anchor our KDE blobs to the center of the head boxes instead of the feet. This completely bypasses the desk-occlusion problem, as heads are almost always visible.
5. **BiSeNetV2 Integration**: Finalize the automatic ground segmentation so the user never has to manually draw an ROI polygon again.
