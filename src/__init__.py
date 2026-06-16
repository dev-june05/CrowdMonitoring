"""
Crowd Monitoring & Density Estimation System
=============================================
Perspective-aware real-time crowd density estimation using
YOLO detection with adaptive Gaussian KDE heatmaps.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class DetectionResult:
    """Output from the person detector."""
    boxes: np.ndarray          # (N, 4) array of [x1, y1, x2, y2]
    confidences: np.ndarray    # (N,) detection confidences
    track_ids: np.ndarray      # (N,) ByteTrack IDs (int), or empty if no tracking
    frame_shape: tuple         # (height, width) of the source frame

    @property
    def count(self) -> int:
        return len(self.boxes)

    @property
    def has_tracks(self) -> bool:
        return self.track_ids is not None and len(self.track_ids) > 0


@dataclass
class FootPoint:
    """A person's estimated ground-contact position."""
    x: float                   # foot x position (image coords)
    y: float                   # foot y position (image coords)
    bbox_height: float         # bounding box height in pixels (perspective proxy)
    track_id: int = -1         # ByteTrack ID (-1 if untracked)
    confidence: float = 1.0    # detection confidence


@dataclass
class CongestionAlert:
    """Alert for a congested grid cell."""
    cell_row: int
    cell_col: int
    cell_bounds: tuple         # (x1, y1, x2, y2) pixel bounds
    density_value: float       # perspective-corrected density
    severity: str              # "WARNING" or "CRITICAL"


@dataclass
class FlowVector:
    """Motion vector for a tracked person."""
    track_id: int
    x: float                   # current foot x
    y: float                   # current foot y
    vx: float                  # velocity x (pixels/frame)
    vy: float                  # velocity y (pixels/frame)

    @property
    def speed(self) -> float:
        return np.sqrt(self.vx ** 2 + self.vy ** 2)

    @property
    def angle(self) -> float:
        """Angle in radians, 0 = right, pi/2 = down."""
        return np.arctan2(self.vy, self.vx)


@dataclass
class FlowMetrics:
    """Aggregate crowd flow metrics for a region."""
    mean_velocity: tuple = (0.0, 0.0)    # (vx, vy)
    mean_speed: float = 0.0
    speed_variance: float = 0.0
    motion_entropy: float = 0.0          # normalized 0-1
    counter_flow_ratio: float = 0.0      # fraction opposing dominant direction
    num_tracked: int = 0


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""
    # Input
    source: object = 0                    # webcam int index or video file path string
    model_path: str = "models/yolov8n.pt"
    device: int = 0                       # GPU device

    # Performance / resolution
    capture_width: int = 1280             # frame width fed into pipeline
    capture_height: int = 720             # frame height fed into pipeline
    yolo_imgsz: int = 640                 # YOLO inference image size (lower = faster)
    skip_frames: int = 0                  # run detection every skip_frames+1 frames

    # ROI
    roi_config_path: str = "config/roi_config.json"
    roi_mode: str = "manual"              # "manual" or "auto"
    redefine_roi: bool = False

    # Detection
    confidence_threshold: float = 0.25
    person_class_id: int = 0

    # Density
    grid_rows: int = 8
    grid_cols: int = 8
    kde_gamma: float = 0.3               # sigma = gamma * bbox_height
    kde_fixed_sigma: float = 20.0        # for fixed-sigma mode
    use_adaptive_kde: bool = True

    # Temporal
    ema_alpha: float = 0.3
    use_adaptive_alpha: bool = True

    # Congestion
    warning_threshold: float = 3.0
    critical_threshold: float = 6.0

    # Flow
    counter_flow_threshold: float = 0.3
    entropy_alert_threshold: float = 0.8
    speed_anomaly_factor: float = 3.0

    # Visualization
    heatmap_opacity: float = 0.4
    show_heatmap: bool = True
    show_grid: bool = False
    show_foot_points: bool = True
    show_roi: bool = True
    show_velocity: bool = False
    show_stats: bool = True

    # Auto segmentation (BiSeNetV2)
    seg_model_config: str = ""
    seg_model_checkpoint: str = ""
    seg_run_interval: int = 30            # run every N-th frame

    # Output
    output_dir: str = "outputs"
    record: bool = False
