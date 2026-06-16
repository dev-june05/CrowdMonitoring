"""
Pipeline Orchestrator
=====================
Main processing loop that ties all modules together.

Frame Capture → Detection → Foot Localization → ROI Filter →
Density KDE → Temporal Smoothing → Congestion Check →
Flow Analysis → Visualization → Display
"""

import cv2
import time
import numpy as np
from pathlib import Path

from src import PipelineConfig, FootPoint
from src.detector import PersonDetector
from src.roi_manager import ROIManager
from src.foot_localizer import FootLocalizer
from src.density_estimator import DensityEstimator
from src.temporal_filter import TemporalFilter
from src.congestion import CongestionDetector
from src.flow_analyzer import FlowAnalyzer
from src.visualizer import Visualizer
from src.ground_segmentor import GroundSegmentor


class Pipeline:
    """Main processing pipeline for crowd monitoring.

    Orchestrates all modules in sequence: detection, localization,
    density estimation, congestion detection, flow analysis, and
    visualization.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._init_modules()

    def _init_modules(self):
        """Initialize all pipeline modules."""
        # Person detector (YOLO + ByteTrack)
        self.detector = PersonDetector(
            model_path=self.config.model_path,
            device=self.config.device,
            conf=self.config.confidence_threshold,
            imgsz=self.config.yolo_imgsz,
        )

        # ROI manager (manual polygon)
        self.roi_manager = ROIManager(config_path=self.config.roi_config_path)

        # Ground segmentor (BiSeNetV2, optional)
        self.ground_segmentor = GroundSegmentor(
            config_path=self.config.seg_model_config,
            checkpoint_path=self.config.seg_model_checkpoint,
            run_interval=self.config.seg_run_interval
        )

        # Current ROI mode
        self.roi_mode = self.config.roi_mode  # "manual" or "auto"

        # Foot localizer (initialized after ROI is defined)
        self.foot_localizer = FootLocalizer()

        # Density estimator (initialized after first frame for shape)
        self.density_estimator = None

        # Temporal filter
        self.temporal_filter = TemporalFilter(
            alpha=self.config.ema_alpha,
            adaptive=self.config.use_adaptive_alpha
        )

        # Congestion detector (initialized after ROI)
        self.congestion_detector = None

        # Flow analyzer
        self.flow_analyzer = FlowAnalyzer(
            counter_flow_threshold=self.config.counter_flow_threshold,
            entropy_threshold=self.config.entropy_alert_threshold,
            speed_anomaly_factor=self.config.speed_anomaly_factor
        )

        # Visualizer
        self.visualizer = Visualizer(self.config)

        # State
        self._roi_mask = None
        self._roi_polygon = None
        self._frame_shape = None
        self._initialized = False
        self._last_detection = None   # cache for frame-skipping

    def _open_source(self):
        """Open video source (webcam or file)."""
        source = self.config.source

        # Already an int means webcam index, otherwise treat as file path string
        if not isinstance(source, int):
            try:
                source = int(source)
            except (ValueError, TypeError):
                pass  # Keep as string (file path)

        cap = cv2.VideoCapture(source)

        if isinstance(source, int):
            # Webcam: request the configured resolution (not always honoured by driver)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.capture_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.capture_height)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.config.source}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        print(f"Source opened: {w}x{h} @ {fps:.1f} FPS")

        # Store resolved source so the rest of pipeline can check isinstance
        self._source_is_camera = isinstance(source, int)

        return cap

    def _setup_roi(self, first_frame: np.ndarray):
        """Setup ROI from manual polygon or auto segmentation."""
        h, w = first_frame.shape[:2]
        self._frame_shape = (h, w)

        if self.roi_mode == "manual":
            # Try loading existing ROI
            if not self.config.redefine_roi and self.roi_manager.load():
                print("ROI loaded from config.")
            else:
                print("Define ROI polygon interactively...")
                self.roi_manager.define_interactive(first_frame)

            if self.roi_manager.is_defined:
                self._roi_mask = self.roi_manager.generate_mask((h, w))
                self._roi_polygon = self.roi_manager.get_polygon()
            else:
                # Full frame as ROI
                self._roi_mask = np.ones((h, w), dtype=np.uint8)
                self._roi_polygon = np.array([
                    [0, 0], [w, 0], [w, h], [0, h]
                ], dtype=np.int32)

        elif self.roi_mode == "auto":
            if self.ground_segmentor.is_available:
                print("Running initial segmentation...")
                self._roi_mask = self.ground_segmentor.segment(first_frame, force=True)
                # Create convex hull polygon from mask contours
                self._roi_polygon = self._mask_to_polygon(self._roi_mask)
            else:
                print("Auto-segmentation unavailable. Falling back to manual ROI.")
                self.roi_mode = "manual"
                self._setup_roi(first_frame)
                return

        # Initialize dependent modules
        self.foot_localizer.set_roi_mask(self._roi_mask)

        self.density_estimator = DensityEstimator(
            frame_shape=(h, w),
            gamma=self.config.kde_gamma,
            fixed_sigma=self.config.kde_fixed_sigma
        )

        self.congestion_detector = CongestionDetector(
            roi_polygon=self._roi_polygon,
            frame_shape=(h, w),
            rows=self.config.grid_rows,
            cols=self.config.grid_cols,
            warning_threshold=self.config.warning_threshold,
            critical_threshold=self.config.critical_threshold
        )

        self._initialized = True

    def _mask_to_polygon(self, mask: np.ndarray) -> np.ndarray:
        """Convert binary mask to the largest contour polygon."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            h, w = mask.shape[:2]
            return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32)

        # Use largest contour
        largest = max(contours, key=cv2.contourArea)
        # Simplify polygon
        epsilon = 0.01 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        return approx.reshape(-1, 2)

    def _switch_roi_mode(self, frame: np.ndarray):
        """Toggle between manual and auto ROI modes."""
        if self.roi_mode == "manual":
            if self.ground_segmentor.is_available:
                self.roi_mode = "auto"
                self._roi_mask = self.ground_segmentor.segment(frame, force=True)
                self._roi_polygon = self._mask_to_polygon(self._roi_mask)
                print("Switched to AUTO (BiSeNetV2) ROI mode")
            else:
                print("Auto-segmentation not available. Staying in manual mode.")
        else:
            self.roi_mode = "manual"
            if self.roi_manager.is_defined:
                h, w = frame.shape[:2]
                self._roi_mask = self.roi_manager.generate_mask((h, w))
                self._roi_polygon = self.roi_manager.get_polygon()
                print("Switched to MANUAL ROI mode")
            else:
                print("No manual ROI defined. Define one with 'R' (redefine).")

        # Update dependent modules
        self.foot_localizer.set_roi_mask(self._roi_mask)
        if self._roi_polygon is not None:
            self.congestion_detector = CongestionDetector(
                roi_polygon=self._roi_polygon,
                frame_shape=self._frame_shape,
                rows=self.config.grid_rows,
                cols=self.config.grid_cols,
                warning_threshold=self.config.warning_threshold,
                critical_threshold=self.config.critical_threshold
            )

    def _redefine_roi(self, frame: np.ndarray):
        """Redefine manual ROI interactively."""
        self.roi_manager.define_interactive(frame)
        if self.roi_manager.is_defined:
            h, w = frame.shape[:2]
            self._roi_mask = self.roi_manager.generate_mask((h, w))
            self._roi_polygon = self.roi_manager.get_polygon()
            self.foot_localizer.set_roi_mask(self._roi_mask)
            self.congestion_detector = CongestionDetector(
                roi_polygon=self._roi_polygon,
                frame_shape=(h, w),
                rows=self.config.grid_rows,
                cols=self.config.grid_cols,
                warning_threshold=self.config.warning_threshold,
                critical_threshold=self.config.critical_threshold
            )
            self.roi_mode = "manual"
            print("ROI redefined.")

    def _save_screenshot(self, frame: np.ndarray):
        """Save current frame to outputs directory."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"screenshot_{timestamp}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"Screenshot saved: {path}")

    def run(self):
        """Main processing loop."""
        cap = self._open_source()

        # Read first frame for initialization
        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame from source.")

        # Resize first frame to configured resolution
        fh, fw = first_frame.shape[:2]
        if (fw, fh) != (self.config.capture_width, self.config.capture_height):
            first_frame = cv2.resize(
                first_frame,
                (self.config.capture_width, self.config.capture_height),
                interpolation=cv2.INTER_LINEAR
            )

        # Setup ROI on first frame
        self._setup_roi(first_frame)

        # FPS tracking
        prev_time = time.time()
        fps = 0.0
        frame_idx = 0

        # For video files: rewind to start after using first frame for ROI setup
        if not self._source_is_camera:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        print("\nPipeline running. Press Q to quit.")
        print("Keyboard: H=heatmap G=grid F=feet V=velocity M=mode R=redefine-ROI S=screenshot\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                # Webcam: just stop. Video file: loop back.
                if self._source_is_camera:
                    break
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

            frame_idx += 1

            # Resize frame to configured resolution (handles DroidCam / 4K cameras)
            fh, fw = frame.shape[:2]
            if (fw, fh) != (self.config.capture_width, self.config.capture_height):
                frame = cv2.resize(
                    frame,
                    (self.config.capture_width, self.config.capture_height),
                    interpolation=cv2.INTER_LINEAR
                )

            # --- 1. Person Detection (with optional frame skipping) ---
            skip = self.config.skip_frames
            if skip <= 0 or frame_idx % (skip + 1) == 1:
                detection = self.detector.detect(frame)
                self._last_detection = detection
            else:
                detection = self._last_detection  # reuse previous result

            # --- 2. Foot Point Extraction ---
            foot_points = self.foot_localizer.extract(detection)

            # --- 3. Auto-segmentation update (every N-th frame) ---
            if self.roi_mode == "auto" and self.ground_segmentor.is_available:
                new_mask = self.ground_segmentor.segment(frame)
                if new_mask is not self._roi_mask:
                    self._roi_mask = new_mask
                    self.foot_localizer.set_roi_mask(self._roi_mask)

            # --- 4. Density Estimation ---
            heatmap = self.density_estimator.compute(
                foot_points,
                roi_mask=self._roi_mask,
                adaptive=self.config.use_adaptive_kde
            )

            # --- 5. Temporal Smoothing ---
            smoothed_heatmap = self.temporal_filter.update(heatmap)

            # --- 6. Congestion Detection ---
            density_matrix, alerts = self.congestion_detector.analyze(foot_points)

            # --- 7. Flow Analysis ---
            flow_vectors, flow_metrics = self.flow_analyzer.update(foot_points)
            anomalies = self.flow_analyzer.detect_anomalies(flow_vectors, flow_metrics)

            # --- 8. FPS Calculation ---
            current_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(current_time - prev_time, 1e-6))
            prev_time = current_time

            # --- 9. Visualization ---
            display = self.visualizer.render(
                frame=frame,
                foot_points=foot_points,
                heatmap=smoothed_heatmap,
                roi_polygon=self._roi_polygon,
                roi_mask=self._roi_mask,
                density_matrix=density_matrix,
                congestion_alerts=alerts,
                flow_vectors=flow_vectors,
                flow_metrics=flow_metrics,
                anomalies=anomalies,
                fps=fps,
                person_count=detection.count,
                roi_mode=self.roi_mode
            )

            # --- 10. Display ---
            cv2.namedWindow("Crowd Monitoring", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Crowd Monitoring", 1280, 720)
            cv2.imshow("Crowd Monitoring", display)

            # --- 11. Keyboard Handling ---
            key = cv2.waitKey(1)
            action = self.visualizer.handle_key(key)

            if action == "quit":
                break
            elif action == "switch_roi_mode":
                self._switch_roi_mode(frame)
            elif action == "screenshot":
                self._save_screenshot(display)
            elif action == "redefine_roi":
                self._redefine_roi(frame)

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Pipeline stopped.")
