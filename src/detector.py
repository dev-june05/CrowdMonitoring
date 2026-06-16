"""Person detection and tracking using Ultralytics YOLOv8 + ByteTrack.

Provides a single ``PersonDetector`` class that wraps the Ultralytics YOLO
model for person detection (class 0) and ByteTrack multi-object tracking.
All results are returned as :class:`src.DetectionResult` dataclass instances
to keep data exchange consistent across the pipeline.

Typical usage::

    detector = PersonDetector('models/yolov8n.pt', device=0, conf=0.25)
    result = detector.detect(frame)        # detection + tracking
    result = detector.detect_no_track(frame)  # detection only
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from ultralytics import YOLO

from src import DetectionResult

logger = logging.getLogger(__name__)


class PersonDetector:
    """Wrap Ultralytics YOLO with ByteTrack tracking for person detection.

    Parameters
    ----------
    model_path : str
        Path to a YOLOv8 model file (e.g. ``'models/yolov8n.pt'``).
    device : int
        CUDA device index.  Use ``-1`` or ``'cpu'`` for CPU inference.
    conf : float
        Minimum confidence threshold for detections (0‑1).
    """

    # Person class in the COCO dataset
    _PERSON_CLASS_ID: int = 0

    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        device: int = 0,
        conf: float = 0.25,
        imgsz: int = 640,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

        # Load the YOLO model once at init time.
        logger.info("Loading YOLO model from '%s' (device=%s, imgsz=%s)", model_path, device, imgsz)
        self.model = YOLO(model_path)
        logger.info("YOLO model loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run detection **with** ByteTrack tracking.

        Uses ``model.track()`` with ``persist=True`` so ByteTrack maintains
        identity across consecutive calls.

        Parameters
        ----------
        frame : np.ndarray
            BGR image as a NumPy array of shape ``(H, W, 3)``.

        Returns
        -------
        DetectionResult
            Detection boxes, confidences, and ByteTrack track IDs.
        """
        results = self.model.track(
            frame,
            persist=True,                   # keep tracker state between frames
            tracker="bytetrack.yaml",        # use ByteTrack config shipped with Ultralytics
            classes=[self._PERSON_CLASS_ID], # only detect persons
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,                   # suppress per-frame logs
        )
        return self._parse_results(results, frame.shape[:2], tracked=True)

    def detect_no_track(self, frame: np.ndarray) -> DetectionResult:
        """Run detection **without** tracking (single-frame inference).

        Useful for one-shot analysis where temporal continuity is not needed.

        Parameters
        ----------
        frame : np.ndarray
            BGR image as a NumPy array of shape ``(H, W, 3)``.

        Returns
        -------
        DetectionResult
            Detection boxes and confidences; ``track_ids`` will be an empty
            array.
        """
        results = self.model(
            frame,
            classes=[self._PERSON_CLASS_ID],
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        return self._parse_results(results, frame.shape[:2], tracked=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_results(
        results,
        frame_shape: tuple[int, int],
        tracked: bool,
    ) -> DetectionResult:
        """Extract numpy arrays from Ultralytics Results objects.

        Parameters
        ----------
        results : list[ultralytics.engine.results.Results]
            Raw output from ``model()`` or ``model.track()``.
        frame_shape : tuple[int, int]
            ``(height, width)`` of the input frame.
        tracked : bool
            Whether ByteTrack was used (determines how we read track IDs).

        Returns
        -------
        DetectionResult
        """
        # Ultralytics always returns a list; we only pass a single image.
        result = results[0]
        boxes_obj = result.boxes

        # Handle no detections — boxes_obj can have zero entries.
        if boxes_obj is None or len(boxes_obj) == 0:
            return DetectionResult(
                boxes=np.empty((0, 4), dtype=np.float32),
                confidences=np.empty(0, dtype=np.float32),
                track_ids=np.empty(0, dtype=np.int64),
                frame_shape=frame_shape,
            )

        # xyxy bounding boxes (N, 4) — always available
        bboxes = boxes_obj.xyxy.cpu().numpy().astype(np.float32)

        # Detection confidences (N,)
        confs = boxes_obj.conf.cpu().numpy().astype(np.float32)

        # Track IDs — only available when tracking is active AND the tracker
        # has successfully assigned IDs.  In the first few frames ByteTrack
        # may return None for .id while it initialises.
        if tracked and boxes_obj.id is not None:
            track_ids = boxes_obj.id.cpu().numpy().astype(np.int64).ravel()
        else:
            # No tracking or tracker not ready yet → empty array
            track_ids = np.empty(0, dtype=np.int64)

        return DetectionResult(
            boxes=bboxes,
            confidences=confs,
            track_ids=track_ids,
            frame_shape=frame_shape,
        )
