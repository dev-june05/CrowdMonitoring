"""Foot-point localisation from bounding-box detections.

Extracts ground-contact positions (bottom-centre of bounding boxes) for use
in spatial density estimation.  Bottom-centre is more spatially accurate than
box centre for overhead / angled camera views.

All data types (``FootPoint``, ``DetectionResult``) are imported from the
shared ``src`` package so every module speaks the same language.

Typical usage::

    localizer = FootLocalizer(roi_mask=roi_manager.get_mask())
    foot_points = localizer.extract(detection_result)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src import DetectionResult, FootPoint

logger = logging.getLogger(__name__)


class FootLocalizer:
    """Extract ground-contact positions from YOLO bounding-box detections.

    The *foot point* is the bottom-centre of each bounding box — the
    approximate ground-contact position of a standing person.  An optional
    binary ROI mask can be supplied to filter out detections that fall
    outside the region of interest.

    Parameters
    ----------
    roi_mask : np.ndarray | None
        Binary mask of shape ``(H, W)`` where values > 0 mark the ROI.
        When ``None`` every detection is kept.
    """

    def __init__(self, roi_mask: Optional[np.ndarray] = None) -> None:
        self._roi_mask: Optional[np.ndarray] = None
        if roi_mask is not None:
            self.set_roi_mask(roi_mask)

    # ------------------------------------------------------------------
    # ROI management
    # ------------------------------------------------------------------

    def set_roi_mask(self, mask: np.ndarray) -> None:
        """Update the ROI mask used for spatial filtering.

        Parameters
        ----------
        mask : np.ndarray
            2-D binary mask (``uint8`` recommended) where values > 0
            indicate the region of interest.

        Raises
        ------
        ValueError
            If *mask* is not two-dimensional.
        """
        mask = np.asarray(mask)
        if mask.ndim != 2:
            raise ValueError(f"ROI mask must be 2-D, got {mask.ndim}-D")
        self._roi_mask = mask
        logger.debug("ROI mask updated — shape %s", mask.shape)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(self, detection: DetectionResult) -> list[FootPoint]:
        """Convert bounding-box detections into foot points.

        For each box ``(x1, y1, x2, y2)``:

        * ``foot_x = (x1 + x2) / 2``  — horizontal centre
        * ``foot_y = y2``              — bottom edge (ground contact)
        * ``bbox_height = y2 - y1``

        If a ROI mask has been set, only foot points whose pixel position
        falls inside the mask (value > 0) are returned.

        Parameters
        ----------
        detection : DetectionResult
            Output from :class:`src.detector.PersonDetector`.

        Returns
        -------
        list[FootPoint]
            One foot point per (valid, in-ROI) detection.
        """
        if detection.count == 0:
            return []

        boxes = detection.boxes  # (N, 4)

        # Vectorised foot-point arithmetic
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        foot_x = (x1 + x2) / 2.0
        foot_y = y2
        bbox_h = y2 - y1

        # Resolve track IDs — fall back to -1 when tracking is inactive
        has_tracks = detection.has_tracks
        track_ids = detection.track_ids if has_tracks else None

        # Build FootPoint list, optionally filtering by ROI
        points: list[FootPoint] = []
        for i in range(detection.count):
            fx, fy = float(foot_x[i]), float(foot_y[i])

            # --- ROI filter ---
            if self._roi_mask is not None:
                h, w = self._roi_mask.shape
                # Clamp to image bounds before sampling the mask
                px = int(np.clip(round(fx), 0, w - 1))
                py = int(np.clip(round(fy), 0, h - 1))
                if self._roi_mask[py, px] == 0:
                    continue  # outside ROI — skip

            points.append(
                FootPoint(
                    x=fx,
                    y=fy,
                    bbox_height=float(bbox_h[i]),
                    track_id=int(track_ids[i]) if has_tracks else -1,
                    confidence=float(detection.confidences[i]),
                )
            )

        logger.debug(
            "Extracted %d foot points from %d detections (ROI active: %s)",
            len(points),
            detection.count,
            self._roi_mask is not None,
        )
        return points

    # ------------------------------------------------------------------
    # Static utility
    # ------------------------------------------------------------------

    @staticmethod
    def compute_foot_point(box: np.ndarray) -> tuple[float, float, float]:
        """Compute the foot point for a single bounding box.

        Parameters
        ----------
        box : np.ndarray
            Array of shape ``(4,)`` with format ``[x1, y1, x2, y2]``.

        Returns
        -------
        tuple[float, float, float]
            ``(foot_x, foot_y, bbox_height)``
        """
        x1, y1, x2, y2 = box[:4]
        foot_x = float((x1 + x2) / 2.0)
        foot_y = float(y2)
        bbox_height = float(y2 - y1)
        return foot_x, foot_y, bbox_height


# ----------------------------------------------------------------------
# Quick smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    loc = FootLocalizer()

    # Simulate a DetectionResult with two persons
    boxes = np.array(
        [[100, 200, 150, 400], [300, 100, 380, 350]], dtype=np.float32
    )
    confs = np.array([0.9, 0.75], dtype=np.float32)
    ids = np.array([1, 2], dtype=np.int64)

    det = DetectionResult(
        boxes=boxes,
        confidences=confs,
        track_ids=ids,
        frame_shape=(480, 640),
    )
    pts = loc.extract(det)
    for p in pts:
        print(
            f"Track {p.track_id}: foot=({p.x:.1f}, {p.y:.1f}), "
            f"h={p.bbox_height:.1f}, conf={p.confidence:.2f}"
        )

    # Empty case
    empty_det = DetectionResult(
        boxes=np.empty((0, 4), dtype=np.float32),
        confidences=np.empty(0, dtype=np.float32),
        track_ids=np.empty(0, dtype=np.int64),
        frame_shape=(480, 640),
    )
    print(f"Empty extract → {len(loc.extract(empty_det))} points")

    # ROI filtering
    mask = np.zeros((500, 500), dtype=np.uint8)
    mask[350:450, 100:200] = 1  # small ROI around first foot point
    loc.set_roi_mask(mask)
    filtered = loc.extract(det)
    print(f"After ROI filter: {len(filtered)} / {det.count} points kept")

    # Static helper
    fx, fy, bh = FootLocalizer.compute_foot_point(np.array([100, 200, 150, 400]))
    print(f"Static helper: foot=({fx:.1f}, {fy:.1f}), h={bh:.1f}")
