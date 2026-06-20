"""
Congestion Detector
===================
Grid-based congestion detection with perspective correction.

Divides the ROI bounding box into an NxM grid, counts people per cell,
and flags cells exceeding configurable thresholds with multi-level
severity (WARNING / CRITICAL).
"""

import numpy as np
from src import HeadPoint, CongestionAlert


class CongestionDetector:
    """Grid-based congestion detection with multi-level alerting.

    Overlays an NxM grid on the ROI polygon's bounding box, counts
    head points per cell, and generates alerts when density exceeds
    configurable thresholds.

    Args:
        roi_polygon: Polygon vertices as (N, 2) array.
        frame_shape: (height, width) of the video frame.
        rows: Number of grid rows.
        cols: Number of grid columns.
        warning_threshold: People/cell to trigger WARNING.
        critical_threshold: People/cell to trigger CRITICAL.
    """

    def __init__(self, roi_polygon: np.ndarray, frame_shape: tuple,
                 rows: int = 8, cols: int = 8,
                 warning_threshold: float = 3.0,
                 critical_threshold: float = 6.0):
        self.rows = rows
        self.cols = cols
        self.warning_thresh = warning_threshold
        self.critical_thresh = critical_threshold
        self.frame_h, self.frame_w = frame_shape[:2]

        # Compute grid bounds from ROI polygon bounding box
        if roi_polygon is not None and len(roi_polygon) > 0:
            self.roi_x_min = int(np.min(roi_polygon[:, 0]))
            self.roi_y_min = int(np.min(roi_polygon[:, 1]))
            self.roi_x_max = int(np.max(roi_polygon[:, 0]))
            self.roi_y_max = int(np.max(roi_polygon[:, 1]))
        else:
            self.roi_x_min, self.roi_y_min = 0, 0
            self.roi_x_max, self.roi_y_max = self.frame_w, self.frame_h

        # Compute cell dimensions
        roi_w = max(self.roi_x_max - self.roi_x_min, 1)
        roi_h = max(self.roi_y_max - self.roi_y_min, 1)
        self.cell_w = roi_w / cols
        self.cell_h = roi_h / rows

        # Current density matrix
        self.density_matrix = np.zeros((rows, cols), dtype=np.float32)

    def get_cell(self, x: float, y: float) -> tuple:
        """Map a head point to its grid cell (row, col)."""
        col = int((x - self.roi_x_min) / self.cell_w)
        row = int((y - self.roi_y_min) / self.cell_h)
        col = np.clip(col, 0, self.cols - 1)
        row = np.clip(row, 0, self.rows - 1)
        return int(row), int(col)

    def get_cell_bounds(self, row: int, col: int) -> tuple:
        """Get pixel bounds (x1, y1, x2, y2) for a given cell."""
        x1 = int(self.roi_x_min + col * self.cell_w)
        y1 = int(self.roi_y_min + row * self.cell_h)
        x2 = int(x1 + self.cell_w)
        y2 = int(y1 + self.cell_h)
        return (x1, y1, x2, y2)

    def analyze(self, head_points: list) -> tuple:
        """Analyze head points and return density matrix + congestion alerts.

        Args:
            head_points: List of HeadPoint objects (already ROI-filtered).

        Returns:
            density_matrix: (rows, cols) array of per-cell counts.
            alerts: List of CongestionAlert for cells exceeding thresholds.
        """
        self.density_matrix = np.zeros((self.rows, self.cols), dtype=np.float32)

        # Count head points per cell
        for fp in head_points:
            row, col = self.get_cell(fp.x, fp.y)
            self.density_matrix[row, col] += 1.0

        # Generate alerts for cells exceeding thresholds
        alerts = []
        for r in range(self.rows):
            for c in range(self.cols):
                val = self.density_matrix[r, c]
                if val >= self.critical_thresh:
                    alerts.append(CongestionAlert(
                        cell_row=r, cell_col=c,
                        cell_bounds=self.get_cell_bounds(r, c),
                        density_value=val,
                        severity="CRITICAL"
                    ))
                elif val >= self.warning_thresh:
                    alerts.append(CongestionAlert(
                        cell_row=r, cell_col=c,
                        cell_bounds=self.get_cell_bounds(r, c),
                        density_value=val,
                        severity="WARNING"
                    ))

        return self.density_matrix, alerts
