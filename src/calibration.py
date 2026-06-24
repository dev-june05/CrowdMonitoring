"""
Ground-plane homography calibration.
Converts pixel coordinates to real-world meters and computes physical cell areas.
"""

from __future__ import annotations

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level state
_H: Optional[np.ndarray] = None
_H_inv: Optional[np.ndarray] = None
_cell_areas: Optional[list[list[float]]] = None

def load_homography(filepath: str) -> Optional[np.ndarray]:
    """Load homography matrix from .npy file."""
    global _H, _H_inv
    path = Path(filepath)
    if not path.exists():
        _H = None
        _H_inv = None
        return None
    try:
        _H = np.load(path)
        _H_inv = np.linalg.inv(_H)
        logger.info(f"Loaded homography from {filepath}")
        return _H
    except Exception as e:
        logger.error(f"Failed to load homography: {e}")
        _H = None
        _H_inv = None
        return None

def save_homography(H: np.ndarray, filepath: str) -> None:
    """Save homography matrix to .npy file."""
    global _H, _H_inv
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, H)
    _H = H
    _H_inv = np.linalg.inv(H)
    logger.info(f"Saved homography to {filepath}")

def is_calibrated() -> bool:
    return _H is not None

def get_homography() -> Optional[np.ndarray]:
    return _H

def px_to_world(pixel_pts: np.ndarray) -> np.ndarray:
    """Map Nx2 pixel coordinates to Nx2 world coordinates (meters)."""
    if _H is None:
        raise RuntimeError("Not calibrated")
    if len(pixel_pts) == 0:
        return np.empty((0, 2))
    
    # Reshape for perspectiveTransform: (N, 1, 2)
    pts = np.array(pixel_pts, dtype=np.float32).reshape(-1, 1, 2)
    world_pts = cv2.perspectiveTransform(pts, _H)
    return world_pts.reshape(-1, 2)

def world_to_px(world_pts: np.ndarray) -> np.ndarray:
    """Map Nx2 world coordinates (meters) to Nx2 pixel coordinates."""
    if _H_inv is None:
        raise RuntimeError("Not calibrated")
    if len(world_pts) == 0:
        return np.empty((0, 2))
    
    pts = np.array(world_pts, dtype=np.float32).reshape(-1, 1, 2)
    px_pts = cv2.perspectiveTransform(pts, _H_inv)
    return px_pts.reshape(-1, 2)

def precompute_cell_areas(frame_shape: tuple, grid_rows: int, grid_cols: int) -> None:
    """Compute real floor area in m2 for each grid cell."""
    global _cell_areas
    if _H is None:
        _cell_areas = None
        return
        
    h, w = frame_shape[:2]
    cell_w = w / grid_cols
    cell_h = h / grid_rows
    
    areas = []
    for r in range(grid_rows):
        row_areas = []
        for c in range(grid_cols):
            x1, y1 = c * cell_w, r * cell_h
            x2, y2 = (c + 1) * cell_w, (r + 1) * cell_h
            
            px_corners = np.array([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ], dtype=np.float32)
            
            world_corners = px_to_world(px_corners)
            area = cv2.contourArea(world_corners.astype(np.float32))
            row_areas.append(float(area))
        areas.append(row_areas)
        
    _cell_areas = areas

def cell_area_m2(row: int, col: int, fallback: float = 2.0) -> float:
    """Return real floor area of cell in m2."""
    if _cell_areas is None:
        return fallback
    try:
        return _cell_areas[row][col]
    except IndexError:
        return fallback

def _compute_homography(img_corners: np.ndarray, world_w: float, world_h: float) -> Optional[np.ndarray]:
    world_corners = np.array([
        [0, 0], [world_w, 0], [world_w, world_h], [0, world_h]
    ], dtype=np.float32)
    
    H, _ = cv2.findHomography(img_corners, world_corners)
    return H

def manual_calibrate(frame: np.ndarray, world_w: float = 10.0, world_h: float = 8.0) -> Optional[np.ndarray]:
    """Interactive 4-point manual calibration."""
    pts = []
    window_name = "Calibration: Click TL, TR, BR, BL corners. 'r' to reset, ESC to cancel"
    
    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_cb)
    
    while True:
        display = frame.copy()
        for i, p in enumerate(pts):
            cv2.circle(display, p, 5, (0, 255, 0), -1)
            cv2.putText(display, str(i+1), (p[0]+10, p[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if i > 0:
                cv2.line(display, pts[i-1], p, (0, 255, 0), 2)
        if len(pts) == 4:
            cv2.line(display, pts[3], pts[0], (0, 255, 0), 2)
            
        cv2.imshow(window_name, display)
        k = cv2.waitKey(1) & 0xFF
        
        if k == 27: # ESC
            cv2.destroyWindow(window_name)
            return None
        elif k == ord('r'):
            pts.clear()
        elif len(pts) == 4 and k in [13, 32]: # Enter or Space
            break
            
    cv2.destroyWindow(window_name)
    if len(pts) == 4:
        img_corners = np.array(pts, dtype=np.float32)
        return _compute_homography(img_corners, world_w, world_h)
    return None

def auto_calibrate(frame: np.ndarray, world_w: float = 10.0, world_h: float = 8.0) -> Optional[np.ndarray]:
    """Auto-detect checkerboard pattern for calibration."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    board_sizes = [(8, 6), (7, 5), (6, 4), (5, 3)]
    
    for (cols, rows) in board_sizes:
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if ret:
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            
            # Extract the 4 outer corners
            c1 = corners[0][0]
            c2 = corners[cols - 1][0]
            c3 = corners[-1][0]
            c4 = corners[-cols][0]
            
            img_corners = np.array([c1, c2, c3, c4], dtype=np.float32)
            return _compute_homography(img_corners, world_w, world_h)
            
    return None

def run_calibration(frame: np.ndarray, filepath: str, world_w: float = 10.0, world_h: float = 8.0) -> bool:
    """Run auto, then manual, and save."""
    H = auto_calibrate(frame, world_w, world_h)
    if H is None:
        logger.info("Auto calibration failed. Falling back to manual.")
        H = manual_calibrate(frame, world_w, world_h)
        
    if H is not None:
        save_homography(H, filepath)
        return True
    return False

def draw_world_grid(frame: np.ndarray, grid_rows: int, grid_cols: int, world_w: float, world_h: float) -> None:
    """Draw real-world grid lines projected onto the image."""
    if _H is None:
        return
        
    cell_w = world_w / grid_cols
    cell_h = world_h / grid_rows
    
    lines_world = []
    # Vertical lines
    for c in range(grid_cols + 1):
        lines_world.append([[c * cell_w, 0], [c * cell_w, world_h]])
    # Horizontal lines
    for r in range(grid_rows + 1):
        lines_world.append([[0, r * cell_h], [world_w, r * cell_h]])
        
    for w_line in lines_world:
        w_pts = np.array(w_line, dtype=np.float32)
        px_pts = world_to_px(w_pts).astype(np.int32)
        cv2.line(frame, tuple(px_pts[0]), tuple(px_pts[1]), (255, 255, 255), 1, cv2.LINE_AA)
