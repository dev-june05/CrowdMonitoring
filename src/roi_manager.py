"""Interactive ROI (Region of Interest) polygon manager.

Provides :class:`ROIManager` which lets the user draw a polygon directly on
a video frame to define the analysis region.  The polygon is persisted as
JSON so it survives restarts, and a binary mask is generated for fast
per-pixel membership tests used by downstream modules.

JSON format on disk::

    {
      "roi_polygon": [[x1, y1], [x2, y2], ...],
      "source_resolution": [width, height],
      "created_at": "2026-06-15T14:00:00"
    }

Typical usage::

    roi = ROIManager('config/roi_config.json')
    if not roi.load():
        roi.define_interactive(first_frame)
    mask = roi.generate_mask(first_frame.shape[:2])
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ROIManager:
    """Manage an interactive polygonal Region of Interest.

    Parameters
    ----------
    config_path : str
        Path to the JSON file that stores the polygon.  Parent directories
        are created automatically on :meth:`save`.
    """

    # ----- interactive-drawing constants -----
    _WINDOW_NAME = "Define ROI"
    _VERTEX_COLOR = (0, 255, 0)      # green circles at each vertex
    _EDGE_COLOR = (255, 255, 0)      # cyan lines between vertices (BGR)
    _FILL_COLOR = (255, 255, 0)      # cyan fill
    _FILL_ALPHA = 0.3                # semi-transparent overlay
    _VERTEX_RADIUS = 5
    _LINE_THICKNESS = 2
    _INSTRUCTIONS = (
        "Left-click: add point | Right-click: undo | "
        "Enter: confirm | Esc: cancel"
    )

    def __init__(self, config_path: str = "config/roi_config.json") -> None:
        self.config_path = Path(config_path)
        self.polygon: Optional[np.ndarray] = None   # (N, 2) int32 vertices
        self.mask: Optional[np.ndarray] = None       # binary (H, W) uint8
        self._frame_shape: Optional[tuple[int, int]] = None  # (H, W)

    # ==================================================================
    # Persistence
    # ==================================================================

    def load(self) -> bool:
        """Load polygon from the JSON config file.

        Returns
        -------
        bool
            ``True`` if the polygon was loaded successfully, ``False`` if
            the file is missing, malformed, or has no polygon data.
        """
        if not self.config_path.is_file():
            logger.info("ROI config not found at '%s'.", self.config_path)
            return False

        try:
            with open(self.config_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read ROI config: %s", exc)
            return False

        raw_polygon = data.get("roi_polygon")
        if not raw_polygon or len(raw_polygon) < 3:
            logger.warning("ROI config has no valid polygon (need ≥ 3 pts).")
            return False

        self.polygon = np.array(raw_polygon, dtype=np.int32)

        # Store source resolution for later mismatch checks
        res = data.get("source_resolution")
        if res and len(res) == 2:
            # Stored as [width, height] → internal as (height, width)
            self._frame_shape = (int(res[1]), int(res[0]))

        logger.info(
            "Loaded ROI polygon with %d vertices from '%s'.",
            len(self.polygon),
            self.config_path,
        )
        return True

    def save(self) -> None:
        """Save the current polygon (and metadata) to the JSON config file.

        Creates parent directories if they don't exist.
        """
        if self.polygon is None:
            logger.warning("No polygon to save.")
            return

        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Build width/height from cached frame shape
        if self._frame_shape is not None:
            source_res = [int(self._frame_shape[1]), int(self._frame_shape[0])]
        else:
            source_res = None

        payload = {
            "roi_polygon": self.polygon.tolist(),
            "source_resolution": source_res,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

        with open(self.config_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        logger.info("ROI polygon saved to '%s'.", self.config_path)

    # ==================================================================
    # Interactive definition
    # ==================================================================

    def define_interactive(self, frame: np.ndarray) -> None:
        """Open an OpenCV window and let the user draw a polygon on *frame*.

        Controls
        --------
        * **Left-click** — add a vertex.
        * **Right-click** — undo the last vertex.
        * **Enter** (key 13) — confirm the polygon.
        * **Escape** (key 27) — cancel and use the full frame as ROI.

        After confirmation the mask is generated and the polygon is saved to
        disk automatically.

        Parameters
        ----------
        frame : np.ndarray
            BGR image used as the drawing canvas.
        """
        self._frame_shape = frame.shape[:2]
        vertices: list[list[int]] = []

        # ---- mouse callback ----
        def _mouse_cb(event: int, x: int, y: int, flags: int, param) -> None:
            if event == cv2.EVENT_LBUTTONDOWN:
                vertices.append([x, y])
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Undo last vertex
                if vertices:
                    vertices.pop()

        # ---- window setup ----
        cv2.namedWindow(self._WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._WINDOW_NAME, 1280, 720)
        cv2.setMouseCallback(self._WINDOW_NAME, _mouse_cb)

        confirmed = False

        while True:
            # Start from a fresh copy of the source frame each iteration
            canvas = frame.copy()

            # Draw instruction text at the top
            cv2.putText(
                canvas,
                self._INSTRUCTIONS,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if vertices:
                pts_arr = np.array(vertices, dtype=np.int32)

                # Semi-transparent fill when we have ≥ 3 vertices
                if len(vertices) >= 3:
                    overlay = canvas.copy()
                    cv2.fillPoly(overlay, [pts_arr], self._FILL_COLOR)
                    cv2.addWeighted(
                        overlay, self._FILL_ALPHA,
                        canvas, 1.0 - self._FILL_ALPHA,
                        0,
                        canvas,
                    )

                # Draw edges between consecutive vertices
                for i in range(len(vertices) - 1):
                    cv2.line(
                        canvas,
                        tuple(vertices[i]),
                        tuple(vertices[i + 1]),
                        self._EDGE_COLOR,
                        self._LINE_THICKNESS,
                        cv2.LINE_AA,
                    )
                # Close the polygon visually (last → first)
                if len(vertices) >= 3:
                    cv2.line(
                        canvas,
                        tuple(vertices[-1]),
                        tuple(vertices[0]),
                        self._EDGE_COLOR,
                        self._LINE_THICKNESS,
                        cv2.LINE_AA,
                    )

                # Draw vertex circles on top
                for v in vertices:
                    cv2.circle(
                        canvas,
                        tuple(v),
                        self._VERTEX_RADIUS,
                        self._VERTEX_COLOR,
                        -1,          # filled
                        cv2.LINE_AA,
                    )

            cv2.imshow(self._WINDOW_NAME, canvas)

            key = cv2.waitKey(30) & 0xFF
            if key == 13:  # Enter — confirm
                confirmed = True
                break
            elif key == 27:  # Escape — cancel
                break

        cv2.destroyWindow(self._WINDOW_NAME)

        if confirmed and len(vertices) >= 3:
            self.polygon = np.array(vertices, dtype=np.int32)
            logger.info("ROI polygon defined with %d vertices.", len(vertices))
        else:
            # No valid polygon → treat the full frame as the ROI
            h, w = self._frame_shape
            self.polygon = np.array(
                [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                dtype=np.int32,
            )
            logger.info(
                "ROI cancelled or too few points — using full frame as ROI."
            )

        # Generate the binary mask and persist to disk
        self.generate_mask(self._frame_shape)
        self.save()

    # ==================================================================
    # Mask generation & point-in-polygon tests
    # ==================================================================

    def generate_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Create a binary mask from the stored polygon.

        If the polygon was saved at a different resolution the vertices are
        rescaled to the new *shape* so the ROI stays geometrically valid.

        Parameters
        ----------
        shape : tuple[int, int]
            ``(height, width)`` of the target frame.

        Returns
        -------
        np.ndarray
            Binary mask of dtype ``uint8`` and shape ``(H, W)`` where
            values of ``255`` mark the ROI interior.
        """
        if self.polygon is None:
            # No polygon — the whole frame is the ROI
            self.mask = np.full(shape[:2], 255, dtype=np.uint8)
            self._frame_shape = shape[:2]
            return self.mask

        polygon = self.polygon.copy()

        # Handle resolution mismatch: scale polygon vertices proportionally
        if (
            self._frame_shape is not None
            and self._frame_shape != shape[:2]
        ):
            old_h, old_w = self._frame_shape
            new_h, new_w = shape[:2]
            sx = new_w / old_w
            sy = new_h / old_h
            polygon = (polygon.astype(np.float64) * [sx, sy]).astype(np.int32)
            logger.info(
                "Rescaled ROI polygon from %s to %s (scale %.2f×%.2f).",
                self._frame_shape,
                shape[:2],
                sx,
                sy,
            )

        self._frame_shape = shape[:2]
        self.mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.fillPoly(self.mask, [polygon], 255)
        return self.mask

    def is_inside(self, x: float, y: float) -> bool:
        """Check whether a point lies inside the polygon.

        Uses :func:`cv2.pointPolygonTest` for a robust geometric test that
        is independent of the rasterised mask.

        Parameters
        ----------
        x, y : float
            Coordinates in the image reference frame.

        Returns
        -------
        bool
            ``True`` if the point is inside or on the polygon boundary.
        """
        if self.polygon is None:
            # No ROI defined → everything is "inside"
            return True
        # pointPolygonTest returns +1 inside, 0 on edge, -1 outside
        dist = cv2.pointPolygonTest(
            self.polygon.astype(np.float32), (x, y), measureDist=False
        )
        return dist >= 0

    # ==================================================================
    # Accessors
    # ==================================================================

    def get_mask(self) -> Optional[np.ndarray]:
        """Return the cached binary mask, or ``None`` if not yet generated."""
        return self.mask

    def get_polygon(self) -> Optional[np.ndarray]:
        """Return the polygon vertices as an ``(N, 2)`` int32 array."""
        return self.polygon

    @property
    def is_defined(self) -> bool:
        """Whether a polygon (explicit or full-frame) has been set."""
        return self.polygon is not None
