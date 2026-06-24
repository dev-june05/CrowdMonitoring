"""Unified Rendering Engine for Crowd Monitoring.

Handles compositing of all visual overlays onto video frames:
ROI polygon, head points, heatmap, grid, congestion alerts,
velocity arrows, stats panel, anomaly banners, and keyboard legend.
"""

import cv2
import numpy as np
import time
from typing import Any
from src import HeadPoint, CongestionAlert, FlowVector, FlowMetrics, PipelineConfig


class Visualizer:
    """Unified rendering engine for all visual overlays.

    Handles: ROI polygon, head points, heatmap overlay, grid lines,
    congestion alerts, velocity arrows, stats panel, and keyboard controls.

    Each overlay can be toggled at runtime via keyboard shortcuts.
    The ``render`` method composites enabled layers in a fixed z-order
    so that later layers (e.g. stats) always appear on top.
    """

    def __init__(self, config: PipelineConfig):
        """Initialise the visualizer from pipeline configuration.

        Args:
            config: Pipeline configuration dataclass that supplies default
                    visibility flags and rendering parameters.
        """
        self.config = config
        self.start_time = time.time()

        # Toggle states – initialised from config, modified by keyboard at runtime
        self.show_heatmap: bool = config.show_heatmap
        self.show_grid: bool = config.show_grid
        self.show_head_points: bool = config.show_head_points
        self.show_roi: bool = config.show_roi
        self.show_velocity: bool = config.show_velocity
        self.show_stats: bool = config.show_stats

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        frame: np.ndarray,
        head_points: list[HeadPoint] | None = None,
        heatmap: np.ndarray | None = None,
        roi_polygon: np.ndarray | None = None,
        roi_mask: np.ndarray | None = None,
        density_matrix: np.ndarray | None = None,
        congestion_alerts: list[CongestionAlert] | None = None,
        flow_vectors: list[FlowVector] | None = None,
        flow_metrics: FlowMetrics | None = None,
        anomalies: list[str] | None = None,
        fps: float = 0.0,
        person_count: int = 0,
        roi_mode: str = "manual",
        footprint: Any | None = None,
        risk_level: str = "",
        is_alerting: bool = False,
    ) -> np.ndarray:
        """Render all enabled overlays onto *frame*.

        Layers are composited in a fixed z-order (bottom → top):
        heatmap → ROI → grid → head points → congestion →
        velocity → stats → anomalies → keyboard legend.

        Args:
            frame:              Raw BGR video frame.
            head_points:        Detected/tracked foot locations.
            heatmap:            2-D float density heatmap (same size as frame).
            roi_polygon:        Nx2 array of ROI polygon vertices.
            roi_mask:           Binary mask (0/1 float) for the ROI region.
            density_matrix:     Grid-cell density matrix (rows × cols).
            congestion_alerts:  List of cells that exceed density thresholds.
            flow_vectors:       Per-track velocity vectors.
            flow_metrics:       Aggregated flow statistics.
            anomalies:          Human-readable anomaly description strings.
            fps:                Current processing frame rate.
            person_count:       Number of detected persons in this frame.
            roi_mode:           Active ROI mode label (``"manual"`` / ``"auto"``).

        Returns:
            Annotated BGR frame (copy of input — original is not modified).
        """
        display = frame.copy()

        # 1. Heatmap overlay (blended with configurable opacity)
        if self.show_heatmap and heatmap is not None:
            display = self._draw_heatmap(display, heatmap, roi_mask)

        # 2. ROI polygon outline + semi-transparent fill
        if self.show_roi and roi_polygon is not None:
            self._draw_roi(display, roi_polygon)

        # 3. Grid lines (requires polygon bounding box + grid dimensions)
        if self.show_grid and density_matrix is not None and roi_polygon is not None:
            self._draw_grid(display, roi_polygon, density_matrix.shape)

        # 4. Crowd Cluster Boundaries (Alpha Shapes)
        if footprint and footprint.cluster_boundaries_px:
            for boundary in footprint.cluster_boundaries_px:
                pts = boundary.reshape((-1, 1, 2))
                # Draw semi-transparent fill
                overlay = display.copy()
                cv2.fillPoly(overlay, [pts], (255, 0, 150))
                cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)
                # Draw outline
                cv2.polylines(display, [pts], isClosed=True, color=(255, 0, 200), thickness=2)

        # 5. Foot-point markers
        if self.show_head_points and head_points:
            self._draw_head_points(display, head_points)

        # 6. Congestion alert rectangles (always shown when present)
        if congestion_alerts:
            self._draw_congestion(display, congestion_alerts)

        # 7. Velocity arrows
        if self.show_velocity and flow_vectors:
            self._draw_velocity(display, flow_vectors)

        # 8. Stats panel (top-left)
        if self.show_stats:
            self._draw_stats(display, fps, person_count, flow_metrics, roi_mode, footprint, risk_level)

        # 9. Alert Screen Flash
        if is_alerting:
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (display.shape[1], display.shape[0]), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)
            cv2.putText(display, "SUSTAINED RISK ALERT!", (display.shape[1]//2 - 200, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

        # 8. Anomaly banners (top-right, max 3)
        if anomalies:
            self._draw_anomalies(display, anomalies)

        # 9. Keyboard shortcut legend (bottom-left, always visible)
        self._draw_legend(display)

        return display

    def handle_key(self, key: int) -> str:
        """Process a keyboard event and apply the corresponding toggle.

        Args:
            key: Return value of ``cv2.waitKey()``.  ``-1`` means no key.

        Returns:
            Action name string (e.g. ``'toggle_heatmap'``, ``'quit'``),
            or empty string if the key is unrecognised / no key pressed.
        """
        if key == -1 or (key & 0xFF) == 255:
            return ""

        char = chr(key & 0xFF)
        
        # Uppercase R is a special command to redefine the ROI interactively
        if char == 'R':
            return "redefine_roi"

        char_lower = char.lower()

        actions = {
            "h": "toggle_heatmap",
            "g": "toggle_grid",
            "f": "toggle_feet",
            "v": "toggle_velocity",
            "r": "toggle_roi",
            "d": "toggle_stats",
            "s": "screenshot",
            "m": "switch_roi_mode",
            "q": "quit",
        }

        action = actions.get(char_lower, "")

        # Apply boolean toggles immediately so the next render reflects them
        if action == "toggle_heatmap":
            self.show_heatmap = not self.show_heatmap
        elif action == "toggle_grid":
            self.show_grid = not self.show_grid
        elif action == "toggle_feet":
            self.show_head_points = not self.show_head_points
        elif action == "toggle_velocity":
            self.show_velocity = not self.show_velocity
        elif action == "toggle_roi":
            self.show_roi = not self.show_roi
        elif action == "toggle_stats":
            self.show_stats = not self.show_stats

        return action

    # ------------------------------------------------------------------
    # Private drawing helpers
    # ------------------------------------------------------------------

    def _draw_heatmap(
        self, frame: np.ndarray, heatmap: np.ndarray, roi_mask: np.ndarray | None
    ) -> np.ndarray:
        """Blend a JET-coloured heatmap onto the frame within the ROI."""
        # Normalise heatmap intensity to 0-255 for colour-mapping
        if heatmap.max() > 0:
            norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        else:
            norm = np.zeros_like(heatmap)

        colored = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)

        # Restrict the coloured overlay to the ROI region only
        if roi_mask is not None:
            mask3 = np.stack([roi_mask] * 3, axis=-1).astype(np.float32)
            colored = (colored.astype(np.float32) * mask3).astype(np.uint8)

        # Alpha-blend: out = (1-α)·frame + α·colored
        alpha = self.config.heatmap_opacity
        frame = cv2.addWeighted(frame, 1.0 - alpha, colored, alpha, 0)
        return frame

    def _draw_roi(self, frame: np.ndarray, polygon: np.ndarray) -> None:
        """Draw the ROI polygon as a cyan outline with a faint fill."""
        pts = polygon.reshape((-1, 1, 2)).astype(np.int32)

        # Cyan outline (BGR: 255, 255, 0), 2-pixel thick
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 0), thickness=2)

        # Very faint fill (10 % opacity) so the ROI area is subtly highlighted
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (255, 255, 0))
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

    def _draw_grid(
        self, frame: np.ndarray, polygon: np.ndarray, grid_shape: tuple[int, int]
    ) -> None:
        """Overlay evenly-spaced grid lines inside the ROI bounding box."""
        rows, cols = grid_shape

        # Compute axis-aligned bounding box of the polygon
        x_min = int(np.min(polygon[:, 0]))
        y_min = int(np.min(polygon[:, 1]))
        x_max = int(np.max(polygon[:, 0]))
        y_max = int(np.max(polygon[:, 1]))

        cell_w = (x_max - x_min) / cols
        cell_h = (y_max - y_min) / rows

        grid_color = (200, 200, 200)  # light grey

        # Vertical lines
        for c in range(cols + 1):
            x = int(x_min + c * cell_w)
            cv2.line(frame, (x, y_min), (x, y_max), grid_color, 1)

        # Horizontal lines
        for r in range(rows + 1):
            y = int(y_min + r * cell_h)
            cv2.line(frame, (x_min, y), (x_max, y), grid_color, 1)

    def _draw_head_points(self, frame: np.ndarray, head_points: list[HeadPoint]) -> None:
        """Draw a colour-coded circle at each head point location."""
        for fp in head_points:
            center = (int(fp.x), int(fp.y))

            if fp.track_id >= 0:
                # Deterministic colour derived from track ID so each person
                # keeps a consistent colour across frames.
                hue = (fp.track_id * 37) % 180  # distribute hues evenly
                color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
                color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()
            else:
                color_bgr = (0, 255, 0)  # green default for untracked points

            cv2.circle(frame, center, 5, color_bgr, -1)       # filled dot
            cv2.circle(frame, center, 5, (255, 255, 255), 1)  # thin white border

    def _draw_congestion(
        self, frame: np.ndarray, alerts: list[CongestionAlert]
    ) -> None:
        """Draw pulsing rectangles around congested grid cells."""
        # Sinusoidal pulse oscillates between 0 and 1 at ~3 Hz
        pulse = abs(np.sin(time.time() * 3.0))

        for alert in alerts:
            x1, y1, x2, y2 = alert.cell_bounds

            if alert.severity == "CRITICAL":
                # Pulsing red (high urgency)
                color = (0, 0, int(255 * (0.5 + 0.5 * pulse)))
                thickness = 3
            else:
                # Pulsing yellow (moderate)
                color = (0, int(255 * (0.5 + 0.5 * pulse)), 255)
                thickness = 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Density label inside the top-left corner of the cell
            label = f"{alert.severity}: {alert.density_value:.0f}"
            cv2.putText(
                frame, label, (x1 + 2, y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )

    def _draw_velocity(self, frame: np.ndarray, vectors: list[FlowVector]) -> None:
        """Draw arrowed lines showing per-person velocity direction."""
        for v in vectors:
            if v.speed < 0.5:
                # Skip near-stationary objects to avoid visual clutter
                continue

            start = (int(v.x), int(v.y))
            # Scale velocity by 10× so small motions are visible
            end = (int(v.x + v.vx * 10), int(v.y + v.vy * 10))
            cv2.arrowedLine(frame, start, end, (0, 200, 255), 2, tipLength=0.3)

    def _draw_stats(
        self,
        frame: np.ndarray,
        fps: float,
        count: int,
        flow_metrics: FlowMetrics | None,
        roi_mode: str,
        footprint: Any | None,
        risk_level: str,
    ) -> None:
        """Render a semi-transparent statistics panel in the top-left corner."""
        # Semi-transparent black background
        panel_h = 160
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        y_pos = 35
        line_h = 25
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_pos), font, 0.6, (0, 255, 0), 2)
        y_pos += line_h

        cv2.putText(frame, f"People: {count}", (20, y_pos), font, 0.6, (0, 255, 255), 2)
        y_pos += line_h

        cv2.putText(
            frame, f"ROI Mode: {roi_mode.upper()}", (20, y_pos),
            font, 0.5, (255, 200, 100), 1,
        )
        y_pos += line_h

        if footprint and footprint.total_area_m2 > 0:
            cv2.putText(
                frame, f"Density: {footprint.physical_density:.2f} p/m2",
                (20, y_pos), font, 0.6, (255, 200, 255), 2
            )
            y_pos += line_h
            
        if risk_level:
            from src import context_risk
            color = context_risk.get_risk_color_bgr(risk_level)
            cv2.putText(
                frame, f"Risk: {risk_level}",
                (20, y_pos), font, 0.6, color, 2
            )
            y_pos += line_h

        if flow_metrics and flow_metrics.num_tracked > 0:
            cv2.putText(
                frame, f"Tracked: {flow_metrics.num_tracked}",
                (20, y_pos), font, 0.5, (200, 200, 200), 1,
            )
            y_pos += line_h
            cv2.putText(
                frame, f"Entropy: {flow_metrics.motion_entropy:.2f}",
                (20, y_pos), font, 0.5, (200, 200, 200), 1,
            )

    def _draw_anomalies(self, frame: np.ndarray, anomalies: list[str]) -> None:
        """Show anomaly alert banners in the top-right corner (max 3)."""
        w = frame.shape[1]
        y_pos = 35

        for anom in anomalies[:3]:
            text_size = cv2.getTextSize(anom, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            x = w - text_size[0] - 20

            # Red banner background
            cv2.rectangle(frame, (x - 5, y_pos - 18), (w - 10, y_pos + 5), (0, 0, 180), -1)
            cv2.putText(
                frame, anom, (x, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
            y_pos += 30

    def _draw_legend(self, frame: np.ndarray) -> None:
        """Render a compact keyboard-shortcut legend at the bottom-left."""
        h = frame.shape[0]
        legend_lines = [
            "H: heatmap | G: grid | F: feet | V: velocity",
            "R: ROI | D: stats | M: mode | S: screenshot | Q: quit",
        ]
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, line in enumerate(legend_lines):
            # Stack lines upward from the bottom edge
            y = h - 15 - (len(legend_lines) - 1 - i) * 18
            cv2.putText(frame, line, (10, y), font, 0.35, (180, 180, 180), 1)
