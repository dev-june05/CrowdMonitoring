"""
Sustained-risk alert manager.
Tracks consecutive frames where crowd risk is HIGH or CRITICAL.
Optionally records video clips and saves snapshots.
"""

from __future__ import annotations

import cv2
import datetime
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class AlertManager:
    """Track sustained crowd risk and fire alerts."""
    
    def __init__(
        self,
        output_dir: str = "outputs",
        sustain_frames: int = 10,
        record: bool = False,
        alert_levels: Optional[set] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.sustain_frames = sustain_frames
        self.record = record
        self.alert_levels = alert_levels or {"HIGH", "CRITICAL"}
        
        self._sustain_count = 0
        self._alerted = False
        self._writer: Optional[cv2.VideoWriter] = None
        self._writer_path: Optional[str] = None
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def update(self, risk: str, frame: np.ndarray, display_frame: Optional[np.ndarray] = None) -> bool:
        """Call once per frame. Returns True if currently in alert state."""
        in_danger = risk in self.alert_levels
        
        if in_danger:
            self._sustain_count += 1
        else:
            self._sustain_count = 0
            self._alerted = False
            self._stop_recording()
        
        # Fire on first crossing
        if self._sustain_count >= self.sustain_frames and not self._alerted:
            self._alerted = True
            snap_path = self._save_snapshot(frame, risk)
            logger.warning("ALERT: %s risk sustained for %d frames. Snapshot: %s",
                          risk, self.sustain_frames, snap_path)
            if self.record:
                h, w = frame.shape[:2]
                self._start_recording(w, h)
        
        # Feed frames into recording
        if self._writer and self._alerted:
            rec_frame = display_frame if display_frame is not None else frame
            self._writer.write(rec_frame)
        
        return self._alerted
    
    def _save_snapshot(self, frame, risk: str) -> str:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"alert_{ts}_{risk}.jpg"
        cv2.imwrite(str(path), frame)
        return str(path)
    
    def _start_recording(self, width: int, height: int) -> None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = str(self.output_dir / f"alert_clip_{ts}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self._writer = cv2.VideoWriter(path, fourcc, 20, (width, height))
        self._writer_path = path
        logger.info("Recording alert clip: %s", path)
    
    def _stop_recording(self) -> None:
        if self._writer:
            self._writer.release()
            logger.info("Alert clip saved: %s", self._writer_path)
            self._writer = None
            self._writer_path = None
    
    def release(self) -> None:
        """Clean up resources."""
        self._stop_recording()
    
    @property
    def is_active(self) -> bool:
        return self._alerted
