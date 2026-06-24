"""
Per-frame CSV logging for crowd monitoring.
Lightweight append-only logging for post-analysis.
"""

from __future__ import annotations

import csv
import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CSVLogger:
    """Append-only CSV logger for per-frame crowd metrics."""
    
    HEADERS = [
        "timestamp", "frame_idx",
        "person_count", "density_per_m2",
        "hull_area_m2", "risk",
        "fps", "calibrated",
    ]
    
    def __init__(self, filepath: str = "outputs/crowd_log.csv") -> None:
        self.filepath = Path(filepath)
        self._initialized = False
    
    def _ensure_init(self) -> None:
        if self._initialized:
            return
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        if not self.filepath.exists():
            with open(self.filepath, "w", newline="") as f:
                csv.writer(f).writerow(self.HEADERS)
        self._initialized = True
    
    def log_frame(
        self,
        frame_idx: int,
        person_count: int,
        density_per_m2: float = 0.0,
        hull_area_m2: float = 0.0,
        risk: str = "",
        fps: float = 0.0,
        calibrated: bool = False,
    ) -> None:
        """Append one row to the CSV."""
        self._ensure_init()
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        try:
            with open(self.filepath, "a", newline="") as f:
                csv.writer(f).writerow([
                    ts, frame_idx, person_count,
                    f"{density_per_m2:.4f}",
                    f"{hull_area_m2:.2f}",
                    risk,
                    f"{fps:.1f}",
                    1 if calibrated else 0,
                ])
        except IOError as e:
            logger.debug(f"Failed to write to log: {e}")
