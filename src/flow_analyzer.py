"""
Flow Analyzer
=============
Track-based crowd flow analysis using ByteTrack trajectories.

Computes per-person velocity vectors from frame-to-frame position changes.
Derives aggregate metrics: mean velocity, speed variance, motion entropy,
and counter-flow ratio for anomaly detection.

Anomaly indicators:
  - Counter-flow > 30%: opposing crowd streams (collision risk)
  - Motion entropy > 0.8: chaotic/disorganized movement
  - Speed > 3× mean: individual running (panic indicator)
"""

import numpy as np
from collections import defaultdict

from src import FootPoint, FlowVector, FlowMetrics


class FlowAnalyzer:
    """Crowd flow analysis from ByteTrack tracking trajectories.

    Maintains per-track position history and computes velocity vectors,
    aggregate flow metrics, and anomaly indicators.

    Args:
        counter_flow_threshold: Fraction of opposing vectors to trigger alert.
        entropy_threshold: Normalized entropy threshold for disorder alert.
        speed_anomaly_factor: Multiplier of mean speed for individual alert.
        history_frames: Number of frames to average velocity over.
    """

    def __init__(self, counter_flow_threshold: float = 0.3,
                 entropy_threshold: float = 0.8,
                 speed_anomaly_factor: float = 3.0,
                 history_frames: int = 5):
        self.counter_flow_thresh = counter_flow_threshold
        self.entropy_thresh = entropy_threshold
        self.speed_factor = speed_anomaly_factor
        self.history_frames = history_frames

        # Track position history: {track_id: [(x, y), ...]}
        self.position_history = defaultdict(list)
        self.max_history = 30  # Max frames to keep per track

    def update(self, foot_points: list) -> tuple:
        """Update with new foot points and compute flow vectors.

        Args:
            foot_points: List of FootPoint objects with track_ids.

        Returns:
            flow_vectors: List of FlowVector (per-person velocity).
            metrics: FlowMetrics (aggregate statistics).
        """
        current_ids = set()
        flow_vectors = []

        for fp in foot_points:
            if fp.track_id < 0:
                continue  # Skip untracked detections

            current_ids.add(fp.track_id)
            history = self.position_history[fp.track_id]
            history.append((fp.x, fp.y))

            # Trim history to avoid unbounded memory growth
            if len(history) > self.max_history:
                self.position_history[fp.track_id] = history[-self.max_history:]
                history = self.position_history[fp.track_id]

            # Compute velocity from recent positions
            if len(history) >= 2:
                # Average over last N frames for stability
                n = min(self.history_frames, len(history))
                old_x, old_y = history[-n]
                vx = (fp.x - old_x) / n
                vy = (fp.y - old_y) / n
                flow_vectors.append(FlowVector(
                    track_id=fp.track_id,
                    x=fp.x, y=fp.y,
                    vx=vx, vy=vy
                ))

        # Clean up stale tracks (IDs not seen in this frame)
        stale = [tid for tid in self.position_history if tid not in current_ids]
        for tid in stale:
            del self.position_history[tid]

        # Compute aggregate metrics
        metrics = self._compute_metrics(flow_vectors)

        return flow_vectors, metrics

    def _compute_metrics(self, vectors: list) -> FlowMetrics:
        """Compute aggregate flow metrics from velocity vectors."""
        if len(vectors) < 2:
            return FlowMetrics(num_tracked=len(vectors))

        velocities = np.array([(v.vx, v.vy) for v in vectors])
        speeds = np.array([v.speed for v in vectors])

        mean_vel = np.mean(velocities, axis=0)
        mean_speed = float(np.mean(speeds))
        speed_var = float(np.var(speeds))

        # Motion entropy: Shannon entropy of direction histogram
        entropy = self._motion_entropy(velocities)

        # Counter-flow ratio
        cf_ratio = self._counter_flow_ratio(velocities)

        return FlowMetrics(
            mean_velocity=(float(mean_vel[0]), float(mean_vel[1])),
            mean_speed=mean_speed,
            speed_variance=speed_var,
            motion_entropy=float(entropy),
            counter_flow_ratio=float(cf_ratio),
            num_tracked=len(vectors)
        )

    def _motion_entropy(self, velocities: np.ndarray, n_bins: int = 8) -> float:
        """Normalized Shannon entropy of motion direction distribution.

        Returns:
            0.0 = all moving same direction (organized)
            1.0 = uniform distribution (chaotic)
        """
        # Filter out near-stationary vectors (< 1 pixel/frame)
        speeds = np.linalg.norm(velocities, axis=1)
        moving = velocities[speeds > 1.0]

        if len(moving) < 2:
            return 0.0

        angles = np.arctan2(moving[:, 1], moving[:, 0])
        hist, _ = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi))

        # Convert to probabilities
        probs = hist / hist.sum()
        probs = probs[probs > 0]  # Avoid log(0)

        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(n_bins)

        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _counter_flow_ratio(self, velocities: np.ndarray) -> float:
        """Fraction of velocity vectors opposing the dominant direction.

        > 0.3 suggests opposing crowd streams (collision risk).
        """
        speeds = np.linalg.norm(velocities, axis=1)
        moving = velocities[speeds > 1.0]

        if len(moving) < 2:
            return 0.0

        # Dominant direction = mean velocity vector
        mean_v = np.mean(moving, axis=0)
        dominant_angle = np.arctan2(mean_v[1], mean_v[0])

        counter = 0
        for v in moving:
            angle = np.arctan2(v[1], v[0])
            diff = abs(angle - dominant_angle)
            if diff > np.pi:
                diff = 2 * np.pi - diff
            # Consider "counter" if angle difference > ~126 degrees
            if diff > np.pi * 0.7:
                counter += 1

        return counter / len(moving)

    def detect_anomalies(self, vectors: list, metrics: FlowMetrics) -> list:
        """Detect crowd flow anomalies.

        Args:
            vectors: List of FlowVector.
            metrics: FlowMetrics from current frame.

        Returns:
            List of anomaly warning strings.
        """
        anomalies = []

        if metrics.counter_flow_ratio > self.counter_flow_thresh:
            anomalies.append(
                f"COUNTER-FLOW: {metrics.counter_flow_ratio:.0%} opposing"
            )

        if metrics.motion_entropy > self.entropy_thresh:
            anomalies.append(
                f"HIGH DISORDER: entropy {metrics.motion_entropy:.2f}"
            )

        # Individual speed anomalies
        if metrics.mean_speed > 0:
            for v in vectors:
                if v.speed > self.speed_factor * metrics.mean_speed:
                    anomalies.append(
                        f"FAST MOVEMENT: ID {v.track_id} "
                        f"({v.speed / metrics.mean_speed:.1f}x avg)"
                    )

        return anomalies

    def reset(self):
        """Clear all tracking history."""
        self.position_history.clear()
