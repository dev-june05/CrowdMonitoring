"""Quick integration test — runs the pipeline on a static image to verify all modules work together."""
import sys
import numpy as np
import cv2

sys.path.insert(0, '.')

from src import PipelineConfig
from src.detector import PersonDetector
from src.foot_localizer import FootLocalizer
from src.roi_manager import ROIManager
from src.density_estimator import DensityEstimator
from src.temporal_filter import TemporalFilter
from src.congestion import CongestionDetector
from src.flow_analyzer import FlowAnalyzer
from src.visualizer import Visualizer

print("=" * 50)
print("  Integration Test — Static Image")
print("=" * 50)

# Load test image
frame = cv2.imread("data/crowd1.png")
if frame is None:
    print("ERROR: Cannot load data/crowd1.png")
    sys.exit(1)
h, w = frame.shape[:2]
print(f"Image: {w}x{h}")

# 1. Detection (no tracking for single image)
print("\n1. Person Detection...")
detector = PersonDetector("models/yolov8n.pt", device=0, conf=0.25)
result = detector.detect_no_track(frame)
print(f"   Detected: {result.count} people")

# 2. Foot localization (no ROI filter for this test)
print("\n2. Foot Localization...")
localizer = FootLocalizer()
foot_points = localizer.extract(result)
print(f"   Foot points: {len(foot_points)}")
if foot_points:
    fp = foot_points[0]
    print(f"   First: ({fp.x:.1f}, {fp.y:.1f}), bbox_h={fp.bbox_height:.1f}")

# 3. Density estimation (adaptive KDE)
print("\n3. Density Estimation (adaptive KDE)...")
density_est = DensityEstimator((h, w), gamma=0.3, fixed_sigma=20.0)
heatmap = density_est.compute_adaptive(foot_points)
print(f"   Heatmap shape: {heatmap.shape}, max={heatmap.max():.4f}")

# 4. Fixed KDE comparison
print("\n4. Density Estimation (fixed KDE)...")
heatmap_fixed = density_est.compute_fixed(foot_points)
print(f"   Heatmap shape: {heatmap_fixed.shape}, max={heatmap_fixed.max():.4f}")

# 5. Temporal filter
print("\n5. Temporal Filter (EMA)...")
tf = TemporalFilter(alpha=0.3)
smoothed = tf.update(heatmap)
print(f"   Smoothed max: {smoothed.max():.4f}")

# 6. Congestion detection (full-frame ROI)
print("\n6. Congestion Detection...")
roi_poly = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32)
congestion = CongestionDetector(roi_poly, (h, w), rows=8, cols=8,
                                 warning_threshold=2, critical_threshold=4)
density_matrix, alerts = congestion.analyze(foot_points)
print(f"   Grid: {density_matrix.shape}, max cell: {density_matrix.max():.0f}")
print(f"   Alerts: {len(alerts)} ({sum(1 for a in alerts if a.severity == 'WARNING')} WARNING, "
      f"{sum(1 for a in alerts if a.severity == 'CRITICAL')} CRITICAL)")

# 7. Flow analysis (single frame = no velocity)
print("\n7. Flow Analysis...")
flow = FlowAnalyzer()
vectors, metrics = flow.update(foot_points)
print(f"   Tracked: {metrics.num_tracked}")

# 8. Visualization
print("\n8. Visualization...")
config = PipelineConfig()
viz = Visualizer(config)
display = viz.render(
    frame=frame,
    foot_points=foot_points,
    heatmap=smoothed,
    roi_polygon=roi_poly,
    roi_mask=np.ones((h, w), dtype=np.uint8),
    density_matrix=density_matrix,
    congestion_alerts=alerts,
    flow_vectors=vectors,
    flow_metrics=metrics,
    anomalies=[],
    fps=30.0,
    person_count=result.count,
    roi_mode="manual"
)
print(f"   Display shape: {display.shape}")

# Save result
output_path = "outputs/integration_test.jpg"
cv2.imwrite(output_path, display)
print(f"\n   Saved: {output_path}")

print("\n" + "=" * 50)
print("  ALL TESTS PASSED ✓")
print("=" * 50)
