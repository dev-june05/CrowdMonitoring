"""
Crowd clustering and physical footprint calculation.
Uses DBSCAN to group people and Alpha Shapes/Convex Hulls to measure their footprint.
"""

from __future__ import annotations

import logging
import numpy as np
import cv2
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

logger = logging.getLogger(__name__)

try:
    from sklearn.cluster import DBSCAN
    from shapely.geometry import Polygon, MultiPolygon
    import alphashape
    _HAS_CLUSTERING = True
except ImportError:
    _HAS_CLUSTERING = False

@dataclass
class CrowdFootprint:
    """Result of crowd clustering analysis."""
    total_area_m2: float = 0.0
    physical_density: float = 0.0      # people/m2
    cluster_count: int = 0
    cluster_boundaries_px: list[np.ndarray] = field(default_factory=list)
    isolated_count: int = 0            # people classified as noise/outliers

def is_available() -> bool:
    return _HAS_CLUSTERING

def compute_crowd_footprint(
    head_points: list,
    world_coords: Optional[np.ndarray] = None,
    pixel_coords: Optional[np.ndarray] = None,
    eps: float = 1.5,
    min_samples: int = 2,
    alpha: float = 0.7,
    personal_space_m2: float = 1.5,
    world_to_px_fn: Optional[Callable] = None,
    max_alpha_shape_time_ms: float = 5.0
) -> CrowdFootprint:
    """
    Cluster crowds and compute physical footprint area.
    Falls back to Convex Hull if Alpha Shape takes too long (for low latency).
    """
    if not _HAS_CLUSTERING:
        logger.warning("Clustering dependencies missing (scikit-learn, alphashape, shapely).")
        return CrowdFootprint()
        
    n_points = len(head_points)
    if n_points == 0:
        return CrowdFootprint()
        
    # Use world coords if available, else pixel coords with a rough scale
    if world_coords is not None and len(world_coords) == n_points:
        coords = world_coords
        is_calibrated = True
    elif pixel_coords is not None and len(pixel_coords) == n_points:
        coords = pixel_coords
        is_calibrated = False
    else:
        return CrowdFootprint()

    # If only 1 person, they are isolated
    if n_points < 2:
        return CrowdFootprint(
            total_area_m2=personal_space_m2,
            physical_density=1.0 / personal_space_m2,
            isolated_count=1
        )

    # 1. Run DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = clustering.labels_
    
    unique_labels = set(labels)
    total_area = 0.0
    cluster_boundaries = []
    isolated_count = 0
    cluster_count = 0

    for label in unique_labels:
        cluster_mask = (labels == label)
        cluster_pts = coords[cluster_mask]
        num_people = len(cluster_pts)
        
        # Noise or too few points -> treat as isolated
        if label == -1 or num_people < 3:
            isolated_count += num_people
            total_area += num_people * personal_space_m2
            continue
            
        cluster_count += 1
        cluster_area = 0.0
        boundary_w = None
        
        # 2. Try Alpha Shape
        t0 = time.time()
        try:
            shape = alphashape.alphashape(cluster_pts, alpha)
            elapsed_ms = (time.time() - t0) * 1000.0
            
            if isinstance(shape, MultiPolygon):
                # Take largest polygon
                shape = max(shape.geoms, key=lambda a: a.area)
                
            if isinstance(shape, Polygon) and shape.area > 0:
                cluster_area = shape.area
                # Extract boundary points
                x, y = shape.exterior.coords.xy
                boundary_w = np.column_stack((x, y))
        except Exception as e:
            pass # fallback to convex hull
            
        # 3. Fallback to Convex Hull if Alpha Shape failed or is invalid
        if boundary_w is None:
            hull = cv2.convexHull(cluster_pts.astype(np.float32))
            if hull is not None and len(hull) >= 3:
                cluster_area = cv2.contourArea(hull)
                boundary_w = hull.reshape(-1, 2)
                
        # 4. Enforce minimum area
        min_cluster_area = num_people * 0.35 # At least 0.35m2 per person
        cluster_area = max(cluster_area, min_cluster_area)
        total_area += cluster_area
        
        # 5. Convert boundary to pixels for drawing
        if boundary_w is not None and world_to_px_fn is not None and is_calibrated:
            try:
                boundary_px = world_to_px_fn(boundary_w)
                cluster_boundaries.append(boundary_px.astype(np.int32))
            except Exception:
                pass
        elif boundary_w is not None and not is_calibrated:
            cluster_boundaries.append(boundary_w.astype(np.int32))

    # Calculate overall density
    if not is_calibrated:
        # If uncalibrated, area is in pixels, so density is meaningless. Return 0.
        phys_density = 0.0
        total_area = 0.0
    else:
        phys_density = n_points / total_area if total_area > 0 else 0.0

    return CrowdFootprint(
        total_area_m2=total_area,
        physical_density=phys_density,
        cluster_count=cluster_count,
        cluster_boundaries_px=cluster_boundaries,
        isolated_count=isolated_count
    )
