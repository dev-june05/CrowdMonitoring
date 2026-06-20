"""
Density Estimator
=================
Perspective-aware Gaussian Kernel Density Estimation for crowd heatmaps.

Provides two modes:
  - Fixed-sigma: Fast bin-and-blur at quarter resolution (~2-5ms)
  - Adaptive-sigma: Per-person sigma scaled by bbox height (~5-15ms)

The adaptive mode is the core research contribution: it uses bounding box
height as a perspective proxy, automatically compensating for foreshortening
without requiring camera calibration.

Mathematical basis:
  F(p) = Σᵢ (1/2πσᵢ²) · exp(-||p - pᵢ||² / 2σᵢ²)
  where σᵢ = γ · bbox_height_i
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

from src import HeadPoint


class DensityEstimator:
    """Perspective-aware crowd density estimation using Gaussian KDE.

    Two estimation modes:
      - compute_fixed(): Fast fixed-bandwidth KDE via bin-and-blur
      - compute_adaptive(): Perspective-aware adaptive-bandwidth KDE
    """

    def __init__(self, frame_shape: tuple, gamma: float = 0.3,
                 fixed_sigma: float = 20.0, scale_factor: int = 4):
        """
        Args:
            frame_shape: (height, width) of the video frame.
            gamma: Adaptive bandwidth factor: sigma = gamma * bbox_height.
            fixed_sigma: Sigma for fixed-bandwidth mode.
            scale_factor: Downscale factor for performance (4 = quarter res).
        """
        self.h, self.w = frame_shape[:2]
        self.gamma = gamma
        self.fixed_sigma = fixed_sigma
        self.scale = scale_factor
        self.small_h = max(1, self.h // scale_factor)
        self.small_w = max(1, self.w // scale_factor)

    def compute_fixed(self, head_points: list, roi_mask: np.ndarray = None) -> np.ndarray:
        """Fast fixed-sigma KDE using bin-and-blur at quarter resolution.

        Mathematically equivalent to KDE because convolving a sum of delta
        functions with a Gaussian equals summing individual Gaussians.
        Runs in ~2-5ms at quarter resolution.

        Args:
            head_points: List of HeadPoint objects.
            roi_mask: Optional binary mask (uint8, 0 or 1).

        Returns:
            Density map at full resolution (height, width), float32.
        """
        # 1. Create sparse point map at reduced resolution
        density = np.zeros((self.small_h, self.small_w), dtype=np.float32)

        for fp in head_points:
            sy = int(fp.y / self.scale)
            sx = int(fp.x / self.scale)
            sy = np.clip(sy, 0, self.small_h - 1)
            sx = np.clip(sx, 0, self.small_w - 1)
            density[sy, sx] += 1.0

        # 2. Single gaussian_filter call (equivalent to KDE with fixed bandwidth)
        sigma_small = max(self.fixed_sigma / self.scale, 0.5)
        if density.max() > 0:
            heatmap = gaussian_filter(density, sigma=sigma_small)
        else:
            heatmap = density

        # 3. Upscale to full resolution with smooth interpolation
        heatmap = cv2.resize(heatmap, (self.w, self.h),
                             interpolation=cv2.INTER_LINEAR)

        # 4. Mask to ROI if provided
        if roi_mask is not None:
            heatmap *= roi_mask.astype(np.float32)

        return heatmap

    def compute_adaptive(self, head_points: list, roi_mask: np.ndarray = None) -> np.ndarray:
        """Perspective-aware adaptive-sigma KDE.

        Each person's kernel sigma = gamma * bbox_height.
        Nearby people (large bbox) get large kernels → correct real-world coverage.
        Far people (small bbox) get small kernels → tight localization.

        Uses localized Gaussian patches (4*sigma radius) for efficiency.
        Runs in ~5-15ms at quarter resolution for N<100 detections.

        Args:
            head_points: List of HeadPoint objects with bbox_height set.
            roi_mask: Optional binary mask (uint8, 0 or 1).

        Returns:
            Density map at full resolution (height, width), float32.
        """
        density = np.zeros((self.small_h, self.small_w), dtype=np.float32)

        for fp in head_points:
            # Perspective-aware sigma: scale with bbox height
            sigma_full = max(self.gamma * fp.bbox_height, 2.0)
            sigma_small = sigma_full / self.scale

            # Localized patch: 4*sigma captures 99.99% of Gaussian mass
            radius = max(int(4 * sigma_small), 1)

            cy = int(fp.y / self.scale)
            cx = int(fp.x / self.scale)
            cy = np.clip(cy, 0, self.small_h - 1)
            cx = np.clip(cx, 0, self.small_w - 1)

            # Compute patch bounds (clipped to image edges)
            y_min = max(0, cy - radius)
            y_max = min(self.small_h, cy + radius + 1)
            x_min = max(0, cx - radius)
            x_max = min(self.small_w, cx + radius + 1)

            if y_max <= y_min or x_max <= x_min:
                continue

            # Create delta at person location within local patch
            patch = np.zeros((y_max - y_min, x_max - x_min), dtype=np.float32)
            local_y = cy - y_min
            local_x = cx - x_min

            if 0 <= local_y < patch.shape[0] and 0 <= local_x < patch.shape[1]:
                patch[local_y, local_x] = 1.0
                # Apply Gaussian filter to this small patch only
                if sigma_small >= 0.5:
                    patch = gaussian_filter(patch, sigma=sigma_small)
                density[y_min:y_max, x_min:x_max] += patch

        # Upscale to full resolution
        heatmap = cv2.resize(density, (self.w, self.h),
                             interpolation=cv2.INTER_LINEAR)

        # Mask to ROI
        if roi_mask is not None:
            heatmap *= roi_mask.astype(np.float32)

        return heatmap

    def compute(self, head_points: list, roi_mask: np.ndarray = None,
                adaptive: bool = True) -> np.ndarray:
        """Unified interface — dispatches to fixed or adaptive KDE.

        Args:
            head_points: List of HeadPoint objects.
            roi_mask: Optional binary mask.
            adaptive: If True, use perspective-aware adaptive-sigma KDE.

        Returns:
            Density heatmap at full resolution, float32.
        """
        if adaptive:
            return self.compute_adaptive(head_points, roi_mask)
        return self.compute_fixed(head_points, roi_mask)
