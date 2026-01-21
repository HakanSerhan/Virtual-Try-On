"""Warping module for garment transformation."""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class WarpEngine:
    """
    Engine for warping garment images to fit person body.
    
    Supports two modes:
    - Affine: Fast, simple transformation (3-point)
    - TPS: Thin-Plate Spline for more accurate warping (n-point)
    """
    
    def __init__(self, mode: str = "affine"):
        """
        Initialize warp engine.
        
        Args:
            mode: "affine" or "tps"
        """
        self.mode = mode
        logger.info(f"WarpEngine initialized (mode={mode})")
    
    def warp(
        self,
        garment_img: np.ndarray,
        garment_mask: Optional[np.ndarray],
        src_points: np.ndarray,
        dst_points: np.ndarray,
        output_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warp garment image from source to destination points.
        
        Args:
            garment_img: RGB garment image (H, W, 3)
            garment_mask: Grayscale mask (H, W) or None
            src_points: Source control points (N, 2)
            dst_points: Destination control points (N, 2)
            output_size: Output (width, height)
        
        Returns:
            Tuple of (warped_image, warped_mask)
        """
        if self.mode == "affine":
            return self._warp_affine(
                garment_img, garment_mask, src_points, dst_points, output_size
            )
        elif self.mode == "tps":
            return self._warp_tps(
                garment_img, garment_mask, src_points, dst_points, output_size
            )
        else:
            raise ValueError(f"Unknown warp mode: {self.mode}")
    
    def _warp_affine(
        self,
        garment_img: np.ndarray,
        garment_mask: Optional[np.ndarray],
        src_points: np.ndarray,
        dst_points: np.ndarray,
        output_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform affine warping (uses first 3 points or estimates best affine).
        """
        width, height = output_size
        
        # Ensure we have at least 3 points for affine
        if len(src_points) < 3:
            raise ValueError("Affine warp requires at least 3 control points")
        
        # Use first 3 points or estimate affine from all points
        if len(src_points) == 3:
            src_pts = src_points.astype(np.float32)
            dst_pts = dst_points.astype(np.float32)
            M = cv2.getAffineTransform(src_pts, dst_pts)
        else:
            # Estimate best affine from multiple points
            src_pts = src_points.astype(np.float32)
            dst_pts = dst_points.astype(np.float32)
            M, _ = cv2.estimateAffine2D(src_pts, dst_pts)
        
        if M is None:
            raise ValueError("Failed to compute affine transformation")
        
        # Warp image
        warped_img = cv2.warpAffine(
            garment_img, M, (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        
        # Warp mask
        if garment_mask is not None:
            warped_mask = cv2.warpAffine(
                garment_mask, M, (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        else:
            # Create mask from non-black pixels
            warped_mask = np.any(warped_img > 10, axis=2).astype(np.uint8) * 255
        
        return warped_img, warped_mask
    
    def _warp_tps(
        self,
        garment_img: np.ndarray,
        garment_mask: Optional[np.ndarray],
        src_points: np.ndarray,
        dst_points: np.ndarray,
        output_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Thin-Plate Spline warping for more flexible deformation.
        """
        width, height = output_size
        
        # Create TPS transformer
        tps = cv2.createThinPlateSplineShapeTransformer()
        
        # TPS expects shape (1, N, 2)
        src_pts = src_points.reshape(1, -1, 2).astype(np.float32)
        dst_pts = dst_points.reshape(1, -1, 2).astype(np.float32)
        
        # Create match indices
        n_points = src_points.shape[0]
        matches = [cv2.DMatch(i, i, 0) for i in range(n_points)]
        
        # Estimate transformation
        tps.estimateTransformation(dst_pts, src_pts, matches)
        
        # Create coordinate grid for output
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        grid_points = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
        grid_points = grid_points.reshape(1, -1, 2).astype(np.float32)
        
        # Transform grid to source coordinates
        transformed = tps.applyTransformation(grid_points)[1]
        transformed = transformed.reshape(height, width, 2)
        
        # Remap image
        map_x = transformed[:, :, 0].astype(np.float32)
        map_y = transformed[:, :, 1].astype(np.float32)
        
        warped_img = cv2.remap(
            garment_img, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        
        # Warp mask
        if garment_mask is not None:
            warped_mask = cv2.remap(
                garment_mask, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        else:
            warped_mask = np.any(warped_img > 10, axis=2).astype(np.uint8) * 255
        
        return warped_img, warped_mask
    
    @staticmethod
    def estimate_garment_points(
        garment_img: np.ndarray,
        garment_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Tuple[int, int]]:
        """
        Estimate standard anchor points on garment image.
        
        Uses heuristics to find:
        - left_shoulder, right_shoulder
        - neck (top center)
        - left_hem, right_hem (bottom corners)
        
        Args:
            garment_img: RGB garment image
            garment_mask: Optional mask (if None, creates from non-black pixels)
        
        Returns:
            Dictionary of anchor point names to pixel coordinates
        """
        h, w = garment_img.shape[:2]
        
        # Create mask if not provided
        if garment_mask is None:
            gray = cv2.cvtColor(garment_img, cv2.COLOR_RGB2GRAY)
            _, garment_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(
            garment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            # Fallback to image corners
            logger.warning("No contours found, using image corners")
            return {
                "left_shoulder": (int(w * 0.1), int(h * 0.1)),
                "right_shoulder": (int(w * 0.9), int(h * 0.1)),
                "neck": (w // 2, int(h * 0.05)),
                "left_hem": (int(w * 0.2), int(h * 0.9)),
                "right_hem": (int(w * 0.8), int(h * 0.9)),
            }
        
        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, bw, bh = cv2.boundingRect(contour)
        
        # Find top points (shoulders)
        top_points = contour[contour[:, 0, 1] < y + bh * 0.3]
        if len(top_points) > 0:
            top_points = top_points.reshape(-1, 2)
            left_idx = np.argmin(top_points[:, 0])
            right_idx = np.argmax(top_points[:, 0])
            left_shoulder = tuple(top_points[left_idx])
            right_shoulder = tuple(top_points[right_idx])
        else:
            left_shoulder = (x, y)
            right_shoulder = (x + bw, y)
        
        # Find neck (top center)
        top_center_points = top_points[
            (top_points[:, 0] > x + bw * 0.3) & 
            (top_points[:, 0] < x + bw * 0.7)
        ] if len(top_points) > 0 else np.array([[x + bw // 2, y]])
        
        if len(top_center_points) > 0:
            neck_y = np.min(top_center_points[:, 1])
            neck_x = x + bw // 2
            neck = (neck_x, neck_y)
        else:
            neck = (x + bw // 2, y)
        
        # Find bottom points (hem)
        bottom_points = contour[contour[:, 0, 1] > y + bh * 0.7]
        if len(bottom_points) > 0:
            bottom_points = bottom_points.reshape(-1, 2)
            left_idx = np.argmin(bottom_points[:, 0])
            right_idx = np.argmax(bottom_points[:, 0])
            left_hem = tuple(bottom_points[left_idx])
            right_hem = tuple(bottom_points[right_idx])
        else:
            left_hem = (x, y + bh)
            right_hem = (x + bw, y + bh)
        
        return {
            "left_shoulder": left_shoulder,
            "right_shoulder": right_shoulder,
            "neck": neck,
            "left_hem": left_hem,
            "right_hem": right_hem,
        }

