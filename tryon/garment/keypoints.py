"""Garment keypoint detection module."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GarmentKeypoints:
    """Detected keypoints on a garment."""
    left_shoulder: Optional[Tuple[int, int]] = None
    right_shoulder: Optional[Tuple[int, int]] = None
    neck: Optional[Tuple[int, int]] = None
    left_hem: Optional[Tuple[int, int]] = None
    right_hem: Optional[Tuple[int, int]] = None
    left_sleeve: Optional[Tuple[int, int]] = None
    right_sleeve: Optional[Tuple[int, int]] = None
    center: Optional[Tuple[int, int]] = None
    
    def to_dict(self) -> Dict[str, Tuple[int, int]]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key in ["left_shoulder", "right_shoulder", "neck", 
                    "left_hem", "right_hem", "left_sleeve", 
                    "right_sleeve", "center"]:
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array of points."""
        points = []
        for key in ["left_shoulder", "right_shoulder", "neck", 
                    "left_hem", "right_hem"]:
            value = getattr(self, key)
            if value is not None:
                points.append(value)
        return np.array(points, dtype=np.float32) if points else np.array([])


class GarmentKeypointDetector:
    """
    Detect keypoints on garment images.
    
    Uses contour analysis and heuristics to identify:
    - Shoulder points
    - Neckline
    - Hem line
    - Sleeve endpoints
    """
    
    def __init__(self):
        """Initialize keypoint detector."""
        logger.info("GarmentKeypointDetector initialized")
    
    def detect(
        self,
        garment_image: np.ndarray,
        garment_mask: Optional[np.ndarray] = None,
    ) -> GarmentKeypoints:
        """
        Detect keypoints on garment image.
        
        Args:
            garment_image: RGB garment image
            garment_mask: Optional grayscale mask
        
        Returns:
            GarmentKeypoints with detected points
        """
        h, w = garment_image.shape[:2]
        
        # Generate mask if not provided
        if garment_mask is None:
            gray = cv2.cvtColor(garment_image, cv2.COLOR_RGB2GRAY)
            _, garment_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(
            garment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            logger.warning("No contours found in garment mask")
            return self._fallback_keypoints(w, h)
        
        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, bw, bh = cv2.boundingRect(contour)
        
        # Detect keypoints
        keypoints = GarmentKeypoints()
        
        # Find shoulder points (top corners of garment)
        keypoints.left_shoulder, keypoints.right_shoulder = \
            self._find_shoulders(contour, x, y, bw, bh)
        
        # Find neck (top center)
        keypoints.neck = self._find_neck(contour, x, y, bw, bh)
        
        # Find hem (bottom corners)
        keypoints.left_hem, keypoints.right_hem = \
            self._find_hem(contour, x, y, bw, bh)
        
        # Find sleeves (side protrusions)
        keypoints.left_sleeve, keypoints.right_sleeve = \
            self._find_sleeves(contour, x, y, bw, bh)
        
        # Calculate center
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            keypoints.center = (cx, cy)
        else:
            keypoints.center = (x + bw // 2, y + bh // 2)
        
        return keypoints
    
    def _find_shoulders(
        self,
        contour: np.ndarray,
        x: int, y: int, bw: int, bh: int,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Find shoulder points on garment contour."""
        # Get points in top 30% of garment
        top_threshold = y + bh * 0.3
        top_points = contour[contour[:, 0, 1] < top_threshold]
        
        if len(top_points) < 2:
            # Fallback
            return ((x, y + int(bh * 0.1)), (x + bw, y + int(bh * 0.1)))
        
        top_points = top_points.reshape(-1, 2)
        
        # Left shoulder: leftmost point in top region
        left_idx = np.argmin(top_points[:, 0])
        left_shoulder = tuple(top_points[left_idx])
        
        # Right shoulder: rightmost point in top region
        right_idx = np.argmax(top_points[:, 0])
        right_shoulder = tuple(top_points[right_idx])
        
        return left_shoulder, right_shoulder
    
    def _find_neck(
        self,
        contour: np.ndarray,
        x: int, y: int, bw: int, bh: int,
    ) -> Tuple[int, int]:
        """Find neck/collar point."""
        # Look for top-center region
        center_x = x + bw // 2
        margin = bw * 0.3
        
        # Points in center column, top area
        center_points = contour[
            (contour[:, 0, 0] > center_x - margin) &
            (contour[:, 0, 0] < center_x + margin) &
            (contour[:, 0, 1] < y + bh * 0.3)
        ]
        
        if len(center_points) > 0:
            # Find topmost center point
            top_idx = np.argmin(center_points[:, 0, 1])
            return tuple(center_points[top_idx, 0])
        
        # Fallback: top center of bounding box
        return (center_x, y)
    
    def _find_hem(
        self,
        contour: np.ndarray,
        x: int, y: int, bw: int, bh: int,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Find hem points (bottom of garment)."""
        # Get points in bottom 30%
        bottom_threshold = y + bh * 0.7
        bottom_points = contour[contour[:, 0, 1] > bottom_threshold]
        
        if len(bottom_points) < 2:
            # Fallback
            return ((x + int(bw * 0.2), y + bh), (x + int(bw * 0.8), y + bh))
        
        bottom_points = bottom_points.reshape(-1, 2)
        
        # Left hem: leftmost bottom point
        left_idx = np.argmin(bottom_points[:, 0])
        left_hem = tuple(bottom_points[left_idx])
        
        # Right hem: rightmost bottom point
        right_idx = np.argmax(bottom_points[:, 0])
        right_hem = tuple(bottom_points[right_idx])
        
        return left_hem, right_hem
    
    def _find_sleeves(
        self,
        contour: np.ndarray,
        x: int, y: int, bw: int, bh: int,
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """Find sleeve endpoints (for long sleeves)."""
        # Sleeves are typically in middle-height, extreme left/right
        mid_top = y + bh * 0.2
        mid_bottom = y + bh * 0.6
        
        mid_points = contour[
            (contour[:, 0, 1] > mid_top) &
            (contour[:, 0, 1] < mid_bottom)
        ]
        
        if len(mid_points) < 2:
            return (None, None)
        
        mid_points = mid_points.reshape(-1, 2)
        
        # Find extreme left and right in mid section
        left_idx = np.argmin(mid_points[:, 0])
        right_idx = np.argmax(mid_points[:, 0])
        
        left_sleeve = tuple(mid_points[left_idx])
        right_sleeve = tuple(mid_points[right_idx])
        
        return left_sleeve, right_sleeve
    
    def _fallback_keypoints(self, w: int, h: int) -> GarmentKeypoints:
        """Generate fallback keypoints based on image dimensions."""
        return GarmentKeypoints(
            left_shoulder=(int(w * 0.15), int(h * 0.1)),
            right_shoulder=(int(w * 0.85), int(h * 0.1)),
            neck=(w // 2, int(h * 0.05)),
            left_hem=(int(w * 0.2), int(h * 0.9)),
            right_hem=(int(w * 0.8), int(h * 0.9)),
            center=(w // 2, h // 2),
        )
    
    def visualize(
        self,
        image: np.ndarray,
        keypoints: GarmentKeypoints,
    ) -> np.ndarray:
        """
        Visualize keypoints on garment image.
        
        Args:
            image: RGB garment image
            keypoints: Detected keypoints
        
        Returns:
            Image with keypoints drawn
        """
        vis = image.copy()
        
        # Colors for different keypoint types
        colors = {
            "left_shoulder": (255, 0, 0),    # Red
            "right_shoulder": (255, 0, 0),
            "neck": (0, 255, 0),              # Green
            "left_hem": (0, 0, 255),          # Blue
            "right_hem": (0, 0, 255),
            "left_sleeve": (255, 255, 0),     # Yellow
            "right_sleeve": (255, 255, 0),
            "center": (255, 0, 255),          # Magenta
        }
        
        for name, point in keypoints.to_dict().items():
            color = colors.get(name, (255, 255, 255))
            cv2.circle(vis, point, 8, color, -1)
            cv2.circle(vis, point, 10, (0, 0, 0), 2)
            cv2.putText(
                vis, name, (point[0] + 12, point[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
        
        # Draw connections
        kp_dict = keypoints.to_dict()
        if "left_shoulder" in kp_dict and "right_shoulder" in kp_dict:
            cv2.line(vis, kp_dict["left_shoulder"], kp_dict["right_shoulder"], 
                    (200, 200, 200), 2)
        if "left_hem" in kp_dict and "right_hem" in kp_dict:
            cv2.line(vis, kp_dict["left_hem"], kp_dict["right_hem"],
                    (200, 200, 200), 2)
        if "neck" in kp_dict and "center" in kp_dict:
            cv2.line(vis, kp_dict["neck"], kp_dict["center"],
                    (200, 200, 200), 2)
        
        return vis

