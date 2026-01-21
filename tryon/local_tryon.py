"""Local CPU-based Virtual Try-On using classical computer vision."""

import logging
import cv2
import numpy as np
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)


class LocalTryOn:
    """
    Local virtual try-on using classical computer vision techniques.
    
    Works on CPU without requiring heavy AI models.
    Uses:
    - Pose estimation via color/shape heuristics
    - Thin-plate spline or affine warping
    - Alpha blending with edge feathering
    """
    
    def __init__(self):
        """Initialize local try-on."""
        logger.info("LocalTryOn initialized (CPU mode)")
    
    def run(
        self,
        person_image: np.ndarray,
        garment_image: np.ndarray,
        garment_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Run local virtual try-on.
        
        Args:
            person_image: RGB person image (H, W, 3)
            garment_image: RGB garment image (H, W, 3)
            garment_mask: Optional grayscale mask for garment
        
        Returns:
            RGB result image (H, W, 3)
        """
        logger.info("Running local try-on...")
        
        # Step 1: Preprocess images
        person_img = self._preprocess(person_image)
        garment_img = self._preprocess(garment_image)
        
        ph, pw = person_img.shape[:2]
        gh, gw = garment_img.shape[:2]
        
        # Step 2: Generate garment mask if not provided
        if garment_mask is None:
            garment_mask = self._generate_garment_mask(garment_img)
        else:
            garment_mask = cv2.resize(garment_mask, (gw, gh))
        
        # Step 3: Detect person torso region
        torso_bbox = self._detect_torso(person_img)
        logger.info(f"Torso bbox: {torso_bbox}")
        
        # Step 4: Detect garment bounds
        garment_bbox = self._detect_garment_bounds(garment_mask)
        logger.info(f"Garment bbox: {garment_bbox}")
        
        # Step 5: Calculate transformation
        transform_matrix = self._calculate_transform(
            garment_bbox, torso_bbox, (gw, gh), (pw, ph)
        )
        
        # Step 6: Warp garment
        warped_garment = cv2.warpPerspective(
            garment_img, transform_matrix, (pw, ph),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        warped_mask = cv2.warpPerspective(
            garment_mask, transform_matrix, (pw, ph),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # Step 7: Composite
        result = self._composite(person_img, warped_garment, warped_mask)
        
        logger.info("Local try-on completed")
        return result
    
    def _preprocess(self, image: np.ndarray, max_size: int = 800) -> np.ndarray:
        """Preprocess image - resize if too large."""
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return image
    
    def _generate_garment_mask(self, garment_img: np.ndarray) -> np.ndarray:
        """Generate mask for garment using background detection."""
        h, w = garment_img.shape[:2]
        
        # Convert to different color spaces
        gray = cv2.cvtColor(garment_img, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(garment_img, cv2.COLOR_RGB2HSV)
        
        # Method 1: Detect non-white background
        white_mask = cv2.inRange(gray, 240, 255)
        
        # Method 2: Low saturation = background
        low_sat_mask = cv2.inRange(hsv[:, :, 1], 0, 30)
        
        # Combine: background is white AND low saturation
        background = cv2.bitwise_and(white_mask, low_sat_mask)
        
        # Invert to get foreground
        mask = cv2.bitwise_not(background)
        
        # Clean up with morphology
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Fill holes - find largest contour and fill
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            filled = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(filled, [largest], -1, 255, -1)
            mask = filled
        
        return mask
    
    def _detect_torso(self, person_img: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Detect torso region in person image.
        
        Returns (x, y, w, h) bounding box.
        """
        h, w = person_img.shape[:2]
        
        # Simple heuristic: torso is in center, upper portion
        # Assume person is centered and facing camera
        
        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(person_img, cv2.COLOR_RGB2HSV)
        
        # Detect skin tones
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
        
        lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
        upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
        skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        
        skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
        
        # Find face region (top skin area)
        kernel = np.ones((10, 10), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find top skin blob (likely face)
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        face_y = h // 6  # Default: assume face at top 1/6
        
        if contours:
            # Find topmost large contour (likely face)
            for cnt in sorted(contours, key=lambda c: cv2.boundingRect(c)[1]):
                x, y, cw, ch = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                if area > (w * h * 0.01):  # At least 1% of image
                    face_y = y + ch
                    break
        
        # Torso starts below face, goes to about 60% of image height
        torso_top = int(face_y + h * 0.05)
        torso_bottom = int(h * 0.65)
        torso_left = int(w * 0.15)
        torso_right = int(w * 0.85)
        
        torso_x = torso_left
        torso_y = torso_top
        torso_w = torso_right - torso_left
        torso_h = torso_bottom - torso_top
        
        return (torso_x, torso_y, torso_w, torso_h)
    
    def _detect_garment_bounds(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box of garment from mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            return cv2.boundingRect(largest)
        
        # Fallback: full image
        h, w = mask.shape
        return (0, 0, w, h)
    
    def _calculate_transform(
        self,
        src_bbox: Tuple[int, int, int, int],
        dst_bbox: Tuple[int, int, int, int],
        src_size: Tuple[int, int],
        dst_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Calculate perspective transform matrix.
        
        Maps garment bounding box to torso bounding box.
        """
        sx, sy, sw, sh = src_bbox
        dx, dy, dw, dh = dst_bbox
        
        # Source points (garment corners)
        src_pts = np.array([
            [sx, sy],
            [sx + sw, sy],
            [sx + sw, sy + sh],
            [sx, sy + sh],
        ], dtype=np.float32)
        
        # Destination points (torso corners with some padding)
        padding_x = int(dw * 0.05)
        padding_y = int(dh * 0.02)
        
        dst_pts = np.array([
            [dx - padding_x, dy - padding_y],
            [dx + dw + padding_x, dy - padding_y],
            [dx + dw + padding_x, dy + dh + padding_y],
            [dx - padding_x, dy + dh + padding_y],
        ], dtype=np.float32)
        
        # Calculate perspective transform
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        return matrix
    
    def _composite(
        self,
        person_img: np.ndarray,
        warped_garment: np.ndarray,
        warped_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Composite warped garment onto person.
        
        Uses alpha blending with edge feathering.
        """
        h, w = person_img.shape[:2]
        
        # Feather the mask edges for smoother blending
        feathered_mask = self._feather_mask(warped_mask, radius=15)
        
        # Normalize to [0, 1]
        alpha = feathered_mask.astype(np.float32) / 255.0
        alpha = alpha[:, :, np.newaxis]
        
        # Alpha blend
        person_float = person_img.astype(np.float32)
        garment_float = warped_garment.astype(np.float32)
        
        result = garment_float * alpha + person_float * (1 - alpha)
        
        return result.astype(np.uint8)
    
    def _feather_mask(self, mask: np.ndarray, radius: int = 10) -> np.ndarray:
        """Apply feathering to mask edges."""
        # Gaussian blur for soft edges
        blurred = cv2.GaussianBlur(mask, (radius * 2 + 1, radius * 2 + 1), 0)
        
        # Keep center solid, only feather edges
        # Use distance transform to find edge region
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dist = np.clip(dist / radius, 0, 1)
        
        # Combine: solid center + feathered edges
        result = np.where(dist >= 1, mask, blurred)
        
        return result.astype(np.uint8)

