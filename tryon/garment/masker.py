"""Garment masking module using rembg or fallback methods."""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class GarmentMasker:
    """
    Generate masks for garment images.
    
    Uses rembg for background removal, with fallback to
    simple thresholding methods.
    """
    
    def __init__(self, use_rembg: bool = True):
        """
        Initialize garment masker.
        
        Args:
            use_rembg: Whether to use rembg for background removal
        """
        self._rembg_session = None
        self._use_rembg = use_rembg
        
        if use_rembg:
            self._try_load_rembg()
        
        logger.info(f"GarmentMasker initialized (rembg={self._rembg_session is not None})")
    
    def _try_load_rembg(self) -> None:
        """Attempt to load rembg."""
        try:
            from rembg import new_session
            self._rembg_session = new_session("u2net_cloth_seg")
            logger.info("Loaded rembg with u2net_cloth_seg model")
        except ImportError:
            logger.warning("rembg not available, using fallback")
            self._rembg_session = None
        except Exception as e:
            logger.warning(f"Failed to load rembg: {e}")
            # Try default model
            try:
                from rembg import new_session
                self._rembg_session = new_session()
                logger.info("Loaded rembg with default model")
            except Exception as e2:
                logger.warning(f"rembg fallback failed: {e2}")
                self._rembg_session = None
    
    def generate_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Generate mask for garment image.
        
        Args:
            image: RGB garment image (H, W, 3)
        
        Returns:
            Grayscale mask (H, W) where 255 = garment, 0 = background
        """
        if self._rembg_session is not None:
            return self._mask_rembg(image)
        else:
            return self._mask_fallback(image)
    
    def _mask_rembg(self, image: np.ndarray) -> np.ndarray:
        """Generate mask using rembg."""
        from rembg import remove
        from PIL import Image
        
        # Convert to PIL
        pil_image = Image.fromarray(image)
        
        # Remove background (returns RGBA)
        result = remove(pil_image, session=self._rembg_session)
        result_array = np.array(result)
        
        # Extract alpha channel as mask
        if result_array.shape[-1] == 4:
            mask = result_array[:, :, 3]
        else:
            # If no alpha, create mask from non-black pixels
            gray = cv2.cvtColor(result_array, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def _mask_fallback(self, image: np.ndarray) -> np.ndarray:
        """
        Generate mask using fallback methods.
        
        Assumes garment is on white/light background.
        """
        h, w = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Try multiple methods and combine
        masks = []
        
        # Method 1: Simple threshold for white background
        _, mask1 = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        masks.append(mask1)
        
        # Method 2: Adaptive threshold
        mask2 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        masks.append(mask2)
        
        # Method 3: Color-based (detect non-white)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        _, mask3 = cv2.threshold(saturation, 20, 255, cv2.THRESH_BINARY)
        masks.append(mask3)
        
        # Combine masks (intersection for robustness)
        combined = masks[0]
        for m in masks[1:]:
            combined = cv2.bitwise_or(combined, m)
        
        # Clean up
        kernel = np.ones((5, 5), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        # Fill holes
        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            # Keep only largest contour
            largest = max(contours, key=cv2.contourArea)
            filled = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(filled, [largest], -1, 255, -1)
            combined = filled
        
        return combined
    
    def remove_background(self, image: np.ndarray) -> np.ndarray:
        """
        Remove background from garment image.
        
        Args:
            image: RGB garment image
        
        Returns:
            RGBA image with transparent background
        """
        mask = self.generate_mask(image)
        
        # Create RGBA image
        rgba = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
        rgba[:, :, :3] = image
        rgba[:, :, 3] = mask
        
        return rgba

