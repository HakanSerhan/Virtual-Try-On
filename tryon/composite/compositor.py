"""Compositing module for layering garment onto person."""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Compositor:
    """
    Compositor for blending warped garment onto person image.
    
    Supports:
    - Simple alpha blending
    - Occlusion-aware compositing (arms/hair in front)
    - Edge feathering for smooth transitions
    """
    
    def __init__(self, feather_radius: int = 3):
        """
        Initialize compositor.
        
        Args:
            feather_radius: Radius for edge feathering (0 to disable)
        """
        self.feather_radius = feather_radius
        logger.info(f"Compositor initialized (feather_radius={feather_radius})")
    
    def compose(
        self,
        person_img: np.ndarray,
        warped_garment: np.ndarray,
        warped_mask: np.ndarray,
        foreground_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compose warped garment onto person image.
        
        Args:
            person_img: RGB person image (H, W, 3)
            warped_garment: RGB warped garment image (H, W, 3)
            warped_mask: Grayscale garment mask (H, W)
            foreground_mask: Optional mask of foreground elements (arms, hair)
                            that should appear in front of garment
        
        Returns:
            Composed RGB image (H, W, 3)
        """
        # Ensure same size
        if person_img.shape[:2] != warped_garment.shape[:2]:
            raise ValueError(
                f"Size mismatch: person={person_img.shape[:2]}, "
                f"garment={warped_garment.shape[:2]}"
            )
        
        h, w = person_img.shape[:2]
        
        # Ensure mask is same size
        if warped_mask.shape[:2] != (h, w):
            warped_mask = cv2.resize(warped_mask, (w, h))
        
        # Normalize mask to [0, 1]
        alpha = warped_mask.astype(np.float32) / 255.0
        
        # Apply feathering
        if self.feather_radius > 0:
            alpha = self._feather_mask(alpha)
        
        # Expand alpha for broadcasting
        alpha = alpha[:, :, np.newaxis]
        
        # Basic alpha blend: output = garment * alpha + person * (1 - alpha)
        output = warped_garment.astype(np.float32) * alpha + \
                 person_img.astype(np.float32) * (1 - alpha)
        
        # Apply foreground occlusion
        if foreground_mask is not None:
            output = self._apply_foreground(
                output, person_img, foreground_mask
            )
        
        return output.astype(np.uint8)
    
    def _feather_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply feathering (blur) to mask edges for smooth blending.
        
        Args:
            mask: Normalized mask [0, 1]
        
        Returns:
            Feathered mask
        """
        # Use Gaussian blur for feathering
        kernel_size = self.feather_radius * 2 + 1
        feathered = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        return feathered
    
    def _apply_foreground(
        self,
        composite: np.ndarray,
        person_img: np.ndarray,
        foreground_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Apply foreground mask to keep arms/hair in front of garment.
        
        Formula: output = person * foreground + composite * (1 - foreground)
        
        Args:
            composite: Current composite image
            person_img: Original person image
            foreground_mask: Mask of foreground elements (arms, hair)
        
        Returns:
            Updated composite with foreground elements preserved
        """
        h, w = composite.shape[:2]
        
        # Ensure mask is same size
        if foreground_mask.shape[:2] != (h, w):
            foreground_mask = cv2.resize(foreground_mask, (w, h))
        
        # Normalize and feather foreground mask
        fg_alpha = foreground_mask.astype(np.float32) / 255.0
        if self.feather_radius > 0:
            fg_alpha = self._feather_mask(fg_alpha)
        
        fg_alpha = fg_alpha[:, :, np.newaxis]
        
        # Apply: keep person in foreground areas
        output = person_img.astype(np.float32) * fg_alpha + \
                 composite * (1 - fg_alpha)
        
        return output
    
    def color_transfer(
        self,
        source: np.ndarray,
        target: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Simple color transfer for lighting consistency.
        
        Matches mean and std of source to target in LAB color space.
        
        Args:
            source: Source image (garment)
            target: Target image (person) for color reference
            mask: Optional mask to limit color matching region
        
        Returns:
            Color-adjusted source image
        """
        # Convert to LAB
        source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Calculate statistics
        if mask is not None:
            mask_bool = mask > 127
            src_mean = source_lab[mask_bool].mean(axis=0)
            src_std = source_lab[mask_bool].std(axis=0) + 1e-6
            tgt_mean = target_lab.mean(axis=(0, 1))
            tgt_std = target_lab.std(axis=(0, 1)) + 1e-6
        else:
            src_mean = source_lab.mean(axis=(0, 1))
            src_std = source_lab.std(axis=(0, 1)) + 1e-6
            tgt_mean = target_lab.mean(axis=(0, 1))
            tgt_std = target_lab.std(axis=(0, 1)) + 1e-6
        
        # Transfer color (only L channel for subtle effect)
        result_lab = source_lab.copy()
        result_lab[:, :, 0] = (result_lab[:, :, 0] - src_mean[0]) * \
                              (tgt_std[0] / src_std[0]) + tgt_mean[0]
        
        # Clip and convert back
        result_lab = np.clip(result_lab, 0, 255)
        result = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        return result

