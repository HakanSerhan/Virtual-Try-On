"""Human parsing module using SCHP or fallback methods."""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, List
from enum import IntEnum

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BodyPart(IntEnum):
    """Body part labels for human parsing."""
    BACKGROUND = 0
    HAT = 1
    HAIR = 2
    FACE = 3
    UPPER_CLOTHES = 4
    DRESS = 5
    COAT = 6
    SOCKS = 7
    PANTS = 8
    NECK = 9
    SKIN = 10
    LEFT_ARM = 11
    RIGHT_ARM = 12
    LEFT_LEG = 13
    RIGHT_LEG = 14
    LEFT_SHOE = 15
    RIGHT_SHOE = 16
    TORSO_SKIN = 17
    # Extended labels (SCHP)
    SUNGLASSES = 18
    SCARF = 19


@dataclass
class ParseResult:
    """Result of human parsing."""
    segmentation: np.ndarray  # (H, W) with class labels
    class_masks: Dict[str, np.ndarray]  # Named masks
    image_shape: tuple
    
    def get_mask(self, *part_names: str) -> np.ndarray:
        """Get combined mask for specified body parts."""
        h, w = self.image_shape[:2]
        combined = np.zeros((h, w), dtype=np.uint8)
        
        for name in part_names:
            if name in self.class_masks:
                combined = np.maximum(combined, self.class_masks[name])
        
        return combined
    
    def get_foreground_mask(self) -> np.ndarray:
        """
        Get mask of elements that should appear in front of garment.
        
        Includes: arms, hands, hair, face (partial)
        """
        return self.get_mask("left_arm", "right_arm", "hair")
    
    def get_torso_mask(self) -> np.ndarray:
        """Get mask of torso area (where garment goes)."""
        return self.get_mask("upper_clothes", "dress", "coat", "torso_skin")


class HumanParser:
    """
    Human parsing for body part segmentation.
    
    Attempts to use SCHP model if available, falls back to 
    simpler methods (GrabCut + pose-based estimation).
    """
    
    # Class label mapping
    LABEL_MAP = {
        "background": 0,
        "hat": 1,
        "hair": 2,
        "face": 3,
        "upper_clothes": 4,
        "dress": 5,
        "coat": 6,
        "socks": 7,
        "pants": 8,
        "neck": 9,
        "skin": 10,
        "left_arm": 11,
        "right_arm": 12,
        "left_leg": 13,
        "right_leg": 14,
        "left_shoe": 15,
        "right_shoe": 16,
        "torso_skin": 17,
    }
    
    def __init__(self, model_path: Optional[str] = None, use_schp: bool = True):
        """
        Initialize human parser.
        
        Args:
            model_path: Path to SCHP model weights (optional)
            use_schp: Whether to attempt loading SCHP model
        """
        self._schp_model = None
        self._use_fallback = True
        
        if use_schp:
            self._try_load_schp(model_path)
        
        logger.info(f"HumanParser initialized (fallback={self._use_fallback})")
    
    def _try_load_schp(self, model_path: Optional[str]) -> None:
        """Attempt to load SCHP model."""
        try:
            # Try importing SCHP dependencies
            import torch
            
            # For PoC, we'll use a simplified approach
            # Full SCHP integration would require the model weights
            logger.info("SCHP model loading skipped (using fallback for PoC)")
            self._use_fallback = True
            
        except ImportError as e:
            logger.warning(f"SCHP dependencies not available: {e}")
            self._use_fallback = True
    
    def predict(self, image: np.ndarray) -> ParseResult:
        """
        Predict human parsing segmentation.
        
        Args:
            image: RGB image (H, W, 3)
        
        Returns:
            ParseResult with segmentation masks
        """
        if self._use_fallback:
            return self._predict_fallback(image)
        else:
            return self._predict_schp(image)
    
    def _predict_fallback(self, image: np.ndarray) -> ParseResult:
        """
        Fallback parsing using color-based and pose-based heuristics.
        
        This is a simplified approach for PoC when SCHP is not available.
        """
        h, w = image.shape[:2]
        
        # Initialize segmentation
        segmentation = np.zeros((h, w), dtype=np.uint8)
        class_masks = {}
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Detect skin using HSV (simplified)
        skin_mask = self._detect_skin(hsv)
        
        # Use pose estimation to identify arm regions
        try:
            from tryon.pose import PoseEstimator
            pose_estimator = PoseEstimator()
            pose_result = pose_estimator.predict(image)
            
            # Create arm masks based on pose
            left_arm_mask, right_arm_mask = self._create_arm_masks(
                image, pose_result, skin_mask
            )
            
            # Create hair mask (top of image, non-skin, darker regions)
            hair_mask = self._detect_hair(image, pose_result)
            
        except Exception as e:
            logger.warning(f"Pose-based parsing failed: {e}")
            # Simple fallback - divide image into regions
            left_arm_mask = np.zeros((h, w), dtype=np.uint8)
            right_arm_mask = np.zeros((h, w), dtype=np.uint8)
            hair_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Arms on sides
            left_arm_mask[:, :w//4] = skin_mask[:, :w//4]
            right_arm_mask[:, 3*w//4:] = skin_mask[:, 3*w//4:]
            
            # Hair at top
            hair_mask[:h//4, w//4:3*w//4] = 255
        
        # Build class masks
        class_masks["left_arm"] = left_arm_mask
        class_masks["right_arm"] = right_arm_mask
        class_masks["hair"] = hair_mask
        class_masks["skin"] = skin_mask
        
        # Build segmentation from masks
        segmentation[left_arm_mask > 127] = BodyPart.LEFT_ARM
        segmentation[right_arm_mask > 127] = BodyPart.RIGHT_ARM
        segmentation[hair_mask > 127] = BodyPart.HAIR
        
        return ParseResult(
            segmentation=segmentation,
            class_masks=class_masks,
            image_shape=image.shape,
        )
    
    def _detect_skin(self, hsv: np.ndarray) -> np.ndarray:
        """Detect skin regions using HSV color space."""
        # Skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Second range for different skin tones
        lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
        upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
        
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        
        skin_mask = cv2.bitwise_or(mask1, mask2)
        
        # Clean up
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        return skin_mask
    
    def _create_arm_masks(
        self,
        image: np.ndarray,
        pose_result,
        skin_mask: np.ndarray,
    ) -> tuple:
        """Create arm masks based on pose keypoints."""
        h, w = image.shape[:2]
        
        left_arm_mask = np.zeros((h, w), dtype=np.uint8)
        right_arm_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Get arm keypoints
        left_shoulder = pose_result.get_pixel_coords("left_shoulder")
        left_elbow = pose_result.get_pixel_coords("left_elbow")
        left_wrist = pose_result.get_pixel_coords("left_wrist")
        
        right_shoulder = pose_result.get_pixel_coords("right_shoulder")
        right_elbow = pose_result.get_pixel_coords("right_elbow")
        right_wrist = pose_result.get_pixel_coords("right_wrist")
        
        # Draw arm regions as thick lines
        arm_thickness = max(30, w // 20)
        
        if left_shoulder and left_elbow:
            cv2.line(left_arm_mask, left_shoulder, left_elbow, 255, arm_thickness)
        if left_elbow and left_wrist:
            cv2.line(left_arm_mask, left_elbow, left_wrist, 255, arm_thickness)
        
        if right_shoulder and right_elbow:
            cv2.line(right_arm_mask, right_shoulder, right_elbow, 255, arm_thickness)
        if right_elbow and right_wrist:
            cv2.line(right_arm_mask, right_elbow, right_wrist, 255, arm_thickness)
        
        # Combine with skin detection for better accuracy
        left_arm_mask = cv2.bitwise_and(left_arm_mask, skin_mask)
        right_arm_mask = cv2.bitwise_and(right_arm_mask, skin_mask)
        
        # Dilate to cover more area
        kernel = np.ones((15, 15), np.uint8)
        left_arm_mask = cv2.dilate(left_arm_mask, kernel, iterations=2)
        right_arm_mask = cv2.dilate(right_arm_mask, kernel, iterations=2)
        
        return left_arm_mask, right_arm_mask
    
    def _detect_hair(self, image: np.ndarray, pose_result) -> np.ndarray:
        """Detect hair region based on pose and color."""
        h, w = image.shape[:2]
        hair_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Get head region from pose (above shoulders)
        left_shoulder = pose_result.get_pixel_coords("left_shoulder")
        right_shoulder = pose_result.get_pixel_coords("right_shoulder")
        
        if left_shoulder and right_shoulder:
            # Head is above shoulders
            shoulder_y = min(left_shoulder[1], right_shoulder[1])
            head_top = max(0, shoulder_y - int(h * 0.3))
            head_left = max(0, min(left_shoulder[0], right_shoulder[0]) - 50)
            head_right = min(w, max(left_shoulder[0], right_shoulder[0]) + 50)
            
            # Create head region mask
            head_region = np.zeros((h, w), dtype=np.uint8)
            head_region[head_top:shoulder_y, head_left:head_right] = 255
            
            # Detect dark regions (hair is usually darker)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, dark_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
            
            # Hair is dark regions in head area, but not skin
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            skin_mask = self._detect_skin(hsv)
            non_skin = cv2.bitwise_not(skin_mask)
            
            hair_mask = cv2.bitwise_and(dark_mask, head_region)
            hair_mask = cv2.bitwise_and(hair_mask, non_skin)
            
            # Clean up
            kernel = np.ones((7, 7), np.uint8)
            hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
            hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel)
        
        return hair_mask
    
    def _predict_schp(self, image: np.ndarray) -> ParseResult:
        """
        Predict using SCHP model.
        
        This is a placeholder for full SCHP integration.
        """
        # TODO: Implement full SCHP integration when model is available
        raise NotImplementedError("SCHP model not yet integrated")

