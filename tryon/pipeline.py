"""Main orchestrator for the Virtual Try-On pipeline."""

import logging
from typing import Optional, Dict, Any

import numpy as np
import cv2

from tryon.pose import PoseEstimator
from tryon.warp import WarpEngine
from tryon.composite import Compositor
from tryon.garment import GarmentMasker, GarmentKeypointDetector
from utils.timing import Timer

logger = logging.getLogger(__name__)


class TryOnPipeline:
    """
    Main orchestrator for the virtual try-on pipeline.
    
    Pipeline steps:
    1. Preprocess images
    2. Pose estimation (person)
    3. Garment keypoint estimation
    4. Warp garment to person
    5. Composite result
    6. Postprocess
    """
    
    def __init__(self):
        """Initialize pipeline components."""
        logger.info("Initializing TryOnPipeline...")
        
        # Initialize components
        self.pose_estimator = PoseEstimator()
        self.warp_engine_fast = WarpEngine(mode="affine")
        self.warp_engine_tps = WarpEngine(mode="tps")
        self.compositor = Compositor(feather_radius=5)
        
        # Garment processing components
        self.garment_keypoint_detector = GarmentKeypointDetector()
        self._garment_masker = None
        
        # Human parser will be initialized on demand (for "better" mode)
        self._human_parser = None
        
        logger.info("TryOnPipeline initialized")
    
    @property
    def human_parser(self):
        """Lazy-load human parser."""
        if self._human_parser is None:
            try:
                from tryon.parsing import HumanParser
                self._human_parser = HumanParser()
            except Exception as e:
                logger.warning(f"Failed to load HumanParser: {e}")
                self._human_parser = None
        return self._human_parser
    
    @property
    def garment_masker(self):
        """Lazy-load garment masker."""
        if self._garment_masker is None:
            try:
                self._garment_masker = GarmentMasker()
            except Exception as e:
                logger.warning(f"Failed to load GarmentMasker: {e}")
                self._garment_masker = None
        return self._garment_masker
    
    def run(
        self,
        person_image: np.ndarray,
        garment_image: np.ndarray,
        garment_mask: Optional[np.ndarray] = None,
        quality_mode: str = "fast",
        timer: Optional[Timer] = None,
    ) -> np.ndarray:
        """
        Run the try-on pipeline.
        
        Args:
            person_image: RGB person image (H, W, 3)
            garment_image: RGB garment image (H, W, 3)
            garment_mask: Optional grayscale mask (H, W)
            quality_mode: "fast" or "better"
            timer: Optional Timer for tracking step durations
        
        Returns:
            RGB result image (H, W, 3)
        """
        if timer is None:
            timer = Timer()
        
        # Step 1: Preprocess
        with timer.measure("preprocess"):
            person_img, garment_img, garment_msk = self._preprocess(
                person_image, garment_image, garment_mask
            )
        
        h, w = person_img.shape[:2]
        
        # Step 2: Pose estimation
        with timer.measure("pose"):
            pose_result = self.pose_estimator.predict(person_img)
            anchor_points = self.pose_estimator.get_anchor_points(pose_result)
        
        logger.info(f"Pose detected: confidence={pose_result.overall_confidence:.2f}")
        
        # Step 3: Garment keypoints
        with timer.measure("garment_keypoints"):
            garment_kp = self.garment_keypoint_detector.detect(
                garment_img, garment_msk
            )
            garment_points = garment_kp.to_dict()
        
        # Step 4: Build control point mapping
        with timer.measure("control_points"):
            src_points, dst_points = self._build_control_points(
                garment_points, anchor_points, garment_img.shape, person_img.shape
            )
        
        # Step 5: Warp garment
        with timer.measure("warp"):
            if quality_mode == "fast":
                warp_engine = self.warp_engine_fast
            else:
                warp_engine = self.warp_engine_tps
            
            warped_garment, warped_mask = warp_engine.warp(
                garment_img, garment_msk,
                src_points, dst_points,
                output_size=(w, h)
            )
        
        # Step 6: Get foreground mask (if available, Phase 2)
        foreground_mask = None
        if quality_mode == "better" and self.human_parser is not None:
            with timer.measure("parsing"):
                try:
                    parse_result = self.human_parser.predict(person_img)
                    foreground_mask = parse_result.get_foreground_mask()
                except Exception as e:
                    logger.warning(f"Human parsing failed: {e}")
        
        # Step 7: Composite
        with timer.measure("composite"):
            result = self.compositor.compose(
                person_img, warped_garment, warped_mask, foreground_mask
            )
        
        # Step 8: Postprocess
        with timer.measure("postprocess"):
            result = self._postprocess(result)
        
        return result
    
    def _preprocess(
        self,
        person_image: np.ndarray,
        garment_image: np.ndarray,
        garment_mask: Optional[np.ndarray],
    ) -> tuple:
        """
        Preprocess images for pipeline.
        
        - Resize if too large
        - Ensure RGB format
        - Generate garment mask if not provided
        """
        # Max size for processing
        max_size = 1024
        
        # Resize person image if needed
        h, w = person_image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            person_image = cv2.resize(
                person_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )
        
        # Resize garment to reasonable size
        gh, gw = garment_image.shape[:2]
        target_size = min(max_size, max(h, w))
        if max(gh, gw) != target_size:
            scale = target_size / max(gh, gw)
            new_gw = int(gw * scale)
            new_gh = int(gh * scale)
            garment_image = cv2.resize(
                garment_image, (new_gw, new_gh), interpolation=cv2.INTER_LINEAR
            )
            
            if garment_mask is not None:
                garment_mask = cv2.resize(
                    garment_mask, (new_gw, new_gh), interpolation=cv2.INTER_LINEAR
                )
        
        # Generate garment mask if not provided
        if garment_mask is None:
            # Try to use rembg or simple thresholding
            garment_mask = self._generate_garment_mask(garment_image)
        
        return person_image, garment_image, garment_mask
    
    def _generate_garment_mask(self, garment_image: np.ndarray) -> np.ndarray:
        """
        Generate mask for garment image.
        
        Uses simple background detection or rembg if available.
        """
        # Try rembg first
        if self.garment_masker is not None:
            try:
                return self.garment_masker.generate_mask(garment_image)
            except Exception as e:
                logger.warning(f"Garment masker failed: {e}")
        
        # Fallback: simple background detection
        # Assume white or near-white background
        gray = cv2.cvtColor(garment_image, cv2.COLOR_RGB2GRAY)
        
        # Threshold to detect non-background
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _build_control_points(
        self,
        garment_points: Dict[str, tuple],
        person_anchors: Dict[str, tuple],
        garment_shape: tuple,
        person_shape: tuple,
    ) -> tuple:
        """
        Build source and destination control points for warping.
        
        Maps garment keypoints to person anchor points.
        """
        src_points = []
        dst_points = []
        
        gh, gw = garment_shape[:2]
        ph, pw = person_shape[:2]
        
        # Map shoulders
        if "left_shoulder" in garment_points and "left_shoulder" in person_anchors:
            src_points.append(garment_points["left_shoulder"])
            dst_points.append(person_anchors["left_shoulder"])
        
        if "right_shoulder" in garment_points and "right_shoulder" in person_anchors:
            src_points.append(garment_points["right_shoulder"])
            dst_points.append(person_anchors["right_shoulder"])
        
        # Map neck/collar
        if "neck" in garment_points and "neck" in person_anchors:
            src_points.append(garment_points["neck"])
            dst_points.append(person_anchors["neck"])
        
        # Map hem to hip area
        if "left_hem" in garment_points and "left_hip" in person_anchors:
            src_points.append(garment_points["left_hem"])
            # Position hem slightly above hip
            hip = person_anchors["left_hip"]
            dst_points.append((hip[0], int(hip[1] * 0.95)))
        
        if "right_hem" in garment_points and "right_hip" in person_anchors:
            src_points.append(garment_points["right_hem"])
            hip = person_anchors["right_hip"]
            dst_points.append((hip[0], int(hip[1] * 0.95)))
        
        # Ensure we have at least 3 points for affine
        if len(src_points) < 3:
            logger.warning("Not enough control points, adding fallback points")
            # Add center points
            src_points.append((gw // 2, gh // 2))
            dst_points.append((pw // 2, ph // 2))
            
            if len(src_points) < 3:
                src_points.append((gw // 2, gh))
                dst_points.append((pw // 2, int(ph * 0.8)))
        
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        
        logger.info(f"Control points: {len(src_points)} pairs")
        
        return src_points, dst_points
    
    def _postprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Postprocess the result image.
        
        - Ensure uint8 format
        - Optional sharpening
        """
        # Ensure uint8
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image

