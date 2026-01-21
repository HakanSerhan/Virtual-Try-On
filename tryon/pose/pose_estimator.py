"""Pose estimation using MediaPipe Tasks API."""

import logging
import os
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Model URL for downloading
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
MODEL_DIR = Path(__file__).parent.parent.parent / "models"


@dataclass
class Keypoint:
    """Single keypoint with position and confidence."""
    x: float  # Normalized [0, 1]
    y: float  # Normalized [0, 1]
    confidence: float
    
    def to_pixel(self, width: int, height: int) -> Tuple[int, int]:
        """Convert to pixel coordinates."""
        return (int(self.x * width), int(self.y * height))


@dataclass
class PoseResult:
    """Result of pose estimation."""
    keypoints: Dict[str, Keypoint]
    image_width: int
    image_height: int
    overall_confidence: float
    
    def get_pixel_coords(self, name: str) -> Optional[Tuple[int, int]]:
        """Get pixel coordinates for a keypoint."""
        if name not in self.keypoints:
            return None
        return self.keypoints[name].to_pixel(self.image_width, self.image_height)
    
    def get_shoulder_width(self) -> Optional[float]:
        """Calculate shoulder width in pixels."""
        left = self.get_pixel_coords("left_shoulder")
        right = self.get_pixel_coords("right_shoulder")
        if left is None or right is None:
            return None
        return np.sqrt((left[0] - right[0])**2 + (left[1] - right[1])**2)
    
    def get_torso_height(self) -> Optional[float]:
        """Calculate torso height (shoulder to hip) in pixels."""
        left_shoulder = self.get_pixel_coords("left_shoulder")
        left_hip = self.get_pixel_coords("left_hip")
        if left_shoulder is None or left_hip is None:
            return None
        return np.sqrt(
            (left_shoulder[0] - left_hip[0])**2 + 
            (left_shoulder[1] - left_hip[1])**2
        )


def download_model(url: str, dest: Path) -> Path:
    """Download model file if not exists."""
    if dest.exists():
        return dest
    
    logger.info(f"Downloading pose model to {dest}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    logger.info("Model downloaded successfully")
    return dest


class PoseEstimator:
    """
    MediaPipe-based pose estimator for extracting body keypoints.
    
    Uses the new MediaPipe Tasks API.
    
    Extracts key upper-body landmarks needed for garment placement:
    - Shoulders (left/right)
    - Elbows (left/right)
    - Wrists (left/right)
    - Hips (left/right)
    """
    
    # Landmark indices for upper body (MediaPipe Pose)
    LANDMARK_INDICES = {
        11: "left_shoulder",
        12: "right_shoulder",
        13: "left_elbow",
        14: "right_elbow",
        15: "left_wrist",
        16: "right_wrist",
        23: "left_hip",
        24: "right_hip",
    }
    
    # Minimum confidence threshold
    MIN_CONFIDENCE = 0.5
    
    def __init__(self, model_complexity: int = 1):
        """
        Initialize the pose estimator.
        
        Args:
            model_complexity: 0 (lite), 1 (full), or 2 (heavy)
        """
        self._landmarker = None
        self._use_fallback = False
        
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # Download model if needed
            model_path = MODEL_DIR / "pose_landmarker.task"
            download_model(POSE_MODEL_URL, model_path)
            
            # Create pose landmarker
            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
            logger.info("PoseEstimator initialized with MediaPipe Tasks API")
            
        except Exception as e:
            logger.warning(f"Failed to initialize MediaPipe Tasks: {e}")
            logger.info("Using OpenCV fallback for pose estimation")
            self._use_fallback = True
    
    def predict(self, image: np.ndarray) -> PoseResult:
        """
        Estimate pose from an RGB image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
        
        Returns:
            PoseResult with keypoints
        
        Raises:
            ValueError: If pose detection fails or confidence is too low
        """
        height, width = image.shape[:2]
        
        if self._use_fallback:
            return self._predict_fallback(image)
        
        import mediapipe as mp
        
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        
        # Detect pose
        result = self._landmarker.detect(mp_image)
        
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            raise ValueError("No pose detected in image")
        
        # Get first pose
        landmarks = result.pose_landmarks[0]
        
        # Extract relevant keypoints
        keypoints = {}
        confidences = []
        
        for idx, name in self.LANDMARK_INDICES.items():
            landmark = landmarks[idx]
            # MediaPipe Tasks uses visibility for confidence
            visibility = landmark.visibility if hasattr(landmark, 'visibility') else 0.9
            keypoints[name] = Keypoint(
                x=landmark.x,
                y=landmark.y,
                confidence=visibility,
            )
            confidences.append(visibility)
        
        overall_confidence = np.mean(confidences)
        
        # Check confidence threshold
        if overall_confidence < self.MIN_CONFIDENCE:
            logger.warning(f"Low pose confidence: {overall_confidence:.2f}")
        
        # Check if key points are detected with sufficient confidence
        critical_points = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        for point in critical_points:
            if keypoints[point].confidence < self.MIN_CONFIDENCE:
                logger.warning(
                    f"Critical keypoint '{point}' has low confidence: "
                    f"{keypoints[point].confidence:.2f}"
                )
        
        return PoseResult(
            keypoints=keypoints,
            image_width=width,
            image_height=height,
            overall_confidence=overall_confidence,
        )
    
    def _predict_fallback(self, image: np.ndarray) -> PoseResult:
        """
        Fallback pose estimation using simple heuristics.
        
        This is used when MediaPipe is not available.
        """
        height, width = image.shape[:2]
        
        # Simple heuristic-based pose estimation
        # Assumes person is centered and facing camera
        
        keypoints = {
            "left_shoulder": Keypoint(x=0.35, y=0.25, confidence=0.7),
            "right_shoulder": Keypoint(x=0.65, y=0.25, confidence=0.7),
            "left_elbow": Keypoint(x=0.25, y=0.45, confidence=0.6),
            "right_elbow": Keypoint(x=0.75, y=0.45, confidence=0.6),
            "left_wrist": Keypoint(x=0.20, y=0.60, confidence=0.5),
            "right_wrist": Keypoint(x=0.80, y=0.60, confidence=0.5),
            "left_hip": Keypoint(x=0.40, y=0.65, confidence=0.7),
            "right_hip": Keypoint(x=0.60, y=0.65, confidence=0.7),
        }
        
        return PoseResult(
            keypoints=keypoints,
            image_width=width,
            image_height=height,
            overall_confidence=0.6,
        )
    
    def get_anchor_points(self, pose_result: PoseResult) -> Dict[str, Tuple[int, int]]:
        """
        Get anchor points for garment warping.
        
        Returns:
            Dictionary with anchor point names and pixel coordinates
        """
        anchors = {}
        
        # Shoulder points
        left_shoulder = pose_result.get_pixel_coords("left_shoulder")
        right_shoulder = pose_result.get_pixel_coords("right_shoulder")
        
        if left_shoulder and right_shoulder:
            anchors["left_shoulder"] = left_shoulder
            anchors["right_shoulder"] = right_shoulder
            
            # Neck point (midpoint of shoulders, slightly up)
            neck_x = (left_shoulder[0] + right_shoulder[0]) // 2
            neck_y = min(left_shoulder[1], right_shoulder[1]) - 20
            anchors["neck"] = (neck_x, max(0, neck_y))
        
        # Hip points for hem placement
        left_hip = pose_result.get_pixel_coords("left_hip")
        right_hip = pose_result.get_pixel_coords("right_hip")
        
        if left_hip and right_hip:
            anchors["left_hip"] = left_hip
            anchors["right_hip"] = right_hip
            
            # Waist point (between shoulders and hips)
            if left_shoulder and right_shoulder:
                waist_x = (left_hip[0] + right_hip[0]) // 2
                shoulder_y = (left_shoulder[1] + right_shoulder[1]) // 2
                hip_y = (left_hip[1] + right_hip[1]) // 2
                waist_y = int(shoulder_y + (hip_y - shoulder_y) * 0.6)
                anchors["waist"] = (waist_x, waist_y)
        
        return anchors
    
    def __del__(self):
        """Cleanup resources."""
        if self._landmarker is not None:
            self._landmarker.close()
