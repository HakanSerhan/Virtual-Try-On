"""Image I/O utilities for the Virtual Try-On pipeline."""

import io
from typing import Optional, Tuple
import numpy as np
import cv2
from PIL import Image


def load_image(
    source: bytes | str | np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    max_size: int = 1024,
) -> np.ndarray:
    """
    Load an image from various sources and return as RGB numpy array.
    
    Args:
        source: Image bytes, file path, or numpy array
        target_size: Optional (width, height) to resize to
        max_size: Maximum dimension (will resize if larger, preserving aspect ratio)
    
    Returns:
        RGB image as numpy array (H, W, 3), dtype uint8
    """
    # Load from source
    if isinstance(source, bytes):
        nparr = np.frombuffer(source, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    elif isinstance(source, str):
        img = cv2.imread(source, cv2.IMREAD_UNCHANGED)
    elif isinstance(source, np.ndarray):
        img = source.copy()
    else:
        raise ValueError(f"Unsupported image source type: {type(source)}")
    
    if img is None:
        raise ValueError("Failed to load image")
    
    # Handle alpha channel - convert to RGB
    if len(img.shape) == 2:
        # Grayscale to RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        # BGRA to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize if needed
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    elif max_size is not None:
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    return img.astype(np.uint8)


def load_image_with_alpha(
    source: bytes | str | np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load an image and extract alpha channel if present.
    
    Args:
        source: Image bytes, file path, or numpy array
        target_size: Optional (width, height) to resize to
    
    Returns:
        Tuple of (RGB image, alpha mask or None)
    """
    # Load from source
    if isinstance(source, bytes):
        nparr = np.frombuffer(source, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    elif isinstance(source, str):
        img = cv2.imread(source, cv2.IMREAD_UNCHANGED)
    elif isinstance(source, np.ndarray):
        img = source.copy()
    else:
        raise ValueError(f"Unsupported image source type: {type(source)}")
    
    if img is None:
        raise ValueError("Failed to load image")
    
    alpha = None
    
    if len(img.shape) == 2:
        # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        # BGRA - extract alpha
        alpha = img[:, :, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize if needed
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        if alpha is not None:
            alpha = cv2.resize(alpha, target_size, interpolation=cv2.INTER_LINEAR)
    
    return img.astype(np.uint8), alpha


def save_image(image: np.ndarray, path: str) -> None:
    """
    Save an RGB image to file.
    
    Args:
        image: RGB image as numpy array
        path: Output file path
    """
    # Convert RGB to BGR for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        bgr = image
    
    cv2.imwrite(path, bgr)


def image_to_bytes(image: np.ndarray, format: str = "PNG") -> bytes:
    """
    Convert an RGB image to bytes.
    
    Args:
        image: RGB image as numpy array
        format: Output format ("PNG" or "JPEG")
    
    Returns:
        Image as bytes
    """
    # Convert to PIL Image
    pil_image = Image.fromarray(image)
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    buffer.seek(0)
    
    return buffer.getvalue()


def bytes_to_image(data: bytes) -> np.ndarray:
    """
    Convert bytes to RGB image.
    
    Args:
        data: Image bytes
    
    Returns:
        RGB image as numpy array
    """
    return load_image(data)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] float range.
    
    Args:
        image: Image as uint8 numpy array
    
    Returns:
        Normalized image as float32
    """
    return image.astype(np.float32) / 255.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Denormalize image from [0, 1] to uint8 [0, 255].
    
    Args:
        image: Normalized float image
    
    Returns:
        Image as uint8
    """
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)

