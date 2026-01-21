"""
Local CUDA-powered Virtual Try-On using Stable Diffusion Inpainting.

Requires NVIDIA GPU with CUDA support.
Works offline, no API needed.
"""

import logging
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class CUDATryOn:
    """
    Local AI Virtual Try-On using Stable Diffusion Inpainting.
    
    Runs entirely on local GPU - no internet required after model download.
    Requires ~6-8GB VRAM.
    """
    
    # Model to use - SD Inpainting is good for this task
    MODEL_ID = "runwayml/stable-diffusion-inpainting"
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize CUDA Try-On.
        
        Args:
            device: "cuda", "cpu", or None for auto-detect
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe = None
        self._segmentation_model = None
        
        logger.info(f"CUDATryOn initializing on device: {self.device}")
        
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {gpu_name}, Memory: {gpu_mem:.1f}GB")
        else:
            logger.warning("CUDA not available! Running on CPU will be VERY slow.")
    
    @property
    def pipe(self):
        """Lazy-load the inpainting pipeline."""
        if self._pipe is None:
            from diffusers import StableDiffusionInpaintPipeline
            
            logger.info(f"Loading model: {self.MODEL_ID}")
            logger.info("This may take a few minutes on first run...")
            
            self._pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )
            
            self._pipe = self._pipe.to(self.device)
            
            # Optimizations for faster inference
            if self.device == "cuda":
                self._pipe.enable_attention_slicing()
                try:
                    self._pipe.enable_xformers_memory_efficient_attention()
                    logger.info("xformers enabled for faster inference")
                except:
                    logger.info("xformers not available, using standard attention")
            
            logger.info("Model loaded successfully!")
        
        return self._pipe
    
    def _create_clothing_mask(self, person_image: np.ndarray) -> np.ndarray:
        """
        Create a mask for the clothing area on the person.
        
        Uses color-based segmentation to identify torso/clothing region.
        """
        from PIL import Image
        import cv2
        
        h, w = person_image.shape[:2]
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(person_image, cv2.COLOR_RGB2HSV)
        
        # Detect skin
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
        
        lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
        upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
        skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        
        skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
        
        # Clean up skin mask
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # Create torso region mask (middle portion of image)
        torso_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Estimate torso region
        top = int(h * 0.15)
        bottom = int(h * 0.70)
        left = int(w * 0.20)
        right = int(w * 0.80)
        
        torso_mask[top:bottom, left:right] = 255
        
        # Clothing mask = torso region minus skin
        clothing_mask = cv2.bitwise_and(torso_mask, cv2.bitwise_not(skin_mask))
        
        # Expand slightly
        clothing_mask = cv2.dilate(clothing_mask, kernel, iterations=3)
        
        # Smooth edges
        clothing_mask = cv2.GaussianBlur(clothing_mask, (21, 21), 0)
        
        return clothing_mask
    
    def _prepare_garment_prompt(self, garment_image: np.ndarray) -> str:
        """
        Analyze garment image and create a prompt describing it.
        """
        # Simple color analysis
        avg_color = garment_image.mean(axis=(0, 1))
        r, g, b = avg_color
        
        # Determine dominant color
        if r > g and r > b:
            if r > 200:
                color = "red"
            else:
                color = "dark red"
        elif g > r and g > b:
            color = "green"
        elif b > r and b > g:
            if b > 150:
                color = "blue"
            else:
                color = "dark blue"
        elif r > 200 and g > 200 and b > 200:
            color = "white"
        elif r < 50 and g < 50 and b < 50:
            color = "black"
        elif abs(r - g) < 30 and abs(g - b) < 30:
            color = "gray"
        else:
            color = "colorful"
        
        return f"a person wearing a {color} shirt, high quality, detailed clothing"
    
    def run(
        self,
        person_image: np.ndarray,
        garment_image: np.ndarray,
        num_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Run local AI virtual try-on.
        
        Args:
            person_image: RGB person image (H, W, 3)
            garment_image: RGB garment image (H, W, 3)
            num_steps: Number of diffusion steps (20-50)
            guidance_scale: How closely to follow prompt (5-15)
            seed: Random seed for reproducibility
        
        Returns:
            RGB result image (H, W, 3)
        """
        logger.info("Starting CUDA try-on...")
        
        # Resize images to 512x512 for SD
        target_size = (512, 512)
        
        person_pil = Image.fromarray(person_image).resize(target_size, Image.Resampling.LANCZOS)
        garment_pil = Image.fromarray(garment_image).resize(target_size, Image.Resampling.LANCZOS)
        
        person_np = np.array(person_pil)
        garment_np = np.array(garment_pil)
        
        # Create clothing mask
        logger.info("Creating clothing mask...")
        mask_np = self._create_clothing_mask(person_np)
        mask_pil = Image.fromarray(mask_np).convert("RGB")
        
        # Generate prompt based on garment
        prompt = self._prepare_garment_prompt(garment_np)
        negative_prompt = "blurry, low quality, distorted, deformed, ugly, bad anatomy"
        
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Running inference with {num_steps} steps...")
        
        # Set seed for reproducibility
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Run inpainting
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=person_pil,
                mask_image=mask_pil,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
        
        logger.info("Inference complete!")
        
        # Convert result to numpy
        result_np = np.array(result)
        
        # Resize back to original size if needed
        orig_h, orig_w = person_image.shape[:2]
        if (orig_w, orig_h) != target_size:
            result_pil = Image.fromarray(result_np).resize((orig_w, orig_h), Image.Resampling.LANCZOS)
            result_np = np.array(result_pil)
        
        return result_np
    
    def get_device_info(self) -> dict:
        """Get information about the current device."""
        info = {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
            info["cuda_version"] = torch.version.cuda
        
        return info

