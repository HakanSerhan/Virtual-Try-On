"""AI-powered Virtual Try-On using Replicate API."""

import logging
import base64
import io
import os
import time
import json
from typing import Optional

import numpy as np
from PIL import Image
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


class AITryOn:
    """
    AI-powered virtual try-on using Replicate API.
    
    Uses direct HTTP requests to avoid Pydantic compatibility issues.
    """
    
    # Virtual try-on models to try (in order of preference)
    MODELS_TO_TRY = [
        "cuuupid/idm-vton",
        "levelsio/neon-tshirt", 
        "jagilley/controlnet-hough",
    ]
    
    # Replicate API endpoints
    API_BASE = "https://api.replicate.com/v1"
    
    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize AI Try-On.
        
        Args:
            api_token: Replicate API token.
        """
        self.api_token = api_token or os.environ.get("REPLICATE_API_TOKEN")
        
        if not self.api_token:
            logger.warning("No Replicate API token provided.")
    
    def _make_request(self, endpoint: str, data: Optional[dict] = None, method: str = "GET") -> dict:
        """Make HTTP request to Replicate API."""
        url = f"{self.API_BASE}/{endpoint}"
        
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        }
        
        req = urllib.request.Request(url, method=method)
        for key, value in headers.items():
            req.add_header(key, value)
        
        if data:
            body = json.dumps(data, ensure_ascii=True).encode("utf-8")
            req.data = body
        
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="ignore")
            logger.error(f"API error: {e.code} - {error_body}")
            raise ValueError(f"API error {e.code}: {error_body}")
    
    def _get_model_version(self, model_name: str) -> str:
        """Get the latest version of a model."""
        try:
            # Get model info
            owner, name = model_name.split("/")
            model_info = self._make_request(f"models/{owner}/{name}")
            
            if not model_info:
                raise ValueError(f"No response for model {model_name}")
            
            logger.info(f"Model info keys: {list(model_info.keys())}")
            
            # Try different ways to get version
            version_id = None
            
            # Method 1: latest_version object
            latest_version = model_info.get("latest_version")
            if latest_version and isinstance(latest_version, dict):
                version_id = latest_version.get("id")
            
            # Method 2: default_example has version
            if not version_id:
                default_example = model_info.get("default_example")
                if default_example and isinstance(default_example, dict):
                    version_id = default_example.get("version")
            
            # Method 3: Get from versions endpoint
            if not version_id:
                versions_resp = self._make_request(f"models/{owner}/{name}/versions")
                if versions_resp:
                    results = versions_resp.get("results", [])
                    if results and len(results) > 0:
                        version_id = results[0].get("id")
            
            if version_id:
                logger.info(f"Found model version: {version_id[:12]}...")
                return version_id
            else:
                raise ValueError(f"No version found for model {model_name}")
                
        except Exception as e:
            logger.error(f"Failed to get model version for {model_name}: {e}")
            raise
    
    def _image_to_data_uri(self, image: np.ndarray) -> str:
        """Convert numpy image to data URI."""
        pil_image = Image.fromarray(image)
        
        # Resize if too large (max 1024px)
        max_size = 1024
        if max(pil_image.size) > max_size:
            ratio = max_size / max(pil_image.size)
            new_size = (int(pil_image.size[0] * ratio), int(pil_image.size[1] * ratio))
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=90)
        buffer.seek(0)
        base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_data}"
    
    def _download_image(self, url: str) -> np.ndarray:
        """Download image from URL."""
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=60) as response:
            image_data = response.read()
        
        pil_image = Image.open(io.BytesIO(image_data))
        return np.array(pil_image.convert("RGB"))
    
    def run(
        self,
        person_image: np.ndarray,
        garment_image: np.ndarray,
        garment_description: str = "A stylish garment",
        category: str = "upper_body",
        denoise_steps: int = 30,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Run AI virtual try-on.
        
        Args:
            person_image: RGB person image (H, W, 3)
            garment_image: RGB garment image (H, W, 3)
            garment_description: Text description of the garment
            category: "upper_body", "lower_body", or "dresses"
            denoise_steps: Number of denoising steps
            seed: Random seed
        
        Returns:
            RGB result image (H, W, 3)
        """
        if not self.api_token:
            raise ValueError(
                "Replicate API token required! "
                "Get your free API token at: https://replicate.com/account/api-tokens"
            )
        
        logger.info("Starting AI try-on...")
        
        # Clean garment description (ASCII only)
        clean_description = garment_description.encode('ascii', 'ignore').decode('ascii')
        if not clean_description:
            clean_description = "A stylish garment"
        
        # Convert images to data URIs
        logger.info("Preparing images...")
        person_uri = self._image_to_data_uri(person_image)
        garment_uri = self._image_to_data_uri(garment_image)
        
        # Try to get model version
        logger.info("Getting model version...")
        
        last_error = None
        
        for model_name in self.MODELS_TO_TRY:
            try:
                logger.info(f"Trying model: {model_name}")
                version_id = self._get_model_version(model_name)
                
                # Create prediction
                logger.info("Sending request to Replicate API...")
                
                # Build input based on model
                if "idm-vton" in model_name:
                    model_input = {
                        "human_img": person_uri,
                        "garm_img": garment_uri,
                        "garment_des": clean_description,
                        "category": category,
                        "denoise_steps": denoise_steps,
                        "seed": seed,
                    }
                elif "neon-tshirt" in model_name:
                    model_input = {
                        "image": person_uri,
                        "product_image": garment_uri,
                    }
                elif "controlnet" in model_name:
                    model_input = {
                        "image": person_uri,
                        "prompt": f"person wearing {clean_description}",
                    }
                else:
                    model_input = {
                        "image": person_uri,
                        "garment": garment_uri,
                    }
                
                prediction = self._make_request(
                    "predictions",
                    data={
                        "version": version_id,
                        "input": model_input
                    },
                    method="POST"
                )
                
                prediction_id = prediction.get("id")
                logger.info(f"Prediction started: {prediction_id}")
                
                # Poll for completion
                result = self._wait_for_prediction(prediction_id)
                return result
                
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                last_error = e
                continue
        
        # All models failed
        raise ValueError(f"All models failed. Last error: {last_error}")
    
    def _wait_for_prediction(self, prediction_id: str) -> np.ndarray:
        """Wait for prediction to complete and return result."""
        max_wait = 300  # 5 minutes
        start_time = time.time()
        
        while True:
            if time.time() - start_time > max_wait:
                raise TimeoutError("Prediction timed out after 5 minutes")
            
            status = self._make_request(f"predictions/{prediction_id}")
            state = status.get("status")
            
            logger.info(f"Status: {state}")
            
            if state == "succeeded":
                output = status.get("output")
                if output:
                    # Output can be a list or a single URL
                    if isinstance(output, list):
                        result_url = output[0] if output else None
                    else:
                        result_url = output
                    
                    if result_url:
                        logger.info("Downloading result...")
                        return self._download_image(result_url)
                    else:
                        raise ValueError("No output URL received")
                else:
                    raise ValueError("No output received from model")
            
            elif state == "failed":
                error = status.get("error", "Unknown error")
                raise ValueError(f"Prediction failed: {error}")
            
            elif state == "canceled":
                raise ValueError("Prediction was canceled")
            
            # Wait before polling again
            time.sleep(2)


def check_api_token() -> bool:
    """Check if Replicate API token is configured."""
    return bool(os.environ.get("REPLICATE_API_TOKEN"))
