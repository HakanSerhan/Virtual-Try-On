"""AI Virtual Try-On using Hugging Face Leffa Space (FREE)."""

import logging
import tempfile
import os

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class HuggingFaceTryOn:
    """
    AI-powered virtual try-on using Leffa on Hugging Face.
    Completely FREE, no API key needed!
    """
    
    SPACE_ID = "franciszzj/Leffa"
    
    def __init__(self):
        self._client = None
        logger.info("HuggingFaceTryOn initialized (Leffa)")
    
    @property
    def client(self):
        if self._client is None:
            from gradio_client import Client
            logger.info(f"Connecting to: {self.SPACE_ID}")
            self._client = Client(self.SPACE_ID)
            logger.info("Connected!")
        return self._client
    
    def run(
        self,
        person_image: np.ndarray,
        garment_image: np.ndarray,
        category: str = "upper_body",
        n_steps: int = 30,
        seed: int = 42,
    ) -> np.ndarray:
        """Run AI virtual try-on."""
        logger.info("Starting Leffa AI try-on...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            person_path = os.path.join(tmpdir, "person.png")
            garment_path = os.path.join(tmpdir, "garment.png")
            
            # Preprocess and save
            person_pil = self._preprocess(person_image)
            garment_pil = self._preprocess(garment_image)
            
            person_pil.save(person_path, "PNG")
            garment_pil.save(garment_path, "PNG")
            
            logger.info("Sending to Leffa API...")
            
            # Map category
            garment_type = "upper_body"
            if "lower" in category.lower():
                garment_type = "lower_body"
            elif "dress" in category.lower():
                garment_type = "dresses"
            
            try:
                result = self.client.predict(
                    {"path": person_path},  # src_image_path
                    {"path": garment_path},  # ref_image_path
                    "False",  # ref_acceleration
                    float(n_steps),  # step
                    2.5,  # scale
                    float(seed),  # seed
                    "viton_hd",  # vt_model_type
                    garment_type,  # vt_garment_type
                    "False",  # vt_repaint
                    api_name="/leffa_predict_vt"
                )
                
                logger.info(f"Result: {type(result)}")
                
                # Result is tuple: (generated_image, generated_mask, generated_densepose)
                if isinstance(result, tuple) and len(result) >= 1:
                    img_data = result[0]
                    if isinstance(img_data, dict):
                        result_path = img_data.get("path") or img_data.get("url")
                    else:
                        result_path = img_data
                else:
                    result_path = result
                
                if not result_path:
                    raise ValueError("No result received")
                
                logger.info(f"Loading result: {result_path}")
                result_img = Image.open(result_path)
                return np.array(result_img.convert("RGB"))
                
            except Exception as e:
                logger.error(f"Leffa error: {e}")
                raise RuntimeError(f"AI hatasi: {e}")
    
    def _preprocess(self, image: np.ndarray, size: int = 512) -> Image.Image:
        """Preprocess image."""
        pil = Image.fromarray(image)
        
        # Resize keeping aspect
        w, h = pil.size
        if w > h:
            new_w, new_h = size, int(h * size / w)
        else:
            new_h, new_w = size, int(w * size / h)
        
        pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Pad to square with white
        result = Image.new("RGB", (size, size), (255, 255, 255))
        x = (size - new_w) // 2
        y = (size - new_h) // 2
        result.paste(pil, (x, y))
        
        return result
