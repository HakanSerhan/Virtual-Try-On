"""FastAPI application for Virtual Try-On."""

import logging
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import Category, QualityMode, HealthResponse, ErrorResponse
from tryon.pipeline import TryOnPipeline
from utils.image_io import load_image, load_image_with_alpha, image_to_bytes
from utils.timing import Timer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Virtual Try-On API",
    description="API for virtual garment try-on",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline (lazy loading)
_pipeline: Optional[TryOnPipeline] = None


def get_pipeline() -> TryOnPipeline:
    """Get or create the try-on pipeline instance."""
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing TryOnPipeline...")
        _pipeline = TryOnPipeline()
        logger.info("TryOnPipeline initialized")
    return _pipeline


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse()


@app.post("/tryon")
async def try_on(
    person_image: UploadFile = File(..., description="Person photo"),
    garment_image: UploadFile = File(..., description="Garment image"),
    garment_mask: Optional[UploadFile] = File(None, description="Optional garment mask"),
    category: str = Form(default="upper", description="Garment category"),
    quality_mode: str = Form(default="fast", description="Quality mode: fast or better"),
):
    """
    Virtual try-on endpoint.
    
    Takes a person photo and garment image, returns the person wearing the garment.
    
    - **person_image**: Photo of the person (JPEG/PNG)
    - **garment_image**: Image of the garment (JPEG/PNG, ideally with transparent background)
    - **garment_mask**: Optional pre-computed mask for the garment
    - **category**: Garment category ("upper" for now)
    - **quality_mode**: "fast" (affine warp) or "better" (TPS warp with occlusion)
    
    Returns: PNG image of the person wearing the garment
    """
    # Generate request ID
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] New try-on request: category={category}, quality={quality_mode}")
    
    # Initialize timer
    timer = Timer(request_id=request_id)
    
    try:
        # Validate inputs
        try:
            cat = Category(category)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
        
        try:
            quality = QualityMode(quality_mode)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid quality_mode: {quality_mode}")
        
        # Load images
        with timer.measure("load_images"):
            person_bytes = await person_image.read()
            garment_bytes = await garment_image.read()
            
            person_img = load_image(person_bytes)
            garment_img, garment_alpha = load_image_with_alpha(garment_bytes)
            
            # Load optional mask
            mask_img = None
            if garment_mask is not None:
                mask_bytes = await garment_mask.read()
                mask_img = load_image(mask_bytes)
                # Convert to grayscale if needed
                if len(mask_img.shape) == 3:
                    import cv2
                    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
            elif garment_alpha is not None:
                # Use alpha channel as mask
                mask_img = garment_alpha
        
        logger.info(f"[{request_id}] Person image: {person_img.shape}, Garment: {garment_img.shape}")
        
        # Get pipeline
        pipeline = get_pipeline()
        
        # Run try-on pipeline
        with timer.measure("pipeline"):
            result_img = pipeline.run(
                person_image=person_img,
                garment_image=garment_img,
                garment_mask=mask_img,
                quality_mode=quality.value,
                timer=timer,
            )
        
        # Convert result to bytes
        with timer.measure("encode"):
            result_bytes = image_to_bytes(result_img, format="PNG")
        
        # Log timing summary
        timer.log_summary()
        
        logger.info(f"[{request_id}] Try-on completed successfully")
        
        return Response(
            content=result_bytes,
            media_type="image/png",
            headers={
                "X-Request-ID": request_id,
                "X-Processing-Time": f"{timer.get_total():.3f}s",
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Virtual Try-On API",
        "version": "0.1.0",
        "endpoints": {
            "/tryon": "POST - Virtual try-on",
            "/health": "GET - Health check",
        }
    }

