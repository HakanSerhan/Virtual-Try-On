"""Pydantic schemas for the Virtual Try-On API."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Category(str, Enum):
    """Garment category enum."""
    UPPER = "upper"
    # Future: LOWER = "lower", DRESS = "dress"


class QualityMode(str, Enum):
    """Processing quality mode."""
    FAST = "fast"      # Affine warp, no occlusion
    BETTER = "better"  # TPS warp, with occlusion handling


class TryOnRequest(BaseModel):
    """Request schema for try-on endpoint (for JSON mode, not used in form-data)."""
    category: Category = Field(default=Category.UPPER, description="Garment category")
    quality_mode: QualityMode = Field(default=QualityMode.FAST, description="Processing quality")


class TryOnResponse(BaseModel):
    """Response schema for try-on endpoint (when returning JSON)."""
    success: bool
    message: str
    request_id: str
    timing: Optional[dict] = None


class ErrorResponse(BaseModel):
    """Error response schema."""
    success: bool = False
    error: str
    request_id: Optional[str] = None
    details: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "0.1.0"

