"""
AI Photo & Document Validator
Python implementation of person photo validation vs signatures, documents, landscapes, and objects.
"""

from .models import PhotoValidationResult, PixelMetrics, DetectedType, DominantColor
from .validator import validate_person_photo, is_human_skin_pixel, analyze_canvas_pixels

__all__ = [
    "validate_person_photo",
    "is_human_skin_pixel",
    "analyze_canvas_pixels",
    "PhotoValidationResult",
    "PixelMetrics",
    "DetectedType",
    "DominantColor",
]
