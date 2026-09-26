from dataclasses import dataclass, asdict
from typing import Literal, Optional, Dict, Any

DetectedType = Literal[
    'person',
    'document',
    'signature',
    'object',
    'landscape',
    'blank',
    'unknown'
]

DominantColor = Literal['skin', 'white', 'nature', 'monochrome', 'mixed']

@dataclass
class PixelMetrics:
    skin_ratio: float
    central_skin_ratio: float
    white_ratio: float
    dark_stroke_ratio: float
    horizontal_edge_density: float
    dominant_color: DominantColor

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class PhotoValidationResult:
    is_valid: bool
    detected_type: DetectedType
    confidence: float
    message: str
    metrics: Optional[PixelMetrics] = None

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        # Match camelCase if desired by frontend APIs as well as snake_case
        return {
            "isValid": self.is_valid,
            "detectedType": self.detected_type,
            "confidence": round(self.confidence, 4),
            "message": self.message,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            # snake_case aliases for Python standard
            "is_valid": self.is_valid,
            "detected_type": self.detected_type,
        }
