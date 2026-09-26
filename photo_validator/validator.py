"""
AI Photo Validator
Validates whether an uploaded file is a valid photograph of a person (e.g. Father, Mother, Student photo).
Rejects non-person images (e.g., cars, landscapes, pets, signatures, documents, invoices, or random objects)
with a clear explanatory message.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Union, BinaryIO, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
    _OPENCV_AVAILABLE = True
except ImportError:
    cv2 = None
    _OPENCV_AVAILABLE = False

from .models import PhotoValidationResult, PixelMetrics, DetectedType


def is_human_skin_pixel(r: int, g: int, b: int) -> bool:
    """
    Checks whether an RGB pixel falls within the universal human skin chromaticity cluster.
    Uses YCbCr color space calibrated across all Fitzpatrick scale skin tones (Fair to Deep Indian/Global tones).
    """
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    # YCbCr skin bounds
    is_skin_ycbcr = (133 <= cr <= 175) and (77 <= cb <= 130) and (35 <= y <= 240)
    if not is_skin_ycbcr:
        return False

    # Additional RGB relationship checks for natural skin tones
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    diff = max_val - min_val

    return (r > g) and (g >= b) and (diff >= 10)


def analyze_canvas_pixels(
    rgba_data: np.ndarray,
    width: int,
    height: int
) -> PixelMetrics:
    """
    Analyzes pixel data from an image canvas to detect whether it is a human portrait,
    a document/certificate, a signature, or an inanimate object/landscape.

    High-performance vectorized NumPy implementation matching the exact pixel math.
    """
    total_pixels = width * height
    if total_pixels == 0:
        return PixelMetrics(
            skin_ratio=0.0,
            central_skin_ratio=0.0,
            white_ratio=0.0,
            dark_stroke_ratio=0.0,
            horizontal_edge_density=0.0,
            dominant_color='mixed'
        )

    r = rgba_data[:, :, 0].astype(np.float32)
    g = rgba_data[:, :, 1].astype(np.float32)
    b = rgba_data[:, :, 2].astype(np.float32)
    a = rgba_data[:, :, 3].astype(np.float32)

    is_transparent = a < 50
    not_transparent = ~is_transparent

    # Luminance calculation: 0.299 * r + 0.587 * g + 0.114 * b
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    active_luminance = np.where(not_transparent, luminance, 0.0)

    # Row average luminance and horizontal transitions
    row_luminance_sum = np.sum(active_luminance, axis=1)
    avg_luminance = row_luminance_sum / width

    diffs = np.empty(height, dtype=np.float32)
    diffs[0] = abs(avg_luminance[0] - 0.0)
    if height > 1:
        diffs[1:] = np.abs(np.diff(avg_luminance))
    horizontal_edge_transitions = int(np.sum(diffs > 25))

    # White / paper background:
    # Transparent treated as white (a < 50), or opaque near-white (r > 225 & g > 225 & b > 225)
    near_white = not_transparent & (r > 225) & (g > 225) & (b > 225)
    white_mask = is_transparent | near_white
    white_count = int(np.sum(white_mask))

    # Dark stroke count: opaque, not near-white, luminance < 75
    dark_stroke_mask = not_transparent & (~near_white) & (luminance < 75)
    dark_stroke_count = int(np.sum(dark_stroke_mask))

    # Foliage green and upper sky blue:
    y_coords = np.arange(height)[:, np.newaxis]
    foliage = not_transparent & (g > r + 25) & (g > b + 15)
    sky = not_transparent & (b > r + 35) & (b > g + 20) & (y_coords < height * 0.4)
    green_blue_count = int(np.sum(foliage | sky))

    # Human skin chromaticity cluster
    y_cbcr_y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    skin_ycbcr = (cr >= 133) & (cr <= 175) & (cb >= 77) & (cb <= 130) & (y_cbcr_y >= 35) & (y_cbcr_y <= 240)

    rgb_max = np.maximum(np.maximum(r, g), b)
    rgb_min = np.minimum(np.minimum(r, g), b)
    rgb_diff = rgb_max - rgb_min

    skin_rgb = (r > g) & (g >= b) & (rgb_diff >= 10)
    skin_mask = not_transparent & skin_ycbcr & skin_rgb

    skin_count = int(np.sum(skin_mask))

    # Center-of-interest region for passport/portrait photos (middle 60% horizontally, top 12% to 82% vertically)
    center_x_start = int(width * 0.2)
    center_x_end = int(width * 0.8)
    center_y_start = int(height * 0.12)
    center_y_end = int(height * 0.82)
    central_total_pixels = (center_x_end - center_x_start) * (center_y_end - center_y_start)

    central_skin_mask = skin_mask[center_y_start:center_y_end + 1, center_x_start:center_x_end + 1]
    central_skin_count = int(np.sum(central_skin_mask))

    skin_ratio = skin_count / total_pixels
    central_skin_ratio = (central_skin_count / central_total_pixels) if central_total_pixels > 0 else 0.0
    white_ratio = white_count / total_pixels
    dark_stroke_ratio = dark_stroke_count / total_pixels
    horizontal_edge_density = horizontal_edge_transitions / height

    dominant_color = 'mixed'
    if skin_ratio > 0.12:
        dominant_color = 'skin'
    elif white_ratio > 0.7:
        dominant_color = 'white'
    elif (green_blue_count / total_pixels) > 0.35:
        dominant_color = 'nature'

    return PixelMetrics(
        skin_ratio=skin_ratio,
        central_skin_ratio=central_skin_ratio,
        white_ratio=white_ratio,
        dark_stroke_ratio=dark_stroke_ratio,
        horizontal_edge_density=horizontal_edge_density,
        dominant_color=dominant_color
    )


# Cached Haar cascade for Layer 1 face detection
_FACE_CASCADE = None

def _get_face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None and _OPENCV_AVAILABLE:
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                _FACE_CASCADE = cascade
        except Exception:
            _FACE_CASCADE = None
    return _FACE_CASCADE


def _detect_faces_opencv(img: Image.Image) -> Optional[float]:
    """
    Detects front faces in the image using OpenCV Haar cascade.
    Returns the maximum face area ratio if found, else None.
    """
    cascade = _get_face_cascade()
    if cascade is None:
        return None

    try:
        rgb_img = img.convert('RGB')
        np_img = np.array(rgb_img)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

        orig_w, orig_h = img.size
        total_area = float(orig_w * orig_h)
        if total_area <= 0:
            return None

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(int(min(orig_w, orig_h) * 0.08), int(min(orig_w, orig_h) * 0.08))
        )

        if len(faces) == 0:
            return None

        max_area_ratio = 0.0
        for (x, y, w, h) in faces:
            area_ratio = (w * h) / total_area
            if area_ratio > max_area_ratio:
                max_area_ratio = area_ratio

        return max_area_ratio
    except Exception:
        return None


def validate_person_photo(
    image_input: Union[str, Path, bytes, BinaryIO, Image.Image, np.ndarray, None],
    filename: Optional[str] = None,
    enable_face_detector: bool = True
) -> PhotoValidationResult:
    """
    Validates whether an uploaded image is a valid photo of a person.
    Runs native FaceDetector (OpenCV Haar Cascade) if available, followed by specialized
    computer vision feature analysis.

    Parameters:
        image_input: File path, bytes, file-like object, PIL Image, or numpy array.
        filename: Optional filename hint to check extensions (e.g. .jpg, .png).
        enable_face_detector: Whether to run Layer 1 OpenCV face detector first.

    Returns:
        PhotoValidationResult with is_valid, detected_type, confidence, message, and pixel metrics.
    """
    # 0. Check empty input
    if image_input is None:
        return PhotoValidationResult(
            is_valid=False,
            detected_type='blank',
            confidence=0.0,
            message='No file provided. Please select an image.'
        )

    # 1. Format check
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    source_filename = None

    if isinstance(image_input, (str, Path)):
        source_filename = str(image_input)
    elif filename:
        source_filename = filename

    if source_filename:
        ext = os.path.splitext(source_filename)[1].lower()
        if ext and ext not in valid_extensions:
            return PhotoValidationResult(
                is_valid=False,
                detected_type='unknown',
                confidence=0.0,
                message='Unsupported format. Please upload a JPG, JPEG, or PNG image.'
            )

    # 2. Decode into a PIL Image
    img: Image.Image
    try:
        if isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input))
            img.load()
        elif isinstance(image_input, bytes):
            if len(image_input) == 0:
                return PhotoValidationResult(
                    is_valid=False,
                    detected_type='blank',
                    confidence=0.0,
                    message='No file provided. Please select an image.'
                )
            img = Image.open(io.BytesIO(image_input))
            img.load()
        elif hasattr(image_input, 'read'):
            raw_bytes = image_input.read()
            if len(raw_bytes) == 0:
                return PhotoValidationResult(
                    is_valid=False,
                    detected_type='blank',
                    confidence=0.0,
                    message='No file provided. Please select an image.'
                )
            img = Image.open(io.BytesIO(raw_bytes))
            img.load()
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                img = Image.fromarray(image_input).convert('RGBA')
            elif image_input.ndim == 3:
                if image_input.shape[2] == 4:
                    img = Image.fromarray(image_input, mode='RGBA')
                elif image_input.shape[2] == 3:
                    img = Image.fromarray(image_input, mode='RGB')
                else:
                    raise ValueError(f"Unsupported array shape: {image_input.shape}")
            else:
                raise ValueError(f"Unsupported array dimensions: {image_input.ndim}")
        else:
            return PhotoValidationResult(
                is_valid=False,
                detected_type='unknown',
                confidence=0.0,
                message='Unsupported input type. Please upload a valid photo.'
            )

        # Validate image format if PIL detected one
        if hasattr(img, 'format') and img.format:
            fmt = img.format.upper()
            if fmt not in {'JPEG', 'JPG', 'PNG', 'WEBP'}:
                return PhotoValidationResult(
                    is_valid=False,
                    detected_type='unknown',
                    confidence=0.0,
                    message='Unsupported format. Please upload a JPG, JPEG, or PNG image.'
                )

    except Exception:
        return PhotoValidationResult(
            is_valid=False,
            detected_type='unknown',
            confidence=0.0,
            message='Corrupted image file. Please upload a valid photo.'
        )

    # 3. Layer 1: Face Detector (OpenCV / Haar cascade)
    if enable_face_detector:
        face_area = _detect_faces_opencv(img)
        if face_area is not None and face_area >= 0.03:
            # Also calculate metrics so they are available for inspection/explainability
            w, h = _calculate_scaled_dims(img.width, img.height, target_dim=250)
            scaled_img = img.resize((w, h), Image.Resampling.BILINEAR).convert('RGBA')
            rgba_data = np.array(scaled_img)
            metrics = analyze_canvas_pixels(rgba_data, w, h)

            return PhotoValidationResult(
                is_valid=True,
                detected_type='person',
                confidence=0.98,
                message='Valid photo of person detected.',
                metrics=metrics
            )

    # 4. Layer 2: Computer Vision Canvas Feature Extraction
    # Scale image down to standard dimensions for fast, deterministic pixel scanning
    w, h = _calculate_scaled_dims(img.width, img.height, target_dim=250)
    scaled_img = img.resize((w, h), Image.Resampling.BILINEAR).convert('RGBA')
    rgba_data = np.array(scaled_img)

    metrics = analyze_canvas_pixels(rgba_data, w, h)

    # Classification Rules:

    # Case A: Signature Detection (predominantly white background with isolated dark ink strokes)
    if metrics.white_ratio > 0.82 and metrics.dark_stroke_ratio < 0.12 and metrics.skin_ratio < 0.03:
        return PhotoValidationResult(
            is_valid=False,
            detected_type='signature',
            confidence=0.94,
            message='AI detected a signature instead of a photo. Please upload a clear photo of the person.',
            metrics=metrics
        )

    # Case B: Document / Certificate / Mark Sheet (mostly white paper with text line transitions)
    if metrics.white_ratio > 0.65 and metrics.horizontal_edge_density > 0.35 and metrics.skin_ratio < 0.04:
        return PhotoValidationResult(
            is_valid=False,
            detected_type='document',
            confidence=0.92,
            message='AI detected a document/certificate instead of a person photo. Please upload a portrait photo.',
            metrics=metrics
        )

    # Case C: Outdoor / Nature / Scenery / Wallpaper
    if metrics.dominant_color == 'nature' and metrics.skin_ratio < 0.04:
        return PhotoValidationResult(
            is_valid=False,
            detected_type='landscape',
            confidence=0.90,
            message='AI detected scenery/landscape. Please upload a clear photo of the person.',
            metrics=metrics
        )

    # Case D: Inanimate Object / Vehicle / Tech / Dark or Blank Image
    if metrics.central_skin_ratio < 0.05 and metrics.skin_ratio < 0.04:
        return PhotoValidationResult(
            is_valid=False,
            detected_type='object',
            confidence=0.88,
            message='AI detected that this is not a person photo (no face/person detected). Please upload a clear photo of the person.',
            metrics=metrics
        )

    # Case E: Human Person Photo (skin pixels form a cohesive cluster in the central portrait area)
    if metrics.central_skin_ratio >= 0.07 or metrics.skin_ratio >= 0.06:
        return PhotoValidationResult(
            is_valid=True,
            detected_type='person',
            confidence=0.95,
            message='Valid person photo detected.',
            metrics=metrics
        )

    # Case F: Borderline / Ambiguous - Reject with explicit guidance
    return PhotoValidationResult(
        is_valid=False,
        detected_type='unknown',
        confidence=0.60,
        message='AI detected that this is not a valid person photo. Please upload a clear, front-facing photo of the person.',
        metrics=metrics
    )


def _calculate_scaled_dims(orig_w: int, orig_h: int, target_dim: int = 250) -> Tuple[int, int]:
    """Calculates resized dimensions preserving aspect ratio matching the JS canvas logic."""
    if orig_h <= 0 or orig_w <= 0:
        return target_dim, target_dim

    aspect = orig_w / orig_h
    if aspect > 1:
        h = max(1, round(target_dim / aspect))
        w = target_dim
    else:
        w = max(1, round(target_dim * aspect))
        h = target_dim
    return w, h
