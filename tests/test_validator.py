"""
Comprehensive test suite for photo_validator.
Tests all classification cases and edge conditions:
- Person photo (skin pixels in central region)
- Signature (white background with thin dark strokes)
- Document / Certificate (white paper with text line transitions)
- Landscape (green foliage and blue sky)
- Object (non-person items, car, dark background)
- Blank / Empty / Invalid inputs
"""

import io
import unittest
import numpy as np
from PIL import Image, ImageDraw

from photo_validator import (
    validate_person_photo,
    is_human_skin_pixel,
    PhotoValidationResult,
    analyze_canvas_pixels,
)


class TestPhotoValidator(unittest.TestCase):

    def test_skin_pixel_formula(self):
        """Test skin tone detection across natural tones and non-skin tones."""
        # Known typical skin tones (Indian/Asian/Caucasian/Darker skin tones)
        # R > G >= B, cr in [133, 175], cb in [77, 130]
        # Example 1: Medium warm skin tone
        r, g, b = 210, 160, 130
        self.assertTrue(is_human_skin_pixel(r, g, b), f"Expected skin pixel for ({r}, {g}, {b})")

        # Example 2: Deeper Indian tone
        r, g, b = 145, 95, 65
        self.assertTrue(is_human_skin_pixel(r, g, b), f"Expected skin pixel for ({r}, {g}, {b})")

        # Pure blue / sky
        self.assertFalse(is_human_skin_pixel(50, 120, 240))
        # Pure green / grass
        self.assertFalse(is_human_skin_pixel(30, 200, 40))
        # Pure black / white / gray
        self.assertFalse(is_human_skin_pixel(0, 0, 0))
        self.assertFalse(is_human_skin_pixel(255, 255, 255))
        self.assertFalse(is_human_skin_pixel(128, 128, 128))

    def test_signature_detection(self):
        """Predominantly white background with isolated dark ink strokes."""
        # 300x150 white image with a dark wavy signature stroke
        img = Image.new("RGBA", (300, 150), (250, 250, 250, 255))
        draw = ImageDraw.Draw(img)
        # Draw some dark strokes (ink)
        for x in range(30, 270, 5):
            y = int(75 + 20 * np.sin(x / 10.0))
            draw.line([(x, y), (x + 8, y + 5)], fill=(20, 20, 30, 255), width=2)

        res = validate_person_photo(img, enable_face_detector=False)
        self.assertFalse(res.is_valid)
        self.assertEqual(res.detected_type, "signature")
        self.assertIn("signature", res.message.lower())
        self.assertGreaterEqual(res.confidence, 0.90)

    def test_document_detection(self):
        """White paper with horizontal dark text lines causing edge transitions."""
        img = Image.new("RGBA", (200, 250), (255, 255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw dense horizontal text lines across the page
        for y in range(8, 240, 4):
            draw.line([(5, y), (195, y)], fill=(0, 0, 0, 255), width=1)

        res = validate_person_photo(img, enable_face_detector=False)
        self.assertFalse(res.is_valid)
        self.assertEqual(res.detected_type, "document")
        self.assertIn("document", res.message.lower())

    def test_landscape_detection(self):
        """Landscape with green foliage bottom and blue sky top."""
        img = Image.new("RGBA", (250, 250), (0, 0, 0, 255))
        pixels = img.load()

        for y in range(250):
            for x in range(250):
                if y < 100:  # Top 40% sky
                    pixels[x, y] = (50, 100, 240, 255)
                else:  # Foliage green
                    pixels[x, y] = (40, 180, 50, 255)

        res = validate_person_photo(img, enable_face_detector=False)
        self.assertFalse(res.is_valid)
        self.assertEqual(res.detected_type, "landscape")
        self.assertIn("scenery", res.message.lower())

    def test_object_detection(self):
        """Inanimate object (e.g. metallic gray / dark blue car or tool)."""
        img = Image.new("RGBA", (250, 250), (70, 75, 80, 255))
        draw = ImageDraw.Draw(img)
        # Draw some metallic shapes
        draw.rectangle([40, 40, 210, 210], fill=(130, 135, 140, 255))
        draw.ellipse([60, 60, 190, 190], fill=(50, 50, 55, 255))

        res = validate_person_photo(img, enable_face_detector=False)
        self.assertFalse(res.is_valid)
        self.assertEqual(res.detected_type, "object")
        self.assertIn("not a person", res.message.lower())

    def test_person_photo_detection(self):
        """Person photo with central skin cluster."""
        # 250x250 portrait with neutral light gray background and skin colored face oval in center
        img = Image.new("RGBA", (250, 250), (200, 200, 205, 255))
        draw = ImageDraw.Draw(img)

        # Central face oval (skin tone r=210, g=155, b=120)
        draw.ellipse([70, 40, 180, 180], fill=(210, 155, 120, 255))
        # Neck and shoulders
        draw.rectangle([85, 170, 165, 240], fill=(200, 145, 110, 255))
        draw.rectangle([40, 210, 210, 250], fill=(30, 50, 90, 255))  # Shirt

        res = validate_person_photo(img, enable_face_detector=False)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.detected_type, "person")
        self.assertIn("valid person photo", res.message.lower())
        self.assertGreaterEqual(res.confidence, 0.90)

    def test_empty_and_corrupt_inputs(self):
        """Test handling of None, empty bytes, corrupt files, and bad formats."""
        res_none = validate_person_photo(None)
        self.assertFalse(res_none.is_valid)
        self.assertEqual(res_none.detected_type, "blank")

        res_empty = validate_person_photo(b"")
        self.assertFalse(res_empty.is_valid)
        self.assertEqual(res_empty.detected_type, "blank")

        res_corrupt = validate_person_photo(b"not an image bytes")
        self.assertFalse(res_corrupt.is_valid)
        self.assertEqual(res_corrupt.detected_type, "unknown")
        self.assertIn("corrupted", res_corrupt.message.lower())

        res_bad_format = validate_person_photo(b"dummy", filename="test.pdf")
        self.assertFalse(res_bad_format.is_valid)
        self.assertIn("unsupported format", res_bad_format.message.lower())

    def test_to_dict_serialization(self):
        """Test JSON / dictionary serialization structure."""
        img = Image.new("RGBA", (200, 200), (255, 255, 255, 255))
        res = validate_person_photo(img, enable_face_detector=False)
        d = res.to_dict()

        self.assertIn("isValid", d)
        self.assertIn("detectedType", d)
        self.assertIn("confidence", d)
        self.assertIn("message", d)
        self.assertIn("metrics", d)
        self.assertIsInstance(d["metrics"], dict)


if __name__ == "__main__":
    unittest.main()
