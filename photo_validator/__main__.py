"""
CLI entry point for photo_validator.
Usage:
    python -m photo_validator <path_to_image>
    python -m photo_validator <path_to_image> --json
"""

import sys
import json
import argparse
from pathlib import Path
from photo_validator import validate_person_photo


def main():
    parser = argparse.ArgumentParser(
        description="AI Photo & Document Validator - Check if an image is a valid human photo vs signature, document, landscape, or object."
    )
    parser.add_argument("image_path", type=str, help="Path to the image file to validate")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--no-face-detector", action="store_true", help="Disable Layer 1 Haar cascade face detector")

    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: File not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    result = validate_person_photo(
        image_path,
        enable_face_detector=not args.no_face_detector
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        status = "PASSED" if result.is_valid else "FAILED"
        print(f"\n==========================================")
        print(f" Photo Validation: {status}")
        print(f"==========================================")
        print(f" Valid:          {result.is_valid}")
        print(f" Detected Type:  {result.detected_type.upper()}")
        print(f" Confidence:     {result.confidence * 100:.1f}%")
        print(f" Message:        {result.message}")
        if result.metrics:
            m = result.metrics
            print(f"\n--- Pixel Metrics ---")
            print(f" Skin Ratio:          {m.skin_ratio * 100:.2f}%")
            print(f" Central Skin Ratio:  {m.central_skin_ratio * 100:.2f}%")
            print(f" White Ratio:         {m.white_ratio * 100:.2f}%")
            print(f" Dark Stroke Ratio:   {m.dark_stroke_ratio * 100:.2f}%")
            print(f" Horiz. Edge Density: {m.horizontal_edge_density * 100:.2f}%")
            print(f" Dominant Color:      {m.dominant_color}")
        print(f"==========================================\n")

    sys.exit(0 if result.is_valid else 1)


if __name__ == "__main__":
    main()
