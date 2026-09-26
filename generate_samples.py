"""
Generates sample images representing each validation class for demonstration.
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_samples(output_dir: str = "samples"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Person Portrait (Light background, realistic skin tone oval in center, shoulders)
    person = Image.new("RGBA", (300, 350), (220, 222, 225, 255))
    draw = ImageDraw.Draw(person)
    # Head / Face
    draw.ellipse([85, 55, 215, 215], fill=(215, 160, 125, 255))
    # Hair
    draw.chord([80, 45, 220, 150], 180, 360, fill=(35, 25, 20, 255))
    # Neck
    draw.rectangle([115, 200, 185, 260], fill=(205, 150, 115, 255))
    # Shoulders / Blue Shirt
    draw.ellipse([30, 240, 270, 450], fill=(45, 75, 130, 255))
    person.save(out / "person_portrait.png")

    # 2. Signature (Pure white canvas with dark blue cursive ink strokes)
    sig = Image.new("RGBA", (350, 180), (255, 255, 255, 255))
    draw_sig = ImageDraw.Draw(sig)
    # Draw realistic cursive signature path
    pts = []
    for t in np.linspace(0, 14 * np.pi, 250):
        x = int(40 + t * 6.5 + 15 * np.cos(t))
        y = int(90 + 35 * np.sin(t * 0.9) + 12 * np.cos(t * 2.3))
        pts.append((x, y))
    for i in range(len(pts) - 1):
        draw_sig.line([pts[i], pts[i + 1]], fill=(15, 25, 80, 255), width=3)
    # Underline flourish
    draw_sig.line([(45, 140), (310, 130)], fill=(15, 25, 80, 255), width=2)
    sig.save(out / "signature.png")

    # 3. Document / Certificate (White paper with dense horizontal text lines)
    doc = Image.new("RGBA", (250, 320), (255, 255, 255, 255))
    draw_doc = ImageDraw.Draw(doc)
    # Header line
    draw_doc.line([(25, 25), (225, 25)], fill=(30, 45, 80, 255), width=3)
    # Dense text lines
    for y in range(40, 290, 6):
        line_end = 225 if (y % 18 != 0) else 170
        draw_doc.line([(25, y), (line_end, y)], fill=(20, 20, 20, 255), width=1)
    doc.save(out / "document.png")

    # 4. Landscape / Scenery (Sky blue gradient on top, rich green grass/trees bottom)
    scenery = Image.new("RGBA", (300, 250), (0, 0, 0, 255))
    pix = scenery.load()
    for y in range(250):
        for x in range(300):
            if y < 100:
                # Sky blue
                pix[x, y] = (70, 130, 245, 255)
            else:
                # Nature green
                pix[x, y] = (35, 175, 55, 255)
    scenery.save(out / "landscape.png")

    # 5. Inanimate Object (Dark vehicle / metallic tech object)
    obj = Image.new("RGBA", (300, 250), (60, 65, 70, 255))
    draw_obj = ImageDraw.Draw(obj)
    # Metallic car-like outline
    draw_obj.polygon([(50, 170), (80, 110), (220, 110), (260, 170)], fill=(180, 20, 30, 255))
    draw_obj.rectangle([40, 170, 270, 210], fill=(160, 15, 25, 255))
    draw_obj.ellipse([65, 190, 105, 230], fill=(20, 20, 20, 255))
    draw_obj.ellipse([205, 190, 245, 230], fill=(20, 20, 20, 255))
    obj.save(out / "object_car.png")

    print(f"Generated 5 sample images in: {out.resolve()}")

if __name__ == "__main__":
    generate_samples()
