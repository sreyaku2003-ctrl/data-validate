# AI Photo & Document Validator (Python)

A Python computer vision engine that validates whether an uploaded file is a valid photograph of a person (e.g., Student, Father, Mother photo for KYC/admissions/onboarding). It automatically detects and rejects non-person images such as **signatures**, **documents / mark sheets**, **landscapes / scenery**, **inanimate objects / cars / tech**, and **blank / corrupted** images with clear explanatory feedback.

Converted from TypeScript into high-performance Python (using **NumPy**, **Pillow**, and **OpenCV**).

---

## Architecture & Detection Pipeline

The engine utilizes a dual-layer vision pipeline matching the exact logic and classification rules:

### Layer 1: Face Detection (OpenCV Haar Cascade)
- Scans for front-facing human faces using OpenCV's frontal face cascade classifier (`haarcascade_frontalface_default.xml`).
- If a face occupying $\ge 3\%$ of the total image area is detected, it is immediately approved as a person photo with `confidence: 0.98`.
- If no face is detected or OpenCV is bypassed, it cascades to Layer 2.

### Layer 2: Vectorized Chromaticity & Feature Extraction
The image is scaled to standard dimensions (`target_dim = 250` preserving aspect ratio) and analyzed across multiple biometric feature dimensions:
1. **Skin Chromaticity Cluster (Fitzpatrick Calibrated)**:
   - Computes YCbCr color space:
     $$Y = 0.299R + 0.587G + 0.114B$$
     $$C_b = 128 - 0.168736R - 0.331264G + 0.5B$$
     $$C_r = 128 + 0.5R - 0.418688G - 0.081312B$$
   - Bounds: $133 \le C_r \le 175$, $77 \le C_b \le 130$, $35 \le Y \le 240$.
   - Natural skin relationship: $R > G \ge B$ and $(max - min) \ge 10$.
2. **Central Portrait Region**: Focuses on the center-of-interest where faces appear (middle 60% width, top 12% to 82% height).
3. **Paper Background & Ink Stroke Density**: Distinguishes paper documents and signatures (`white_ratio > 0.82` with dark stroke density `< 0.12`).
4. **Horizontal Text-Line Edge Transitions**: Detects alternating lines of text in certificates and mark sheets (`horizontal_edge_density > 0.35`).
5. **Nature / Foliage & Sky Clusters**: Distinguishes outdoor landscapes (`green_blue_count / total_pixels > 0.35`).

---

## Classification Rules

| Case | Detected Type | Condition | Result | Message |
|------|--------------|-----------|--------|---------|
| **A** | `signature` | White $> 82\%$, Dark Strokes $< 12\%$, Skin $< 3\%$ | ❌ Invalid | *"AI detected a signature instead of a photo. Please upload a clear photo of the person."* |
| **B** | `document` | White $> 65\%$, Edge Density $> 35\%$, Skin $< 4\%$ | ❌ Invalid | *"AI detected a document/certificate instead of a person photo. Please upload a portrait photo."* |
| **C** | `landscape` | Dominant = `nature`, Skin $< 4\%$ | ❌ Invalid | *"AI detected scenery/landscape. Please upload a clear photo of the person."* |
| **D** | `object` | Central Skin $< 5\%$, Skin $< 4\%$ | ❌ Invalid | *"AI detected that this is not a person photo (no face/person detected). Please upload a clear photo of the person."* |
| **E** | `person` | Central Skin $\ge 7\%$ OR Skin $\ge 6\%$ | ✅ **Valid** | *"Valid person photo detected."* |
| **F** | `unknown` | Ambiguous / borderline | ❌ Invalid | *"AI detected that this is not a valid person photo. Please upload a clear, front-facing photo of the person."* |

---

## Quickstart

### 1. Python Usage

```python
from photo_validator import validate_person_photo

# 1. From a file path
result = validate_person_photo("path/to/student_photo.jpg")

print(f"Valid:       {result.is_valid}")
print(f"Type:        {result.detected_type}")
print(f"Confidence:  {result.confidence:.2f}")
print(f"Message:     {result.message}")

# Access detailed pixel metrics
if result.metrics:
    print(f"Skin Ratio:         {result.metrics.skin_ratio * 100:.1f}%")
    print(f"Central Skin Ratio: {result.metrics.central_skin_ratio * 100:.1f}%")
    print(f"White Paper Ratio:  {result.metrics.white_ratio * 100:.1f}%")
    print(f"Edge Density:       {result.metrics.horizontal_edge_density * 100:.1f}%")
    print(f"Dominant Color:     {result.metrics.dominant_color}")
```

### 2. Supported Inputs

The validator automatically accepts:
- `str` or `Path`: Local image file path
- `bytes`: Raw uploaded file bytes (e.g. from FastAPI/Flask/Django `file.read()`)
- `io.BytesIO` / file-like objects: `UploadFile.file`
- `PIL.Image.Image`: Decoded Pillow image object
- `np.ndarray`: OpenCV or NumPy image array (RGB or RGBA)

### 3. Command Line Interface (CLI)

```bash
# Human-readable output
python main.py samples/person_portrait.png

# JSON output
python main.py samples/signature.png --json
```

---

## Interactive Web Dashboard & REST API

Run the included FastAPI web dashboard:

```bash
python app.py
```

Then open your browser at **`http://localhost:8000`** to access:
- **Interactive Drag & Drop Upload**: Test any image live with real-time verdicts.
- **One-click Sample Testing**: Test built-in presets (Person, Signature, Document, Landscape, Object).
- **REST Endpoints**:
  - `POST /api/validate` (multipart/form-data)
  - `POST /api/validate-base64` (JSON base64)
  - `GET /api/samples` (sample image catalog)

---

## Running Tests

Run the unit test suite:

```bash
python -m unittest discover tests
```
