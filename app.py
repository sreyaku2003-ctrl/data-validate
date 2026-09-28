"""
AI Photo & Document Validator - FastAPI Web Application & REST API
Provides:
1. REST API endpoint `POST /api/validate` for frontend upload buttons.
   - Evaluates whether the uploaded file is valid for the expected document type (person photo, signature, document).
   - Returns boolean `allowed: true/false` with a user-friendly `message`.
2. CORS enabled for cross-origin requests from Next.js, React, Angular, etc.
3. Interactive testing table UI at `GET /`.
"""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from photo_validator import validate_person_photo

app = FastAPI(
    title="AI Document & Photo Validation API",
    description="Pre-upload validation API to allow or reject document/photo uploads.",
    version="1.2.0"
)

# Enable CORS for external frontends (Next.js, React, mobile apps, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
SAMPLES_DIR = BASE_DIR / "samples"
INDEX_HTML = BASE_DIR / "index.html"


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "service": "photo-validator",
        "version": "1.2.0",
        "supported_expected_types": ["person", "signature", "document"]
    }


@app.get("/api/samples")
def list_samples():
    """List sample test images available on the server."""
    if not SAMPLES_DIR.exists():
        return []
    samples = []
    for f in SAMPLES_DIR.glob("*.png"):
        samples.append({
            "name": f.stem.replace("_", " ").title(),
            "filename": f.name,
            "url": f"/samples/{f.name}"
        })
    return samples


@app.post("/api/validate")
async def validate_file(
    file: UploadFile = File(...),
    enable_face_detector: bool = True,
    doc_type: Optional[str] = Form("person"),       # 'person' | 'signature' | 'document'
    field_name: Optional[str] = Form(None)          # Optional e.g. 'mother_photo', 'father_signature'
):
    """
    Main Pre-Upload Validation API Endpoint.
    
    Hit this endpoint inside your frontend's upload button handler before saving the file.
    
    Form Data Parameters:
      - file: Binary image file (JPEG, PNG, WEBP)
      - doc_type: Expected document type:
          * 'person' (default): For Mother Photo, Father Photo, Student Photo, Profile Picture.
          * 'signature': For Father Signature, Mother Signature, Student Signature.
          * 'document': For Marksheet, Certificate, Birth Certificate.
      - field_name: (Optional) Identifier of the field being validated.
      - enable_face_detector: (Optional, default True) Run OpenCV Haar Cascade face detector.

    Returns:
      {
        "allowed": true | false,
        "status": "APPROVED" | "REJECTED",
        "message": "Human-friendly explanation message",
        "detectedType": "person" | "signature" | "document" | "landscape" | "object" | "blank" | "unknown",
        "expectedType": "person",
        "confidence": 0.95,
        "fieldName": "mother_photo",
        "metrics": { ... }
      }
    """
    content = await file.read()
    if not content:
        return JSONResponse(
            status_code=400,
            content={
                "allowed": False,
                "status": "REJECTED",
                "message": "No file uploaded. Please select an image.",
                "detectedType": "blank",
                "expectedType": doc_type,
                "confidence": 0.0,
                "fieldName": field_name,
                "metrics": None
            }
        )

    # 1. Run the core AI vision validator
    result = validate_person_photo(
        content,
        filename=file.filename,
        enable_face_detector=enable_face_detector
    )

    detected = result.detected_type
    expected = (doc_type or "person").lower().strip()

    # 2. Gatekeeper Logic: Decide if upload is allowed for this document type
    allowed = False
    status_label = "REJECTED"
    custom_message = result.message

    if expected == "person":
        # Requires person photo
        if result.is_valid and detected == "person":
            allowed = True
            status_label = "APPROVED"
            badge_text = "Approved (Face Detected)"
            custom_message = "Valid person photo detected."
        else:
            allowed = False
            status_label = "REJECTED"
            badge_text = "Rejected: No Face Detected"
            custom_message = "AI didn't detect a face. Please upload a clear photo of the person."

    elif expected in ["signature"]:
        # Requires signature
        if detected == "signature":
            allowed = True
            status_label = "APPROVED"
            badge_text = "Approved (Signature)"
            custom_message = "Valid signature detected."
        else:
            allowed = False
            status_label = "REJECTED"
            badge_text = "Rejected (Not a Signature)"
            if detected == "person":
                custom_message = "AI detected a person photo instead of a signature. Please upload an ink signature on white paper."
            elif detected == "document":
                custom_message = "AI detected a full document instead of a single signature. Please upload an ink signature."
            else:
                custom_message = f"AI detected {detected} instead of a signature. Please upload a clear signature on white paper."

    else:
        # All Certificate & Document Fields (Aadhaar, TC, Marksheet, Birth Certificate, etc.)
        from photo_validator.document_verifier import verify_document_upload
        doc_result = verify_document_upload(content, filename=file.filename, doc_type=expected)
        allowed = doc_result["allowed"]
        status_label = doc_result["status"]
        badge_text = doc_result["badgeText"]
        custom_message = doc_result["message"]
        detected = doc_result.get("detectedType", detected)

    # Format response payload
    response_payload = {
        "allowed": allowed,
        "status": status_label,
        "badgeText": badge_text,
        "message": custom_message,
        "detectedType": detected,
        "expectedType": expected,
        "confidence": round(result.confidence, 4),
        "fieldName": field_name,
        # Legacy/helper aliases for frontend compatibility
        "isValid": allowed,
        "isFieldApproved": allowed,
        "fieldVerdict": f"{status_label.title()}: {badge_text}",
        "metrics": result.metrics.to_dict() if result.metrics else None
    }

    return JSONResponse(content=response_payload)


# Mount samples directory if it exists
if SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")


@app.get("/", response_class=HTMLResponse)
def index_page():
    if not INDEX_HTML.exists():
        return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)
    return HTMLResponse(content=INDEX_HTML.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import socket
    import sys
    import uvicorn

    def is_port_available(p: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('127.0.0.1', p))
                return True
        except OSError:
            return False

    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    elif not is_port_available(port):
        for candidate in [8001, 8080, 5000, 8888, 3000]:
            if is_port_available(candidate):
                print(f"Port {port} is occupied. Automatically using free port {candidate}.")
                port = candidate
                break

    print(f"\n=======================================================")
    print(f" AI Photo & Document Validator Web Server")
    print(f" REST API: POST http://localhost:{port}/api/validate")
    print(f" Test UI:  http://localhost:{port}")
    print(f"=======================================================\n")

    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False)
