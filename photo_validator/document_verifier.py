"""
AI Document Verification Engine - Strict Certificate Classifier
Validates official student & parent certificates and government documents:
- Student / Father / Mother Aadhaar Card
- Birth Certificate
- Transfer Certificate (TC)
- Caste Certificate
- Income Certificate
- Previous Examination Marksheet / Report Card
- Medical / Vaccination Certificate
- Address Proof / Ration Card / Utility Bill
- Study & Conduct Certificate
- Migration Certificate

Strictly enforces document-specific criteria. Rejects mismatched or random documents.
Supports: PDF (.pdf), Word (.docx), and Images (.jpg, .jpeg, .png, .webp).
"""

from __future__ import annotations

import re
import io
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

try:
    import cv2
    _OPENCV_AVAILABLE = True
except ImportError:
    cv2 = None
    _OPENCV_AVAILABLE = False

try:
    import docx
    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False

from photo_validator.validator import validate_person_photo

# Strict Document Classification Rules
DOCUMENT_RULES = {
    "aadhar": {
        "name": "Aadhaar Card",
        "primary_keywords": [
            "aadhaar", "uidai", "unique identification authority of india",
            "government of india", "mera aadhaar", "bharat sarkar",
            "enrolment no", "my aadhaar", "help@uidai.gov.in"
        ],
        "secondary_keywords": [
            "resident", "vid :", "vid:", "unique identification", "male", "female", "dob"
        ],
        "regex": [
            r"\b\d{4}\s\d{4}\s\d{4}\b",          # 1234 5678 9012
            r"\b[X\d]{4}\s[X\d]{4}\s\d{4}\b",     # XXXX XXXX 1234
            r"\b\d{12}\b",                        # continuous 12 digits
        ]
    },
    "birth_certificate": {
        "name": "Birth Certificate",
        "primary_keywords": [
            "birth certificate", "certificate of birth", "birth and death", "registration of birth",
            "registration of births", "form no. 5", "form no 5", "form 5", "janm praman",
            "birth register", "birth registration", "chief registrar of births",
            "vital statistics", "rbd act"
        ],
        "secondary_keywords": [
            "date of birth", "child name", "name of child", "sex of child",
            "place of birth", "name of mother", "name of father", "municipal corporation",
            "gram panchayat", "department of health", "order of birth", "hospital of birth"
        ]
    },
    "tc": {
        "name": "Transfer Certificate (TC)",
        "primary_keywords": [
            "transfer certificate", "school leaving certificate", "tc no",
            "t.c. no", "t.c.", "leaving certificate", "college leaving certificate"
        ],
        "secondary_keywords": [
            "admission no", "institution last attended", "pupil last attended",
            "class in which pupil last studied", "conduct and character",
            "date of leaving", "scholar no", "student id"
        ]
    },
    "caste_certificate": {
        "name": "Caste Certificate",
        "primary_keywords": [
            "caste certificate", "community certificate", "scheduled caste",
            "scheduled tribe", "backward class", "other backward class", "obc certificate",
            "sc certificate", "st certificate", "jati praman"
        ],
        "secondary_keywords": [
            "tahsildar", "revenue department", "belongs to the caste", "sub-caste",
            "district magistrate", "revenue officer"
        ]
    },
    "income_certificate": {
        "name": "Income Certificate",
        "primary_keywords": [
            "income certificate", "annual income", "family income",
            "gross annual income", "aamdani praman", "annual family income"
        ],
        "secondary_keywords": [
            "revenue department", "tahsildar", "family members", "rupees",
            "per annum", "gross income"
        ]
    },
    "marksheet": {
        "name": "Marksheet / Report Card",
        "primary_keywords": [
            "marksheet", "mark sheet", "marks card", "report card",
            "progress report", "statement of marks", "annual examination", "grade card"
        ],
        "secondary_keywords": [
            "board of secondary education", "cbse", "icse", "sslc", "grades",
            "marks obtained", "maximum marks", "percentage", "roll no", "total marks",
            "subject", "result: pass", "passed"
        ]
    },
    "medical_certificate": {
        "name": "Medical / Vaccination Certificate",
        "primary_keywords": [
            "medical certificate", "fitness certificate", "vaccination certificate",
            "immunization record", "health certificate", "covid-19 vaccination", "vaccination"
        ],
        "secondary_keywords": [
            "registered medical practitioner", "doctor", "physical fitness",
            "blood group", "hospital", "clinic", "beneficiary", "dose 1", "dose 2"
        ]
    },
    "address_proof": {
        "name": "Address Proof / Utility Bill",
        "primary_keywords": [
            "electricity bill", "electric bill", "water bill", "gas bill",
            "ration card", "telephone bill", "broadband bill", "consumer no",
            "consumer number", "address proof", "voter id", "driving licence", "driving license"
        ],
        "secondary_keywords": [
            "bill date", "amount payable", "account number", "residence",
            "meter number", "units consumed"
        ]
    },
    "study_certificate": {
        "name": "Study & Conduct Certificate",
        "primary_keywords": [
            "study certificate", "conduct certificate", "character certificate",
            "bonafide certificate", "bonafide student"
        ],
        "secondary_keywords": [
            "studying in this school", "student of this institution",
            "good moral character", "principal", "bonafide"
        ]
    },
    "migration_certificate": {
        "name": "Migration Certificate",
        "primary_keywords": [
            "migration certificate", "migrated from", "interstate migration"
        ],
        "secondary_keywords": [
            "board of education", "council of higher secondary",
            "university migration", "roll no", "council"
        ]
    }
}

_OCR_READER = None

def get_ocr_reader():
    """Lazy loader for EasyOCR reader."""
    global _OCR_READER
    if _OCR_READER is None:
        try:
            import easyocr
            _OCR_READER = easyocr.Reader(['en'], gpu=False)
        except Exception:
            _OCR_READER = None
    return _OCR_READER


def extract_content(
    file_bytes: bytes,
    filename: str
) -> Tuple[str, Optional[Image.Image], str, bool]:
    """
    Extracts text, renders first page image, and checks for QR codes.
    Returns: (text, first_page_img, file_type, has_qr)
    """
    ext = Path(filename).suffix.lower() if filename else ""
    text = ""
    first_page_img = None
    file_type = "unknown"
    has_qr = False

    # 1. PDF
    if ext == ".pdf" or file_bytes.startswith(b"%PDF"):
        file_type = "pdf"
        try:
            pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
            if len(pdf_doc) > 0:
                pages_text = []
                for p_idx in range(min(len(pdf_doc), 3)):
                    page = pdf_doc[p_idx]
                    pages_text.append(page.get_text())
                text = " ".join(pages_text).strip()

                # Render page 1
                page0 = pdf_doc[0]
                pix = page0.get_pixmap(dpi=150)
                first_page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Check QR code on page 1
                if _OPENCV_AVAILABLE and first_page_img:
                    cv_img = cv2.cvtColor(np.array(first_page_img), cv2.COLOR_RGB2BGR)
                    qr_det = cv2.QRCodeDetector()
                    val, _, _ = qr_det.detectAndDecode(cv_img)
                    if val and len(val) > 10:
                        has_qr = True
                        text += " " + val

                # If scanned PDF with no text, run OCR
                if len(text) < 15 and first_page_img:
                    reader = get_ocr_reader()
                    if reader:
                        res = reader.readtext(np.array(first_page_img))
                        text += " " + " ".join([r[1] for r in res])
        except Exception:
            pass

    # 2. DOCX
    elif ext in [".docx", ".doc"]:
        file_type = "docx"
        if _DOCX_AVAILABLE and ext == ".docx":
            try:
                doc = docx.Document(io.BytesIO(file_bytes))
                text = " ".join([p.text for p in doc.paragraphs]).strip()
            except Exception:
                pass

    # 3. Images (JPEG, PNG, WEBP)
    else:
        file_type = "image"
        try:
            first_page_img = Image.open(io.BytesIO(file_bytes))
            first_page_img.load()

            # Fast OpenCV QR check
            if _OPENCV_AVAILABLE:
                cv_img = cv2.cvtColor(np.array(first_page_img.convert("RGB")), cv2.COLOR_RGB2BGR)
                qr_det = cv2.QRCodeDetector()
                val, _, _ = qr_det.detectAndDecode(cv_img)
                if val and len(val) > 10:
                    has_qr = True
                    text += " " + val
        except Exception:
            first_page_img = None

    return text.lower(), first_page_img, file_type, has_qr


def verify_document_upload(
    file_bytes: bytes,
    filename: str,
    doc_type: str
) -> Dict[str, Any]:
    """
    Main document verification function.
    Validates that the file uploaded matches the specific document requirement.
    Strictly rejects mismatched or random documents.
    """
    text, first_page_img, file_type, has_qr = extract_content(file_bytes, filename)
    doc_type_clean = doc_type.lower().replace("-", "_").strip()

    # Pre-check: If it's an image file, run visual validator to reject person selfies
    if file_type == "image" and first_page_img:
        vis_res = validate_person_photo(first_page_img, enable_face_detector=True)
        if vis_res.is_valid and vis_res.detected_type == "person":
            return {
                "allowed": False,
                "status": "REJECTED",
                "badgeText": "Rejected: Photo Detected",
                "message": "AI detected a person photograph instead of a document. Please upload the official document.",
                "detectedType": "person",
                "confidence": 0.96
            }

        # If it's a document image with little/no text, run OCR to read the text
        if len(text) < 15:
            reader = get_ocr_reader()
            if reader:
                ocr_results = reader.readtext(np.array(first_page_img.convert("RGB")))
                text += " " + " ".join([r[1] for r in ocr_results]).lower()

        # Only reject as signature if OCR found no document text and visual signature detector triggered
        if len(text) < 10 and vis_res.detected_type == "signature":
            return {
                "allowed": False,
                "status": "REJECTED",
                "badgeText": "Rejected: Signature Detected",
                "message": "AI detected a signature instead of a document. Please upload the official document.",
                "detectedType": "signature",
                "confidence": 0.94
            }

    # 1. SPECIFIC CHECK: AADHAAR CARD
    if "aadhar" in doc_type_clean or "aadhaar" in doc_type_clean:
        rule = DOCUMENT_RULES["aadhar"]
        primary_matches = [kw for kw in rule["primary_keywords"] if kw in text]
        secondary_matches = [kw for kw in rule["secondary_keywords"] if kw in text]
        has_id_regex = any(re.search(pat, text, re.IGNORECASE) for pat in rule["regex"])

        is_valid_aadhar = (
            len(primary_matches) >= 1 and (has_id_regex or len(secondary_matches) >= 1)
        ) or (
            has_qr and "uid" in text
        ) or (
            "aadhaar" in text and ("government of india" in text or "uidai" in text or has_id_regex)
        )

        if is_valid_aadhar:
            return {
                "allowed": True,
                "status": "APPROVED",
                "badgeText": "Approved (Aadhaar Verified)",
                "message": "Valid Aadhaar Card detected & verified.",
                "detectedType": "aadhar",
                "confidence": 0.98 if has_id_regex else 0.92
            }
        else:
            return {
                "allowed": False,
                "status": "REJECTED",
                "badgeText": "Rejected: Not an Aadhaar Card",
                "message": "AI didn't detect an Aadhaar card. Please upload a valid Aadhaar card (PDF or image).",
                "detectedType": "mismatched_document",
                "confidence": 0.88
            }

    # 2. SPECIFIC CHECK: OTHER CERTIFICATES (Birth Certificate, TC, Marksheet, etc.)
    rule_key = None
    for k in DOCUMENT_RULES:
        if k in doc_type_clean:
            rule_key = k
            break

    if rule_key and rule_key in DOCUMENT_RULES:
        rule = DOCUMENT_RULES[rule_key]
        primary_matches = [kw for kw in rule.get("primary_keywords", []) if kw in text]
        secondary_matches = [kw for kw in rule.get("secondary_keywords", []) if kw in text]

        # STRICT VALIDATION:
        # Must have at least 1 primary keyword OR at least 3 secondary keywords.
        # NEVER accept a random document without these matches!
        is_verified = (len(primary_matches) >= 1) or (len(secondary_matches) >= 3)

        if is_verified:
            return {
                "allowed": True,
                "status": "APPROVED",
                "badgeText": f"Approved ({rule['name']})",
                "message": f"Valid {rule['name']} detected & verified.",
                "detectedType": rule_key,
                "confidence": 0.94 if len(primary_matches) >= 1 else 0.88
            }
        else:
            # REJECT: Mismatched or random document
            return {
                "allowed": False,
                "status": "REJECTED",
                "badgeText": f"Rejected: Not a {rule['name']}",
                "message": f"AI didn't detect a valid {rule['name']}. Please upload the official {rule['name']}.",
                "detectedType": "mismatched_document",
                "confidence": 0.88
            }

    # 3. GENERIC DOCUMENT CHECK (Only when doc_type is generic 'document')
    if doc_type_clean in ["document", "doc"] and (file_type in ["pdf", "docx"] or len(text) > 20):
        return {
            "allowed": True,
            "status": "APPROVED",
            "badgeText": "Approved (Document Verified)",
            "message": "Valid document detected & verified.",
            "detectedType": "document",
            "confidence": 0.90
        }

    return {
        "allowed": False,
        "status": "REJECTED",
        "badgeText": "Rejected: Unrecognized Document",
        "message": "AI could not verify this as a valid document. Please upload a clear PDF or image of the document.",
        "detectedType": "unknown",
        "confidence": 0.70
    }
