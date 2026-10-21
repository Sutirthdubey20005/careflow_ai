import io
import pytest
from PIL import Image
from backend.services.ocr_service import MedicalDocumentOCRService, ocr_service
from backend.schemas.clinical import CaseCreateRequest


def test_ocr_service_plain_text():
    """Test OCR service parses plain text files directly."""
    service = MedicalDocumentOCRService()
    content = "Patient prescribed Amoxicillin 500mg TID for 7 days."
    res = service.extract_with_metadata(content.encode("utf-8"), "prescription.txt")

    assert res.char_count == len(content)
    assert "Amoxicillin" in res.extracted_text
    assert res.confidence == 1.0
    assert res.engine_used == "plain_text"


def test_ocr_service_empty_input():
    """Test OCR service handles empty bytes safely."""
    service = MedicalDocumentOCRService()
    res = service.extract_with_metadata(b"", "empty.txt")
    assert res.char_count == 0
    assert res.extracted_text == ""


def test_ocr_service_image_extraction():
    """Test OCR service image extraction metadata returns valid engine and confidence."""
    service = MedicalDocumentOCRService()
    img = Image.new("RGB", (120, 40), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    res = service.extract_with_metadata(png_bytes, "lab_report.png")

    assert res.filename == "lab_report.png"
    assert res.engine_used in ["winocr", "paddleocr", "paddleocr_multilingual", "image_metadata_fallback", "winocr_native", "mock_ocr", "none"]
    assert res.confidence >= 0.0
    assert isinstance(res.extracted_text, str)


def test_preview_document_endpoint_success(client):
    """Test document preview endpoint returns extraction metadata without committing."""
    case_res = client.post("/cases", json={"preferred_language": "English", "chief_complaint": "Testing OCR"})
    assert case_res.status_code == 201
    case_id = case_res.json()["id"]

    file_content = b"DISCHARGE SUMMARY: Patient discharged in stable condition. Diagnosis: Acute Pharyngitis."
    files = {"file": ("discharge_summary.txt", file_content, "text/plain")}

    resp = client.post(f"/cases/{case_id}/extract-document", files=files)
    assert resp.status_code == 200
    data = resp.json()

    assert data["char_count"] == len(file_content)
    assert "Pharyngitis" in data["extracted_text"]
    assert data["engine_used"] == "plain_text"

    # Verify no message was auto-committed to the case
    case_detail = client.get(f"/cases/{case_id}").json()
    assert len(case_detail["messages"]) == 0


def test_preview_document_endpoint_invalid_format(client):
    """Test document preview rejects prohibited file extensions."""
    case_res = client.post("/cases", json={"preferred_language": "English"})
    assert case_res.status_code == 201
    case_id = case_res.json()["id"]

    files = {"file": ("malware.exe", b"MZ......", "application/octet-stream")}
    resp = client.post(f"/cases/{case_id}/extract-document", files=files)
    assert resp.status_code == 400
    assert "unsupported" in resp.json()["detail"].lower()


def test_preview_document_endpoint_not_found(client):
    """Test document preview returns 404 for non-existent case."""
    files = {"file": ("notes.txt", b"Notes", "text/plain")}
    resp = client.post("/cases/NON-EXISTENT-CASE/extract-document", files=files)
    assert resp.status_code == 404
