import json
import pytest

def test_api_health_check(client):
    """Verifies backend health check endpoint."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["database"] == "ok"
    assert data["medgemma_mode"] == "mock"
    assert "v1.0.0" in data["triage_rule_version"]


def test_api_case_intake_full_flow(client):
    """Tests end-to-end patient conversational intake: create case -> add messages -> next question -> submit."""
    # 1. Create case
    create_resp = client.post("/cases", json={"preferred_language": "English"})
    assert create_resp.status_code == 201
    case_data = create_resp.json()
    case_id = case_data["id"]
    assert case_id.startswith("CF-")

    # 2. Add first patient message (chief complaint)
    msg_resp = client.post(f"/cases/{case_id}/messages", json={"content": "I have a sore throat and fever for 2 days."})
    assert msg_resp.status_code == 200
    msg_data = msg_resp.json()
    assert msg_data["status"] == "success"
    assert msg_data["message"]["content"] == "I have a sore throat and fever for 2 days."

    # 3. Request next follow-up question
    q_resp = client.post(f"/cases/{case_id}/next-question")
    assert q_resp.status_code == 200
    q_data = q_resp.json()
    assert "next_question" in q_data

    # 4. Patient answers with severity
    ans_resp = client.post(f"/cases/{case_id}/messages", json={"content": "The pain is about 4/10."})
    assert ans_resp.status_code == 200

    # 5. Submit intake early
    submit_resp = client.post(f"/cases/{case_id}/submit")
    assert submit_resp.status_code == 200
    submit_data = submit_resp.json()
    assert submit_data["status"] == "submitted"
    assert submit_data["priority_category"] in ["P0", "P1", "P2", "P3"]

    # 6. Retrieve full case details
    detail_resp = client.get(f"/cases/{case_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert len(detail["messages"]) >= 2
    assert len(detail["evidence_items"]) > 0
    assert len(detail["audit_events"]) >= 3
    assert detail["ai_summary"]["chief_complaint"] != ""


def test_api_reviewer_queue_sorting_and_filtering(client):
    """Verifies that the reviewer queue sorts cases by priority (P0 before P1 before P3) and supports filters."""
    # Create P0 case (Chest pain radiating to arm)
    c0 = client.post("/cases", json={"preferred_language": "English"}).json()
    client.post(f"/cases/{c0['id']}/messages", json={"content": "Crushing chest pain radiating to left arm!"})
    client.post(f"/cases/{c0['id']}/submit")

    # Create P3 case (Mild runny nose)
    c3 = client.post("/cases", json={"preferred_language": "English"}).json()
    client.post(f"/cases/{c3['id']}/messages", json={"content": "Mild runny nose, no fever, pain is 1/10."})
    client.post(f"/cases/{c3['id']}/submit")

    # Create P1 case (Severe pain 9/10)
    c1 = client.post("/cases", json={"preferred_language": "English"}).json()
    client.post(f"/cases/{c1['id']}/messages", json={"content": "Severe knee pain 9/10 after twisting it."})
    client.post(f"/cases/{c1['id']}/submit")

    # Fetch queue
    queue_resp = client.get("/cases")
    assert queue_resp.status_code == 200
    queue = queue_resp.json()
    assert len(queue) >= 3

    # Check that P0 is first in the list
    case_ids = [item["id"] for item in queue]
    p0_index = case_ids.index(c0["id"])
    p1_index = case_ids.index(c1["id"])
    p3_index = case_ids.index(c3["id"])
    assert p0_index < p1_index < p3_index

    # Test filtering by priority P0
    filtered_p0 = client.get("/cases?priority=P0").json()
    assert all(item["priority_category"] == "P0" for item in filtered_p0)
    assert any(item["id"] == c0["id"] for item in filtered_p0)


def test_api_reviewer_override_and_approval(client):
    """Verifies priority override with mandatory reason, preserving original assessment, and approval."""
    # Create and submit a routine case
    c = client.post("/cases", json={"preferred_language": "English"}).json()
    case_id = c["id"]
    client.post(f"/cases/{case_id}/messages", json={"content": "Mild headache for 1 day."})
    client.post(f"/cases/{case_id}/submit")

    # Attempt override with too-short reason (should fail validation)
    fail_override = client.post(f"/cases/{case_id}/override-priority", json={
        "new_priority": "P1",
        "reason": "bad"  # min_length=5
    })
    assert fail_override.status_code == 422

    # Override priority with valid justification
    override_resp = client.post(f"/cases/{case_id}/override-priority", json={
        "new_priority": "P1",
        "reason": "Patient is elderly on anticoagulation; urgent head CT evaluation warranted.",
        "reviewer_name": "Dr. Sarah Adams, MD"
    })
    assert override_resp.status_code == 200
    data = override_resp.json()
    assert data["original_priority"] == "P3"
    assert data["new_priority"] == "P1"

    # Review and approve case
    review_resp = client.post(f"/cases/{case_id}/review", json={
        "reviewer_summary": "Approved clinical summary with urgent CT brain protocol orders.",
        "reviewer_notes": "Admit to rapid assessment unit.",
        "approve": True
    })
    assert review_resp.status_code == 200
    rev_data = review_resp.json()
    assert rev_data["review_status"] == "approved"
    assert rev_data["reviewer_summary"] is not None

    # Reopen case
    reopen_resp = client.post(f"/cases/{case_id}/reopen")
    assert reopen_resp.status_code == 200
    assert reopen_resp.json()["review_status"] == "reopened"


def test_api_upload_document_image(client):
    """Verifies image upload and OCR text extraction."""
    from io import BytesIO
    from PIL import Image, ImageDraw

    create_resp = client.post("/cases", json={"preferred_language": "English"})
    case_id = create_resp.json()["id"]

    img = Image.new("RGB", (300, 80), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 25), "CLINICAL LAB REPORT: NORMAL", fill=(0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    resp = client.post(
        f"/cases/{case_id}/upload-document",
        files={"file": ("test_report.png", buf.getvalue(), "image/png")}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["filename"] == "test_report.png"
    assert "char_count" in data


def test_api_upload_document_pdf(client):
    """Verifies PDF document upload and text parsing."""
    from io import BytesIO
    from pypdf import PdfWriter

    create_resp = client.post("/cases", json={"preferred_language": "Hindi"})
    case_id = create_resp.json()["id"]

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    pdf_buf = BytesIO()
    writer.write(pdf_buf)

    resp = client.post(
        f"/cases/{case_id}/upload-document",
        files={"file": ("prescription.pdf", pdf_buf.getvalue(), "application/pdf")}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["filename"] == "prescription.pdf"


def test_api_create_case_initial_triage_acuity(client):
    """Acute chief complaint at case creation is triaged immediately as P1 or P0, never default P3."""
    resp = client.post("/cases", json={
        "preferred_language": "Hinglish",
        "chief_complaint": "dil me bahut dard ho rha"
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["priority_category"] == "P1"
    assert data["chief_complaint"] == "dil me bahut dard ho rha"
    assert any("RULE-P1" in r for r in data["priority_reasons"])


def test_api_reviewer_queue_reflects_actual_db_priority_counts(client):
    """Verifies that list_cases accurately returns cases with their prioritized categories."""
    # Create a routine case
    c1 = client.post("/cases", json={"chief_complaint": "mild sneezing for 2 days"}).json()["id"]
    # Create an urgent case
    c2 = client.post("/cases", json={"chief_complaint": "very high chest pain"}).json()["id"]

    list_resp = client.get("/cases")
    assert list_resp.status_code == 200
    cases = list_resp.json()
    case_map = {c["id"]: c["priority_category"] for c in cases}
    assert case_map[c1] == "P3"
    assert case_map[c2] == "P1"

