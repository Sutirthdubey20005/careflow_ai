import io
import json
import pytest
from PIL import Image, ImageDraw
from backend.services.summarization import clinical_summarization_service
from backend.database.models import Case, AuditEvent


def test_no_fabricated_denials_when_chest_pain_present():
    """Safety check: MedGemma mock summary must never fabricate 'Denies prior cardiac history'."""
    patient_messages = [
        "I have severe chest pain that started an hour ago. The pain is about 7/10."
    ]
    summary = clinical_summarization_service.generate_summary(patient_messages)

    # Must NOT contain fabricated cardiac history denial
    assert "Denies prior cardiac history" not in summary.explicit_negative_findings
    for denial in summary.explicit_negative_findings:
        assert "cardiac" not in denial.lower()

    # Chest pain must be captured
    assert any("chest pain" in s.lower() for s in summary.symptoms_reported)


def test_interim_p1_persistence(client, db_session):
    """
    Safety check: Interim triage must persist P1 (urgent) to the case record immediately,
    so that urgent cases are immediately visible in the reviewer queue before intake completion.
    """
    create_resp = client.post("/cases", json={"preferred_language": "English"})
    assert create_resp.status_code == 201
    case_id = create_resp.json()["id"]

    # Before message, case priority is routine P3
    initial_case = db_session.query(Case).filter(Case.id == case_id).first()
    assert initial_case.priority_category == "P3"

    # Patient reports severe pain (triggering RULE-P1-PAIN-02: pain >= 8/10)
    msg_resp = client.post(
        f"/cases/{case_id}/messages",
        json={"content": "I have unbearable abdominal pain, rated 9/10."}
    )
    assert msg_resp.status_code == 200
    msg_data = msg_resp.json()
    assert msg_data["interim_priority"] == "P1"

    # Verify that Case.priority_category in the database was immediately updated to P1
    db_session.expire_all()
    updated_case = db_session.query(Case).filter(Case.id == case_id).first()
    assert updated_case.priority_category == "P1"
    assert "P1" in updated_case.priority_reasons_json


def test_interim_p0_never_downgraded_by_subsequent_p1_message(client, db_session):
    """
    Safety check: Once elevated to P0 (emergency), subsequent messages with P1 or P3 symptoms
    must NEVER downgrade the case priority.
    """
    create_resp = client.post("/cases", json={"preferred_language": "English"})
    case_id = create_resp.json()["id"]

    # First message: P0 emergency (chest pain radiating to arm)
    resp1 = client.post(
        f"/cases/{case_id}/messages",
        json={"content": "Crushing chest pain radiating to my left arm."}
    )
    assert resp1.status_code == 200
    assert resp1.json()["interim_priority"] == "P0"

    db_session.expire_all()
    case = db_session.query(Case).filter(Case.id == case_id).first()
    assert case.priority_category == "P0"

    # Second message: P1 symptom (fever with stiff neck)
    resp2 = client.post(
        f"/cases/{case_id}/messages",
        json={"content": "I also have high fever and neck stiffness."}
    )
    assert resp2.status_code == 200

    db_session.expire_all()
    case = db_session.query(Case).filter(Case.id == case_id).first()
    # Must remain P0, never downgraded to P1
    assert case.priority_category == "P0"


def test_upload_document_actor_role_attribution(client, db_session):
    """
    Data integrity check: Evidence updates triggered by document extraction
    must attribute the extraction to 'ai_system', not 'patient'.
    """
    create_resp = client.post("/cases", json={"preferred_language": "English"})
    case_id = create_resp.json()["id"]

    img = Image.new("RGB", (200, 60), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 20), "Rx: Paracetamol 500mg", fill=(0, 0, 0))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    upload_resp = client.post(
        f"/cases/{case_id}/upload-document",
        files={"file": ("rx_test.png", img_bytes, "image/png")}
    )
    assert upload_resp.status_code == 200

    db_session.expire_all()
    evidence_audits = (
        db_session.query(AuditEvent)
        .filter(AuditEvent.case_id == case_id, AuditEvent.event_type == "evidence_state_updated")
        .all()
    )
    assert len(evidence_audits) >= 1
    # Actor role must be 'ai_system'
    assert evidence_audits[-1].actor_role == "ai_system"


def test_upload_document_validation(client):
    """Safety check: Uploaded document endpoint must validate file extension and reject invalid formats."""
    create_resp = client.post("/cases", json={"preferred_language": "English"})
    case_id = create_resp.json()["id"]

    # Reject unsupported extension
    resp_bad_ext = client.post(
        f"/cases/{case_id}/upload-document",
        files={"file": ("malware.exe", b"binarycontent", "application/octet-stream")}
    )
    assert resp_bad_ext.status_code == 400
    assert "Unsupported file format" in resp_bad_ext.json()["detail"]

    # Reject empty file
    resp_empty = client.post(
        f"/cases/{case_id}/upload-document",
        files={"file": ("empty.png", b"", "image/png")}
    )
    assert resp_empty.status_code == 400
    assert "Uploaded file is empty" in resp_empty.json()["detail"]
