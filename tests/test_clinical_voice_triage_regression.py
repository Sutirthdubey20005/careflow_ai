import io
import json
import pytest
from unittest.mock import patch
from backend.services.triage_rules import DeterministicTriageEngine
from backend.services.transcription import transcription_service
from backend.services.questioning import AdaptiveQuestioningService
from backend.services.summarization import clinical_summarization_service
from backend.database.models import Case, Message, Evidence, AuditEvent


def test_hematemesis_vomiting_blood_p0_escalation():
    """Verify that vomiting blood / hematemesis in English, Hindi, and Hinglish triggers P0 escalation."""
    engine = DeterministicTriageEngine()

    test_cases = [
        ["I have been vomiting blood since this morning"],
        ["threw up blood twice today and feeling dizzy"],
        ["blood in my vomit"],
        ["मरीज को कल रात से खून की उल्टी हो रही है"],
        ["ulti me khoon aa raha hai bahut zyada"],
        ["khoon ki ulti ho rahi hai"]
    ]

    for raw_msgs in test_cases:
        res = engine.evaluate(evidence_state={}, raw_messages=raw_msgs)
        assert res.priority_category == "P0", f"Expected P0 for {raw_msgs}, got {res.priority_category}"
        assert any(t.rule_id == "RULE-P0-HEM-04" for t in res.triggered_rules)


def test_negation_safeguards_do_not_escalate():
    """Verify that explicit negations do not trigger acute P0/P1 emergency rules."""
    engine = DeterministicTriageEngine()

    negated_cases = [
        ["I have a mild runny nose and cough. No vomiting blood."],
        ["Patient denies any chest pain. Mild sore throat."],
        ["seene me dard nahi hai, bas thoda bukhar hai."],
        ["Mujhe ulti me khoon nahi aaya."],
        ["बुखार है लेकिन सीने में दर्द नहीं है।"]
    ]

    for raw_msgs in negated_cases:
        res = engine.evaluate(evidence_state={}, raw_messages=raw_msgs)
        assert res.priority_category != "P0", f"Did not expect P0 for negated statement: {raw_msgs}"
        assert not any(t.rule_id == "RULE-P0-HEM-04" for t in res.triggered_rules)

    # But if there is a positive emergency clause following a negated clause, it must trigger
    mixed_msg = ["I have no fever, but severe crushing chest pain radiating to my left arm"]
    res_mixed = engine.evaluate(evidence_state={}, raw_messages=mixed_msg)
    assert res_mixed.priority_category == "P0"


def test_symptom_specific_questioning_routing():
    """Verify adaptive questioning selects symptom-specific clinical questions across languages."""
    # 1. Hematemesis -> asks about volume and appearance
    q_out_hem = AdaptiveQuestioningService.generate_next_question(
        patient_messages=["I threw up blood this morning."],
        current_question_count=0,
        preferred_language="English"
    )
    assert q_out_hem.target_field == "hematemesis_appearance_and_volume"
    assert "coffee grounds" in q_out_hem.next_question.lower() or "bright red" in q_out_hem.next_question.lower()

    # 2. Chest pain in Hindi -> asks about radiation in Hindi
    q_out_cp_hi = AdaptiveQuestioningService.generate_next_question(
        patient_messages=["मुझे सीने में दर्द हो रहा है"],
        current_question_count=0,
        preferred_language="Hindi"
    )
    assert q_out_cp_hi.target_field == "pain_radiation"
    assert "बाएं हाथ" in q_out_cp_hi.next_question

    # 3. Chest pain in Hinglish -> asks about radiation in Hinglish
    q_out_cp_hing = AdaptiveQuestioningService.generate_next_question(
        patient_messages=["seene me dard ho raha hai kal se"],
        current_question_count=0,
        preferred_language="Hinglish"
    )
    assert q_out_cp_hing.target_field == "pain_radiation"
    assert "radiate" in q_out_cp_hing.next_question.lower() or "fail" in q_out_cp_hing.next_question.lower()

    # 4. Anti-repetition: if radiation question already asked, moves to associated symptoms
    q_out_cp_anti = AdaptiveQuestioningService.generate_next_question(
        patient_messages=["seene me dard ho raha hai kal se"],
        current_question_count=1,
        assistant_messages=["Does the chest pain radiate or spread to your left arm, shoulder, neck, jaw, or back?"],
        preferred_language="English"
    )
    assert q_out_cp_anti.target_field == "associated_cardiac_symptoms"


def test_clinical_summary_retains_evidence_facts():
    """Verify that clinical summary never marks structured evidence facts as 'Not reported'."""
    patient_msgs = ["I have had a severe headache since yesterday.", "10"]
    evidence_st = {
        "facts": {
            "severity": {"field": "severity", "value": "10/10", "status": "reported", "source_quote": "10"},
            "onset_and_duration": {"field": "onset_and_duration", "value": "since yesterday", "status": "reported"}
        }
    }

    summary_out = clinical_summarization_service.generate_summary(patient_msgs, evidence_state=evidence_st)
    assert summary_out.severity in ["10/10", "10", "Severe"]
    assert summary_out.severity != "Not reported"
    assert summary_out.onset_and_duration != "Not reported"


def test_transcription_language_parameter_normalization():
    """Verify that language strings are correctly mapped to ISO codes or None for auto-detect."""
    assert transcription_service._normalize_language("English") == "en"
    assert transcription_service._normalize_language("en") == "en"
    assert transcription_service._normalize_language("Hindi") == "hi"
    assert transcription_service._normalize_language("hi") == "hi"
    assert transcription_service._normalize_language("Hinglish") is None
    assert transcription_service._normalize_language("Auto-detect / Hinglish") is None
    assert transcription_service._normalize_language(None) is None


def test_document_upload_ocr_provenance_and_audit(client, db_session):
    """Verify document upload records engine_used, confidence, and audit event."""
    create_resp = client.post("/cases", json={"preferred_language": "English"})
    assert create_resp.status_code == 201
    case_id = create_resp.json()["id"]

    # Mock OCR extraction to return deterministic metadata
    fake_txt_bytes = b"Rx: Amoxicillin 500mg TID. Patient has severe acute bronchitis."
    files = {"file": ("rx_test.txt", fake_txt_bytes, "text/plain")}

    resp = client.post(f"/cases/{case_id}/upload-document", files=files)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "success"
    assert data["filename"] == "rx_test.txt"
    assert "engine_used" in data
    assert "confidence" in data
    assert data["char_count"] > 0

    # Verify audit event in DB
    db_session.expire_all()
    audit_row = db_session.query(AuditEvent).filter(
        AuditEvent.case_id == case_id,
        AuditEvent.event_type == "document_uploaded"
    ).first()
    assert audit_row is not None
    assert audit_row.actor_role == "patient"
    details = json.loads(audit_row.new_value_json)
    assert details["filename"] == "rx_test.txt"
    assert details["ocr_engine"] == data["engine_used"]
