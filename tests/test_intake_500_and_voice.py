import io
import json
import pytest
from unittest.mock import patch
from backend.database.models import Case, Message, Evidence
from backend.services.transcription import transcription_service, TranscriptionError


def test_empty_and_whitespace_message_validation(client):
    """Priority 1: Verify that empty and whitespace-only messages are rejected with 4xx, not 500."""
    create_resp = client.post("/cases", json={"preferred_language": "English"})
    assert create_resp.status_code == 201
    case_id = create_resp.json()["id"]

    # Empty string
    empty_resp = client.post(f"/cases/{case_id}/messages", json={"content": ""})
    assert empty_resp.status_code in [400, 422]

    # Whitespace-only string
    ws_resp = client.post(f"/cases/{case_id}/messages", json={"content": "    \n\t   "})
    assert ws_resp.status_code in [400, 422]


def test_intake_message_success_and_provenance(client, db_session):
    """Priority 1: Regression test reproducing CF-2026-8F9D42 message submission and verifying column persistence."""
    create_resp = client.post("/cases", json={"preferred_language": "English"})
    assert create_resp.status_code == 201
    case_id = create_resp.json()["id"]

    msg_payload = {
        "content": "I have had a mild headache and runny nose for 2 days.",
        "source_type": "patient_typed",
        "source_language": "en",
        "transcription_confidence": 1.0,
        "is_confirmed": True
    }
    resp = client.post(f"/cases/{case_id}/messages", json=msg_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["message"]["content"] == msg_payload["content"]
    assert data["message"]["source_type"] == "patient_typed"
    assert data["message"]["source_language"] == "en"

    # Verify directly in database that all columns were persisted
    db_session.expire_all()
    msg_row = db_session.query(Message).filter(Message.case_id == case_id).first()
    assert msg_row is not None
    assert msg_row.source_type == "patient_typed"
    assert msg_row.source_language == "en"
    assert msg_row.transcription_confidence == 1.0
    assert msg_row.is_confirmed is True


def test_multilingual_messages_english_hindi_hinglish(client, db_session):
    """Priority 1: Verify intake processes English, Hindi (Devanagari), and Hinglish messages cleanly."""
    create_resp = client.post("/cases", json={"preferred_language": "English"})
    case_id = create_resp.json()["id"]

    # 1. English
    en_resp = client.post(
        f"/cases/{case_id}/messages",
        json={"content": "I have had a severe sore throat and fever for 2 days.", "source_language": "en"}
    )
    assert en_resp.status_code == 200
    assert en_resp.json()["status"] == "success"

    # 2. Hindi (Devanagari)
    hi_resp = client.post(
        f"/cases/{case_id}/messages",
        json={"content": "मुझे दो दिन से बहुत तेज बुखार है और सिर में दर्द है।", "source_language": "hi"}
    )
    assert hi_resp.status_code == 200
    assert hi_resp.json()["status"] == "success"

    # 3. Hinglish
    hing_resp = client.post(
        f"/cases/{case_id}/messages",
        json={"content": "mujhe kal se pet me dard ho raha hai aur ulti jaisa lag raha hai.", "source_language": "hi"}
    )
    assert hing_resp.status_code == 200
    assert hing_resp.json()["status"] == "success"

    # Verify all 3 messages are preserved with exact content
    case_resp = client.get(f"/cases/{case_id}")
    assert case_resp.status_code == 200
    messages = case_resp.json()["messages"]
    assert len(messages) == 3
    assert "बहुत तेज बुखार" in messages[1]["content"]
    assert "pet me dard" in messages[2]["content"]


def test_model_extraction_failure_clean_rollback(client, db_session):
    """Priority 1: Verify extraction failure rolls back cleanly and marks case as ERR without 500 or PendingRollbackError."""
    create_resp = client.post("/cases", json={"preferred_language": "English"})
    case_id = create_resp.json()["id"]

    with patch("backend.services.extraction.ClinicalExtractionService.extract_clinical_facts") as mock_extract:
        mock_extract.side_effect = RuntimeError("Simulated MedGemma model crash during extraction")

        resp = client.post(
            f"/cases/{case_id}/messages",
            json={"content": "I feel dizzy and lightheaded."}
        )
        assert resp.status_code == 200

        db_session.expire_all()
        case_row = db_session.query(Case).filter(Case.id == case_id).first()
        assert case_row.ai_status == "failed"
        assert case_row.priority_category == "ERR"


def test_audio_transcription_validation(client):
    """Priority 2: Test audio file validation (empty audio, bad extensions, large files)."""
    # Empty file
    empty_file = ("recording.wav", io.BytesIO(b""), "audio/wav")
    resp_empty = client.post("/transcribe-audio", files={"file": empty_file})
    assert resp_empty.status_code == 400
    assert "empty" in resp_empty.json()["detail"].lower()

    # Unsupported format
    bad_ext_file = ("payload.exe", io.BytesIO(b"not_audio"), "application/octet-stream")
    resp_bad = client.post("/transcribe-audio", files={"file": bad_ext_file})
    assert resp_bad.status_code == 400
    assert "unsupported" in resp_bad.json()["detail"].lower()


def test_audio_transcription_language_normalization(client):
    """Priority 2: Verify language normalization handles 'Hindi', 'English', 'Hinglish', 'auto' without ValueError."""
    # Test with mock synthetic audio
    dummy_wav = b"MOCK_TRANSCRIPT: I have chest discomfort"
    test_cases = [
        ("English", "en"),
        ("en", "en"),
        ("Hindi", "en"),  # Audio has English text
        ("hi", "en"),
        ("Hinglish", "en"),
        ("auto", "en"),
        (None, "en")
    ]

    for lang_param, expected_detected in test_cases:
        params = {"language": lang_param} if lang_param else {}
        resp = client.post(
            "/transcribe-audio",
            files={"file": ("recording.wav", io.BytesIO(dummy_wav), "audio/wav")},
            params=params
        )
        assert resp.status_code == 200, f"Failed for language: {lang_param}"
        data = resp.json()
        assert "transcript" in data
        assert "detected_language" in data
        assert "confidence" in data


def test_audio_transcription_tiny_and_base_models(client):
    """Priority 2: Verify configurable ASR model sizes 'tiny' and 'base'."""
    dummy_audio = b"MOCK_TRANSCRIPT: Severe headache for two days"
    for m_size in ["tiny", "base"]:
        resp = client.post(
            "/transcribe-audio",
            files={"file": ("recording.wav", io.BytesIO(dummy_audio), "audio/wav")},
            params={"model_size": m_size}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_size"] == m_size
        assert "Severe headache" in data["transcript"]


def test_voice_transcription_patient_confirmation_flow(client, db_session):
    """Priority 2: End-to-end voice flow: transcribe -> review/edit -> confirm & send -> verify evidence."""
    # 1. Create case
    create_resp = client.post("/cases", json={"preferred_language": "English"})
    case_id = create_resp.json()["id"]

    # 2. Transcribe voice audio
    audio_content = b"MOCK_TRANSCRIPT: I have sharp abdominal pain"
    transcribe_resp = client.post(
        "/transcribe-audio",
        files={"file": ("voice_note.wav", io.BytesIO(audio_content), "audio/wav")},
        params={"language": "en", "model_size": "base"}
    )
    assert transcribe_resp.status_code == 200
    t_data = transcribe_resp.json()
    raw_transcript = t_data["transcript"]
    confidence = t_data["confidence"]
    assert "sharp abdominal pain" in raw_transcript

    # 3. Patient edits the transcript before confirmation
    edited_transcript = raw_transcript + " that started 3 hours ago."

    # 4. Patient confirms and submits as voice message
    send_resp = client.post(
        f"/cases/{case_id}/messages",
        json={
            "content": edited_transcript,
            "source_type": "voice",
            "source_language": "en",
            "transcription_confidence": confidence,
            "is_confirmed": True
        }
    )
    assert send_resp.status_code == 200
    send_data = send_resp.json()
    assert send_data["status"] == "success"

    # 5. Verify database record preserves voice provenance and confidence
    db_session.expire_all()
    msg = db_session.query(Message).filter(Message.case_id == case_id).first()
    assert msg.source_type == "voice"
    assert msg.is_confirmed is True
    assert msg.transcription_confidence == confidence
    assert "3 hours ago" in msg.content


def test_real_faster_whisper_tiny_local_transcription():
    """Priority 2: Benchmark and verify real local faster-whisper execution with 'tiny' model."""
    pytest.importorskip("faster_whisper", reason="faster-whisper is not installed in this environment")
    import sys
    import subprocess

    code = """
import os, io, wave
from backend.services.transcription import transcription_service
from backend.config import settings

os.environ["ASR_MODE"] = "local"
settings.ASR_MODE = "local"
buf = io.BytesIO()
with wave.open(buf, "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(16000)
    wav.writeframes(b"\\x00" * 32000)
pcm_wav = buf.getvalue()

res = transcription_service.transcribe_audio(
    file_bytes=pcm_wav,
    filename="test_audio.wav",
    model_size="tiny"
)
assert res.is_mock is False
assert res.model_size == "tiny"
assert res.duration_seconds > 0.0
assert res.confidence is not None
print("REAL_WHISPER_SUCCESS")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=45
    )
    assert result.returncode == 0, f"Process failed: {result.stderr}\n{result.stdout}"
    assert "REAL_WHISPER_SUCCESS" in result.stdout


def test_full_multimodal_patient_and_reviewer_journey(client):
    """Priority 3: Comprehensive end-to-end user journey across languages, modalities, and reviewer actions."""
    # 1. Patient creates case in Hindi
    create_resp = client.post(
        "/cases",
        json={"preferred_language": "Hindi", "chief_complaint": "मुझे दो दिन से तेज़ बुखार है"}
    )
    assert create_resp.status_code == 201
    case_id = create_resp.json()["id"]

    # 2. Patient sends first message in Hindi
    msg1_resp = client.post(
        f"/cases/{case_id}/messages",
        json={
            "content": "मुझे दो दिन से बहुत तेज बुखार है और बदन दर्द है।",
            "source_type": "patient_typed",
            "source_language": "hi",
            "is_confirmed": True
        }
    )
    assert msg1_resp.status_code == 200
    assert msg1_resp.json()["status"] == "success"

    # 3. Patient sends voice recording (transcribe -> confirm)
    voice_bytes = b"MOCK_TRANSCRIPT: pet me halka dard hai"
    t_resp = client.post(
        "/transcribe-audio",
        files={"file": ("note.wav", io.BytesIO(voice_bytes), "audio/wav")},
        params={"language": "Hindi", "model_size": "base"}
    )
    assert t_resp.status_code == 200
    t_data = t_resp.json()
    assert "pet me halka dard" in t_data["transcript"]

    # Confirm voice transcript
    msg2_resp = client.post(
        f"/cases/{case_id}/messages",
        json={
            "content": t_data["transcript"],
            "source_type": "voice",
            "source_language": "hi",
            "transcription_confidence": t_data["confidence"],
            "is_confirmed": True
        }
    )
    assert msg2_resp.status_code == 200

    # 4. Patient uploads lab report (OCR preview -> confirm)
    doc_bytes = b"CBC REPORT: WBC Count 11,500 /mcL (Elevated). Platelet Count: 250,000."
    doc_preview_resp = client.post(
        f"/cases/{case_id}/extract-document",
        files={"file": ("lab_report.txt", io.BytesIO(doc_bytes), "text/plain")}
    )
    assert doc_preview_resp.status_code == 200
    preview_data = doc_preview_resp.json()
    assert "11,500" in preview_data["extracted_text"]

    # Confirm document text
    msg3_resp = client.post(
        f"/cases/{case_id}/messages",
        json={
            "content": f"[Verified Document: lab_report.txt]\n{preview_data['extracted_text']}",
            "source_type": "ocr",
            "transcription_confidence": preview_data["confidence"],
            "is_confirmed": True
        }
    )
    assert msg3_resp.status_code == 200

    # 5. Patient answers follow-up in Hinglish
    msg4_resp = client.post(
        f"/cases/{case_id}/messages",
        json={
            "content": "koi ulti ya dast nahi hai, sirf thoda chakkar aa raha hai.",
            "source_type": "patient_typed",
            "source_language": "hi",
            "is_confirmed": True
        }
    )
    assert msg4_resp.status_code == 200

    # 6. Patient finishes and submits intake
    submit_resp = client.post(f"/cases/{case_id}/submit")
    assert submit_resp.status_code == 200
    submit_data = submit_resp.json()
    assert submit_data["status"] == "submitted"
    assert submit_data["priority_category"] in ["P0", "P1", "P2", "P3"]

    # 7. Clinician Reviewer queries the triage queue
    queue_resp = client.get("/cases")
    assert queue_resp.status_code == 200
    cases_in_queue = queue_resp.json()
    assert any(c["id"] == case_id for c in cases_in_queue)

    # 8. Reviewer opens full case record
    case_detail_resp = client.get(f"/cases/{case_id}")
    assert case_detail_resp.status_code == 200
    detail = case_detail_resp.json()

    # Verify message modalities and provenance
    messages = detail["messages"]
    assert len(messages) >= 4
    source_types = [m.get("source_type") for m in messages]
    assert "patient_typed" in source_types
    assert "voice" in source_types
    assert "ocr" in source_types

    # Verify audit events captured all transitions
    assert len(detail["audit_events"]) >= 5

    # 9. Reviewer approves the case
    review_resp = client.post(
        f"/cases/{case_id}/review",
        json={
            "reviewer_summary": "Approved after reviewing multimodal findings.",
            "reviewer_priority": detail["priority_category"],
            "reviewer_notes": "Prescribed supportive care and monitoring.",
            "approve": True
        }
    )
    assert review_resp.status_code == 200
    assert review_resp.json()["review_status"] == "approved"


def test_voice_raw_transcript_provenance_and_patient_editing(client, db_session):
    """Verify that when a patient edits a voice transcript, both raw_transcript and edited content are preserved."""
    create_resp = client.post("/cases", json={"preferred_language": "English"})
    assert create_resp.status_code == 201
    case_id = create_resp.json()["id"]

    raw_asr = "I have severe chest pain and radiating arm numbness"
    edited_text = "I have had severe chest pain radiating to my left arm since this morning"

    msg_resp = client.post(
        f"/cases/{case_id}/messages",
        json={
            "content": edited_text,
            "raw_transcript": raw_asr,
            "source_type": "voice",
            "source_language": "en",
            "transcription_confidence": 0.94,
            "is_confirmed": True
        }
    )
    assert msg_resp.status_code == 200
    msg_data = msg_resp.json()["message"]
    assert msg_data["content"] == edited_text
    assert msg_data["raw_transcript"] == raw_asr
    assert msg_data["source_type"] == "voice"

    # Verify directly in SQLite DB
    db_session.expire_all()
    db_msg = db_session.query(Message).filter(Message.case_id == case_id).first()
    assert db_msg is not None
    assert db_msg.content == edited_text
    assert db_msg.raw_transcript == raw_asr

    # Verify case GET endpoint returns raw_transcript
    case_resp = client.get(f"/cases/{case_id}")
    assert case_resp.status_code == 200
    fetched_msg = case_resp.json()["messages"][0]
    assert fetched_msg["content"] == edited_text
    assert fetched_msg["raw_transcript"] == raw_asr


def test_priority_escalation_preserved_on_final_submit(client, db_session):
    """Verify that an interim P0 or P1 emergency escalation is never downgraded upon final case submission."""
    create_resp = client.post("/cases", json={"preferred_language": "English"})
    case_id = create_resp.json()["id"]

    # Message 1 triggers P0 (chest pain radiating to left arm)
    msg1_resp = client.post(
        f"/cases/{case_id}/messages",
        json={
            "content": "I have sudden crushing chest pain radiating to my left arm and neck.",
            "source_type": "patient_typed"
        }
    )
    assert msg1_resp.status_code == 200
    assert msg1_resp.json()["interim_priority"] == "P0"

    # Verify case priority is now P0
    case_resp = client.get(f"/cases/{case_id}")
    assert case_resp.json()["priority_category"] == "P0"

    # Message 2 is a mild statement without emergency keywords
    msg2_resp = client.post(
        f"/cases/{case_id}/messages",
        json={
            "content": "No fever or cough.",
            "source_type": "patient_typed"
        }
    )
    assert msg2_resp.status_code == 200

    # Submit case - should preserve P0 and NOT downgrade
    submit_resp = client.post(f"/cases/{case_id}/submit")
    assert submit_resp.status_code == 200
    submit_data = submit_resp.json()
    assert submit_data["priority_category"] == "P0"

    # Double check database persistence
    db_session.expire_all()
    case_row = db_session.query(Case).filter(Case.id == case_id).first()
    assert case_row.priority_category == "P0"


def test_ai_failure_defaults_to_err_never_p3(client, db_session):
    """Verify that if AI extraction/summary fails, final triage defaults to ERR (never P3)."""
    create_resp = client.post("/cases", json={"preferred_language": "English"})
    case_id = create_resp.json()["id"]

    # Send a routine complaint
    client.post(
        f"/cases/{case_id}/messages",
        json={"content": "I have had a mild cough for three days.", "source_type": "patient_typed"}
    )

    # Force AI failure during summarization
    with patch("backend.services.summarization.ClinicalSummarizationService.generate_summary") as mock_sum:
        mock_sum.side_effect = RuntimeError("MedGemma endpoint unavailable")
        submit_resp = client.post(f"/cases/{case_id}/submit")
        assert submit_resp.status_code == 200
        submit_data = submit_resp.json()
        assert submit_data["ai_status"] == "failed"
        assert submit_data["priority_category"] == "ERR"


def test_normalize_language_hinglish_and_auto_detection():
    """Verify _normalize_language handles Hinglish, auto, and case variations without forcing wrong language."""
    assert transcription_service._normalize_language("Hinglish") is None
    assert transcription_service._normalize_language("hinglish") is None
    assert transcription_service._normalize_language("mixed") is None
    assert transcription_service._normalize_language("auto") is None
    assert transcription_service._normalize_language("Automatic") is None
    assert transcription_service._normalize_language(None) is None
    assert transcription_service._normalize_language("") is None
    assert transcription_service._normalize_language("English") == "en"
    assert transcription_service._normalize_language("hindi") == "hi"
    assert transcription_service._normalize_language("HI") == "hi"


def test_no_stale_transcript_reuse_on_new_audio(client):
    """Verify that transcribing distinct audio inputs yields distinct transcripts and doesn't return stale text."""
    # Audio 1: Chest pain
    audio1 = b"MOCK_TRANSCRIPT: severe chest pain radiating to jaw"
    resp1 = client.post(
        "/transcribe-audio",
        files={"file": ("cardiac.wav", io.BytesIO(audio1), "audio/wav")}
    )
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert "chest pain" in data1["transcript"].lower()

    # Audio 2: Cough & cold
    audio2 = b"MOCK_TRANSCRIPT: mild fever and dry cough for 3 days"
    resp2 = client.post(
        "/transcribe-audio",
        files={"file": ("cough.wav", io.BytesIO(audio2), "audio/wav")}
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert "dry cough" in data2["transcript"].lower()

    # Transcripts must not be identical
    assert data1["transcript"] != data2["transcript"]


