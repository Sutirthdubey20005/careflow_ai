import uuid
import json
import logging
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status, UploadFile, File, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc
from backend.services.ocr_service import ocr_service
from backend.services.transcription import transcription_service, TranscriptionError
from backend.schemas.multilingual import TranscriptionResult, DocumentExtractionResult

from backend.database.db import get_db
from backend.database.models import Case, Message, Evidence, AuditEvent
from backend.schemas.clinical import (
    CaseCreateRequest,
    MessageCreateRequest,
    CaseReviewRequest,
    PriorityOverrideRequest,
    CaseListItemResponse,
    HealthResponse,
    FollowUpQuestionOutput,
    TriageResult
)
from backend.services.medgemma import medgemma_service, ModelInferenceError
from backend.services.extraction import clinical_extraction_service
from backend.services.questioning import adaptive_questioning_service
from backend.services.summarization import clinical_summarization_service
from backend.services.triage_rules import triage_engine
from backend.services.evidence_state import EvidenceStateManager
from backend.config import settings

logger = logging.getLogger("careflow.api")

router = APIRouter()

def _generate_case_id() -> str:
    """Generates a non-identifying random Case ID, e.g. CF-2026-A1B2."""
    current_year = datetime.utcnow().year
    suffix = uuid.uuid4().hex[:6].upper()
    return f"CF-{current_year}-{suffix}"


# -------------------------------------------------------------
# Health Check
# -------------------------------------------------------------
@router.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)):
    """Reports backend, database, and MedGemma readiness."""
    db_status = "ok"
    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {e}"

    model_info = medgemma_service.get_status()

    return HealthResponse(
        status="healthy" if db_status == "ok" else "degraded",
        database=db_status,
        medgemma_mode=model_info["mode"],
        medgemma_status="ready" if model_info["is_ready"] else "unavailable",
        triage_rule_version=settings.TRIAGE_RULE_VERSION
    )


# -------------------------------------------------------------
# Case Management
# -------------------------------------------------------------
@router.post("/cases", status_code=status.HTTP_201_CREATED)
def create_case(req: CaseCreateRequest, db: Session = Depends(get_db)):
    """Creates a new patient intake case with a random, non-identifying ID."""
    case_id = _generate_case_id()
    init_priority = "P3"
    init_reasons = []
    init_missing = []
    init_uncertainties = []
    if req.chief_complaint and req.chief_complaint.strip():
        initial_triage = triage_engine.evaluate(
            evidence_state={"facts": {"chief_complaint": req.chief_complaint.strip()}},
            raw_messages=[req.chief_complaint.strip()],
            ai_status="ready"
        )
        init_priority = initial_triage.priority_category
        init_reasons = initial_triage.reasons
        init_missing = initial_triage.missing_information
        init_uncertainties = initial_triage.uncertainties

    new_case = Case(
        id=case_id,
        preferred_language=req.preferred_language,
        chief_complaint=req.chief_complaint,
        ai_status="pending",
        priority_category=init_priority,
        priority_reasons_json=json.dumps(init_reasons) if init_reasons else None,
        missing_information_json=json.dumps(init_missing) if init_missing else None,
        uncertainties_json=json.dumps(init_uncertainties) if init_uncertainties else None,
        review_status="pending",
        triage_rule_version=settings.TRIAGE_RULE_VERSION,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    db.add(new_case)

    # Initial audit log
    audit = AuditEvent(
        case_id=case_id,
        event_type="case_created",
        new_value_json=json.dumps({
            "preferred_language": req.preferred_language,
            "chief_complaint": req.chief_complaint,
            "initial_priority": init_priority
        }),
        reason="Patient initiated new intake session",
        actor_role="patient",
        created_at=datetime.utcnow()
    )
    db.add(audit)
    db.commit()
    db.refresh(new_case)

    return new_case.to_dict()


@router.get("/cases")
def list_cases(
    priority: Optional[str] = Query(None, description="Filter by priority (P0, P1, P2, P3, ERR)"),
    review_status: Optional[str] = Query(None, description="Filter by review status (pending, reviewed, approved, reopened)"),
    has_uncertainty: Optional[bool] = Query(None, description="Filter cases with unresolved uncertainty or conflicts"),
    db: Session = Depends(get_db)
):
    """
    Lists patient cases in priority order: P0 -> P1 -> P2 -> P3 -> ERR.
    Within each priority, oldest cases (longest wait times) are listed first.
    """
    query = db.query(Case)

    if priority:
        query = query.filter(Case.priority_category == priority)
    if review_status:
        query = query.filter(Case.review_status == review_status)

    all_cases = query.all()

    # Priority sort ranking
    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "ERR": 4}

    result = []
    now = datetime.utcnow()

    for c in all_cases:
        uncertainties = json.loads(c.uncertainties_json or "[]")
        evidence_st = json.loads(c.evidence_state_json or "{}")
        contradictions = evidence_st.get("contradictions", [])
        has_u_or_c = bool(uncertainties or contradictions)

        if has_uncertainty is not None and has_u_or_c != has_uncertainty:
            continue

        missing_info = json.loads(c.missing_information_json or "[]")
        waiting_mins = (now - c.created_at).total_seconds() / 60.0 if c.created_at else 0.0

        item = {
            "id": c.id,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
            "priority_category": c.priority_category,
            "review_status": c.review_status,
            "chief_complaint": c.chief_complaint or "Not entered",
            "has_missing_info": bool(missing_info),
            "has_uncertainty_or_conflict": has_u_or_c,
            "ai_status": c.ai_status,
            "waiting_minutes": round(waiting_mins, 1)
        }
        result.append(item)

    # Sort by priority rank, then by oldest created_at (ascending)
    result.sort(key=lambda x: (
        priority_order.get(x["priority_category"], 99),
        x["created_at"] or ""
    ))

    return result


@router.get("/cases/{case_id}")
def get_case(case_id: str, db: Session = Depends(get_db)):
    """Fetches full case details including messages, atomic evidence items, and audit history."""
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    case_dict = case.to_dict()
    case_dict["messages"] = [m.to_dict() for m in case.messages]
    case_dict["evidence_items"] = [e.to_dict() for e in case.evidence_items]
    case_dict["audit_events"] = [a.to_dict() for a in case.audit_events]

    return case_dict


# -------------------------------------------------------------
# Multilingual Audio Transcription & Document OCR Preview
# -------------------------------------------------------------
class TextToSpeechRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text content to synthesize to spoken audio")
    model: Optional[str] = Field("aura-asteria-en", description="Deepgram Aura TTS model")


@router.post("/transcribe-audio", response_model=TranscriptionResult)
async def transcribe_audio_endpoint(
    file: UploadFile = File(...),
    language: Optional[str] = Query(None, description="Optional ISO language code, e.g. 'en', 'hi', 'en-IN'"),
    model_size: Optional[str] = Query(None, description="ASR model: 'nova-2', 'base', 'tiny'"),
    translate_to_english: bool = Query(False, description="Translate speech to English via Deepgram")
):
    """
    Transcribes audio input into text with detected language and confidence.
    Does NOT automatically create a message or modify clinical state.
    Allows patient review, editing, and explicit confirmation before submission.
    """
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    try:
        result = transcription_service.transcribe_audio(
            file_bytes=file_bytes,
            filename=file.filename or "recording.wav",
            language=language,
            model_size=model_size,
            translate_to_english=translate_to_english
        )
        return result
    except TranscriptionError as te:
        raise HTTPException(status_code=400, detail=str(te))
    except Exception as e:
        logger.error(f"Audio transcription service error: {e}")
        raise HTTPException(status_code=500, detail=f"Audio transcription failed: {e}")


@router.post("/text-to-speech")
def text_to_speech_endpoint(req: TextToSpeechRequest):
    """
    Synthesizes speech audio from clinical text using Deepgram TTS (Aura model).
    Returns audio/mp3 stream.
    """
    try:
        audio_bytes = transcription_service.text_to_speech(req.text, model=req.model)
        return Response(content=audio_bytes, media_type="audio/mp3")
    except TranscriptionError as te:
        raise HTTPException(status_code=400, detail=str(te))
    except Exception as e:
        logger.error(f"Text to speech synthesis error: {e}")
        raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {e}")


@router.post("/cases/{case_id}/extract-document", response_model=DocumentExtractionResult)
async def preview_document_extraction(
    case_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Extracts text from an uploaded medical report or prescription image/PDF
    WITHOUT committing it to the clinical case transcript.
    Enables patient review and editing of extracted findings before ingestion.
    """
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    allowed_extensions = {"png", "jpg", "jpeg", "webp", "bmp", "tiff", "pdf", "txt", "csv", "json"}
    ext = file.filename.lower().split(".")[-1] if file.filename and "." in file.filename else ""
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed extensions: {', '.join(sorted(allowed_extensions))}"
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(file_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Uploaded file exceeds maximum allowed size (15 MB).")

    result = ocr_service.extract_with_metadata(file_bytes, file.filename)
    return result


# -------------------------------------------------------------
# Conversational Intake & Extraction
# -------------------------------------------------------------
@router.post("/cases/{case_id}/messages")
def add_patient_message(case_id: str, req: MessageCreateRequest, db: Session = Depends(get_db)):
    """
    Appends a patient message, runs clinical extraction, updates evidence state,
    and runs interim triage to detect emergency red flags immediately.
    Preserves source modality (typed, voice_transcript, ocr_extraction) and confidence.
    """
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    content_clean = req.content.strip() if req.content else ""
    if not content_clean:
        raise HTTPException(status_code=400, detail="Message content cannot be empty or whitespace only.")

    # Store verbatim message in SQLite with modality provenance
    msg = Message(
        case_id=case_id,
        role="patient",
        content=content_clean,
        raw_transcript=req.raw_transcript,
        source_type=req.source_type or "patient_typed",
        source_language=req.source_language,
        transcription_confidence=req.transcription_confidence,
        is_confirmed=req.is_confirmed if req.is_confirmed is not None else True,
        created_at=datetime.utcnow()
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)

    # Set chief complaint if empty
    if not case.chief_complaint:
        case.chief_complaint = content_clean

    # Collect all patient messages so far
    all_msgs = [m.content for m in case.messages if m.role == "patient"]

    # Run clinical fact extraction via MedGemma
    try:
        extraction = clinical_extraction_service.extract_clinical_facts(all_msgs)
        # Update persistent evidence state
        EvidenceStateManager.update_evidence_state(
            db=db,
            case=case,
            extraction=extraction,
            source_message=msg,
            actor_role="ai_system"
        )
        case.ai_status = "ready"
        db.commit()
    except Exception as e:
        logger.error(f"Error during extraction for case {case_id}: {e}")
        db.rollback()
        case = db.query(Case).filter(Case.id == case_id).first()
        if case:
            case.ai_status = "failed"
            case.priority_category = "ERR"
            case.priority_reasons_json = json.dumps([f"Clinical extraction error: {type(e).__name__}"])
            db.commit()

    # Interim Red-Flag Check: Emergency warning must NOT be delayed while questioning continues
    current_evidence = json.loads(case.evidence_state_json or "{}")
    triage_res = triage_engine.evaluate(
        evidence_state=current_evidence,
        raw_messages=all_msgs,
        ai_status=case.ai_status
    )
    
    # Escalation hierarchy safeguard: P0 cannot be downgraded; P1 cannot be downgraded to P2/P3; P2 cannot be downgraded to P3
    new_p = triage_res.priority_category
    curr_p = case.priority_category

    if curr_p == "P0" or new_p == "P0":
        case.priority_category = "P0"
    elif curr_p == "P1" or new_p == "P1":
        case.priority_category = "P1"
    elif curr_p == "P2" or new_p == "P2":
        case.priority_category = "P2"
    elif new_p == "ERR" or case.ai_status == "failed":
        case.priority_category = "ERR"
    else:
        case.priority_category = new_p

    case.priority_reasons_json = json.dumps(triage_res.reasons)
    case.missing_information_json = json.dumps(triage_res.missing_information)
    case.uncertainties_json = json.dumps(triage_res.uncertainties)
    case.updated_at = datetime.utcnow()
    db.commit()

    return {
        "status": "success",
        "message": msg.to_dict(),
        "interim_priority": triage_res.priority_category,
        "warning_signs": triage_res.warning_signs,
        "is_emergency": triage_res.priority_category == "P0"
    }


@router.post("/cases/{case_id}/upload-document")
async def upload_medical_document(
    case_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Accepts an uploaded PDF, prescription, or medical image,
    runs OCR / text extraction, appends it as a patient message,
    and updates evidence state with clinical fact extraction.
    """
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    # Validate file extension
    allowed_extensions = {"png", "jpg", "jpeg", "webp", "bmp", "tiff", "pdf", "txt", "csv", "json"}
    ext = file.filename.lower().split(".")[-1] if file.filename and "." in file.filename else ""
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed extensions: {', '.join(sorted(allowed_extensions))}"
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(file_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Uploaded file exceeds maximum allowed size (15 MB).")

    # Run OCR or PDF text extraction with full provenance metadata
    ocr_res = ocr_service.extract_with_metadata(file_bytes, file.filename)
    extracted_text = ocr_res.extracted_text
    formatted_content = f"[Uploaded Medical Document ({file.filename}) OCR]:\n{extracted_text}"

    # Store in messages
    msg = Message(
        case_id=case_id,
        role="patient",
        content=formatted_content,
        source_type="ocr_extraction",
        transcription_confidence=ocr_res.confidence,
        is_confirmed=True,
        created_at=datetime.utcnow()
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)

    # Record Document Upload Audit Event
    doc_audit = AuditEvent(
        case_id=case.id,
        event_type="document_uploaded",
        new_value_json=json.dumps({
            "filename": file.filename,
            "char_count": len(extracted_text),
            "ocr_engine": ocr_res.engine_used,
            "confidence": ocr_res.confidence,
            "preview": extracted_text[:500]
        }),
        reason=f"Patient uploaded medical document '{file.filename}' processed via {ocr_res.engine_used}",
        actor_role="patient",
        created_at=datetime.utcnow()
    )
    db.add(doc_audit)
    db.commit()

    # Collect all patient messages so far
    all_msgs = [m.content for m in case.messages if m.role == "patient"]

    # Run clinical extraction
    try:
        extraction = clinical_extraction_service.extract_clinical_facts(all_msgs)
        EvidenceStateManager.update_evidence_state(
            db=db,
            case=case,
            extraction=extraction,
            source_message=msg,
            actor_role="ai_system"
        )
        case.ai_status = "ready"
        db.commit()
    except Exception as e:
        logger.error(f"Error during extraction for uploaded document in case {case_id}: {e}")
        db.rollback()
        case = db.query(Case).filter(Case.id == case_id).first()
        if case:
            case.ai_status = "failed"
            case.priority_category = "ERR"
            case.priority_reasons_json = json.dumps([f"Document extraction error: {type(e).__name__}"])
            db.commit()

    # Interim Red-Flag Check
    current_evidence = json.loads(case.evidence_state_json or "{}")
    triage_res = triage_engine.evaluate(
        evidence_state=current_evidence,
        raw_messages=all_msgs,
        ai_status=case.ai_status
    )
    # Escalation hierarchy safeguard: P0 cannot be downgraded; P1 cannot be downgraded to P2/P3; P2 cannot be downgraded to P3
    new_p = triage_res.priority_category
    curr_p = case.priority_category

    if curr_p == "P0" or new_p == "P0":
        case.priority_category = "P0"
    elif curr_p == "P1" or new_p == "P1":
        case.priority_category = "P1"
    elif curr_p == "P2" or new_p == "P2":
        case.priority_category = "P2"
    elif new_p == "ERR" or case.ai_status == "failed":
        case.priority_category = "ERR"
    else:
        case.priority_category = new_p

    case.priority_reasons_json = json.dumps(triage_res.reasons)
    case.missing_information_json = json.dumps(triage_res.missing_information)
    case.uncertainties_json = json.dumps(triage_res.uncertainties)
    case.updated_at = datetime.utcnow()
    db.commit()

    return {
        "status": "success",
        "filename": file.filename,
        "extracted_text": extracted_text,
        "char_count": len(extracted_text),
        "engine_used": ocr_res.engine_used,
        "confidence": ocr_res.confidence,
        "interim_priority": triage_res.priority_category,
        "warning_signs": triage_res.warning_signs,
        "is_emergency": triage_res.priority_category == "P0"
    }


@router.post("/cases/{case_id}/next-question")
def get_next_question(case_id: str, db: Session = Depends(get_db)):
    """
    Generates the single next clinically relevant follow-up question.
    """
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    patient_msgs = [m.content for m in case.messages if m.role == "patient"]
    assistant_msgs = [m.content for m in case.messages if m.role == "assistant"]
    turn_count = len(assistant_msgs)

    evidence_st = json.loads(case.evidence_state_json or "{}")

    # Generate question with anti-repetition tracking
    result = adaptive_questioning_service.generate_next_question(
        patient_messages=patient_msgs,
        current_question_count=turn_count,
        assistant_messages=assistant_msgs,
        evidence_state=evidence_st,
        preferred_language=case.preferred_language or "English"
    )

    # If a question was generated, store it in messages
    if result.next_question:
        asst_msg = Message(
            case_id=case_id,
            role="assistant",
            content=result.next_question,
            created_at=datetime.utcnow()
        )
        db.add(asst_msg)
        db.commit()

    return result.model_dump()


# -------------------------------------------------------------
# Submission & Summary Generation
# -------------------------------------------------------------
@router.post("/cases/{case_id}/submit")
def submit_case(case_id: str, db: Session = Depends(get_db)):
    """
    Finalizes patient intake, generates clinical summary via MedGemma,
    applies the deterministic triage engine, and places the case in reviewer queue.
    """
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    patient_msgs = [m.content for m in case.messages if m.role == "patient"]
    if not patient_msgs:
        raise HTTPException(status_code=400, detail="Cannot submit a case with no patient messages")

    # 1. Load Evidence State Facts
    evidence_st = json.loads(case.evidence_state_json or "{}")

    # 2. Generate Clinical Summary with preserved structured evidence facts
    try:
        summary_obj = clinical_summarization_service.generate_summary(patient_msgs, evidence_state=evidence_st)
        case.ai_summary_json = json.dumps(summary_obj.model_dump())
        case.ai_status = "ready"
    except Exception as e:
        logger.error(f"Failed to generate summary on submit for case {case_id}: {e}")
        case.ai_status = "failed"
        case.ai_summary_json = json.dumps({
            "chief_complaint": case.chief_complaint or "Intake submitted",
            "summary_text": "AI clinical summary generation failed. Case requires manual clinical assessment.",
            "missing_critical_information": ["Full AI summary"],
            "contradictions": [],
            "uncertainties": ["Model unavailable"]
        })

    # 3. Apply Deterministic Triage Rules
    triage_res = triage_engine.evaluate(
        evidence_state=evidence_st,
        raw_messages=patient_msgs,
        ai_status=case.ai_status
    )

    # 3. Store Triage & Queue Attributes with Escalation Preservation
    final_priority = triage_res.priority_category
    final_reasons = list(triage_res.reasons)

    if case.ai_status == "failed" or triage_res.priority_category == "ERR":
        final_priority = "ERR"
    else:
        # P0/P1 emergency escalation cannot be downgraded upon final case submission
        existing_p = case.priority_category
        existing_reasons = json.loads(case.priority_reasons_json or "[]")
        if existing_p == "P0" and final_priority != "P0":
            final_priority = "P0"
            final_reasons = existing_reasons + [r for r in final_reasons if r not in existing_reasons]
        elif existing_p == "P1" and final_priority in ("P2", "P3"):
            final_priority = "P1"
            final_reasons = existing_reasons + [r for r in final_reasons if r not in existing_reasons]
        elif existing_p == "P2" and final_priority == "P3" and (triage_res.uncertainties or evidence_st.get("contradictions")):
            final_priority = "P2"
            final_reasons = existing_reasons + [r for r in final_reasons if r not in existing_reasons]

    case.priority_category = final_priority
    case.priority_reasons_json = json.dumps(final_reasons)
    case.missing_information_json = json.dumps(triage_res.missing_information)
    case.uncertainties_json = json.dumps(triage_res.uncertainties)
    case.review_status = "pending"
    case.updated_at = datetime.utcnow()

    # 4. Audit Log
    audit = AuditEvent(
        case_id=case.id,
        event_type="intake_submitted",
        new_value_json=json.dumps({
            "priority": triage_res.priority_category,
            "reasons": triage_res.reasons,
            "ai_status": case.ai_status
        }),
        reason="Patient submitted intake for healthcare professional review",
        actor_role="patient",
        created_at=datetime.utcnow()
    )
    db.add(audit)
    db.commit()
    db.refresh(case)

    return {
        "status": "submitted",
        "case_id": case.id,
        "priority_category": case.priority_category,
        "ai_status": case.ai_status,
        "case": case.to_dict()
    }


@router.post("/cases/{case_id}/generate-summary")
def regenerate_summary(case_id: str, db: Session = Depends(get_db)):
    """Regenerates the draft clinical summary via MedGemma."""
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    patient_msgs = [m.content for m in case.messages if m.role == "patient"]
    try:
        summary_obj = clinical_summarization_service.generate_summary(patient_msgs)
        case.ai_summary_json = json.dumps(summary_obj.model_dump())
        case.ai_status = "ready"
        db.commit()
        return summary_obj.model_dump()
    except Exception as e:
        logger.error(f"Error regenerating summary for {case_id}: {e}")
        case.ai_status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {e}")


# -------------------------------------------------------------
# Reviewer Controls & Overrides
# -------------------------------------------------------------
@router.post("/cases/{case_id}/review")
def review_case(case_id: str, req: CaseReviewRequest, db: Session = Depends(get_db)):
    """
    Saves reviewer edits, notes, and approval status.
    Maintains the original AI summary intact and records audit log.
    """
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    old_state = {
        "review_status": case.review_status,
        "reviewer_summary": case.reviewer_summary,
        "reviewer_priority": case.reviewer_priority,
        "reviewer_notes": case.reviewer_notes
    }

    if req.reviewer_summary is not None:
        case.reviewer_summary = req.reviewer_summary
    if req.reviewer_priority is not None:
        case.reviewer_priority = req.reviewer_priority
    if req.reviewer_notes is not None:
        case.reviewer_notes = req.reviewer_notes

    if req.approve:
        case.review_status = "approved"
        case.reviewed_at = datetime.utcnow()
    else:
        case.review_status = "reviewed"
        case.reviewed_at = datetime.utcnow()

    case.updated_at = datetime.utcnow()

    # Audit Trail
    audit = AuditEvent(
        case_id=case.id,
        event_type="case_approved" if req.approve else "case_reviewed",
        old_value_json=json.dumps(old_state),
        new_value_json=json.dumps({
            "review_status": case.review_status,
            "reviewer_summary": case.reviewer_summary,
            "reviewer_priority": case.reviewer_priority,
            "reviewer_notes": case.reviewer_notes
        }),
        reason="Healthcare reviewer updated notes and review status",
        actor_role="reviewer",
        created_at=datetime.utcnow()
    )
    db.add(audit)
    db.commit()
    db.refresh(case)

    return case.to_dict()


@router.post("/cases/{case_id}/override-priority")
def override_priority(case_id: str, req: PriorityOverrideRequest, db: Session = Depends(get_db)):
    """
    Allows a reviewer to override the operational priority with a mandatory reason.
    Preserves the original AI assessment separately and records an audit log.
    """
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    old_priority = case.priority_category
    case.reviewer_priority = req.new_priority
    case.priority_category = req.new_priority  # Update active queue priority
    case.updated_at = datetime.utcnow()

    # Audit log
    audit = AuditEvent(
        case_id=case.id,
        event_type="priority_overridden",
        old_value_json=json.dumps({"priority": old_priority}),
        new_value_json=json.dumps({"priority": req.new_priority, "reviewer": req.reviewer_name}),
        reason=req.reason,
        actor_role="reviewer",
        created_at=datetime.utcnow()
    )
    db.add(audit)
    db.commit()
    db.refresh(case)

    return {
        "status": "overridden",
        "case_id": case.id,
        "original_priority": old_priority,
        "new_priority": req.new_priority,
        "reason": req.reason
    }


@router.post("/cases/{case_id}/reopen")
def reopen_case(case_id: str, db: Session = Depends(get_db)):
    """Reopens an approved or reviewed case when further clarification is required."""
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    old_status = case.review_status
    case.review_status = "reopened"
    case.updated_at = datetime.utcnow()

    audit = AuditEvent(
        case_id=case.id,
        event_type="case_reopened",
        old_value_json=json.dumps({"review_status": old_status}),
        new_value_json=json.dumps({"review_status": "reopened"}),
        reason="Further clinical clarification requested by reviewer",
        actor_role="reviewer",
        created_at=datetime.utcnow()
    )
    db.add(audit)
    db.commit()
    db.refresh(case)

    return {"status": "reopened", "case_id": case.id, "review_status": case.review_status}
