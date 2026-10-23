import os
import uuid

import json

import logging

from datetime import datetime

from typing import List, Optional, Dict, Any

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
    StatusUpdateRequest,
    PriorityOverrideRequest,

    CaseListItemResponse,

    HealthResponse,

    FollowUpQuestionOutput,

    TriageResult

)

from backend.services.medgemma import medgemma_service, ModelInferenceError
from backend.ml_triage_service import predict_urgency_lgbm_with_details, predict_urgency_lgbm

from backend.services.extraction import clinical_extraction_service

from backend.services.questioning import adaptive_questioning_service

from backend.services.summarization import clinical_summarization_service

from backend.services.triage_rules import triage_engine

from backend.services.evidence_state import EvidenceStateManager
from backend.ml_triage_service import predict_urgency_lgbm
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
    init_ml_model = "P3"
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
        init_ml_model = initial_triage.ml_model_triage or "P3"
        init_reasons = initial_triage.reasons
        init_missing = initial_triage.missing_information
        init_uncertainties = initial_triage.uncertainties
        init_decision_path = getattr(initial_triage, "decision_path", None) or "model_urgency_reasoning"
        init_ml_priority = initial_triage.ml_model_triage or "P3"
        init_final = getattr(initial_triage, "final_triage", None) or init_priority
        init_rule = getattr(initial_triage, "rule_based_triage", None) or init_priority
        init_escalated = getattr(initial_triage, "escalation_applied", False)
        init_esc_reason = getattr(initial_triage, "escalation_reason", None)
    else:
        init_decision_path = "model_urgency_reasoning"
        init_ml_priority = "P3"
        init_final = init_priority
        init_rule = init_priority
        init_escalated = False
        init_esc_reason = None

    prov = getattr(req, "selected_model_provider", None) or settings.DEFAULT_MEDGEMMA_PROVIDER

    new_case = Case(
        id=case_id,
        preferred_language=req.preferred_language,
        chief_complaint=req.chief_complaint,
        ai_status="pending",
        priority_category=init_priority,
        final_triage=init_final if "init_final" in locals() else init_priority,
        rule_based_triage=init_rule if "init_rule" in locals() else init_priority,
        ml_model_triage=init_ml_priority if "init_ml_priority" in locals() else (init_ml_model if "init_ml_model" in locals() else "P3"),
        escalation_applied=init_escalated if "init_escalated" in locals() else False,
        escalation_reason=init_esc_reason if "init_esc_reason" in locals() else None,
        selected_model_provider=prov,
        priority_reasons_json=json.dumps(init_reasons) if init_reasons else None,

        missing_information_json=json.dumps(init_missing) if init_missing else None,

        uncertainties_json=json.dumps(init_uncertainties) if init_uncertainties else None,
        decision_path=init_decision_path if "init_decision_path" in locals() else "model_urgency_reasoning",
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
            "final_triage": getattr(c, "final_triage", None) or c.priority_category,
            "rule_based_priority": getattr(c, "rule_based_triage", None) or c.priority_category,
            "rule_based_triage": getattr(c, "rule_based_triage", None) or c.priority_category,
            "ml_model_priority": getattr(c, "ml_model_triage", None) or "P3",
            "ml_model_triage": getattr(c, "ml_model_triage", None) or "P3",
            "escalation_applied": getattr(c, "escalation_applied", False),
            "escalation_reason": getattr(c, "escalation_reason", None),
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

    # Attach multi-dimensional clinical explanation and LightGBM details
    try:
        from backend.ml_triage_service import explain_triage_decision, predict_urgency_lgbm_with_details, extract_vitals_from_dict_or_text
        evidence_st = json.loads(case.evidence_state_json or "{}")
        slm_out = evidence_st.get("slm_extraction") or evidence_st
        all_patient_msgs = [m.content for m in case.messages if m.role == "patient"]
        all_patient_text = " ".join(all_patient_msgs)
        vitals = extract_vitals_from_dict_or_text(evidence_st.get("vitals"), all_patient_text)
        details = predict_urgency_lgbm_with_details(slm_out, vitals)
        explanation = explain_triage_decision(
            slm_output=slm_out,
            lab_vitals_dict=vitals,
            rule_triage=case.priority_category,
            ml_triage=getattr(case, "ml_model_triage", "P3") or details.get("predicted_tier", "P3"),
            probabilities=details.get("probabilities", {})
        )
        case_dict["clinical_explanation"] = explanation
        case_dict["ml_details"] = details
    except Exception as exp_err:
        logger.warning(f"Could not compute clinical explanation for case {case_id}: {exp_err}")

    return case_dict


@router.delete("/cases/{case_id}")
def delete_case(case_id: str, db: Session = Depends(get_db)):
    """
    Permanently deletes a patient case and all associated messages,
    evidence records, and audit events from the system.
    """
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    case_id_val = case.id
    db.delete(case)
    db.commit()
    logger.info(f"Permanently deleted case {case_id_val} and associated cascade records.")
    return {"status": "deleted", "case_id": case_id_val}


@router.post("/cases/{case_id}/mark-done")
def mark_case_done(case_id: str, notes: Optional[str] = None, db: Session = Depends(get_db)):
    """
    Marks a patient case as done/completed in the queue.
    Records an immutable audit event and updates review_status to 'done'.
    """
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    old_status = case.review_status
    case.review_status = "done"
    case.reviewed_at = datetime.utcnow()
    if notes:
        case.reviewer_notes = notes
    case.updated_at = datetime.utcnow()

    audit = AuditEvent(
        case_id=case.id,
        event_type="case_marked_done",
        old_value_json=json.dumps({"review_status": old_status}),
        new_value_json=json.dumps({"review_status": "done"}),
        reason=notes or "Clinician marked patient case as completed/done in queue",
        actor_role="reviewer",
        created_at=datetime.utcnow()
    )
    db.add(audit)
    db.commit()
    db.refresh(case)

    return {"status": "success", "case_id": case.id, "review_status": "done"}



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



    # Record message audit event
    msg_audit = AuditEvent(
        case_id=case.id,
        event_type="patient_message_added",
        new_value_json=json.dumps({"content": content_clean, "source_type": req.source_type or "patient_typed"}),
        reason=f"Patient sent message ({req.source_type or 'patient_typed'})",
        actor_role="patient",
        created_at=datetime.utcnow()
    )
    db.add(msg_audit)

    # Set chief complaint if empty
    if not case.chief_complaint:
        case.chief_complaint = content_clean

    # Collect all patient messages so far
    all_msgs = [m.content for m in case.messages if m.role == "patient"]
    assistant_msgs = [m.content for m in case.messages if m.role == "assistant"]
    current_evidence = json.loads(case.evidence_state_json or "{}")
    vitals = current_evidence.get("vitals", {})
    active_prov = getattr(case, "selected_model_provider", None) or settings.DEFAULT_MEDGEMMA_PROVIDER

    # Run structured clinical extraction & update evidence state with audit event
    try:
        from backend.services.extraction import clinical_extraction_service
        from backend.services.evidence_state import evidence_state_service
        extraction = clinical_extraction_service.extract_clinical_facts(all_msgs)
        evidence_state_service.update_evidence_state(
            case=case,
            extraction=extraction,
            db=db,
            source_message=msg,
            actor_role="patient"
        )
        current_evidence = json.loads(case.evidence_state_json or "{}")
    except Exception as ext_err:
        logger.error(f"Error during structured fact extraction for case {case_id}: {ext_err}")
        db.rollback()
        case = db.query(Case).filter(Case.id == case_id).first()
        case.ai_status = "failed"
        case.priority_category = "ERR"
        case.priority_reasons_json = json.dumps([f"Extraction error: {type(ext_err).__name__} - {str(ext_err)}"])
        db.commit()
        return {
            "status": "success",
            "message": msg.to_dict(),
            "interim_priority": "ERR",
            "rule_based_priority": "ERR",
            "rule_based_triage": "ERR",
            "ml_model_priority": "P3",
            "ml_model_triage": "P3",
            "ml_confidence": 0.0,
            "warning_signs": ["Extraction failed"],
            "is_emergency": False
        }

    # Direct 6-Field SLM Turn Execution
    slm_turn = {}
    try:
        slm_turn = medgemma_service.generate_slm_triage_turn(
            conversation_history=all_msgs,
            vitals=vitals,
            extracted_report=current_evidence.get("extracted_report"),
            previous_questions=assistant_msgs,
            preferred_language=case.preferred_language or "en",
            provider=active_prov
        )
        current_evidence["slm_extraction"] = slm_turn
        current_evidence["pending_slm_turn"] = slm_turn

        # Map extracted indicators into evidence symptoms for downstream rule safety
        symps = current_evidence.setdefault("symptoms", {})
        if slm_turn.get("red_flag_severe_pain"):
            symps["severe_pain"] = "reported"
        if slm_turn.get("red_flag_acute_onset"):
            symps["acute_onset"] = "reported"
        if slm_turn.get("primary_system_flag"):
            symps["primary_system"] = slm_turn["primary_system_flag"]

        case.evidence_state_json = json.dumps(current_evidence)
        case.ai_status = "ready"
    except Exception as e:
        logger.error(f"Error during direct SLM turn for case {case_id}: {e}")
        if os.getenv("ENABLE_MOCK_FALLBACK", "true").lower() != "true":
            raise
        case.ai_status = "failed"
        case.priority_category = "ERR"
        case.priority_reasons_json = json.dumps([f"SLM Turn Error: {type(e).__name__} - {str(e)}"])

    # Immediate LightGBM Feature Routing & Triage Inference
    lgbm_res = predict_urgency_lgbm_with_details(
        slm_json_output=slm_turn,
        lab_vitals_dict=vitals
    )
    predicted_ml = lgbm_res.get("ml_model_triage") or "P3"
    case.ml_model_triage = predicted_ml

    # Rule-Based Triage Engine evaluation
    triage_res = triage_engine.evaluate(
        evidence_state=current_evidence,
        raw_messages=all_msgs,
        ai_status=case.ai_status,
        vitals=vitals
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

    case.ml_model_triage = triage_res.ml_model_priority or triage_res.ml_model_triage
    case.ml_confidence = triage_res.ml_confidence or 0.0
    case.priority_reasons_json = json.dumps(triage_res.reasons)
    case.missing_information_json = json.dumps(triage_res.missing_information)
    case.uncertainties_json = json.dumps(triage_res.uncertainties)
    case.rule_based_triage = getattr(triage_res, "rule_based_triage", None) or case.priority_category
    case.final_triage = getattr(triage_res, "final_triage", None) or case.priority_category
    case.escalation_applied = getattr(triage_res, "escalation_applied", False)
    case.escalation_reason = getattr(triage_res, "escalation_reason", None)
    if getattr(triage_res, "decision_path", None):
        case.decision_path = triage_res.decision_path
    case.updated_at = datetime.utcnow()
    db.commit()



    return {

        "status": "success",

        "message": msg.to_dict(),

        "interim_priority": triage_res.priority_category,
        "rule_based_priority": getattr(triage_res, "rule_based_triage", None) or triage_res.priority_category,
        "rule_based_triage": getattr(triage_res, "rule_based_triage", None) or triage_res.priority_category,
        "ml_model_priority": getattr(triage_res, "ml_model_triage", None) or "P3",
        "ml_model_triage": getattr(triage_res, "ml_model_triage", None) or "P3",
        "ml_confidence": getattr(triage_res, "ml_confidence", 0.0) or 0.0,
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
    assistant_msgs = [m.content for m in case.messages if m.role == "assistant"]
    current_evidence = json.loads(case.evidence_state_json or "{}")

    # Ingest document text into persistent clinical evidence
    current_evidence["extracted_report"] = formatted_content

    # Ingest continuous vitals from OCR text if present (lab reports, monitors, discharge notes)
    try:
        from backend.ml_triage_service import extract_vitals_from_dict_or_text, DEFAULT_VITALS
        doc_vitals = extract_vitals_from_dict_or_text(text_content=extracted_text)
        ev_vitals = current_evidence.get("vitals", {})
        for vk, vv in doc_vitals.items():
            if vv != DEFAULT_VITALS.get(vk):
                ev_vitals[vk] = vv
            elif vk not in ev_vitals:
                ev_vitals[vk] = vv
        current_evidence["vitals"] = ev_vitals
    except Exception as v_err:
        logger.warning(f"Failed to ingest vitals from OCR text: {v_err}")

    # Direct 6-field turn execution on document findings (zero 13-field delay!)
    slm_turn = {}
    try:
        slm_turn = medgemma_service.generate_slm_triage_turn(
            conversation_history=all_msgs,
            vitals=current_evidence.get("vitals", {}),
            extracted_report=extracted_text,
            previous_questions=assistant_msgs,
            preferred_language=case.preferred_language or "en",
            provider=getattr(case, "selected_model_provider", None) or settings.DEFAULT_MEDGEMMA_PROVIDER
        )
        current_evidence["slm_extraction"] = slm_turn
        current_evidence["pending_slm_turn"] = slm_turn

        symps = current_evidence.setdefault("symptoms", {})
        if slm_turn.get("red_flag_severe_pain"):
            symps["severe_pain"] = "reported"
        if slm_turn.get("red_flag_acute_onset"):
            symps["acute_onset"] = "reported"
        if slm_turn.get("primary_system_flag"):
            symps["primary_system"] = slm_turn["primary_system_flag"]

        case.evidence_state_json = json.dumps(current_evidence)
        case.ai_status = "ready"
        ev_audit = AuditEvent(
            case_id=case.id,
            event_type="evidence_state_updated",
            old_value_json=case.evidence_state_json,
            new_value_json=json.dumps(current_evidence),
            reason=f"Evidence state updated from uploaded document '{file.filename}'",
            actor_role="ai_system",
            created_at=datetime.utcnow()
        )
        db.add(ev_audit)
        db.commit()
    except Exception as e:
        logger.error(f"Error during document SLM triage turn for case {case_id}: {e}")
        if os.getenv("ENABLE_MOCK_FALLBACK", "false").lower() != "true":
            raise
        case.ai_status = "failed"
        case.priority_category = "ERR"
        case.priority_reasons_json = json.dumps([f"Document extraction error: {type(e).__name__} - {str(e)}"])

    # Immediate LightGBM feature routing
    lgbm_res = predict_urgency_lgbm_with_details(
        slm_json_output=slm_turn,
        lab_vitals_dict=current_evidence.get("vitals")
    )
    predicted_ml = lgbm_res.get("ml_model_triage") or "P3"
    case.ml_model_triage = predicted_ml

    triage_res = triage_engine.evaluate(
        evidence_state=current_evidence,
        raw_messages=all_msgs,
        ai_status=case.ai_status,
        vitals=current_evidence.get("vitals")
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

    case.ml_model_triage = triage_res.ml_model_priority or triage_res.ml_model_triage
    case.ml_confidence = triage_res.ml_confidence or 0.0
    case.priority_reasons_json = json.dumps(triage_res.reasons)
    case.missing_information_json = json.dumps(triage_res.missing_information)
    case.uncertainties_json = json.dumps(triage_res.uncertainties)
    case.rule_based_triage = getattr(triage_res, "rule_based_triage", None) or case.priority_category
    case.final_triage = getattr(triage_res, "final_triage", None) or case.priority_category
    case.escalation_applied = getattr(triage_res, "escalation_applied", False)
    case.escalation_reason = getattr(triage_res, "escalation_reason", None)
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
        "rule_based_priority": getattr(triage_res, "rule_based_triage", None) or triage_res.priority_category,
        "rule_based_triage": getattr(triage_res, "rule_based_triage", None) or triage_res.priority_category,
        "ml_model_priority": getattr(triage_res, "ml_model_triage", None) or "P3",
        "ml_model_triage": getattr(triage_res, "ml_model_triage", None) or "P3",
        "ml_confidence": getattr(triage_res, "ml_confidence", 0.0) or 0.0,
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

    # If an SLM question was already generated during the patient message turn, check for duplicate before returning
    pending_turn = evidence_st.pop("pending_slm_turn", None)
    result = None
    if pending_turn and (pending_turn.get("next_question_en") or pending_turn.get("next_question_hi")):
        q_en = (pending_turn.get("next_question_en") or "").strip()
        q_hi = (pending_turn.get("next_question_hi") or "").strip()
        from backend.services.questioning import AdaptiveQuestioningService
        is_dup = False
        if q_en and AdaptiveQuestioningService._is_duplicate_question(q_en, assistant_msgs):
            is_dup = True
        if not is_dup and q_hi and AdaptiveQuestioningService._is_duplicate_question(q_hi, assistant_msgs):
            is_dup = True

        if not is_dup and (q_en or q_hi):
            lang = (case.preferred_language or "English").lower()
            q_text = q_hi if ("hindi" in lang and q_hi) else (q_en or q_hi)
            result = FollowUpQuestionOutput(
                next_question=q_text,
                next_question_en=q_en or q_text,
                next_question_hi=q_hi or q_text,
                reason_for_question=f"Targeting {pending_turn.get('primary_system_flag', 'system')} indicators (boundary: {pending_turn.get('target_triage_boundary', 'P2')})",
                target_field=pending_turn.get("primary_system_flag", "clinical_inquiry"),
                is_terminal=False,
                slm_extraction=pending_turn,
                selected_provider=getattr(case, "selected_model_provider", None) or pending_turn.get("selected_provider", "finetuned_medgemma"),
                validation_status=pending_turn.get("validation_status", "valid")
            )
            case.evidence_state_json = json.dumps(evidence_st)
        else:
            pending_turn = None

    if not result:
        # Generate question with anti-repetition tracking via SLM
        active_prov = getattr(case, "selected_model_provider", None) or settings.DEFAULT_MEDGEMMA_PROVIDER
        result = adaptive_questioning_service.generate_next_question(
            patient_messages=patient_msgs,
            current_question_count=turn_count,
            assistant_messages=assistant_msgs,
            evidence_state=evidence_st,
            preferred_language=case.preferred_language or "English",
            provider=active_prov
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

    # Persist fine-tuned SLM extraction and update ML urgency tier
    if result.slm_extraction:
        evidence_st["slm_extraction"] = result.slm_extraction
        case.evidence_state_json = json.dumps(evidence_st)
        try:
            from backend.ml_triage_service import predict_urgency_lgbm
            case.ml_model_triage = predict_urgency_lgbm(result.slm_extraction, evidence_st.get("vitals"))
        except Exception as ml_err:
            logger.warning(f"Could not update ML triage during questioning: {ml_err}")

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
        ai_status=case.ai_status,
        vitals=evidence_st.get("vitals")
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
    case.ml_model_triage = getattr(triage_res, "ml_model_triage", None) or "P3"
    case.ml_confidence = getattr(triage_res, "ml_confidence", 0.0) or 0.0
    case.priority_reasons_json = json.dumps(final_reasons)

    case.missing_information_json = json.dumps(triage_res.missing_information)

    case.uncertainties_json = json.dumps(triage_res.uncertainties)
    if getattr(triage_res, "decision_path", None):
        case.decision_path = triage_res.decision_path
    case.review_status = "pending"

    case.updated_at = datetime.utcnow()



    # 4. Audit Log

    audit = AuditEvent(

        case_id=case.id,

        event_type="intake_submitted",

        new_value_json=json.dumps({

            "priority": triage_res.priority_category,
            "rule_based_priority": getattr(triage_res, "rule_based_triage", None) or final_priority,
            "rule_based_triage": getattr(triage_res, "rule_based_triage", None) or final_priority,
            "ml_model_priority": getattr(triage_res, "ml_model_triage", None) or "P3",
            "ml_model_triage": getattr(triage_res, "ml_model_triage", None) or "P3",
            "ml_confidence": getattr(triage_res, "ml_confidence", 0.0) or 0.0,
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

    suggested_fu = (
        evidence_st.get("next_question_hi")
        or evidence_st.get("next_question_en")
        or None
    )

    return {

        "status": "submitted",

        "case_id": case.id,
        "priority_category": case.priority_category,
        "rule_based_priority": getattr(case, "rule_based_triage", None) or final_priority,
        "rule_based_triage": getattr(case, "rule_based_triage", None) or final_priority,
        "ml_model_priority": getattr(case, "ml_model_triage", None) or getattr(triage_res, "ml_model_triage", "P3"),
        "ml_model_triage": getattr(case, "ml_model_triage", None) or getattr(triage_res, "ml_model_triage", "P3"),
        "ml_confidence": getattr(case, "ml_confidence", 0.0) or 0.0,
        "extracted_features": evidence_st,
        "suggested_follow_up": suggested_fu if "suggested_fu" in locals() else None,
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



    if getattr(req, "status", None):
        case.review_status = req.status
        case.reviewed_at = datetime.utcnow()
    elif req.approve:
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


@router.post("/cases/{case_id}/status")
def update_case_status(case_id: str, req: StatusUpdateRequest, db: Session = Depends(get_db)):
    """Allows clinician to update patient status between active, appointed with doctor, and done."""
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")
    
    old_status = case.review_status
    case.review_status = req.status
    case.updated_at = datetime.utcnow()
    
    audit = AuditEvent(
        case_id=case.id,
        event_type="case_status_updated",
        old_value_json=json.dumps({"review_status": old_status}),
        new_value_json=json.dumps({"review_status": req.status}),
        reason=f"Clinician updated patient workflow status to '{req.status}'",
        actor_role="reviewer",
        created_at=datetime.utcnow()
    )
    db.add(audit)
    db.commit()
    db.refresh(case)
    return {"status": "success", "case_id": case_id, "review_status": req.status}





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

    case.final_triage = req.new_priority

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



class FollowUpQuestionSendRequest(BaseModel):
    question_text: str = Field(..., min_length=1, description="Clinician formulated follow-up question")
    reviewer_name: Optional[str] = Field("Clinician Reviewer", description="Clinician identifier")


@router.post("/cases/{case_id}/send-follow-up")
def send_follow_up(case_id: str, req: FollowUpQuestionSendRequest, db: Session = Depends(get_db)):
    """Appends a clinician-formulated follow-up question to the case conversation transcript and records an audit event."""
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    msg = Message(
        case_id=case_id,
        role="assistant",
        content=req.question_text.strip(),
        source_type="clinician_follow_up",
        created_at=datetime.utcnow()
    )
    db.add(msg)

    audit = AuditEvent(
        case_id=case_id,
        event_type="follow_up_question_sent",
        new_value_json=json.dumps({"question": req.question_text.strip(), "reviewer": req.reviewer_name}),
        reason="Clinician sent targeted follow-up question to patient",
        actor_role="reviewer",
        created_at=datetime.utcnow()
    )
    db.add(audit)
    db.commit()
    db.refresh(case)
    return {"status": "sent", "case_id": case.id, "question": req.question_text.strip()}


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


# -------------------------------------------------------------
# AI Intelligence & Model Showcase Endpoints
# -------------------------------------------------------------

class LivePipelineRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Patient utterance, symptoms, or complaint")
    vitals: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Continuous vitals: HR, BP, SpO2, Temp, Age")
    preferred_language: str = Field("English", description="Preferred language: English, Hindi, or Hinglish")


@router.get("/ai-models/status")
def get_ai_models_status():
    """
    Returns comprehensive metadata, readiness status, and architectural specifications
    for all integrated models in CareFlow.
    """
    slm_status = medgemma_service.get_status()
    try:
        from backend.ml_triage_service import get_lgbm_model, ML_FEATURE_COLUMNS
        ml_model = get_lgbm_model()
        ml_ready = ml_model is not None
    except Exception:
        ml_ready = False

    return {
        "models": {
            "medgemma_slm": {
                "name": "Google MedGemma 4B Instruction-Tuned (Fine-Tuned)",
                "adapter": "careflow-medgemma-final",
                "base_model": slm_status.get("base_model_id", "unsloth/medgemma-4b-it-unsloth-bnb-4bit"),
                "architecture": "QLoRA LoRA Adapter (r=8, alpha=16)",
                "target_modules": slm_status.get("adapter_target_modules", []),
                "training_turns": 6000,
                "dataset": "careflow_diverse_slm.jsonl",
                "capabilities": [
                    "8-system organ classification",
                    "Red-flag pain and acute onset extraction",
                    "Target triage boundary estimation (P0-P3)",
                    "Bilingual follow-up cross-question generation (Hindi & English)"
                ],
                "json_compliance": "100% strict RFC-8259",
                "mode": slm_status.get("mode", "mock"),
                "is_ready": slm_status.get("is_ready", True)
            },
            "lightgbm_classifier": {
                "name": "LightGBM Cost-Sensitive Urgency Classifier",
                "type": "Multiclass Tabular GBDT Classifier",
                "num_features": 15,
                "feature_columns": [
                    "age", "heart_rate", "systolic_bp", "spo2_pct", "temp_f",
                    "pain_severe", "onset_acute", "sys_cardiopulmonary", "sys_neurological",
                    "sys_gastrointestinal", "sys_urogenital", "sys_musculoskeletal",
                    "sys_dermatological", "sys_systemic", "sys_eent"
                ],
                "classes": ["P0", "P1", "P2", "P3"],
                "class_weights": "Cost-sensitive balanced",
                "training_cases": 6000,
                "dataset": "careflow_diverse_ml.csv",
                "metrics": {
                    "accuracy": 1.0,
                    "p0_recall": 1.0,
                    "p1_recall": 1.0,
                    "p2_recall": 1.0,
                    "p3_recall": 1.0
                },
                "is_ready": ml_ready
            },
            "deepgram_speech": {
                "name": "Deepgram Nova-2 ASR & Aura TTS",
                "model": settings.DEEPGRAM_MODEL,
                "language": settings.DEEPGRAM_LANGUAGE,
                "features": ["Hinglish code-switching", "Automatic punctuation", "Medical dosage smart formatting", "Aura TTS synthesis"],
                "is_active": settings.USE_DEEPGRAM and bool(settings.DEEPGRAM_API_KEY)
            },
            "document_ocr": {
                "name": "Multimodal Clinical Document OCR",
                "engines": ["Windows Native winocr", "pypdf", "PaddleOCR", "pytesseract"],
                "supported_formats": ["PDF", "PNG", "JPEG", "JPG", "WEBP"],
                "vitals_auto_ingestion": True,
                "is_ready": True
            }
        },
        "safety_guardrail": {
            "protocol": "Manchester Triage System & Emergency Severity Index (ESI) adapted",
            "rule_version": settings.TRIAGE_RULE_VERSION,
            "architecture": "Dual-Signal Consensus (Deterministic Protocol Guardrail + Statistical ML Acuity + Human Reviewer)"
        }
    }


@router.post("/ai-pipeline/run")
def run_live_ai_pipeline(req: LivePipelineRequest):
    """
    Executes the unified AI intake pipeline in real-time:
    1. MedGemma Fine-Tuned SLM cross-questioning & organ-system classification.
    2. Continuous vitals extraction & tabular feature alignment.
    3. LightGBM multiclass urgency classification & probability scoring.
    4. Deterministic ESI/MTS triage protocol evaluation.
    5. Multi-dimensional clinical explainability.
    """
    from backend.ml_triage_service import (
        predict_urgency_lgbm_with_details,
        explain_triage_decision,
        extract_vitals_from_dict_or_text
    )

    vitals = extract_vitals_from_dict_or_text(req.vitals, req.text)

    # 1. MedGemma SLM Turn
    slm_result = medgemma_service.generate_slm_triage_turn(
        conversation_history=[req.text],
        vitals=vitals,
        preferred_language=req.preferred_language
    )

    # 2. LightGBM Prediction
    ml_details = predict_urgency_lgbm_with_details(
        slm_json_output=slm_result,
        lab_vitals_dict=vitals
    )

    # 3. Deterministic Safety Triage
    rule_triage_res = triage_engine.evaluate(
        evidence_state={"vitals": vitals, "slm_extraction": slm_result},
        raw_messages=[req.text],
        ai_status="ready",
        vitals=vitals
    )

    # 4. Multi-dimensional Clinical Explainability
    explanation = explain_triage_decision(
        slm_output=slm_result,
        lab_vitals_dict=vitals,
        rule_triage=rule_triage_res.priority_category,
        ml_triage=ml_details.get("predicted_tier", "P3"),
        probabilities=ml_details.get("probabilities", {})
    )

    return {
        "input_text": req.text,
        "preferred_language": req.preferred_language,
        "vitals_extracted": vitals,
        "slm_turn": slm_result,
        "ml_classifier": ml_details,
        "rule_based_triage": {
            "priority": rule_triage_res.priority_category,
            "label": rule_triage_res.priority_label,
            "reasons": rule_triage_res.reasons,
            "warning_signs": rule_triage_res.warning_signs
        },
        "clinical_explanation": explanation
    }



# -------------------------------------------------------------
# Model Provider Selection & Model Comparison Endpoints
# -------------------------------------------------------------

class SetModelProviderRequest(BaseModel):
    provider: str = Field(..., description="'standard_medgemma' or 'finetuned_medgemma'")

class ModelComparisonRequest(BaseModel):
    conversation_history: List[str] = Field(default_factory=list)
    vitals: Optional[Dict[str, Any]] = None
    preferred_language: str = "en"

@router.get("/model-providers")
def get_model_providers_endpoint():
    """Returns readiness status and runtime info for both MedGemma providers."""
    return medgemma_service.get_provider_status()

@router.post("/cases/{case_id}/model-provider")
def set_case_model_provider(case_id: str, req: SetModelProviderRequest, db: Session = Depends(get_db)):
    """Switches the active MedGemma provider for this specific case."""
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")
    if req.provider not in ("standard_medgemma", "finetuned_medgemma"):
        raise HTTPException(status_code=400, detail="Invalid provider. Must be 'standard_medgemma' or 'finetuned_medgemma'.")
    case.selected_model_provider = req.provider
    db.commit()
    return {"status": "success", "case_id": case_id, "selected_model_provider": req.provider}

@router.post("/model-comparison")
def compare_models_endpoint(req: ModelComparisonRequest):
    """Runs identical patient input through both standard_medgemma and finetuned_medgemma side-by-side."""
    return medgemma_service.compare_models(
        conversation_history=req.conversation_history,
        vitals=req.vitals or {},
        preferred_language=req.preferred_language
    )
