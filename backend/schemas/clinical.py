from typing import Optional, List, Dict, Any, Literal, Union
from pydantic import BaseModel, Field, field_validator

# -------------------------------------------------------------
# Evidence & Extraction Schemas
# -------------------------------------------------------------

class EvidenceItemSchema(BaseModel):
    field: str = Field(..., description="Standardized clinical field name (e.g. 'chest_pain', 'fever', 'onset')")
    value: Any = Field(..., description="Extracted clinical value or stated observation")
    status: Literal["reported", "unknown", "conflicting", "uncertain"] = Field(
        "reported", 
        description="Status of fact: reported (explicit), unknown (asked & unknown/skipped), conflicting (clashing reports), uncertain (vague)"
    )
    source_type: Literal["patient_message", "model_extraction"] = Field(
        "patient_message",
        description="Origin of the fact"
    )
    source_message_id: Optional[int] = Field(None, description="Database message id where fact appeared")
    source_quote: Optional[str] = Field(None, description="Exact verifiable verbatim quote from patient message")
    confidence: Optional[float] = Field(None, description="Optional confidence indicator if meaningful")
    updated_at: Optional[str] = Field(None, description="ISO timestamp")


class ClinicalUrgencySignal(BaseModel):
    signal: str = Field(..., description="Identified clinical urgency sign or red-flag condition")
    severity_level: Literal["emergency", "urgent", "moderate", "routine"] = Field(
        ..., description="Model-reasoned clinical urgency level: emergency, urgent, moderate, or routine"
    )
    rationale: str = Field(..., description="Clinical reasoning explaining the urgency or physiological risk")
    source_quote: str = Field(..., description="Verbatim quote from patient utterance or medical document supporting this signal")


class ClinicalExtractionOutput(BaseModel):
    symptoms: List[str] = Field(default_factory=list, description="Primary reported symptoms")
    onset_and_duration: Optional[str] = Field(None, description="When symptoms began or how long they have lasted")
    progression: Optional[str] = Field(None, description="Course of symptoms: worsening, improving, fluctuating, or constant")
    severity: Optional[str] = Field(None, description="Reported pain/severity scale (e.g., 8/10 or mild/severe)")
    associated_symptoms: List[str] = Field(default_factory=list, description="Associated secondary complaints")
    medical_history: List[str] = Field(default_factory=list, description="Past medical history explicitly mentioned")
    medications_and_allergies: List[str] = Field(default_factory=list, description="Current medications or allergies")
    explicit_negative_findings: List[str] = Field(default_factory=list, description="Symptoms explicitly denied by patient")
    unknown_information: List[str] = Field(default_factory=list, description="Important unasked or skipped items")
    contradictions: List[str] = Field(default_factory=list, description="Detected conflicts across messages")
    uncertainties: List[str] = Field(default_factory=list, description="Uncertain or poorly supported statements")
    evidence_items: List[EvidenceItemSchema] = Field(default_factory=list, description="Atomic evidence units")
    urgency_signals: List[ClinicalUrgencySignal] = Field(default_factory=list, description="Model-identified clinical urgency signals")
    is_sufficient_for_summary: bool = Field(False, description="Whether intake has reached sufficient detail")

    # Constrained MedGemma Vocabulary & Schema Fields
    input_language: Optional[str] = Field("en", description="en | hi | hinglish")
    primary_system: Optional[str] = Field("sys_systemic", description="Affected organ system enum")
    pain_scale_0_10: Optional[int] = Field(0, description="Pain score integer 0-10")
    acuity_red_flags: List[str] = Field(default_factory=list, description="Extracted acuity red flags matching vocabulary")
    comorbidities_identified: List[str] = Field(default_factory=list, description="Extracted comorbidities matching vocabulary")
    target_triage_boundary: Optional[str] = Field("P3", description="P0 | P1 | P2 | P3")
    next_question_en: Optional[str] = Field(None, description="Focused question in English")
    next_question_hi: Optional[str] = Field(None, description="Focused question in Hindi or Hinglish")


class FollowUpQuestionOutput(BaseModel):
    next_question: Optional[str] = Field(None, description="The single next clinically relevant question to ask the patient")
    next_question_en: Optional[str] = Field(None, description="English question text")
    next_question_hi: Optional[str] = Field(None, description="Hindi/Hinglish question text")
    reason_for_question: str = Field(..., description="Clinical justification for asking this particular question")
    target_field: str = Field(..., description="The clinical field or missing piece of information targeted")
    is_terminal: bool = Field(False, description="Whether questioning should conclude")
    slm_extraction: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Fine-tuned MedGemma SLM features and bilingual questions")
    selected_provider: Optional[str] = Field("finetuned_medgemma", description="Provider used for generation")
    validation_status: Optional[str] = Field("valid", description="Schema validation status: valid, repaired, fallback")


class SLMTriageTurn(BaseModel):
    primary_system_flag: Literal[
        "sys_cardiopulmonary",
        "sys_neurological",
        "sys_gastrointestinal",
        "sys_urogenital",
        "sys_musculoskeletal",
        "sys_dermatological",
        "sys_systemic",
        "sys_eent",
        "cardiopulmonary",
        "neurological",
        "gastrointestinal",
        "musculoskeletal",
        "dermatological",
        "renal_urological",
        "systemic",
        "other"
    ]
    red_flag_severe_pain: Union[bool, int]
    red_flag_acute_onset: Union[bool, int]
    target_triage_boundary: Literal["P0", "P1", "P2", "P3", "P0_P1", "P1_P2", "P2_P3"]
    next_question_en: str = Field(..., min_length=5, description="English follow-up question")
    next_question_hi: str = Field(..., min_length=3, description="Hindi follow-up question")

    @field_validator("primary_system_flag", mode="after")
    @classmethod
    def normalize_sys_flag(cls, v: str) -> str:
        if v.startswith("sys_"):
            return v
        alias = {
            "cardiopulmonary": "sys_cardiopulmonary",
            "neurological": "sys_neurological",
            "gastrointestinal": "sys_gastrointestinal",
            "musculoskeletal": "sys_musculoskeletal",
            "dermatological": "sys_dermatological",
            "renal_urological": "sys_urogenital",
            "systemic": "sys_systemic",
            "other": "sys_systemic"
        }
        return alias.get(v, f"sys_{v}")

    @field_validator("red_flag_severe_pain", "red_flag_acute_onset", mode="after")
    @classmethod
    def normalize_bool_flags(cls, v: Union[bool, int]) -> bool:
        return bool(v)


class ClinicalSummaryOutput(BaseModel):
    chief_complaint: str = Field(..., description="Concise primary reason for clinical intake")
    symptoms_reported: List[str] = Field(default_factory=list, description="Reported symptoms")
    onset_and_duration: str = Field("Not reported", description="Onset and duration or 'Not reported'")
    progression: str = Field("Not reported", description="Progression course or 'Not reported'")
    severity: str = Field("Not reported", description="Reported severity or 'Not reported'")
    associated_symptoms: List[str] = Field(default_factory=list, description="Associated symptoms")
    relevant_medical_history: str = Field("Not reported", description="Relevant history or 'Not reported'")
    relevant_medications_and_allergies: str = Field("Not reported", description="Medications/allergies or 'Not reported'")
    explicit_negative_findings: List[str] = Field(default_factory=list, description="Explicitly denied findings")
    missing_critical_information: List[str] = Field(default_factory=list, description="Crucial missing clinical data")
    contradictions: List[str] = Field(default_factory=list, description="Contradictions noted in statements")
    uncertainties: List[str] = Field(default_factory=list, description="Uncertain or poorly supported facts")
    urgency_signals: List[ClinicalUrgencySignal] = Field(default_factory=list, description="Model-identified clinical urgency signals")
    evidence_citations: Dict[str, str] = Field(default_factory=dict, description="Fact -> verbatim patient quote")
    summary_text: str = Field(..., description="Clinician-facing factual synthesis")


# -------------------------------------------------------------
# Triage Protocol Schemas
# -------------------------------------------------------------

class TriageTrigger(BaseModel):
    rule_id: str
    rule_name: str
    priority: Literal["P0", "P1", "P2", "P3", "ERR"]
    criterion: str
    matched_evidence: str
    action_protocol: str


class TriageResult(BaseModel):
    priority_category: Literal["P0", "P1", "P2", "P3", "ERR"]
    priority_label: str
    reasons: List[str]
    warning_signs: List[str]
    triggered_rules: List[TriageTrigger]
    missing_information: List[str]
    uncertainties: List[str]
    triage_rule_version: str
    rule_based_priority: Optional[str] = Field(None, description="Deterministic rule-based triage priority (P0-P3, ERR)")
    rule_based_triage: Optional[str] = Field(None, description="Deterministic rule-based triage output (P0-P3, ERR)")
    ml_model_priority: Optional[str] = Field(None, description="LightGBM urgency classifier prediction (P0-P3)")
    ml_model_triage: Optional[str] = Field(None, description="LightGBM urgency classifier prediction (P0-P3)")
    final_triage: Optional[str] = Field(None, description="Final operational triage output")
    escalation_applied: bool = False
    escalation_reason: Optional[str] = None
    model_status: Optional[str] = "available"
    imputed_features: List[str] = Field(default_factory=list)
    decision_path: Optional[str] = Field(
        "model_clinical_reasoning",
        description="Evaluation pathway: model_urgency_reasoning, multidimensional_acuity_synthesis, or protocol_safety_guardrail"
    )
    ml_confidence: Optional[float] = Field(0.0, description="LightGBM prediction probability confidence score")


# -------------------------------------------------------------
# API Request / Response Schemas
# -------------------------------------------------------------

class CaseCreateRequest(BaseModel):
    preferred_language: str = "English"
    chief_complaint: Optional[str] = None
    selected_model_provider: Optional[str] = "finetuned_medgemma"


class MessageCreateRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Patient message text")
    raw_transcript: Optional[str] = Field(None, description="Original unedited transcript before patient edits")
    source_type: Optional[str] = Field("patient_typed", description="Origin: patient_typed, voice_transcript, ocr_extraction")
    source_language: Optional[str] = Field(None, description="Source language code")
    transcription_confidence: Optional[float] = Field(None, description="Confidence if from ASR/OCR")
    is_confirmed: Optional[bool] = Field(True, description="Whether confirmed by patient")

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty or whitespace only.")
        return v


class CaseReviewRequest(BaseModel):
    reviewer_summary: Optional[str] = Field(None, description="Edited or approved clinician summary")
    reviewer_priority: Optional[Literal["P0", "P1", "P2", "P3", "ERR"]] = Field(None, description="Reviewer assigned priority")
    reviewer_notes: Optional[str] = Field(None, description="Clinical notes or instructions")
    status: Optional[str] = Field(None, description="Updated patient workflow status: active, appointed, done")
    approve: bool = Field(False, description="Set to true to mark case approved")


class StatusUpdateRequest(BaseModel):
    status: Literal["active", "appointed", "done", "pending", "reviewed", "approved"] = Field(..., description="Patient care status")


class PriorityOverrideRequest(BaseModel):
    new_priority: Literal["P0", "P1", "P2", "P3", "ERR"] = Field(..., description="Overridden operational priority")
    reason: str = Field(..., min_length=5, description="Mandatory reason justifying the override")
    reviewer_name: Optional[str] = Field("Reviewer", description="Clinician identifier")


class CaseListItemResponse(BaseModel):
    id: str
    created_at: Optional[str]
    updated_at: Optional[str]
    priority_category: str
    rule_based_triage: Optional[str] = None
    ml_model_triage: Optional[str] = None
    review_status: str
    chief_complaint: Optional[str]
    has_missing_info: bool
    has_uncertainty_or_conflict: bool
    ai_status: str
    waiting_minutes: float


class HealthResponse(BaseModel):
    status: str
    database: str
    medgemma_mode: str
    medgemma_status: str
    triage_rule_version: str
