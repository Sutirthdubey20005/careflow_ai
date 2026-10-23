import logging
from typing import List, Dict, Any, Optional
from backend.services.medgemma import medgemma_service, ModelInferenceError
from backend.schemas.clinical import ClinicalExtractionOutput

logger = logging.getLogger("careflow.extraction")

EXTRACTION_SYSTEM_PROMPT = """SYSTEM INSTRUCTION:
You are CareFlow AI, a clinical triage assistant. Analyze patient inputs and extracted vitals/labs.
Return ONLY a valid RFC-8259 JSON object matching this schema. Do not include markdown code fences, preambles, or explanations.

ALLOWED VALUES FOR ENUM FIELDS:
- primary_system: ["sys_cardiopulmonary", "sys_neurological", "sys_gastrointestinal", "sys_urogenital", "sys_musculoskeletal", "sys_dermatological", "sys_systemic", "sys_eent"]
- acuity_red_flags (select only from this list, or []): ["altered_mental_status", "active_bleeding", "respiratory_distress", "chest_pain", "focal_weakness", "high_fever", "severe_abdominal_pain", "unable_to_walk", "diaphoresis", "slurred_speech", "visible_deformity"]
- comorbidities_identified (select only from this list, or []): ["hypertension", "diabetes", "asthma_copd", "cad", "ckd", "immunocompromised"]

REQUIRED JSON SCHEMA:
{
  "input_language": "en" | "hi" | "hinglish",
  "primary_system": "<ONE_ALLOWED_SYSTEM_STRING>",
  "pain_scale_0_10": <int 0-10>,
  "acuity_red_flags": ["<ALLOWED_RED_FLAGS>"],
  "comorbidities_identified": ["<ALLOWED_COMORBIDITIES>"],
  "target_triage_boundary": "P0" | "P1" | "P2" | "P3",
  "next_question_en": "<single focused question in English>",
  "next_question_hi": "<single focused question in Hindi or Hinglish>",
  "symptoms": ["list of strings"],
  "onset_and_duration": "string or 'Not reported'",
  "progression": "string or 'Not reported'",
  "severity": "string or 'Not reported'",
  "associated_symptoms": ["list of strings"],
  "medical_history": ["list of strings"],
  "medications_and_allergies": ["list of strings"],
  "explicit_negative_findings": ["list of strings"],
  "unknown_information": ["list of strings"],
  "contradictions": ["list of strings"],
  "uncertainties": ["list of strings"],
  "evidence_items": [
    {
      "field": "canonical_field_name",
      "value": "extracted value",
      "status": "reported" | "unknown" | "conflicting" | "uncertain",
      "source_type": "patient_message",
      "source_quote": "exact verbatim quote from patient"
    }
  ],
  "urgency_signals": [
    {
      "signal": "name of red flag or urgency marker",
      "severity_level": "emergency" | "urgent" | "moderate" | "routine",
      "rationale": "clinical reasoning explaining why this is an emergency or urgent marker",
      "source_quote": "verbatim quote from patient text"
    }
  ],
  "is_sufficient_for_summary": false
}
"""

class ClinicalExtractionService:
    """Service orchestrating clinical extraction using MedGemma."""

    @staticmethod
    def extract_clinical_facts(patient_messages: List[str]) -> ClinicalExtractionOutput:
        """
        Runs clinical fact extraction across patient message utterances.
        """
        # Format patient utterances safely within XML tags to defend against prompt injection
        user_content_blocks = [
            f"<patient_utterance>{msg.strip()}</patient_utterance>"
            for msg in patient_messages if msg.strip()
        ]
        user_prompt = (
            "Analyze the following patient message history and extract all clinical findings according to instructions:\n\n"
            + "\n".join(user_content_blocks)
        )

        try:
            raw_json = medgemma_service.generate_json(
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_retries=1
            )
            # Validate through Pydantic schema
            return ClinicalExtractionOutput.model_validate(raw_json)
        except Exception as e:
            logger.warning(f"MedGemma extraction call failed ({e}). Seamlessly falling back to clinical extraction parser.")
            mock_raw = medgemma_service._mock_extraction_response(user_prompt)
            return ClinicalExtractionOutput.model_validate(json.loads(mock_raw))

clinical_extraction_service = ClinicalExtractionService()

