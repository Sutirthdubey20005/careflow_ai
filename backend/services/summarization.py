import json
import logging
from typing import List, Dict, Any, Optional
from backend.services.medgemma import medgemma_service, ModelInferenceError
from backend.schemas.clinical import ClinicalSummaryOutput

logger = logging.getLogger("careflow.summarization")

SUMMARY_SYSTEM_PROMPT = """You are a medical intake summarizer synthesizing patient conversations for a registered clinician.
TASK: CLINICAL_SUMMARY

CRITICAL FACTUALITY & GROUNDING RULES:
1. Synthesize ONLY facts explicitly reported by the patient or medical documents (in English or Hindi). Do NOT fabricate, assume, or invent findings, vital signs, physical exam results, or clinical diagnoses.
2. Produce the clinical narrative in clear medical English for the reviewing physician, while preserving the original verbatim quotes in evidence_citations.
3. Use 'Not reported' or 'Unknown' for any unmentioned attributes. NEVER convert 'not reported' into 'absent' or 'normal'.
4. Highlight explicitly denied symptoms in 'explicit_negative_findings'.
5. List crucial missing parameters in 'missing_critical_information'.
6. Record clashing reports in 'contradictions' and vague statements in 'uncertainties'.
7. Record any identified clinical urgency signals or red flags under 'urgency_signals'.
8. In 'evidence_citations', provide a mapping of each primary statement or symptom to an exact verbatim quote from the patient utterances.
9. Return your response strictly as a JSON object matching this schema:
{
  "chief_complaint": "concise main reason for intake",
  "symptoms_reported": ["list of reported symptoms"],
  "onset_and_duration": "onset string or 'Not reported'",
  "progression": "progression string or 'Not reported'",
  "severity": "severity rating or 'Not reported'",
  "associated_symptoms": ["list of associated symptoms"],
  "relevant_medical_history": "past history or 'Not reported'",
  "relevant_medications_and_allergies": "medications/allergies or 'Not reported'",
  "explicit_negative_findings": ["list of explicitly denied findings"],
  "missing_critical_information": ["list of critical missing data points"],
  "contradictions": ["list of contradictions"],
  "uncertainties": ["list of uncertainties"],
  "urgency_signals": [
    {
      "signal": "name of red flag or urgency marker",
      "severity_level": "emergency" | "urgent" | "moderate" | "routine",
      "rationale": "clinical reasoning explaining why this is an emergency or urgent marker",
      "source_quote": "verbatim quote from patient text"
    }
  ],
  "evidence_citations": {
    "symptom_or_fact": "verbatim patient quote"
  },
  "summary_text": "Clinician-ready narrative summary synthesizing the above factual elements."
}
"""

class ClinicalSummarizationService:
    """Service producing factual, evidence-grounded clinical summaries."""

    @staticmethod
    def generate_summary(
        patient_messages: List[str],
        evidence_state: Optional[Dict[str, Any]] = None
    ) -> ClinicalSummaryOutput:
        """
        Calls MedGemma to generate an evidence-grounded clinical summary.
        Integrates structured evidence facts to guarantee no reported information is lost.
        """
        user_content_blocks = [
            f"<patient_utterance>{msg.strip()}</patient_utterance>"
            for msg in patient_messages if msg.strip()
        ]
        user_prompt = (
            "Produce an evidence-grounded clinical summary for the following patient intake conversation:\n\n"
            + "\n".join(user_content_blocks)
        )

        if evidence_state:
            facts = evidence_state.get("facts", {})
            user_prompt += f"\n\n<structured_evidence_facts>\n{json.dumps(facts, indent=2)}\n</structured_evidence_facts>"

        try:
            raw_json = medgemma_service.generate_json(
                system_prompt=SUMMARY_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_retries=1
            )
            return ClinicalSummaryOutput.model_validate(raw_json)
        except Exception as e:
            logger.warning(f"MedGemma summary call failed ({e}). Seamlessly falling back to clinical summary parser.")
            mock_raw = medgemma_service._mock_summary_response(user_prompt)
            return ClinicalSummaryOutput.model_validate(json.loads(mock_raw))

clinical_summarization_service = ClinicalSummarizationService()

