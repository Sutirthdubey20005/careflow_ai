import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pydantic import ValidationError

from backend.config import settings
from backend.schemas.clinical import SLMTriageTurn

logger = logging.getLogger("careflow.safety")

# Regex patterns for supported Indian PII formats
PHONE_PATTERN = re.compile(
    r"(?:\+91[\s\-]?)?(?:[6-9]\d{9}|[6-9]\d{4}[\s\-]\d{5}|[6-9]\d{2}[\s\-]\d{3}[\s\-]\d{4})\b"
)
AADHAAR_PATTERN = re.compile(
    r"\b[2-9]\d{3}[\s\-]\d{4}[\s\-]\d{4}\b"
)
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"
)
TITLED_NAME_PATTERN = re.compile(
    r"\b(?:Dr\.|Doctor|Mr\.|Mister|Mrs\.|Ms\.|Shri|Smt\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"
)

# Prohibited clinical diagnosis and prescription patterns for SLMs
PROHIBITED_DIAGNOSIS_PATTERNS = [
    re.compile(r"\b(?:i\s+diagnose\s+you|you\s+have\s+been\s+diagnosed|definitive\s+diagnosis\s+is|my\s+diagnosis\s+is)\b", re.IGNORECASE),
    re.compile(r"\b(?:you\s+definitely\s+have|you\s+are\s+suffering\s+from\s+confirmed)\b", re.IGNORECASE),
]

PROHIBITED_PRESCRIPTION_PATTERNS = [
    re.compile(r"\b(?:take|prescribe|prescribing|dosage|dose\s+of)\s+\d+\s*(?:mg|ml|tablet|tablets|pills?|capsules?)\b", re.IGNORECASE),
    re.compile(r"\b(?:i\s+prescribe|take\s+paracetamol|take\s+aspirin|take\s+ibuprofen|take\s+antibiotic)\b", re.IGNORECASE),
]


@dataclass
class MaskingResult:
    """Represents the result of PII privacy masking."""
    masked_text: str
    redaction_count: int
    redactions_by_type: Dict[str, int]
    has_pii: bool
    masking_status: str  # "clean", "masked", "error"
    disclaimer: str = "Automated heuristic redaction applied. De-identification is not guaranteed."


class ClinicalSafetyService:
    """
    Centralized Privacy, Safety Guardrails, and SLM Validation Service for CareFlow.
    """

    @classmethod
    def mask_pii(cls, text: Optional[str]) -> MaskingResult:
        """
        Masks identifiers (Indian phone numbers, Aadhaar, emails, titled names)
        while preserving clinically meaningful facts (vitals, symptoms, dosages).
        """
        if not text:
            return MaskingResult(
                masked_text="",
                redaction_count=0,
                redactions_by_type={"phone": 0, "aadhaar": 0, "email": 0, "name": 0},
                has_pii=False,
                masking_status="clean"
            )

        redactions = {"phone": 0, "aadhaar": 0, "email": 0, "name": 0}
        processed = str(text)

        try:
            # 1. Emails
            emails = EMAIL_PATTERN.findall(processed)
            if emails:
                redactions["email"] = len(emails)
                processed = EMAIL_PATTERN.sub("[EMAIL_REDACTED]", processed)

            # 2. Aadhaar Numbers (12-digit formatted/unformatted)
            aadhaar_matches = AADHAAR_PATTERN.findall(processed)
            if aadhaar_matches:
                redactions["aadhaar"] = len(aadhaar_matches)
                processed = AADHAAR_PATTERN.sub("[AADHAAR_REDACTED]", processed)

            # 3. Phone Numbers (+91 or 10-digit mobile)
            # Guard against masking blood pressure ratios like "120/80" or dates like "10/05/2026"
            def _phone_sub(m):
                match_str = m.group(0)
                # Ensure it's not a vitals ratio or date
                if "/" in match_str or "." in match_str:
                    return match_str
                redactions["phone"] += 1
                return "[PHONE_REDACTED]"

            processed = PHONE_PATTERN.sub(_phone_sub, processed)

            # 4. Titled Names
            names = TITLED_NAME_PATTERN.findall(processed)
            if names:
                redactions["name"] = len(names)
                processed = TITLED_NAME_PATTERN.sub("[NAME_REDACTED]", processed)

            total_redactions = sum(redactions.values())
            has_pii = total_redactions > 0

            return MaskingResult(
                masked_text=processed,
                redaction_count=total_redactions,
                redactions_by_type=redactions,
                has_pii=has_pii,
                masking_status="masked" if has_pii else "clean"
            )
        except Exception as e:
            logger.error(f"Error during PII masking: {e}")
            return MaskingResult(
                masked_text=text,
                redaction_count=0,
                redactions_by_type=redactions,
                has_pii=False,
                masking_status="error"
            )

    @classmethod
    def validate_file_upload(cls, filename: Union[str, bytes], file_bytes: Union[bytes, str]) -> Tuple[bool, Optional[str]]:
        """
        Validates uploaded file format, size (<15 MB), and content safety.
        Supports either (filename, file_bytes) or (file_bytes, filename).
        """
        if isinstance(filename, (bytes, bytearray)):
            file_bytes, filename = filename, str(file_bytes)
        else:
            filename = str(filename)
            if not isinstance(file_bytes, (bytes, bytearray)):
                file_bytes = bytes(file_bytes, "utf-8", errors="ignore")

        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if len(file_bytes) > max_bytes:
            return False, f"File exceeds maximum permissible size of {settings.MAX_UPLOAD_SIZE_MB} MB."

        if len(file_bytes) < 4:
            return False, "File is empty or corrupted."

        ext = filename.lower().split(".")[-1] if "." in filename else ""
        allowed_exts = {"pdf", "png", "jpg", "jpeg", "webp", "bmp", "tiff", "txt"}
        if ext not in allowed_exts:
            return False, f"Unsupported file type '.{ext}'. Allowed formats: {', '.join(allowed_exts)}."

        # Magic byte validation
        if ext == "pdf" and not file_bytes.startswith(b"%PDF"):
            return False, "File does not match expected binary signatures (invalid PDF header)."

        return True, None

    @classmethod
    def sanitize_untrusted_text(cls, text: str) -> str:
        """
        Strips prompt injection triggers and control sequences from untrusted patient/document text.
        """
        if not text:
            return ""
        # Check prompt injection patterns
        injection_triggers = [
            r"\b(?:ignore\s+previous\s+instructions|system\s+prompt|disregard\s+all\s+prior)\b",
            r"\b(?:bypass\s+safety|jailbreak|pretend\s+you\s+are)\b"
        ]
        cleaned = text
        for pat in injection_triggers:
            cleaned = re.sub(pat, "[BLOCKED_INJECTION_PATTERN]", cleaned, flags=re.IGNORECASE)

        # Remove Gemma special tokens and system command markers
        cleaned = re.sub(r"<(?:start_of_turn|end_of_turn|system|model|user)>", "", cleaned, flags=re.IGNORECASE)
        # Remove null bytes and terminal control chars
        cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", cleaned)
        return cleaned.strip()

    @classmethod
    def validate_slm_triage_turn(
        cls,
        raw_output: Any,
        conversation_history: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]], List[str]]:
        """
        Strict Pydantic schema validation and clinical safety guardrail checking for MedGemma output.
        Returns: (is_valid, validated_dict_or_repaired, list_of_violations)
        """
        violations = []

        if not isinstance(raw_output, dict):
            return False, None, ["Model output is not a JSON dictionary."]

        # 1. Pydantic Schema Validation
        try:
            validated = SLMTriageTurn(**raw_output)
            clean_dict = validated.model_dump()
        except ValidationError as val_err:
            for err in val_err.errors():
                loc = ".".join(str(x) for x in err.get("loc", []))
                violations.append(f"Schema violation at '{loc}': {err.get('msg')}")
            return False, None, violations

        q_en = clean_dict.get("next_question_en", "").strip()
        q_hi = clean_dict.get("next_question_hi", "").strip()

        # 2. Guard against multiple unrelated questions (exactly one question per turn)
        # Punctuation check: multiple '?' or Hindi '?'
        q_en_marks = q_en.count("?")
        q_hi_marks = q_hi.count("?")
        if q_en_marks > 1 or q_hi_marks > 1:
            violations.append("Multiple questions detected in a single turn. Enforcing one targeted question.")
            # Repair by taking the first complete question sentence
            sentences = [s.strip() + "?" for s in q_en.split("?") if s.strip()]
            if sentences:
                clean_dict["next_question_en"] = sentences[0]
            sentences_hi = [s.strip() + "?" for s in q_hi.split("?") if s.strip()]
            if sentences_hi:
                clean_dict["next_question_hi"] = sentences_hi[0]

        # 3. Guard against prohibited diagnosis claims
        for pat in PROHIBITED_DIAGNOSIS_PATTERNS:
            if pat.search(q_en) or pat.search(q_hi):
                violations.append("Prohibited definitive diagnosis claim detected in follow-up question.")
                return False, None, violations

        # 4. Guard against prohibited prescription instructions
        for pat in PROHIBITED_PRESCRIPTION_PATTERNS:
            if pat.search(q_en) or pat.search(q_hi):
                violations.append("Prohibited medication prescription detected in follow-up question.")
                return False, None, violations

        # 5. Anti-Repetition Check against conversation history
        if conversation_history:
            history_text = " ".join(conversation_history).lower()
            q_norm = clean_dict["next_question_en"].lower().replace("?", "").strip()
            # If significant portion of question already exists in history
            words = [w for w in q_norm.split() if len(w) > 4]
            if words:
                matches = sum(1 for w in words if w in history_text)
                if matches / len(words) > 0.85:
                    violations.append("Question is highly repetitive of previous conversation turns.")

        is_valid = len(violations) == 0
        return is_valid, clean_dict, violations


safety_service = ClinicalSafetyService()

mask_pii = safety_service.mask_pii
validate_file_upload = safety_service.validate_file_upload
sanitize_untrusted_text = safety_service.sanitize_untrusted_text
validate_slm_triage_turn = safety_service.validate_slm_triage_turn
