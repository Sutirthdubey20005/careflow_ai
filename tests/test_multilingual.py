import pytest
from backend.services.extraction import clinical_extraction_service
from backend.services.summarization import clinical_summarization_service
from backend.services.questioning import adaptive_questioning_service
from backend.services.triage_rules import triage_engine
from backend.database.models import Case


def test_devanagari_hindi_extraction_and_triage():
    """Phase A: Tests symptom extraction and deterministic triage on Devanagari Hindi text."""
    hindi_text = "मुझे दो दिन से सीने में दर्द है और दर्द बाएं हाथ तक जा रहा है, सांस फूल रही है।"
    extraction = clinical_extraction_service.extract_clinical_facts([hindi_text])

    assert any("chest pain" in s.lower() for s in extraction.symptoms)
    # Emergency red flag signal must be present
    assert any(sig.severity_level == "emergency" for sig in extraction.urgency_signals)

    # Evaluate triage directly
    triage_res = triage_engine.evaluate(
        evidence_state=extraction.model_dump(),
        raw_messages=[hindi_text],
        ai_status="ready"
    )
    assert triage_res.priority_category == "P0"


def test_hinglish_extraction_and_triage():
    """Phase A: Tests symptom extraction and deterministic triage on Romanized Hinglish text."""
    hinglish_text = "mujhe kal se seene me dard hai aur dard left arm me ja raha hai, saans lene me takleef hai."
    extraction = clinical_extraction_service.extract_clinical_facts([hinglish_text])

    assert any("chest pain" in s.lower() for s in extraction.symptoms)
    assert extraction.onset_and_duration != "Not reported"

    triage_res = triage_engine.evaluate(
        evidence_state=extraction.model_dump(),
        raw_messages=[hinglish_text],
        ai_status="ready"
    )
    assert triage_res.priority_category == "P0"


def test_hindi_and_hinglish_negation_not_fabricated():
    """Phase A: Tests explicit denials in Hindi and Hinglish are captured without hallucinating unstated denials."""
    messages = [
        "मुझे सिरदर्द है, लेकिन सीने में दर्द नहीं है और बुखार नहीं है।",  # Hindi: headache, no chest pain, no fever
        "chot nahi lagi hai, koi allergy nahi hai"                         # Hinglish: no injury, no allergy
    ]
    extraction = clinical_extraction_service.extract_clinical_facts(messages)

    # Symptoms must have headache
    assert any("headache" in s.lower() for s in extraction.symptoms)
    # Denials must capture chest pain, fever, head injury, allergies
    assert "denies chest pain" in extraction.explicit_negative_findings
    assert "denies fever" in extraction.explicit_negative_findings
    assert "denies head injury" in extraction.explicit_negative_findings
    assert "denies allergies" in extraction.explicit_negative_findings

    # Must NOT fabricate denials not mentioned (e.g. denies medications)
    assert "denies medications" not in extraction.explicit_negative_findings


def test_cross_lingual_contradiction_detection():
    """Phase A: Tests contradiction detection when patient reports conflicting facts across languages."""
    # Patient says they have fever in Hindi/Hinglish, then in next message says no fever in English
    messages = [
        "mujhe 2 din se tez bukhar hai.",
        "Actually, I have no fever today, temperature is normal."
    ]
    extraction = clinical_extraction_service.extract_clinical_facts(messages)

    assert len(extraction.contradictions) >= 1
    assert any("fever" in c.lower() for c in extraction.contradictions)

    # Triage engine should place case into P2 due to contradiction
    triage_res = triage_engine.evaluate(
        evidence_state=extraction.model_dump(),
        raw_messages=messages,
        ai_status="ready"
    )
    # Contradiction triggers RULE-P2-CONFLICT-01 unless higher P0/P1 red flag exists
    assert triage_res.priority_category in ("P2", "P1", "P0")


def test_multilingual_unknown_answer_preservation():
    """Phase A: Unknown answers in Hindi/Hinglish must be preserved in unknown_information, not negative."""
    messages = [
        "mujhe pet dard hai.",
        "pata nahi kab shuru hua."  # "I don't know when it started" in Hinglish
    ]
    extraction = clinical_extraction_service.extract_clinical_facts(messages)

    assert len(extraction.unknown_information) >= 1
    assert any("pata nahi" in u.lower() for u in extraction.unknown_information)

    # Crucial safety invariant: Must NOT convert unknown onset to a denial!
    for denial in extraction.explicit_negative_findings:
        assert "onset" not in denial.lower()


def test_multilingual_questioning_honors_preferred_language():
    """Phase A: Adaptive questioning generates Hindi questions when preferred_language is Hindi."""
    result = adaptive_questioning_service.generate_next_question(
        patient_messages=["सीने में दर्द है"],
        current_question_count=0,
        assistant_messages=[],
        evidence_state={},
        preferred_language="हिन्दी (Hindi)"
    )
    assert result.next_question is not None
    # Question should contain Devanagari script
    assert any('\u0900' <= char <= '\u097F' for char in result.next_question)
