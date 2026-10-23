import pytest
from backend.services.questioning import AdaptiveQuestioningService

def test_genuine_reaction_and_inquiry_sore_throat():
    """Verify genuine empathetic reaction and targeted airway inquiry for sore throat."""
    res = AdaptiveQuestioningService.generate_next_question(
        patient_messages=["I have had a severe sore throat since yesterday."],
        current_question_count=0,
        preferred_language="English"
    )
    assert res.target_field == "throat_airway_swallowing"
    assert res.next_question is not None
    # Empathetic reaction acknowledgment
    assert "sorry" in res.next_question.lower() or "throat is hurting" in res.next_question.lower()
    # Targeted inquiry
    assert "saliva" in res.next_question.lower() or "swallowing" in res.next_question.lower() or "breathing" in res.next_question.lower()


def test_genuine_reaction_and_inquiry_ankle_trauma():
    """Verify genuine reaction and weight-bearing inquiry for ankle/fall injury."""
    res = AdaptiveQuestioningService.generate_next_question(
        patient_messages=["I fell down the stairs and twisted my ankle, it is swollen."],
        current_question_count=0,
        preferred_language="English"
    )
    assert res.target_field == "trauma_weight_bearing_deformity"
    assert res.next_question is not None
    # Empathetic reaction
    assert "sorry" in res.next_question.lower() or "painful" in res.next_question.lower()
    # High-yield inquiry
    assert "bear weight" in res.next_question.lower() or "steps" in res.next_question.lower() or "deformity" in res.next_question.lower()


def test_genuine_reaction_and_inquiry_rash():
    """Verify genuine reaction and mucosal/airway inquiry for dermatological rash."""
    res = AdaptiveQuestioningService.generate_next_question(
        patient_messages=["I have a red rash spreading across my chest with extreme itching."],
        current_question_count=0,
        preferred_language="English"
    )
    assert res.target_field == "rash_mucosal_airway_involvement"
    assert res.next_question is not None
    # Empathetic reaction
    assert "uncomfortable" in res.next_question.lower() or "rash" in res.next_question.lower()
    # Targeted red-flag inquiry
    assert "blister" in res.next_question.lower() or "lips" in res.next_question.lower() or "swelling" in res.next_question.lower()


def test_smart_dimension_avoidance_onset_already_given():
    """When onset is already provided in narrative, model never asks when it started."""
    res = AdaptiveQuestioningService.generate_next_question(
        patient_messages=["I have a mild runny nose and cough that started 3 days ago."],
        current_question_count=0,
        preferred_language="English"
    )
    # Since onset is already given ("started 3 days ago"), onset_and_duration must be bypassed!
    assert res.target_field != "onset_and_duration"
    assert "when did your symptoms first start" not in res.next_question.lower()
    assert "when did" not in res.next_question.lower()


def test_smart_dimension_avoidance_severity_already_given():
    """When severity is already provided in narrative, model never asks for a 0-10 scale."""
    res = AdaptiveQuestioningService.generate_next_question(
        patient_messages=["I have a mild runny nose that started 3 days ago.", "My pain is 2/10."],
        current_question_count=1,
        assistant_messages=["Have you been coughing up any blood or rust-colored phlegm?"],
        preferred_language="English"
    )
    # Both onset and severity were provided, so it should move to medications or complete intake!
    assert res.target_field != "onset_and_duration"
    assert res.target_field != "severity"
    if res.next_question:
        assert "0 to 10" not in res.next_question.lower()
        assert "scale from 0" not in res.next_question.lower()


def test_genuine_reaction_hindi_dysuria():
    """Verify Hindi empathetic reaction and inquiry for urinary burning/pain."""
    res = AdaptiveQuestioningService.generate_next_question(
        patient_messages=["मुझे पेशाब में जलन और बहुत दर्द हो रहा है"],
        current_question_count=0,
        preferred_language="Hindi"
    )
    assert res.target_field == "urinary_fever_flank_hematuria"
    assert res.next_question is not None
    # Devanagari Hindi empathetic phrasing
    assert "पेशाब" in res.next_question
    assert "दर्द" in res.next_question or "कष्टकारी" in res.next_question


def test_genuine_reaction_hinglish_back_pain():
    """Verify Hinglish empathetic reaction and inquiry for back pain."""
    res = AdaptiveQuestioningService.generate_next_question(
        patient_messages=["kamar me bohot tez dard hai kal se"],
        current_question_count=0,
        preferred_language="Hinglish"
    )
    assert res.target_field == "back_cauda_equina_red_flags"
    assert res.next_question is not None
    assert "kamar" in res.next_question.lower()
    assert "groin" in res.next_question.lower() or "numbness" in res.next_question.lower() or "control" in res.next_question.lower()
