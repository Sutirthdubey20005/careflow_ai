from backend.services.triage_rules import DeterministicTriageEngine, triage_engine

def test_p0_emergency_cardiac_precedence():
    """Emergency trigger (chest pain radiating to arm) takes precedence over all else."""
    messages = [
        "I am having crushing chest pain radiating to my left arm."
    ]
    evidence_state = {"facts": {"chest_pain": {"value": "crushing"}}}

    result = triage_engine.evaluate(evidence_state, messages, ai_status="ready")
    assert result.priority_category == "P0"
    assert "P0" in result.priority_label
    assert len(result.triggered_rules) >= 1
    assert result.triggered_rules[0].rule_id == "RULE-P0-CARD-01"
    assert "Acute Coronary Syndrome" in result.triggered_rules[0].rule_name
    assert result.triage_rule_version == DeterministicTriageEngine.PROTOCOL_VERSION


def test_p0_never_downgraded_by_absent_symptoms():
    """Even if patient denies other symptoms (e.g. no fever, normal breathing), P0 is never downgraded."""
    messages = [
        "I have severe chest pain radiating to my jaw. No fever. Breathing feels normal. No headache."
    ]
    evidence_state = {
        "explicit_negative_findings": ["denies fever", "denies shortness of breath"]
    }

    result = triage_engine.evaluate(evidence_state, messages, ai_status="ready")
    assert result.priority_category == "P0"
    assert result.triggered_rules[0].rule_id == "RULE-P0-CARD-01"


def test_p1_urgent_meningeal_signs():
    """Fever combined with stiff neck triggers P1 urgent review within 1 hour."""
    messages = [
        "I have a high fever of 103 and a really stiff neck, can't touch chin to chest."
    ]
    evidence_state = {"facts": {}}

    result = triage_engine.evaluate(evidence_state, messages, ai_status="ready")
    assert result.priority_category == "P1"
    assert any(t.rule_id == "RULE-P1-INF-01" for t in result.triggered_rules)


def test_p1_severe_pain_score():
    """Pain score >= 8/10 triggers P1 urgent review."""
    messages = [
        "My toothache is excruciating, it is 9/10 pain."
    ]
    evidence_state = {"facts": {}}

    result = triage_engine.evaluate(evidence_state, messages, ai_status="ready")
    assert result.priority_category == "P1"
    assert any(t.rule_id == "RULE-P1-PAIN-02" for t in result.triggered_rules)


def test_p2_incomplete_or_conflicting_information():
    """Contradictory findings escalate to P2: Incomplete or uncertain."""
    messages = [
        "I have had a mild rash for two days."
    ]
    evidence_state = {
        "contradictions": ["Conflict in onset: previously '2 days', now '2 months'"],
        "uncertainties": []
    }

    result = triage_engine.evaluate(evidence_state, messages, ai_status="ready")
    assert result.priority_category == "P2"
    assert any(t.rule_id == "RULE-P2-CONFLICT-01" for t in result.triggered_rules)


def test_p3_routine_review():
    """Mild symptoms with no red flags assign P3 routine review."""
    messages = [
        "I have a mild runny nose and sneezing for 3 days. No fever, pain is 1/10."
    ]
    evidence_state = {"facts": {}, "contradictions": [], "uncertainties": []}

    result = triage_engine.evaluate(evidence_state, messages, ai_status="ready")
    assert result.priority_category == "P3"
    assert "Routine" in result.priority_label


def test_ai_failure_never_produces_routine_priority():
    """If AI fails or model status is 'failed', triage MUST assign ERR, never P3 routine."""
    messages = [
        "Patient message text here"
    ]
    evidence_state = {}

    result = triage_engine.evaluate(evidence_state, messages, ai_status="failed")
    assert result.priority_category == "ERR"
    assert "Review Required — AI Unavailable" in result.priority_label
    assert result.triggered_rules[0].rule_id == "RULE-SYS-01"


def test_p0_emergency_hindi_cardiac():
    """Bilingual Hindi symptom (सीने में दर्द radiating to बांह) triggers P0 emergency triage."""
    messages = [
        "सीने में बहुत तेज दबाव और दर्द है, दर्द बाएं हाथ तक जा रहा है और सांस फूल रही है।"
    ]
    evidence_state = {"facts": {}}

    result = triage_engine.evaluate(evidence_state, messages, ai_status="ready")
    assert result.priority_category == "P0"
    assert any("RULE-P0" in t.rule_id for t in result.triggered_rules)


def test_generalized_model_urgency_signal_p0():
    """MedGemma-reasoned emergency urgency signal elevates unscripted symptoms to P0."""
    messages = [
        "My throat is closing up after a bee sting and I am gasping for air."
    ]
    evidence_state = {
        "urgency_signals": [
            {
                "signal": "anaphylactic_airway_compromise",
                "severity_level": "emergency",
                "rationale": "Rapidly progressing acute airway obstruction following hymenoptera sting",
                "source_quote": "throat is closing up after a bee sting and I am gasping for air"
            }
        ]
    }

    result = triage_engine.evaluate(evidence_state, messages, ai_status="ready")
    assert result.priority_category == "P0"
    assert any(t.rule_id == "RULE-P0-MODEL-01" for t in result.triggered_rules)


def test_generalized_model_urgency_signal_p1():
    """MedGemma-reasoned urgent signal elevates unscripted condition to P1."""
    messages = [
        "Severe pain started around my belly button and has moved to my right lower side."
    ]
    evidence_state = {
        "urgency_signals": [
            {
                "signal": "suspected_acute_appendicitis",
                "severity_level": "urgent",
                "rationale": "Periumbilical pain migrating to right lower quadrant concerning for acute surgical abdomen",
                "source_quote": "pain started around my belly button and has moved to my right lower side"
            }
        ]
    }

    result = triage_engine.evaluate(evidence_state, messages, ai_status="ready")
    assert result.priority_category == "P1"
    assert any(t.rule_id == "RULE-P1-MODEL-01" for t in result.triggered_rules)


def test_p1_acute_central_chest_pain_without_radiation():
    """Acute chest discomfort without confirmed radiation triggers P1 urgent review, never P3 routine."""
    test_cases = [
        "very high chest pain",
        "dil me bahut dard ho rha",
        "my lower chest is paining i cant spit",
        "सीने में तेज दर्द है",
        "chest tightness and discomfort"
    ]
    for text in test_cases:
        res = triage_engine.evaluate(
            evidence_state={"facts": {}},
            raw_messages=[text],
            ai_status="ready"
        )
        assert res.priority_category == "P1", f"Failed for '{text}': got {res.priority_category}"
        assert any(t.rule_id in ("RULE-P1-CARD-05", "RULE-P1-PAIN-02") for t in res.triggered_rules)


def test_p1_standalone_numeric_pain_scores():
    """Standalone numeric severity replies (10, 9, 8) trigger P1 severe pain, never P3."""
    for score in ["10", "9", "8"]:
        res = triage_engine.evaluate(
            evidence_state={"facts": {}},
            raw_messages=["I have a headache", score],
            ai_status="ready"
        )
        assert res.priority_category == "P1", f"Failed for standalone score '{score}': got {res.priority_category}"
        assert any(t.rule_id == "RULE-P1-PAIN-02" for t in res.triggered_rules)


def test_p1_severe_pain_hinglish_variants():
    """Hinglish headache and severe pain statements correctly trigger P1."""
    hinglish_samples = [
        "bahut shar dard hai yaar",
        "sar me bahut tez dard hai",
        "shar me bahut dard ho raha hai"
    ]
    for msg in hinglish_samples:
        res = triage_engine.evaluate(
            evidence_state={"facts": {}},
            raw_messages=[msg],
            ai_status="ready"
        )
        assert res.priority_category == "P1", f"Failed for Hinglish '{msg}': got {res.priority_category}"
        assert any(t.rule_id == "RULE-P1-PAIN-02" for t in res.triggered_rules)


def test_p1_evidence_fact_severity_trigger():
    """Structured severity >= 8/10 or 'Severe' in evidence_state facts triggers P1."""
    evidence_state = {
        "facts": {
            "severity": {"value": "9/10", "status": "reported"}
        }
    }
    res = triage_engine.evaluate(
        evidence_state=evidence_state,
        raw_messages=["I have a toothache"],
        ai_status="ready"
    )
    assert res.priority_category == "P1"
    assert any(t.rule_id == "RULE-P1-PAIN-02" for t in res.triggered_rules)


def test_p2_clinical_uncertainty_trigger():
    """A clinical uncertainty (e.g. unverified audio or patient ambiguity) triggers P2."""
    evidence_state = {
        "facts": {},
        "uncertainties": ["Low-confidence voice transcript (55%): clinical facts require verbal clinician confirmation."],
        "contradictions": []
    }
    res = triage_engine.evaluate(
        evidence_state=evidence_state,
        raw_messages=["I feel unwell"],
        ai_status="ready"
    )
    assert res.priority_category == "P2"
    assert any(t.rule_id == "RULE-P2-UNCERTAIN-02" for t in res.triggered_rules)


def test_p2_multiple_unknown_intake_dimensions():
    """Multiple skipped or unknown dimensions trigger P2 incomplete intake."""
    evidence_state = {
        "facts": {},
        "unknown_information": [
            "Patient indicated unknown or skipped: 'onset'",
            "Patient indicated unknown or skipped: 'severity'"
        ],
        "contradictions": []
    }
    res = triage_engine.evaluate(
        evidence_state=evidence_state,
        raw_messages=["I have a mild rash"],
        ai_status="ready"
    )
    assert res.priority_category == "P2"
    assert any(t.rule_id == "RULE-P2-INCOMPLETE-03" for t in res.triggered_rules)
    assert res.decision_path == "multidimensional_acuity_synthesis"


def test_triage_decision_paths_multi_tier():
    """Verify that triage engine correctly records decision_path across all tiers."""
    # 1. Tier 1: Model Urgency Reasoning
    model_sig_evidence = {
        "urgency_signals": [
            {
                "signal": "Acute Surgical Abdomen",
                "severity_level": "urgent",
                "rationale": "Peritoneal irritation requiring urgent assessment",
                "source_quote": "intense belly pain and vomiting"
            }
        ]
    }
    res_model = triage_engine.evaluate(model_sig_evidence, ["intense belly pain and vomiting"])
    assert res_model.priority_category == "P1"
    assert res_model.decision_path == "model_urgency_reasoning"

    # 2. Tier 2: Multidimensional Acuity Synthesis (Severe Pain score 9/10)
    acuity_evidence = {"facts": {"severity": "9/10"}}
    res_acuity = triage_engine.evaluate(acuity_evidence, ["My toe hurts really badly", "9/10"])
    assert res_acuity.priority_category == "P1"
    assert res_acuity.decision_path == "multidimensional_acuity_synthesis"

    # 3. Tier 3: Protocol Safety Guardrail Floor (radiating chest pain without prior model signal)
    res_guardrail = triage_engine.evaluate({}, ["Crushing chest pain radiating to left arm"])
    assert res_guardrail.priority_category == "P0"
    assert res_guardrail.decision_path == "protocol_safety_guardrail"

    # 4. Tier 4: System Fallback
    res_sys = triage_engine.evaluate({}, ["General malaise"], ai_status="failed")
    assert res_sys.priority_category == "ERR"
    assert res_sys.decision_path == "system_fallback"


