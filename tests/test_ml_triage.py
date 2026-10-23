import os
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.ml_triage_service import (
    predict_urgency_lgbm,
    construct_tabular_features,
    COLUMNS,
    PRIMARY_MODEL_PATH,
    CLASS_TO_TIER,
    _parse_temperature_f
)
from backend.services.triage_rules import triage_engine


@pytest.fixture
def client():
    return TestClient(app)


def test_lgbm_model_artifact_exists():
    """Confirms the LightGBM model artifact exists on disk and is loadable."""
    assert os.path.exists(PRIMARY_MODEL_PATH), f"Model artifact missing at {PRIMARY_MODEL_PATH}"
    import joblib
    model = joblib.load(PRIMARY_MODEL_PATH)
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_predict_urgency_fallback_on_missing_model():
    """When the model artifact is missing, gracefully returns fallback priority P2."""
    with patch("backend.services.ml_triage_service.get_model", return_value=None):
        res = predict_urgency_lgbm(slm_json={}, vitals_dict={})
        assert res["ml_predicted_priority"] == "P2"
        assert res["status"] == "model_artifact_missing"
        assert res["ml_confidence"] == 0.0
        # Hybrid result object behaves as string and dict
        assert str(res) == "P2"
        assert res == "P2"


def test_derived_metrics_and_vitals_defaults():
    """Validates shock index, MAP calculation, Celsius conversion, and default values."""
    # 1. Defaults applied when empty
    df_empty = construct_tabular_features(slm_json={}, vitals_dict={})
    row = df_empty.iloc[0]
    assert row["age"] == 45.0
    assert row["gender"] == 0.0
    assert row["bmi"] == 25.0
    assert row["heart_rate"] == 75.0
    assert row["sys_bp"] == 120.0
    assert row["dia_bp"] == 80.0
    assert row["spo2_pct"] == 98.0
    assert row["resp_rate"] == 16.0
    assert row["temp_f"] == 98.6
    # shock_index = 75 / 120 = 0.625
    assert abs(row["shock_index"] - (75.0 / 120.0)) < 1e-4
    # map_bp = (120 + 2 * 80) / 3 = 93.333...
    assert abs(row["map_bp"] - ((120.0 + 160.0) / 3.0)) < 1e-4

    # 2. Celsius to Fahrenheit conversion
    vitals_celsius = {"temperature": 39.0, "heart_rate": 110.0, "sys_bp": 90.0, "dia_bp": 60.0}
    df_c = construct_tabular_features(slm_json={}, vitals_dict=vitals_celsius)
    row_c = df_c.iloc[0]
    expected_temp_f = (39.0 * 9.0 / 5.0) + 32.0
    assert abs(row_c["temp_f"] - expected_temp_f) < 1e-3
    assert abs(row_c["shock_index"] - (110.0 / 90.0)) < 1e-4
    assert abs(row_c["map_bp"] - ((90.0 + 120.0) / 3.0)) < 1e-4


def test_robust_vocabulary_filtering_guardrails():
    """Ensures unmapped SLM hallucinations are strictly filtered out of the tabular feature vector."""
    slm_json = {
        "primary_system": "sys_cardiopulmonary",
        "pain_scale_0_10": 8,
        "acuity_red_flags": [
            "chest_pain",            # Valid -> rf_chest_pain
            "diaphoresis",           # Valid -> rf_diaphoresis
            "space_sickness",        # HALLUCINATED -> must be ignored!
            "severe_headache"        # NOT in red flags list -> must be ignored!
        ],
        "comorbidities_identified": [
            "hypertension",          # Valid -> hx_hypertension
            "diabetes",              # Valid -> hx_diabetes
            "alien_hand_syndrome"    # HALLUCINATED -> must be ignored!
        ]
    }
    vitals = {"sys_bp": 115.0, "dia_bp": 75.0, "heart_rate": 95.0}
    df = construct_tabular_features(slm_json, vitals)
    row = df.iloc[0]

    # Valid mappings
    assert row["sys_cardiopulmonary"] == 1.0
    assert row["sys_neurological"] == 0.0
    assert row["rf_chest_pain"] == 1.0
    assert row["rf_diaphoresis"] == 1.0
    assert row["hx_hypertension"] == 1.0
    assert row["hx_diabetes"] == 1.0
    assert row["pain_scale_0_10"] == 8.0

    # Ensure hallucinated items were NOT added to the feature row
    assert "rf_space_sickness" not in row
    assert "rf_severe_headache" not in row
    assert "hx_alien_hand_syndrome" not in row
    # All other flags remain 0.0
    assert row["rf_altered_mental_status"] == 0.0
    assert row["rf_active_bleeding"] == 0.0
    assert row["hx_cad"] == 0.0


def test_predict_urgency_lgbm_output_schema():
    """Verifies predict_urgency_lgbm returns required keys and confidence."""
    slm_json = {
        "primary_system": "sys_cardiopulmonary",
        "pain_scale_0_10": 9,
        "acuity_red_flags": ["chest_pain", "diaphoresis", "respiratory_distress"],
        "comorbidities_identified": ["cad", "hypertension"]
    }
    vitals = {
        "heart_rate": 125.0,
        "sys_bp": 85.0,
        "dia_bp": 55.0,
        "spo2_pct": 89.0,
        "resp_rate": 28.0,
        "poc_troponin_pos": 1.0
    }
    result = predict_urgency_lgbm(slm_json, vitals)
    assert isinstance(result, dict)
    assert "ml_predicted_priority" in result
    assert "ml_confidence" in result
    assert result["status"] == "success"
    assert result["ml_predicted_priority"] in ["P0", "P1", "P2", "P3"]
    assert 0.0 <= result["ml_confidence"] <= 1.0
    # High-acuity cardiac distress with troponin pos should predict P0 or P1
    assert result["ml_predicted_priority"] in ["P0", "P1"]


def test_predict_urgency_lgbm_routine_case():
    """Verifies that a low-acuity routine patient receives P3 prediction."""
    slm_json = {
        "primary_system": "sys_systemic",
        "pain_scale_0_10": 1,
        "acuity_red_flags": [],
        "comorbidities_identified": []
    }
    vitals = {
        "heart_rate": 72.0,
        "sys_bp": 120.0,
        "dia_bp": 80.0,
        "spo2_pct": 99.0,
        "resp_rate": 15.0,
        "temp_f": 98.4
    }
    result = predict_urgency_lgbm(slm_json, vitals)
    assert result["status"] == "success"
    assert result["ml_predicted_priority"] in ["P2", "P3"]


def test_triage_engine_dual_model_guardrail_preserves_rules():
    """
    CRITICAL GUARDRAIL TEST:
    Ensures that deterministic rule-based triage is never altered or overridden by ML.
    Both signals are computed and exposed side-by-side in TriageResult.
    """
    evidence_state = {
        "facts": {
            "symptoms": ["chest pain radiating to left arm"],
            "pain_scale_0_10": 9,
            "acuity_red_flags": ["chest_pain"]
        },
        "urgency_signals": [
            {
                "signal": "Acute Coronary Syndrome Red Flag",
                "severity_level": "emergency",
                "rationale": "Crushing chest discomfort radiating to left arm"
            }
        ]
    }
    triage_res = triage_engine.evaluate(
        evidence_state=evidence_state,
        raw_messages=["I have severe chest pain radiating to my left arm"],
        ai_status="ready"
    )

    # Deterministic rule evaluates acute coronary syndrome as P0
    assert triage_res.priority_category == "P0"
    assert triage_res.rule_based_priority == "P0"
    assert triage_res.rule_based_triage == "P0"

    # ML model priority runs concurrently and is populated
    assert triage_res.ml_model_priority in ["P0", "P1", "P2", "P3"]
    assert triage_res.ml_model_triage in ["P0", "P1", "P2", "P3"]
    assert triage_res.ml_confidence is not None
    assert triage_res.ml_confidence >= 0.0


def test_api_intake_submit_payload_contains_dual_signals(client):
    """
    Tests that POST /cases/{case_id}/submit delivers the exact required schema:
    - case_id
    - rule_based_priority
    - ml_model_priority
    - ml_confidence
    - extracted_features
    - suggested_follow_up
    """
    # 1. Create intake case
    create_res = client.post("/cases", json={"preferred_language": "English", "chief_complaint": "I have chest discomfort"})
    assert create_res.status_code == 201
    case_id = create_res.json()["id"]

    # 2. Add patient message with red flags
    msg_res = client.post(f"/cases/{case_id}/messages", json={
        "content": "I have severe chest pain radiating to my left arm and sweating."
    })
    assert msg_res.status_code == 200
    msg_data = msg_res.json()
    assert "rule_based_priority" in msg_data
    assert "ml_model_priority" in msg_data
    assert "ml_confidence" in msg_data

    # 3. Submit case
    submit_res = client.post(f"/cases/{case_id}/submit")
    assert submit_res.status_code == 200
    submit_data = submit_res.json()

    # Exact payload requirement validation
    assert submit_data["case_id"] == case_id
    assert submit_data["rule_based_priority"] in ["P0", "P1", "P2", "P3"]
    assert submit_data["ml_model_priority"] in ["P0", "P1", "P2", "P3"]
    assert isinstance(submit_data["ml_confidence"], float)
    assert "extracted_features" in submit_data
    assert isinstance(submit_data["extracted_features"], dict)
    assert "suggested_follow_up" in submit_data


def test_api_case_list_contains_both_triage_signals(client):
    """Verifies that the /cases list endpoint returns both rule_based_triage and ml_model_triage."""
    res = client.get("/cases")
    assert res.status_code == 200
    cases = res.json()
    assert len(cases) > 0
    for c in cases:
        assert "priority_category" in c
        assert "rule_based_triage" in c
        assert "ml_model_triage" in c
