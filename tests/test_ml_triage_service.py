import pytest
from fastapi.testclient import TestClient
from backend.database.db import get_db, Base, engine
from backend.services.triage_rules import triage_engine
from backend.ml_triage_service import (
    predict_urgency_lgbm,
    predict_urgency_lgbm_with_details,
    extract_vitals_from_dict_or_text,
    _map_slm_and_vitals_to_dataframe,
    FEATURE_COLUMNS,
    TARGET_PRIORITY_MAP
)


def test_feature_columns_and_schema():
    assert len(FEATURE_COLUMNS) == 15
    assert "sys_cardiopulmonary" in FEATURE_COLUMNS
    assert "heart_rate" in FEATURE_COLUMNS
    assert "pain_severe" in FEATURE_COLUMNS


def test_vitals_extraction_from_dict():
    vitals_input = {
        "age": 45,
        "hr": 115,
        "bp": "145/95",
        "spo2": 91,
        "temp": 101.5
    }
    vitals = extract_vitals_from_dict_or_text(vitals_input)
    assert vitals["age"] == 45
    assert vitals["heart_rate"] == 115
    assert vitals["systolic_bp"] == 145
    assert vitals["spo2_pct"] == 91
    assert vitals["temp_f"] == 101.5


def test_vitals_extraction_from_clinical_text():
    text = "Initial triage triage note: HR 109 bpm, BP 104 mmHg, SpO2 92%, Temp 97.0 F"
    vitals = extract_vitals_from_dict_or_text(None, text)
    assert vitals["heart_rate"] == 109
    assert vitals["systolic_bp"] == 104
    assert vitals["spo2_pct"] == 92
    assert vitals["temp_f"] == 97.0


def test_slm_mapping_dataframe_generation():
    slm_output = {
        "primary_system_flag": "sys_gastrointestinal",
        "red_flag_severe_pain": True,
        "red_flag_acute_onset": True
    }
    vitals = {"age": 76, "heart_rate": 109, "systolic_bp": 104, "spo2_pct": 92, "temp_f": 97.0}
    df = _map_slm_and_vitals_to_dataframe(slm_output, vitals)

    assert df.shape == (1, 15)
    assert df["sys_gastrointestinal"].iloc[0] == 1
    assert df["sys_cardiopulmonary"].iloc[0] == 0
    assert df["pain_severe"].iloc[0] == 1
    assert df["onset_acute"].iloc[0] == 1
    assert df["spo2_pct"].iloc[0] == 92


def test_predict_urgency_lgbm_slm_format():
    slm_output = {
        "primary_system_flag": "sys_gastrointestinal",
        "red_flag_severe_pain": True,
        "red_flag_acute_onset": True
    }
    vitals = {"age": 76, "heart_rate": 109, "systolic_bp": 104, "spo2_pct": 92, "temp_f": 97.0}
    prediction = predict_urgency_lgbm(slm_output, vitals)
    assert prediction in ["P0", "P1", "P2", "P3"]
    assert prediction == "P1"


def test_predict_urgency_lgbm_diverse_dataset_sample():
    # CF_V2_0000 sample from careflow_diverse_slm.jsonl
    slm_sample = {
        "primary_system_flag": "sys_dermatological",
        "red_flag_severe_pain": False,
        "red_flag_acute_onset": True,
        "target_triage_boundary": "P2"
    }
    vitals_sample = {"age": 36, "heart_rate": 98, "systolic_bp": 129, "spo2_pct": 96, "temp_f": 99.4}
    pred = predict_urgency_lgbm(slm_sample, vitals_sample)
    assert pred == "P2"


def test_predict_urgency_lgbm_with_details():
    slm_output = {
        "primary_system_flag": "sys_cardiopulmonary",
        "red_flag_severe_pain": True,
        "red_flag_acute_onset": True
    }
    vitals = {"age": 70, "heart_rate": 145, "systolic_bp": 85, "spo2_pct": 82, "temp_f": 98.6}
    details = predict_urgency_lgbm_with_details(slm_output, vitals)

    assert "ml_model_triage" in details
    assert details["ml_model_triage"] == "P0"
    assert "probabilities" in details
    assert "feature_vector" in details
    assert details["feature_vector"]["sys_cardiopulmonary"] == 1


def test_triage_orchestrator_returns_both_signals():
    evidence_state = {
        "urgency_signals": [],
        "symptoms": ["chest pain"],
        "severity": "mild",
        "onset_and_duration": "3 days ago"
    }
    raw_messages = ["I have had a mild chest ache for 3 days."]

    res = triage_engine.evaluate(evidence_state, raw_messages, ai_status="ready")
    assert hasattr(res, "rule_based_triage")
    assert hasattr(res, "ml_model_triage")
    assert res.rule_based_triage is not None
    assert res.ml_model_triage in ["P0", "P1", "P2", "P3"]


def test_api_submit_case_returns_both_signals(client):
    # 1. Create a case
    create_res = client.post("/cases", json={"preferred_language": "English", "chief_complaint": "Severe stomach pain and vomiting blood"})
    assert create_res.status_code == 201
    case_id = create_res.json()["id"]

    # 2. Add patient message with vitals
    msg_res = client.post(f"/cases/{case_id}/messages", json={"content": "Vitals: HR 110 bpm, BP 95 mmHg, SpO2 93%, Temp 98.4 F. Acute severe pain since 2 hours."})
    assert msg_res.status_code == 200
    msg_data = msg_res.json()
    assert "rule_based_triage" in msg_data
    assert "ml_model_triage" in msg_data

    # 3. Submit case and check response payload
    submit_res = client.post(f"/cases/{case_id}/submit")
    assert submit_res.status_code == 200
    submit_data = submit_res.json()

    assert "rule_based_triage" in submit_data
    assert "ml_model_triage" in submit_data
    assert submit_data["rule_based_triage"] in ["P0", "P1", "P2", "P3", "ERR"]
    assert submit_data["ml_model_triage"] in ["P0", "P1", "P2", "P3"]
    assert "case" in submit_data
    assert "rule_based_triage" in submit_data["case"]
    assert "ml_model_triage" in submit_data["case"]
