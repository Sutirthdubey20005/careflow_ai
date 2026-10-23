import json
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.medgemma import medgemma_service
from backend.services.questioning import adaptive_questioning_service
from backend.ml_triage_service import explain_triage_decision, predict_urgency_lgbm_with_details

client = TestClient(app)


def test_medgemma_service_adapter_status():
    status = medgemma_service.get_status()
    assert "has_adapter" in status
    assert status["adapter_name"] == "careflow-medgemma-final"
    assert status["adapter_rank"] == 8
    assert status["adapter_alpha"] == 16
    assert status["training_turns"] == 6000
    assert status["is_ready"] is True


def test_medgemma_slm_triage_turn_cardiopulmonary():
    dialogue = ["Seene me bahut tez dabaav aur dard ho raha hai kal se, baaye haath me dard jaa raha hai."]
    vitals = {"heart_rate": 115, "systolic_bp": 145, "spo2_pct": 93, "temp_f": 98.6, "age": 55}
    
    turn = medgemma_service.generate_slm_triage_turn(dialogue, vitals, preferred_language="Hinglish")
    
    assert turn["primary_system_flag"] == "sys_cardiopulmonary"
    assert turn["target_triage_boundary"] in ["P0", "P1"]
    assert "next_question_en" in turn and len(turn["next_question_en"]) > 10
    assert "next_question_hi" in turn and len(turn["next_question_hi"]) > 10
    assert isinstance(turn["red_flag_severe_pain"], bool)
    assert isinstance(turn["red_flag_acute_onset"], bool)


def test_medgemma_slm_triage_turn_neurological():
    dialogue = ["Severe throbbing headache since this morning with neck stiffness and photophobia."]
    vitals = {"heart_rate": 92, "systolic_bp": 130, "spo2_pct": 98, "temp_f": 101.8, "age": 32}
    
    turn = medgemma_service.generate_slm_triage_turn(dialogue, vitals, preferred_language="English")
    
    assert turn["primary_system_flag"] == "sys_neurological"
    assert turn["red_flag_severe_pain"] is True
    assert turn["red_flag_acute_onset"] is True
    assert "neck" in turn["next_question_en"].lower() or "vision" in turn["next_question_en"].lower()


def test_medgemma_slm_triage_turn_gastrointestinal_hematemesis():
    dialogue = ["Khoon ki ulti hui hai subah se do baar."]
    vitals = {"heart_rate": 110, "systolic_bp": 92, "spo2_pct": 97, "temp_f": 98.4, "age": 48}
    
    turn = medgemma_service.generate_slm_triage_turn(dialogue, vitals, preferred_language="Hinglish")
    
    assert turn["primary_system_flag"] == "sys_gastrointestinal"
    assert turn["target_triage_boundary"] == "P0"


def test_adaptive_questioning_service_slm_integration():
    patient_msgs = ["I have had an itchy red rash all over my arms since yesterday."]
    vitals = {"heart_rate": 78, "systolic_bp": 120, "spo2_pct": 99, "temp_f": 98.6}
    
    q_output = adaptive_questioning_service.generate_next_question(
        patient_messages=patient_msgs,
        current_question_count=0,
        assistant_messages=[],
        evidence_state={"vitals": vitals},
        preferred_language="English"
    )
    
    assert q_output.next_question is not None
    assert q_output.is_terminal is False
    assert q_output.slm_extraction is not None
    assert q_output.slm_extraction.get("primary_system_flag") == "sys_dermatological"


def test_api_ai_models_status():
    resp = client.get("/ai-models/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert "medgemma_slm" in data["models"]
    assert "lightgbm_classifier" in data["models"]
    assert "deepgram_speech" in data["models"]
    assert "document_ocr" in data["models"]
    assert data["models"]["lightgbm_classifier"]["num_features"] == 15


def test_api_run_live_pipeline():
    payload = {
        "text": "Seene me bahut tez dabaav aur dard hai, baaye haath me dard jaa raha hai.",
        "vitals": {
            "heart_rate": 115,
            "systolic_bp": 140,
            "spo2_pct": 93,
            "temp_f": 98.6,
            "age": 62
        },
        "preferred_language": "Hinglish"
    }
    resp = client.post("/ai-pipeline/run", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    
    assert "slm_turn" in data
    assert data["slm_turn"]["primary_system_flag"] == "sys_cardiopulmonary"
    assert "ml_classifier" in data
    assert data["ml_classifier"]["predicted_tier"] in ["P0", "P1"]
    assert "rule_based_triage" in data
    assert data["rule_based_triage"]["priority"] == "P0"
    assert "clinical_explanation" in data
    assert "vital_deviations" in data["clinical_explanation"]


def test_case_detail_includes_clinical_explanation():
    # 1. Create case
    c_resp = client.post("/cases", json={"preferred_language": "English", "chief_complaint": "Acute central chest pain"})
    assert c_resp.status_code == 201
    cid = c_resp.json()["id"]

    # 2. Add message
    m_resp = client.post(f"/cases/{cid}/messages", json={"content": "Crushing chest pain radiating to my left arm."})
    assert m_resp.status_code == 200

    # 3. Get case
    get_resp = client.get(f"/cases/{cid}")
    assert get_resp.status_code == 200
    case_detail = get_resp.json()
    
    assert "clinical_explanation" in case_detail
    assert "primary_system" in case_detail["clinical_explanation"]
    assert "consensus_summary" in case_detail["clinical_explanation"]
    assert "ml_details" in case_detail


def test_document_ocr_vitals_ingestion():
    # 1. Create case
    c_resp = client.post("/cases", json={"preferred_language": "English", "chief_complaint": "Lab report review"})
    cid = c_resp.json()["id"]

    # 2. Upload text document with vitals
    doc_content = "EMERGENCY TRIAGE ASSESSMENT\nVitals:\nHeart Rate: 128 bpm\nSystolic BP: 165 mmHg\nSpO2: 91%\nTemp: 102.4 F\nPatient complaining of severe headache."
    files = {"file": ("report.txt", doc_content.encode("utf-8"), "text/plain")}
    up_resp = client.post(f"/cases/{cid}/upload-document", files=files)
    assert up_resp.status_code == 200

    # 3. Verify case has ingested vitals and updated ML triage
    get_resp = client.get(f"/cases/{cid}")
    case_data = get_resp.json()
    evidence_st = case_data.get("evidence_state", {})
    vitals = evidence_st.get("vitals", {})
    
    assert vitals.get("heart_rate") == 128
    assert vitals.get("systolic_bp") == 165
    assert vitals.get("spo2_pct") == 91
    assert vitals.get("temp_f") == 102.4
    assert case_data.get("ml_model_triage") in ["P0", "P1"]
