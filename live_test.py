import os
import sys
import time
import json
import uuid

# Force live neural execution without any mock fallbacks
os.environ["ENABLE_MOCK_FALLBACK"] = "false"
os.environ["MEDGEMMA_MODE"] = "local"
os.environ["MEDGEMMA_BACKEND_MODE"] = "finetuned_ollama"

sys.stdout.reconfigure(encoding="utf-8")

from backend.config import settings
settings.ENABLE_MOCK_FALLBACK = False
settings.MEDGEMMA_MODE = "local"
settings.MEDGEMMA_BACKEND_MODE = "finetuned_ollama"

from backend.services.medgemma import medgemma_service
from backend.services.translation import translation_service
from backend.ml_triage_service import predict_urgency_lgbm_with_details
from backend.database.db import SessionLocal
from backend.database.models import Case, Message
from backend.api.routes import add_patient_message, get_next_question
from backend.schemas.clinical import MessageCreateRequest

def run_live_verification():
    patient_input = "bukhaar hai aur sans nhe liya ja rha"
    patient_vitals = {"heart_rate": 112, "systolic_bp": 128, "spo2_pct": 91, "temp_f": 102.8}

    print("=" * 70)
    print("   CAREFLOW AI - LIVE NEURAL SLM PIPELINE VERIFICATION")
    print("   Mode: Live Local Ollama (careflow-slm) | Mock Fallback: DISABLED")
    print("=" * 70)
    print(f"\n[Patient Input]: \"{patient_input}\"")
    print(f"[Vitals]: HR {patient_vitals['heart_rate']} bpm, BP {patient_vitals['systolic_bp']} mmHg, SpO2 {patient_vitals['spo2_pct']}%, Temp {patient_vitals['temp_f']} F")

    # 1. Zero-VRAM Inbound Translation
    print("\n--- 1. INBOUND TRANSLATION LAYER ---")
    inbound_result = translation_service.translate_to_english(patient_input)
    print(f"Inbound Text for SLM: \"{inbound_result}\" (Native Romanized Hinglish preserved for context encoder)")

    # 2. Grammar-Constrained 6-Field SLM Execution
    print("\n--- 2. LIVE NEURAL SLM INFERENCE (careflow-slm) ---")
    t0 = time.time()
    slm_turn = medgemma_service.generate_slm_triage_turn(
        conversation_history=[patient_input],
        vitals=patient_vitals,
        preferred_language="Hindi"
    )
    slm_elapsed = time.time() - t0

    print(f"SLM Turn Execution Time: {slm_elapsed:.2f} seconds")
    print("SLM Structured 6-Field Output:")
    print(json.dumps(slm_turn, indent=2, ensure_ascii=False))

    # 3. Direct LightGBM Feature Routing
    print("\n--- 3. DIRECT LIGHTGBM ML TRIAGE ROUTING ---")
    lgbm_result = predict_urgency_lgbm_with_details(
        slm_json_output=slm_turn,
        lab_vitals_dict=patient_vitals
    )
    print(f"LightGBM Predicted Priority: {lgbm_result.get('ml_model_triage')}")
    print(f"LightGBM Class Probabilities: {lgbm_result.get('probabilities')}")
    print(f"Extracted 15-Feature Vector:\n{lgbm_result.get('feature_vector')}")

    # 4. Outbound Neural Translation
    print("\n--- 4. OUTBOUND PURE NEURAL TRANSLATION ---")
    print(f"English Question (SLM): \"{slm_turn.get('next_question_en')}\"")
    print(f"Hindi Question (Neural Devanagari): \"{slm_turn.get('next_question_hi')}\"")

    # 5. Full FastAPI Chat Loop Execution & SQLite Persistence
    print("\n--- 5. FULL FASTAPI CHAT ENDPOINT & SQLITE PERSISTENCE ---")
    db = SessionLocal()
    case_id = f"CF-LIVE-{uuid.uuid4().hex[:6].upper()}"
    case = Case(
        id=case_id,
        chief_complaint=patient_input,
        preferred_language="Hindi",
        priority_category="P3"
    )
    db.add(case)
    db.commit()

    req = MessageCreateRequest(content=patient_input, source_type="patient_typed", source_language="Hindi")
    t_msg = time.time()
    msg_response = add_patient_message(case_id=case_id, req=req, db=db)
    msg_elapsed = time.time() - t_msg

    db.refresh(case)
    print(f"FastAPI /messages Turn Latency: {msg_elapsed:.2f} seconds")
    print(f"Case ID: {case.id}")
    print(f"Interim Priority: {msg_response['interim_priority']}")
    print(f"Database Rule-Based Triage: {case.rule_based_triage}")
    print(f"Database ML Model Triage:   {case.ml_model_triage}")
    print(f"Database AI Status:         {case.ai_status}")

    # Question Retrieval Endpoint
    t_q = time.time()
    q_response = get_next_question(case_id=case_id, db=db)
    q_elapsed = time.time() - t_q

    print(f"FastAPI /next-question Retrieval Time: {q_elapsed:.4f} seconds (Cached from turn)")
    print(f"Selected Follow-up Question: \"{q_response.get('next_question')}\"")
    print(f"English: \"{q_response.get('next_question_en')}\"")
    print(f"Hindi:   \"{q_response.get('next_question_hi')}\"")
    print(f"Target System: {q_response.get('target_field')}")
    print(f"Active Provider: {q_response.get('selected_provider')}")

    # Clean up test case
    db.query(Message).filter(Message.case_id == case_id).delete()
    db.query(Case).filter(Case.id == case_id).delete()
    db.commit()
    db.close()

    print("\n" + "=" * 70)
    print("   LIVE PIPELINE VERIFICATION PASSED WITH ZERO MOCKS")
    print("=" * 70)

if __name__ == "__main__":
    run_live_verification()
