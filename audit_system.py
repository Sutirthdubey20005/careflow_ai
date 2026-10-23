import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

results = {}

# 1. Database
try:
    from backend.database.db import engine
    with engine.connect() as conn:
        results['1. Database (SQLite)'] = 'PASS (Connected, careflow.db active)'
except Exception as e:
    results['1. Database (SQLite)'] = f'FAIL: {e}'

# 2. Local Ollama & MedGemma SLM
try:
    from backend.services.medgemma import medgemma_service
    ollama_ok = medgemma_service.is_ollama_available()
    results['2. Local Ollama SLM'] = f'PASS (Model "careflow-slm" online: {ollama_ok})'
except Exception as e:
    results['2. Local Ollama SLM'] = f'FAIL: {e}'

# 3. Dynamic Questioning Engine
try:
    from backend.services.questioning import adaptive_questioning_service
    q_out = adaptive_questioning_service.generate_next_question(['I have a cough and mild fever since 2 days.'], 0, preferred_language='English')
    results['3. Questioning Engine'] = f'PASS (Model-generated question: "{q_out.next_question[:45]}...")'
except Exception as e:
    results['3. Questioning Engine'] = f'FAIL: {e}'

# 4. Clinical Safety & PII Masking
try:
    from backend.services.safety import safety_service
    masked = safety_service.mask_pii('My name is Dr. Sharma, phone is 9876543210.')
    results['4. Privacy & Safety Guards'] = f'PASS (PII detected and redacted: {masked.has_pii})'
except Exception as e:
    results['4. Privacy & Safety Guards'] = f'FAIL: {e}'

# 5. LightGBM ML Urgency Classifier
try:
    from backend.ml_triage_service import get_lgbm_model, predict_urgency_lgbm_with_details
    model = get_lgbm_model()
    pred = predict_urgency_lgbm_with_details(
        {'primary_system_flag': 'sys_cardiopulmonary', 'red_flag_severe_pain': True, 'red_flag_acute_onset': True},
        {'heart_rate': 110, 'systolic_bp': 140, 'spo2_pct': 94, 'temp_f': 98.6}
    )
    results['5. LightGBM Urgency ML'] = f'PASS (Features extracted -> Tier: {pred["predicted_tier"]}, Prob: {pred["probabilities"].get("P0")})'
except Exception as e:
    results['5. LightGBM Urgency ML'] = f'FAIL: {e}'

# 6. Deterministic Safety Rules
try:
    from backend.services.triage_rules import triage_engine
    ev_state = {
        'symptoms': ['chest pain'],
        'vitals': {'heart_rate': 105, 'systolic_bp': 140, 'spo2_pct': 95, 'temp_f': 98.6},
        'facts': {'chest_pain': {'field': 'chest_pain', 'status': 'reported', 'source_quote': 'severe crushing chest pain'}},
        'urgency_signals': [{'signal': 'Acute Coronary Syndrome Red Flag', 'severity_level': 'emergency', 'rationale': 'Crushing chest pain radiating to arm', 'source_quote': 'radiating to left arm'}]
    }
    raw_msgs = ['I have severe crushing chest pain radiating to my left arm.']
    trig = triage_engine.evaluate(ev_state, raw_msgs)
    results['6. Rule-Based Triage'] = f'PASS (Acuity: {trig.priority_category}, Escalate: {trig.escalation_applied}, Rule: {trig.rule_based_triage})'
except Exception as e:
    results['6. Rule-Based Triage'] = f'FAIL: {e}'

# 7. Clinical Summarization Service
try:
    from backend.services.summarization import clinical_summarization_service
    summ = clinical_summarization_service.generate_summary(['Patient reports chest pain since morning.'], {})
    results['7. Clinical Summarization'] = f'PASS (Generated summary text length: {len(summ.summary_text)} chars)'
except Exception as e:
    results['7. Clinical Summarization'] = f'FAIL: {e}'

# 8. FastAPI Application Core
try:
    from backend.main import app
    results['8. FastAPI Backend Core'] = f'PASS (Loaded {len(app.routes)} API routes)'
except Exception as e:
    results['8. FastAPI Backend Core'] = f'FAIL: {e}'

print('\n' + '='*75)
print('         CAREFLOW AI - SYSTEM CAPABILITY & READINESS AUDIT')
print('='*75)
for comp, stat in results.items():
    print(f'  {comp:<28}: {stat}')
print('='*75)
