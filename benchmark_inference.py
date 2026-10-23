import os
import sys
import time
import json
import requests
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from backend.config import settings

TEST_SCENARIOS = [
    {
        "id": "CARDIAC_EMERGENCY",
        "description": "Crushing chest pain radiating to left arm",
        "vitals": {"heart_rate": 105, "systolic_bp": 145, "spo2_pct": 95, "temp_f": 98.6},
        "history": ["[Patient]: I have severe crushing chest pain since 1 hour radiating to my left arm and jaw."]
    },
    {
        "id": "HINDI_ACUTE_FEVER",
        "description": "High fever with neck stiffness in Hindi",
        "vitals": {"heart_rate": 96, "systolic_bp": 120, "spo2_pct": 98, "temp_f": 103.2},
        "history": ["[Patient]: kal raat se bahut tez bukhar hai aur gardan me akadpan hai."]
    },
    {
        "id": "MILD_CHRONIC",
        "description": "Mild joint pain for 2 months",
        "vitals": {"heart_rate": 74, "systolic_bp": 118, "spo2_pct": 99, "temp_f": 98.4},
        "history": ["[Patient]: pichle do mahine se ghutne me halka dard hai chalte waqt."]
    }
]

EXPECTED_KEYS = [
    "primary_system_flag",
    "red_flag_severe_pain",
    "red_flag_acute_onset",
    "target_triage_boundary",
    "next_question_en",
    "next_question_hi"
]

def format_prompt(scenario):
    v = scenario["vitals"]
    vitals_str = f"HR {v['heart_rate']} bpm, BP {v['systolic_bp']} mmHg, SpO2 {v['spo2_pct']}%, Temp {v['temp_f']} F"
    history_str = "\n".join(scenario["history"])
    
    system_prompt = "You are CareFlow AI. Classify the affected system, extract red flags, and output ONE bilingual follow-up question in strict JSON with keys: primary_system_flag, red_flag_severe_pain, red_flag_acute_onset, target_triage_boundary, next_question_en, next_question_hi."
    user_prompt = f"<extracted_report>\n- Vitals: {vitals_str}\n</extracted_report>\n<conversation_history>\n{history_str}\n</conversation_history>"
    
    # Gemma standard format: System instructions inside the user turn
    return (
        f"<start_of_turn>user\n"
        f"{system_prompt}\n\n"
        f"{user_prompt}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

def test_hosted_api():
    print("\n" + "="*70)
    print("  [1/2] TESTING HOSTED API PERFORMANCE")
    print("="*70)
    print(f"  Endpoint: {settings.MEDGEMMA_ENDPOINT or 'NOT CONFIGURED'}")
    print(f"  Model ID: {settings.MEDGEMMA_MODEL_ID}")
    
    if not settings.MEDGEMMA_ENDPOINT:
        print("  [!] SKIPPED: MEDGEMMA_ENDPOINT is empty in .env.")
        return

    headers = {"Content-Type": "application/json"}
    if settings.MEDGEMMA_API_KEY:
        headers["Authorization"] = f"Bearer {settings.MEDGEMMA_API_KEY}"

    is_openai_style = "chat/completions" in settings.MEDGEMMA_ENDPOINT or "v1" in settings.MEDGEMMA_ENDPOINT

    for sc in TEST_SCENARIOS:
        prompt = format_prompt(sc)
        if is_openai_style:
            payload = {
                "model": settings.MEDGEMMA_MODEL_ID,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 150
            }
        else:
            payload = {
                "inputs": prompt,
                "parameters": {"max_new_tokens": 150, "temperature": 0.1, "return_full_text": False}
            }

        t0 = time.perf_counter()
        try:
            resp = requests.post(settings.MEDGEMMA_ENDPOINT, headers=headers, json=payload, timeout=settings.MODEL_TIMEOUT_SECONDS)
            lat_ms = (time.perf_counter() - t0) * 1000
            
            if resp.status_code == 200:
                data = resp.json()
                raw_text = ""
                if isinstance(data, list) and len(data) > 0:
                    raw_text = data[0].get("generated_text", "")
                elif isinstance(data, dict):
                    if "choices" in data:
                        raw_text = data["choices"][0].get("message", {}).get("content", "")
                    else:
                        raw_text = data.get("generated_text", str(data))
                
                print(f"\n  Scenario: {sc['id']} ({lat_ms:.1f} ms) [HTTP 200]")
                print(f"  Raw Output: {raw_text.strip()[:180]}...")
            elif resp.status_code == 401:
                print(f"\n  Scenario: {sc['id']} - [HTTP 401 Unauthorized]")
                print("  --> Your Hugging Face / API token is invalid or expired.")
                print("      Please visit https://huggingface.co/settings/tokens to create a fresh User Access Token.")
                break
            elif resp.status_code == 404:
                print(f"\n  Scenario: {sc['id']} - [HTTP 404 Not Found]")
                print(f"  --> The model endpoint '{settings.MEDGEMMA_ENDPOINT}' does not exist or requires approval.")
                break
            else:
                print(f"\n  Scenario: {sc['id']} - HTTP {resp.status_code}: {resp.text[:150]}")
        except Exception as e:
            print(f"\n  Scenario: {sc['id']} - Request Error: {e}")

def test_local_ollama():
    print("\n" + "="*70)
    print("  [2/2] TESTING LOCAL OLLAMA ('careflow-slm') PERFORMANCE")
    print("="*70)
    url = f"{settings.OLLAMA_BASE_URL}/api/generate"
    print(f"  Ollama URL: {url}")
    print(f"  Model:      {settings.OLLAMA_MODEL}")

    for sc in TEST_SCENARIOS:
        prompt = format_prompt(sc)
        payload = {
            "model": settings.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",  # Constrained grammar JSON decoding
            "options": {
                "temperature": 0.1,
                "num_predict": 150  # Prevent unbounded generation loops
            }
        }

        t0 = time.perf_counter()
        try:
            resp = requests.post(url, json=payload, timeout=30)
            lat_ms = (time.perf_counter() - t0) * 1000
            if resp.status_code == 200:
                res_json = resp.json().get("response", "")
                try:
                    parsed = json.loads(res_json)
                    missing = [k for k in EXPECTED_KEYS if k not in parsed]
                    valid_schema = (len(missing) == 0)
                    print(f"\n  Scenario: {sc['id']} ({lat_ms:.1f} ms)")
                    print(f"    - System:   {parsed.get('primary_system_flag')}")
                    print(f"    - Triage:   {parsed.get('target_triage_boundary')}")
                    print(f"    - Pain Red: {parsed.get('red_flag_severe_pain')} (type: {type(parsed.get('red_flag_severe_pain')).__name__})")
                    print(f"    - Acute:    {parsed.get('red_flag_acute_onset')}")
                    print(f"    - Q (EN):   {parsed.get('next_question_en')}")
                    print(f"    - Q (HI):   {parsed.get('next_question_hi')}")
                    print(f"    - Schema Valid: {valid_schema} (Missing: {missing})")
                except json.JSONDecodeError:
                    print(f"\n  Scenario: {sc['id']} ({lat_ms:.1f} ms) - Non-JSON Output: {res_json[:120]}")
            else:
                print(f"\n  Scenario: {sc['id']} - Ollama HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"\n  Scenario: {sc['id']} - Ollama Connection Error: {e}")

if __name__ == "__main__":
    test_hosted_api()
    test_local_ollama()
    print("\n" + "="*70)
    print("  Benchmark Complete")
    print("="*70)
