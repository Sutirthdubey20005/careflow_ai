#!/usr/bin/env python3
"""
CareFlow AI - A/B Model Showdown Script
Compares Fine-Tuned MedGemma ('careflow-slm') vs Base Model ('medgemma-4b-it')
Evaluates clinical relevance, Pydantic 6-field schema compliance, and deterministic vital sign triage.
"""
import os
import sys
import json
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Enforce no mock fallbacks
os.environ["ENABLE_MOCK_FALLBACK"] = "false"

from backend.config import settings
from backend.services.medgemma import medgemma_service
from backend.services.triage_rules import triage_engine
from backend.ml_triage_service import predict_urgency_lgbm_with_details


def run_single_inference(
    model_mode: str,
    provider_name: str,
    ollama_model_name: str,
    patient_input: str,
    patient_vitals: dict
) -> dict:
    """Runs a single inference turn and evaluates clinical triage rules."""
    # Configure backend settings for this run
    settings.MEDGEMMA_BACKEND_MODE = model_mode
    settings.DEFAULT_MEDGEMMA_PROVIDER = provider_name
    if model_mode == "base_ollama":
        settings.BASE_OLLAMA_MODEL = ollama_model_name
    elif model_mode == "finetuned_ollama":
        settings.OLLAMA_MODEL = ollama_model_name

    start_time = time.perf_counter()
    slm_turn = medgemma_service.generate_slm_triage_turn(
        conversation_history=[patient_input],
        vitals=patient_vitals,
        preferred_language="hi",
        provider=provider_name
    )
    elapsed_time = time.perf_counter() - start_time

    # Evaluate deterministic rule engine with continuous vitals
    current_evidence = {
        "vitals": patient_vitals,
        "slm_extraction": slm_turn
    }
    triage_result = triage_engine.evaluate(
        evidence_state=current_evidence,
        raw_messages=[patient_input],
        vitals=patient_vitals,
        ai_status="ready"
    )

    # Evaluate LightGBM
    ml_details = predict_urgency_lgbm_with_details(
        slm_json_output=slm_turn,
        lab_vitals_dict=patient_vitals
    )

    return {
        "mode": model_mode,
        "provider": provider_name,
        "model_name": ollama_model_name,
        "latency_sec": round(elapsed_time, 3),
        "slm_turn": slm_turn,
        "rule_based_triage": triage_result.rule_based_triage,
        "final_triage": triage_result.final_triage,
        "ml_model_triage": ml_details.get("predicted_tier", "P3"),
        "triggered_rules": [t.rule_id for t in triage_result.triggered_rules],
        "escalation_applied": triage_result.escalation_applied,
        "escalation_reason": triage_result.escalation_reason
    }


def main():
    print("=" * 80)
    print(" CAREFLOW AI: A/B MODEL SHOWDOWN (FINE-TUNED vs BASE)")
    print("=" * 80)

    patient_input = "bukhaar hai aur sans nhe liya ja rha"
    patient_vitals = {
        "heart_rate": 112,
        "systolic_bp": 128,
        "spo2_pct": 91,
        "temp_f": 102.8
    }

    print(f"\n[Test Case Input]")
    print(f"  Chief Complaint:  \"{patient_input}\"")
    print(f"  Recorded Vitals:  HR {patient_vitals['heart_rate']} bpm, SBP {patient_vitals['systolic_bp']} mmHg, "
          f"SpO2 {patient_vitals['spo2_pct']}%, Temp {patient_vitals['temp_f']} F")
    print(f"  Critical Marker:  SpO2 91% (CRITICAL HYPOXEMIA < 92% -> MANDATORY P0)")
    print("-" * 80)

    # Determine Base Model: prefers medgemma-4b-it, fallbacks if needed
    base_model = os.getenv("BASE_OLLAMA_MODEL", "medgemma-4b-it")

    # RUN 1: Fine-Tuned Model ('careflow-slm')
    print("\n>>> Executing RUN 1: Fine-Tuned Model ('careflow-slm:latest')...")
    res1 = run_single_inference(
        model_mode="finetuned_ollama",
        provider_name="finetuned_medgemma",
        ollama_model_name="careflow-slm",
        patient_input=patient_input,
        patient_vitals=patient_vitals
    )

    # RUN 2: Base Model ('medgemma-4b-it')
    print(f"\n>>> Executing RUN 2: Base Model ('{base_model}:latest')...")
    res2 = run_single_inference(
        model_mode="base_ollama",
        provider_name="standard_medgemma",
        ollama_model_name=base_model,
        patient_input=patient_input,
        patient_vitals=patient_vitals
    )

    # SIDE-BY-SIDE COMPARISON TABLE
    print("\n" + "=" * 80)
    print(" SIDE-BY-SIDE CLINICAL SHOWDOWN RESULTS")
    print("=" * 80)

    header = f"{'Metric / Field':<28} | {'RUN 1: Fine-Tuned (careflow-slm)':<30} | {'RUN 2: Base (' + base_model + ')':<30}"
    print(header)
    print("-" * len(header))

    def row(label, val1, val2):
        s1 = str(val1)[:30]
        s2 = str(val2)[:30]
        print(f"{label:<28} | {s1:<30} | {s2:<30}")

    slm1 = res1["slm_turn"]
    slm2 = res2["slm_turn"]

    row("Executed Runtime", res1["slm_turn"].get("executed_runtime", "ollama"), res2["slm_turn"].get("executed_runtime", "ollama"))
    row("Latency (seconds)", f"{res1['latency_sec']}s", f"{res2['latency_sec']}s")
    row("primary_system_flag", slm1.get("primary_system_flag"), slm2.get("primary_system_flag"))
    row("red_flag_severe_pain", slm1.get("red_flag_severe_pain"), slm2.get("red_flag_severe_pain"))
    row("red_flag_acute_onset", slm1.get("red_flag_acute_onset"), slm2.get("red_flag_acute_onset"))
    row("target_triage_boundary", slm1.get("target_triage_boundary"), slm2.get("target_triage_boundary"))
    row("rule_based_triage", res1["rule_based_triage"], res2["rule_based_triage"])
    row("ml_model_triage", res1["ml_model_triage"], res2["ml_model_triage"])
    row("final_triage", res1["final_triage"], res2["final_triage"])
    row("Triggered Rules", ", ".join(res1["triggered_rules"]), ", ".join(res2["triggered_rules"]))

    print("-" * len(header))
    print("\n[Detailed Follow-Up Questions Generated]")
    print(f"\n  Run 1 (Fine-Tuned):")
    print(f"    - English: {slm1.get('next_question_en')}")
    print(f"    - Hindi:   {slm1.get('next_question_hi')}")

    print(f"\n  Run 2 (Base Model):")
    print(f"    - English: {slm2.get('next_question_en')}")
    print(f"    - Hindi:   {slm2.get('next_question_hi')}")

    print("\n" + "=" * 80)
    print(" CLINICAL SAFETY VERIFICATION:")
    if res1["rule_based_triage"] == "P0" and res2["rule_based_triage"] == "P0":
        print(" [PASS] Vitals Blindness Fixed: Deterministic rule engine correctly returned P0 for both runs (SpO2=91%).")
    else:
        print(f" [FAIL] Vitals Check Failed: Run 1={res1['rule_based_triage']}, Run 2={res2['rule_based_triage']}")

    print("=" * 80)


if __name__ == "__main__":
    main()
