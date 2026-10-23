"""
CareFlow Tabular Urgency ML Triage Service.
Operationalizes LightGBM tabular classifier (trained on 41-feature schema)
alongside Google MedGemma clinical feature extractor.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib

logger = logging.getLogger("careflow.ml_triage")

# Project directories & model path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PRIMARY_MODEL_PATH = BASE_DIR / "models" / "lightgbm_triage.pkl"
SECONDARY_MODEL_PATH = BASE_DIR / "backend" / "models" / "lightgbm_triage.pkl"

# Complete 41-feature tabular schema in exact training order
COLUMNS = [
    'age', 'gender', 'bmi', 'heart_rate', 'sys_bp', 'dia_bp', 'map_bp', 'spo2_pct',
    'resp_rate', 'temp_f', 'shock_index', 'pain_scale_0_10', 'poc_glucose',
    'poc_lactate', 'poc_wbc', 'poc_troponin_pos', 'sys_cardiopulmonary',
    'sys_neurological', 'sys_gastrointestinal', 'sys_urogenital',
    'sys_musculoskeletal', 'sys_dermatological', 'sys_systemic', 'sys_eent',
    'hx_hypertension', 'hx_diabetes', 'hx_asthma_copd', 'hx_cad', 'hx_ckd',
    'hx_immunocompromised', 'rf_altered_mental_status', 'rf_active_bleeding',
    'rf_respiratory_distress', 'rf_chest_pain', 'rf_focal_weakness',
    'rf_high_fever', 'rf_severe_abdominal_pain', 'rf_unable_to_walk',
    'rf_diaphoresis', 'rf_slurred_speech', 'rf_visible_deformity'
]

CLASS_TO_TIER = {
    0: "P0",
    1: "P1",
    2: "P2",
    3: "P3"
}

_LOADED_MODEL = None


class MLTriageResult(dict):
    """
    Result dictionary from LightGBM prediction.
    Supports dictionary access (res['ml_predicted_priority']) as well as
    convenient string comparison (res == 'P1') and coercion (str(res) -> 'P1').
    """
    @property
    def ml_predicted_priority(self) -> str:
        return self.get("ml_predicted_priority", "P3")

    @property
    def ml_confidence(self) -> float:
        return float(self.get("ml_confidence", 0.0))

    def __str__(self) -> str:
        return self.get("ml_predicted_priority", "P3")

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.get("ml_predicted_priority") == other
        return super().__eq__(other)


def get_model():
    """Loads and caches the trained LightGBM model from disk."""
    global _LOADED_MODEL
    if _LOADED_MODEL is not None:
        return _LOADED_MODEL

    custom_path = os.getenv("LGBM_MODEL_PATH")
    candidate_paths = [
        Path(custom_path) if custom_path else None,
        PRIMARY_MODEL_PATH,
        SECONDARY_MODEL_PATH
    ]

    for p in candidate_paths:
        if p and p.exists():
            try:
                _LOADED_MODEL = joblib.load(p)
                logger.info(f"Loaded LightGBM model from {p}")
                return _LOADED_MODEL
            except Exception as e:
                logger.warning(f"Failed to load LightGBM model from {p}: {e}")

    logger.warning("No LightGBM model artifact found at designated paths.")
    return None


def _parse_pain_score(slm_json: Dict[str, Any], vitals: Dict[str, Any]) -> float:
    """Parses pain scale (0-10) from vitals or SLM JSON."""
    if "pain_scale_0_10" in vitals and vitals["pain_scale_0_10"] is not None:
        try:
            return float(vitals["pain_scale_0_10"])
        except (ValueError, TypeError):
            pass
    if "pain_scale" in vitals and vitals["pain_scale"] is not None:
        try:
            return float(vitals["pain_scale"])
        except (ValueError, TypeError):
            pass
    if "pain_scale_0_10" in slm_json and slm_json["pain_scale_0_10"] is not None:
        try:
            return float(slm_json["pain_scale_0_10"])
        except (ValueError, TypeError):
            pass

    # Inspect severity text
    sev = str(slm_json.get("severity", "")).lower()
    if not sev and "facts" in slm_json and isinstance(slm_json["facts"], dict):
        s_f = slm_json["facts"].get("severity", {})
        sev = str(s_f.get("value", "") if isinstance(s_f, dict) else s_f).lower()

    m = re.search(r"\b([0-9]|10)\b", sev)
    if m:
        return float(m.group(1))
    if any(w in sev for w in ["severe", "unbearable", "excruciating", "extreme"]):
        return 9.0
    if any(w in sev for w in ["moderate", "medium"]):
        return 5.0
    if any(w in sev for w in ["mild", "slight"]):
        return 2.0
    return 0.0


def _parse_temperature_f(vitals: Dict[str, Any]) -> float:
    """Normalizes patient temperature into Fahrenheit degrees."""
    raw_temp = vitals.get("temp_f") or vitals.get("temperature") or vitals.get("temp")
    if raw_temp is None:
        return 98.6
    try:
        val = float(raw_temp)
        # If in Celsius (< 50.0), convert to Fahrenheit
        if val < 50.0:
            return val * 9.0 / 5.0 + 32.0
        return val
    except (ValueError, TypeError):
        return 98.6


def construct_tabular_features(slm_json: Dict[str, Any], vitals_dict: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Constructs a single-row Pandas DataFrame containing all 41 feature columns
    in the exact training order, applying clinical derived formulas and vocabulary filtering.
    """
    slm = slm_json or {}
    vitals = dict(vitals_dict) if vitals_dict else {}

    # Check if vitals are nested in slm_json
    if not vitals:
        if "lab_vitals" in slm and isinstance(slm["lab_vitals"], dict):
            vitals = slm["lab_vitals"]
        elif "vitals" in slm and isinstance(slm["vitals"], dict):
            vitals = slm["vitals"]
        elif "facts" in slm and isinstance(slm["facts"], dict):
            v_val = slm["facts"].get("vitals") or slm["facts"].get("lab_vitals")
            if isinstance(v_val, dict):
                vitals = v_val.get("value", v_val) if isinstance(v_val.get("value"), dict) else v_val

    # 1. Base Vitals with Clinical Defaults
    age = float(vitals.get("age", 45.0))
    raw_gender = vitals.get("gender", 0.0)
    if isinstance(raw_gender, str):
        gender = 1.0 if raw_gender.strip().upper() == "F" else 0.0
    else:
        gender = float(raw_gender)

    bmi = float(vitals.get("bmi", 25.0))
    hr = float(vitals.get("heart_rate", 75.0))
    sys_bp = float(vitals.get("sys_bp", vitals.get("systolic_bp", 120.0)))
    dia_bp = float(vitals.get("dia_bp", vitals.get("diastolic_bp", 80.0)))
    spo2 = float(vitals.get("spo2_pct", vitals.get("spo2", 98.0)))
    rr = float(vitals.get("resp_rate", vitals.get("respiratory_rate", 16.0)))
    temp_f = _parse_temperature_f(vitals)
    pain = _parse_pain_score(slm, vitals)

    # 2. Derived Clinical Metrics
    shock_index = hr / max(sys_bp, 1.0)
    map_bp = (sys_bp + 2.0 * dia_bp) / 3.0

    # 3. Point-of-Care Laboratory Defaults
    poc_glucose = float(vitals.get("poc_glucose", vitals.get("blood_glucose", 100.0)))
    poc_lactate = float(vitals.get("poc_lactate", 1.0))
    poc_wbc = float(vitals.get("poc_wbc", 7.0))
    poc_troponin = float(vitals.get("poc_troponin_pos", 0.0))

    # Initialize feature row with 0.0
    row: Dict[str, float] = {col: 0.0 for col in COLUMNS}
    row.update({
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "heart_rate": hr,
        "sys_bp": sys_bp,
        "dia_bp": dia_bp,
        "map_bp": map_bp,
        "spo2_pct": spo2,
        "resp_rate": rr,
        "temp_f": temp_f,
        "shock_index": shock_index,
        "pain_scale_0_10": pain,
        "poc_glucose": poc_glucose,
        "poc_lactate": poc_lactate,
        "poc_wbc": poc_wbc,
        "poc_troponin_pos": poc_troponin,
    })

    # 4. Primary System Mapping
    primary_sys = str(slm.get("primary_system", "")).strip().lower()
    if not primary_sys.startswith("sys_") and primary_sys:
        primary_sys = f"sys_{primary_sys}"
    if primary_sys in COLUMNS:
        row[primary_sys] = 1.0
    else:
        # Fallback to sys_systemic if none or unmapped
        row["sys_systemic"] = 1.0

    # 5. Robust Vocabulary Filtering: Acuity Red Flags
    raw_red_flags = slm.get("acuity_red_flags", [])
    if isinstance(raw_red_flags, str):
        raw_red_flags = [raw_red_flags]
    for flag in raw_red_flags:
        cleaned_flag = str(flag).strip().lower()
        rf_col = f"rf_{cleaned_flag}" if not cleaned_flag.startswith("rf_") else cleaned_flag
        if rf_col in COLUMNS:
            row[rf_col] = 1.0

    # Also extract red flags from urgency_signals or symptoms if available
    urgency_sigs = slm.get("urgency_signals", [])
    if isinstance(urgency_sigs, list):
        for s in urgency_sigs:
            if isinstance(s, dict):
                sig_text = str(s.get("signal", "")).lower()
                if "chest" in sig_text or "coronary" in sig_text:
                    row["rf_chest_pain"] = 1.0
                if "breath" in sig_text or "respiratory" in sig_text:
                    row["rf_respiratory_distress"] = 1.0
                if "hemorrhage" in sig_text or "bleed" in sig_text or "hematemesis" in sig_text:
                    row["rf_active_bleeding"] = 1.0
                if "stroke" in sig_text or "facial" in sig_text or "weakness" in sig_text:
                    row["rf_focal_weakness"] = 1.0
                if "mening" in sig_text or "stiff neck" in sig_text or "fever" in sig_text:
                    row["rf_high_fever"] = 1.0
                if "abdomen" in sig_text or "peritoneal" in sig_text:
                    row["rf_severe_abdominal_pain"] = 1.0
                if "altered" in sig_text or "confusion" in sig_text or "syncope" in sig_text:
                    row["rf_altered_mental_status"] = 1.0

    # Check raw symptoms for obvious red flag keywords if red flags are empty
    symptoms = slm.get("symptoms", []) or slm.get("symptoms_reported", [])
    if not isinstance(symptoms, list):
        symptoms = [str(symptoms)]
    symp_lower = [str(s).lower() for s in symptoms]
    for s in symp_lower:
        if "chest" in s or "सीने" in s or "chhati" in s or "dil" in s:
            row["rf_chest_pain"] = 1.0
        if "breath" in s or "dyspnea" in s or "saans" in s or "सांस" in s:
            row["rf_respiratory_distress"] = 1.0
        if "vomit" in s and "blood" in s or "hematemesis" in s or "khoon" in s:
            row["rf_active_bleeding"] = 1.0
        if "stiff neck" in s or "gardan akad" in s:
            row["rf_high_fever"] = 1.0
        if "cannot walk" in s or "unable to walk" in s:
            row["rf_unable_to_walk"] = 1.0

    # 6. Robust Vocabulary Filtering: Comorbidities
    raw_comorbidities = slm.get("comorbidities_identified", [])
    if isinstance(raw_comorbidities, str):
        raw_comorbidities = [raw_comorbidities]
    for hx in raw_comorbidities:
        cleaned_hx = str(hx).strip().lower()
        hx_col = f"hx_{cleaned_hx}" if not cleaned_hx.startswith("hx_") else cleaned_hx
        if hx_col in COLUMNS:
            row[hx_col] = 1.0

    # Also check medical_history list
    hist = slm.get("medical_history", []) or slm.get("relevant_medical_history", [])
    if isinstance(hist, list):
        for h in hist:
            h_low = str(h).lower()
            if "hyper" in h_low or "bp" in h_low:
                row["hx_hypertension"] = 1.0
            if "diabet" in h_low or "sugar" in h_low:
                row["hx_diabetes"] = 1.0
            if "asthma" in h_low or "copd" in h_low:
                row["hx_asthma_copd"] = 1.0
            if "cad" in h_low or "cardiac" in h_low or "heart" in h_low:
                row["hx_cad"] = 1.0
            if "ckd" in h_low or "renal" in h_low or "kidney" in h_low:
                row["hx_ckd"] = 1.0
            if "cancer" in h_low or "chemo" in h_low or "immune" in h_low or "hiv" in h_low:
                row["hx_immunocompromised"] = 1.0

    # Construct single-row DataFrame with exact columns
    df = pd.DataFrame([row], columns=COLUMNS)
    return df


def predict_urgency_lgbm(
    slm_json: Dict[str, Any],
    vitals_dict: Optional[Dict[str, Any]] = None
) -> MLTriageResult:
    """
    Predicts patient clinical urgency tier using the trained LightGBM classifier.
    
    Parameters:
    - slm_json: MedGemma extraction dictionary (primary_system, acuity_red_flags, comorbidities_identified, etc.)
    - vitals_dict: Patient continuous vitals and lab dictionary
    
    Returns:
    - MLTriageResult dict with keys:
      {
          "ml_predicted_priority": "P0" | "P1" | "P2" | "P3",
          "ml_confidence": float (e.g. 0.89),
          "status": "success" | "model_artifact_missing"
      }
    """
    model = get_model()
    if model is None:
        logger.warning("LightGBM model artifact not found. Returning graceful fallback P2.")
        return MLTriageResult({
            "ml_predicted_priority": "P2",
            "ml_confidence": 0.0,
            "status": "model_artifact_missing"
        })

    try:
        # 1. Construct single-row DataFrame matching 41-feature schema
        df = construct_tabular_features(slm_json, vitals_dict)

        # 2. Run model prediction & probabilities
        preds = model.predict(df[COLUMNS])
        proba = model.predict_proba(df[COLUMNS])

        pred_class = int(preds[0])
        priority_str = CLASS_TO_TIER.get(pred_class, "P3")
        confidence_score = float(np.max(proba[0])) if (isinstance(proba, np.ndarray) and proba.ndim == 2) else 1.0

        logger.info(f"LightGBM predicted urgency: {priority_str} with {confidence_score*100:.1f}% confidence")
        return MLTriageResult({
            "ml_predicted_priority": priority_str,
            "ml_confidence": round(confidence_score, 4),
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error during LightGBM inference: {e}")
        return MLTriageResult({
            "ml_predicted_priority": "P2",
            "ml_confidence": 0.0,
            "status": "model_inference_error"
        })
