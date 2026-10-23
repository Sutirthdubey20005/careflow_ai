import os
import re
import json
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import pandas as pd
import joblib

logger = logging.getLogger("careflow.ml_triage")

# Ordered features matching careflow_diverse_ml.csv (and careflow_robust_ml.csv)
FEATURE_COLUMNS: List[str] = [
    "age",
    "heart_rate",
    "systolic_bp",
    "spo2_pct",
    "temp_f",
    "pain_severe",
    "onset_acute",
    "sys_cardiopulmonary",
    "sys_neurological",
    "sys_gastrointestinal",
    "sys_urogenital",
    "sys_musculoskeletal",
    "sys_dermatological",
    "sys_systemic",
    "sys_eent",
]
EXPECTED_FEATURE_COLUMNS = FEATURE_COLUMNS

# Population median fallbacks derived from careflow_diverse_ml.csv
DEFAULT_VITALS: Dict[str, Any] = {
    "age": 50,
    "heart_rate": 90,
    "systolic_bp": 125,
    "spo2_pct": 95,
    "temp_f": 98.6,
}

# Integer class mapping to clinical urgency categories
TARGET_PRIORITY_MAP: Dict[int, str] = {
    0: "P0",
    1: "P1",
    2: "P2",
    3: "P3",
}

_MODEL_INSTANCE: Any = None
_DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "lgbm_urgency_model.joblib"

# Prioritized search order for training datasets (prefers updated diverse dataset)
CANDIDATE_DATASET_PATHS = [
    Path(__file__).resolve().parent.parent / "careflow_final_dataset" / "careflow_diverse_ml.csv",
    Path(__file__).resolve().parent.parent / "careflow_diverse_ml.csv",
    Path(__file__).resolve().parent.parent / "careflow_robust_dataset" / "careflow_diverse_ml.csv",
    Path(__file__).resolve().parent.parent / "careflow_robust_dataset" / "careflow_robust_ml.csv",
]


def resolve_primary_dataset_path() -> Path:
    """Returns the most up-to-date available tabular training dataset path."""
    for path in CANDIDATE_DATASET_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError("No tabular triage dataset (careflow_diverse_ml.csv) found in candidate locations.")


def train_lgbm_model(
    dataset_path: Optional[Union[str, Path]] = None,
    save_path: Optional[Union[str, Path]] = None,
    class_weight: Union[str, Dict[int, float]] = "balanced"
) -> Any:
    """
    Trains a multiclass LightGBM classifier on the tabular dataset using
    cost-sensitive class weights for P0-P3 tiers and saves the model artifact.
    """
    import lightgbm as lgb

    data_file = Path(dataset_path) if dataset_path else resolve_primary_dataset_path()
    output_file = Path(save_path) if save_path else _DEFAULT_MODEL_PATH

    if not data_file.exists():
        raise FileNotFoundError(f"Training dataset not found at {data_file}")

    logger.info(f"Training cost-sensitive LightGBM urgency model from {data_file}...")
    df = pd.read_csv(data_file)
    X = df[FEATURE_COLUMNS]
    y = df["target_priority"]

    # Cost-sensitive multiclass classifier:
    # 'balanced' automatically adjusts weights inversely proportional to class frequencies,
    # penalizing under-triage of rare/critical tiers (P0, P1, P3).
    clf = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        class_weight=class_weight,
        random_state=42,
        verbose=-1,
    )
    clf.fit(X, y)

    # Attach training metadata
    setattr(clf, "_dataset_source", str(data_file.name))
    setattr(clf, "_dataset_mtime", data_file.stat().st_mtime)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_file)
    logger.info(f"LightGBM urgency model saved to {output_file} (source: {data_file.name})")
    return clf


def is_model_out_of_date(model_path: Path, dataset_path: Path, loaded_model: Optional[Any] = None) -> bool:
    """
    Checks if the model artifact is missing or out of date relative to the dataset.
    """
    if not model_path.exists():
        return True

    try:
        # Check file timestamp
        if dataset_path.exists() and dataset_path.stat().st_mtime > model_path.stat().st_mtime:
            return True

        # Check metadata source if available
        if loaded_model is not None:
            source = getattr(loaded_model, "_dataset_source", "")
            if source and source != dataset_path.name:
                return True
    except Exception as e:
        logger.debug(f"Error checking model freshness: {e}")

    return False


def get_lgbm_model(allow_auto_train: bool = False) -> Optional[Any]:
    """
    Retrieves or loads the cached LightGBM model singleton.
    Does NOT train a new classifier silently at application startup.
    Returns None if the artifact is missing or unreadable.
    """
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is not None:
        return _MODEL_INSTANCE

    if _DEFAULT_MODEL_PATH.exists():
        try:
            loaded = joblib.load(_DEFAULT_MODEL_PATH)
            logger.info(f"Loaded existing LightGBM model from {_DEFAULT_MODEL_PATH}")
            _MODEL_INSTANCE = loaded
            return _MODEL_INSTANCE
        except Exception as e:
            logger.warning(f"Failed to load existing model ({e})")
            if not allow_auto_train:
                return None

    if allow_auto_train:
        try:
            primary_dataset = resolve_primary_dataset_path()
            logger.info("Explicit auto-training enabled. Training model artifact...")
            _MODEL_INSTANCE = train_lgbm_model(primary_dataset, _DEFAULT_MODEL_PATH, class_weight="balanced")
            return _MODEL_INSTANCE
        except Exception as train_err:
            logger.warning(f"Auto-training failed: {train_err}")
            return None

    logger.warning(f"LightGBM model artifact not found at {_DEFAULT_MODEL_PATH}. Silent auto-training disabled.")
    return None


def extract_vitals_with_metadata(
    vitals_dict: Optional[Union[Dict[str, Any], str]] = None,
    text_content: Optional[str] = None
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Extracts continuous lab/vital parameters (age, heart_rate, systolic_bp, spo2_pct, temp_f)
    from a dictionary or parses them from clinical text/reports.
    Tracks and returns the list of imputed features populated from population defaults.
    """
    vitals = dict(DEFAULT_VITALS)
    imputed_fields = list(DEFAULT_VITALS.keys())

    # If vitals_dict was passed as string, treat as text_content or parse JSON
    if isinstance(vitals_dict, str):
        try:
            parsed = json.loads(vitals_dict)
            if isinstance(parsed, dict):
                vitals_dict = parsed
            else:
                text_content = (text_content or "") + " " + vitals_dict
                vitals_dict = None
        except Exception:
            text_content = (text_content or "") + " " + vitals_dict
            vitals_dict = None

    if isinstance(vitals_dict, dict):
        # Age
        if "age" in vitals_dict and vitals_dict["age"] is not None:
            try:
                vitals["age"] = int(vitals_dict["age"])
                if "age" in imputed_fields:
                    imputed_fields.remove("age")
            except (ValueError, TypeError):
                pass

        # Heart rate / Pulse
        for k in ("heart_rate", "hr", "pulse", "bpm"):
            if k in vitals_dict and vitals_dict[k] is not None:
                try:
                    vitals["heart_rate"] = int(vitals_dict[k])
                    if "heart_rate" in imputed_fields:
                        imputed_fields.remove("heart_rate")
                    break
                except (ValueError, TypeError):
                    pass

        # Systolic BP
        for k in ("systolic_bp", "systolic", "sbp", "bp_sys", "bp"):
            if k in vitals_dict and vitals_dict[k] is not None:
                val = str(vitals_dict[k])
                if "/" in val:
                    val = val.split("/")[0]
                try:
                    vitals["systolic_bp"] = int(float(re.sub(r"[^\d.]", "", val)))
                    if "systolic_bp" in imputed_fields:
                        imputed_fields.remove("systolic_bp")
                    break
                except (ValueError, TypeError):
                    pass

        # SpO2
        for k in ("spo2_pct", "spo2", "oximetry", "oxygen_saturation"):
            if k in vitals_dict and vitals_dict[k] is not None:
                try:
                    vitals["spo2_pct"] = int(float(re.sub(r"[^\d.]", "", str(vitals_dict[k]))))
                    if "spo2_pct" in imputed_fields:
                        imputed_fields.remove("spo2_pct")
                    break
                except (ValueError, TypeError):
                    pass

        # Temperature
        for k in ("temp_f", "temperature", "temp", "temp_deg_f"):
            if k in vitals_dict and vitals_dict[k] is not None:
                try:
                    t_val = float(re.sub(r"[^\d.]", "", str(vitals_dict[k])))
                    # Convert Celsius to Fahrenheit if needed
                    if t_val < 45.0:
                        t_val = (t_val * 9.0 / 5.0) + 32.0
                    vitals["temp_f"] = round(t_val, 1)
                    if "temp_f" in imputed_fields:
                        imputed_fields.remove("temp_f")
                    break
                except (ValueError, TypeError):
                    pass

    # If text is provided, extract missing vitals using regex
    if text_content:
        # HR regex (e.g. "HR 109 bpm", "Heart Rate: 128 bpm", "pulse 95")
        hr_match = re.search(r"\b(?:HR|pulse|heart\s*rate)\s*[:=]?\s*(\d{2,3})\s*(?:bpm)?\b", text_content, re.IGNORECASE)
        if hr_match and "heart_rate" in imputed_fields:
            vitals["heart_rate"] = int(hr_match.group(1))
            imputed_fields.remove("heart_rate")

        # BP regex (e.g. "BP 104/70 mmHg", "Systolic BP: 165 mmHg", "blood pressure 120")
        bp_match = re.search(r"\b(?:systolic\s*bp|systolic|bp|blood\s*pressure)\s*[:=]?\s*(\d{2,3})(?:/\d{2,3})?\s*(?:mmHg)?\b", text_content, re.IGNORECASE)
        if bp_match and "systolic_bp" in imputed_fields:
            vitals["systolic_bp"] = int(bp_match.group(1))
            imputed_fields.remove("systolic_bp")

        # SpO2 regex (e.g. "SpO2 92%", "SpO2: 91%", "oxygen 94%")
        spo2_match = re.search(r"\b(?:SpO2|O2\s*sat|oxygen)\s*[:=]?\s*(\d{2,3})\s*%?\b", text_content, re.IGNORECASE)
        if spo2_match and "spo2_pct" in imputed_fields:
            vitals["spo2_pct"] = int(spo2_match.group(1))
            imputed_fields.remove("spo2_pct")

        # Temp regex (e.g. "Temp 97.0 F", "Temp: 102.4 F", "temperature: 101.8 F")
        temp_match = re.search(r"\b(?:Temp|temperature)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:F|deg F)?\b", text_content, re.IGNORECASE)
        if temp_match and "temp_f" in imputed_fields:
            vitals["temp_f"] = float(temp_match.group(1))
            imputed_fields.remove("temp_f")

    return vitals, imputed_fields


def extract_vitals_from_dict_or_text(
    vitals_dict: Optional[Union[Dict[str, Any], str]] = None,
    text_content: Optional[str] = None
) -> Dict[str, Any]:
    """
    Backward-compatible wrapper returning only the extracted vitals dictionary.
    """
    vitals, _ = extract_vitals_with_metadata(vitals_dict, text_content)
    return vitals


def _parse_bool_flag(val: Any) -> int:
    """Robustly casts booleans, numbers, and string representations into 0 or 1."""
    if val is None:
        return 0
    if isinstance(val, bool):
        return 1 if val else 0
    if isinstance(val, (int, float)):
        return 1 if val > 0 else 0
    if isinstance(val, str):
        v_clean = val.strip().lower()
        if v_clean in ("true", "1", "yes", "y", "t", "acute", "severe", "positive"):
            return 1
        return 0
    return 1 if bool(val) else 0


def _map_slm_and_vitals_to_dataframe(
    slm_json_output: Optional[Union[Dict[str, Any], str]],
    lab_vitals_dict: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Maps MedGemma / SLM extracted features and patient vitals into a single-row
    DataFrame matching the exact 15-column schema required by LightGBM.
    """
    # Parse slm_json_output if passed as string
    slm_data: Dict[str, Any] = {}
    if isinstance(slm_json_output, str):
        try:
            slm_data = json.loads(slm_json_output)
        except Exception:
            slm_data = {"raw_text": slm_json_output}
    elif isinstance(slm_json_output, dict):
        slm_data = dict(slm_json_output)
    elif slm_json_output is not None:
        slm_data = getattr(slm_json_output, "model_dump", lambda: {})() or {}

    # If passed an entire evidence_state dict, unwrap nested slm_extraction and vitals
    if isinstance(slm_data, dict):
        if "slm_extraction" in slm_data and isinstance(slm_data["slm_extraction"], dict):
            # Merge slm_extraction keys into slm_data for direct access
            slm_data = {**slm_data["slm_extraction"], **slm_data}
        if lab_vitals_dict is None and "vitals" in slm_data and isinstance(slm_data["vitals"], dict):
            lab_vitals_dict = slm_data["vitals"]

    # Extract continuous vitals with explicit imputed fields tracking
    vitals, imputed_fields = extract_vitals_with_metadata(lab_vitals_dict)

    # Initialize feature row
    row: Dict[str, Any] = {
        "age": vitals["age"],
        "heart_rate": vitals["heart_rate"],
        "systolic_bp": vitals["systolic_bp"],
        "spo2_pct": vitals["spo2_pct"],
        "temp_f": vitals["temp_f"],
        "pain_severe": 0,
        "onset_acute": 0,
        "sys_cardiopulmonary": 0,
        "sys_neurological": 0,
        "sys_gastrointestinal": 0,
        "sys_urogenital": 0,
        "sys_musculoskeletal": 0,
        "sys_dermatological": 0,
        "sys_systemic": 0,
        "sys_eent": 0,
    }

    # 1. Map Pain Severity (pain_severe)
    if "red_flag_severe_pain" in slm_data:
        row["pain_severe"] = _parse_bool_flag(slm_data["red_flag_severe_pain"])
    elif "pain_severe" in slm_data:
        row["pain_severe"] = _parse_bool_flag(slm_data["pain_severe"])
    else:
        # Check MedGemma standard fields
        sev_str = str(slm_data.get("severity", "")).lower()
        if any(w in sev_str for w in ["severe", "unbearable", "excruciating", "8/10", "9/10", "10/10", "बहुत तेज"]):
            row["pain_severe"] = 1
        elif any(sig.get("severity_level") in ("emergency", "urgent") for sig in slm_data.get("urgency_signals", []) if isinstance(sig, dict)):
            row["pain_severe"] = 1

    # 2. Map Acute Onset (onset_acute)
    if "red_flag_acute_onset" in slm_data:
        row["onset_acute"] = _parse_bool_flag(slm_data["red_flag_acute_onset"])
    elif "onset_acute" in slm_data:
        row["onset_acute"] = _parse_bool_flag(slm_data["onset_acute"])
    else:
        onset_str = str(slm_data.get("onset_and_duration", "")).lower()
        if any(w in onset_str for w in ["sudden", "acute", "today", "hour", "minute", "kal se", "कल से", "achanak"]):
            row["onset_acute"] = 1
        else:
            row["onset_acute"] = 0

    # 3. Map System Flags (sys_*)
    # Check direct primary_system_flag from SLM with alias normalization
    primary_sys = str(slm_data.get("primary_system_flag", "")).strip().lower()
    if primary_sys:
        if not primary_sys.startswith("sys_"):
            primary_sys = f"sys_{primary_sys}"
        
        # Clinical system alias map
        alias_map = {
            "sys_cardio": "sys_cardiopulmonary",
            "sys_cardiac": "sys_cardiopulmonary",
            "sys_pulmonary": "sys_cardiopulmonary",
            "sys_respiratory": "sys_cardiopulmonary",
            "sys_neuro": "sys_neurological",
            "sys_gi": "sys_gastrointestinal",
            "sys_gastro": "sys_gastrointestinal",
            "sys_abdominal": "sys_gastrointestinal",
            "sys_uro": "sys_urogenital",
            "sys_renal": "sys_urogenital",
            "sys_urinary": "sys_urogenital",
            "sys_msk": "sys_musculoskeletal",
            "sys_ortho": "sys_musculoskeletal",
            "sys_derm": "sys_dermatological",
            "sys_skin": "sys_dermatological",
            "sys_ent": "sys_eent",
            "sys_eye": "sys_eent",
            "sys_ear": "sys_eent",
        }
        primary_sys = alias_map.get(primary_sys, primary_sys)

        if primary_sys in row:
            row[primary_sys] = 1

    # Also check direct sys_* keys if already binary in slm_data
    for col in FEATURE_COLUMNS:
        if col.startswith("sys_") and col in slm_data:
            if slm_data[col]:
                row[col] = 1

    # If no system flag was matched, infer from symptoms and complaint text
    if sum(row[c] for c in FEATURE_COLUMNS if c.startswith("sys_")) == 0:
        symptoms_text = ""
        if "symptoms" in slm_data and isinstance(slm_data["symptoms"], list):
            symptoms_text += " ".join(str(s) for s in slm_data["symptoms"])
        if "chief_complaint" in slm_data:
            symptoms_text += " " + str(slm_data["chief_complaint"])
        if "raw_text" in slm_data:
            symptoms_text += " " + str(slm_data["raw_text"])
        s_low = symptoms_text.lower()

        # Keyword matching heuristics
        if any(w in s_low for w in ["chest", "heart", "breath", "dyspnea", "cough", "palpitation", "सीने", "छाती", "सांस"]):
            row["sys_cardiopulmonary"] = 1
        elif any(w in s_low for w in ["headache", "dizzy", "head", "stroke", "seizure", "numbness", "सिरदर्द", "चक्कर"]):
            row["sys_neurological"] = 1
        elif any(w in s_low for w in ["stomach", "vomit", "abdomen", "nausea", "belly", "diarrhea", "pet", "पेट", "उल्टी"]):
            row["sys_gastrointestinal"] = 1
        elif any(w in s_low for w in ["urine", "urinary", "bladder", "kidney", "dysuria", "flank"]):
            row["sys_urogenital"] = 1
        elif any(w in s_low for w in ["joint", "bone", "back pain", "muscle", "fracture", "knee", "कमर", "हड्डी"]):
            row["sys_musculoskeletal"] = 1
        elif any(w in s_low for w in ["rash", "skin", "itch", "hives", "allergy", "चकत्ते", "खुजली"]):
            row["sys_dermatological"] = 1
        elif any(w in s_low for w in ["throat", "ear", "eye", "vision", "nose", "gala", "गला", "आंख", "कान"]):
            row["sys_eent"] = 1
        elif any(w in s_low for w in ["fever", "chills", "fatigue", "weakness", "infection", "बुखार"]):
            row["sys_systemic"] = 1
        else:
            # Default to sys_systemic for safety if completely undetermined
            row["sys_systemic"] = 1

    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    df.attrs["imputed_features"] = imputed_fields
    return df


def predict_urgency_lgbm(
    slm_json_output: Optional[Union[Dict[str, Any], str]] = None,
    lab_vitals_dict: Optional[Dict[str, Any]] = None
) -> str:
    """
    Main LightGBM Triage Predictor.
    
    1. Loads the trained LightGBM urgency model.
    2. Maps MedGemma / SLM JSON features and patient continuous vitals into the
       15-feature Pandas DataFrame schema matching careflow_diverse_ml.csv.
    3. Returns the predicted P0-P3 urgency tier (e.g. 'P0', 'P1', 'P2', 'P3').
    """
    try:
        model = get_lgbm_model()
        if model is None:
            return "P3"
        df_features = _map_slm_and_vitals_to_dataframe(slm_json_output, lab_vitals_dict)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df_features)[0]
            max_p = float(max(probs))
            min_p = float(min(probs))
            if max_p - min_p < 0.05:
                # Flat distribution / tie (e.g. 0.25 on all 4 classes)
                if df_features.get("pain_severe", [0])[0] == 1:
                    pred_class = 1
                elif df_features.get("onset_acute", [0])[0] == 1:
                    pred_class = 2
                else:
                    pred_class = 3
                return TARGET_PRIORITY_MAP.get(pred_class, "P3")
        pred_class = int(model.predict(df_features)[0])
        return TARGET_PRIORITY_MAP.get(pred_class, "P3")
    except Exception as e:
        logger.error(f"Error predicting urgency via LightGBM: {e}")
        # Safe fallback
        return "P2"


def predict_urgency_lgbm_with_details(
    slm_json_output: Optional[Union[Dict[str, Any], str]] = None,
    lab_vitals_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Predicts urgency and returns detailed class probabilities, feature vector,
    imputed features list, and model readiness status.
    """
    try:
        df_features = _map_slm_and_vitals_to_dataframe(slm_json_output, lab_vitals_dict)
        imputed_fields = getattr(df_features, "attrs", {}).get("imputed_features", [])

        model = get_lgbm_model()
        if model is None:
            return {
                "ml_model_triage": None,
                "predicted_tier": None,
                "predicted_class": None,
                "model_status": "unavailable",
                "probabilities": {},
                "feature_vector": df_features.iloc[0].to_dict(),
                "imputed_features": imputed_fields,
                "error": "LightGBM model artifact not loaded. Train via train_lgbm_model() or check backend/models."
            }

        pred_class = int(model.predict(df_features)[0])
        probabilities = {}
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df_features)[0]
            for idx, prob in enumerate(probs):
                tier = TARGET_PRIORITY_MAP.get(idx, f"Class_{idx}")
                probabilities[tier] = round(float(prob), 4)

            max_p = float(max(probs))
            min_p = float(min(probs))
            if max_p - min_p < 0.05:
                # Flat distribution / tie (e.g. 0.25 on all 4 classes)
                if df_features.get("pain_severe", [0])[0] == 1:
                    pred_class = 1
                elif df_features.get("onset_acute", [0])[0] == 1:
                    pred_class = 2
                else:
                    pred_class = 3

        tier_str = TARGET_PRIORITY_MAP.get(pred_class, "P3")
        return {
            "ml_model_triage": tier_str,
            "predicted_tier": tier_str,
            "predicted_class": pred_class,
            "model_status": "available",
            "probabilities": probabilities,
            "feature_vector": df_features.iloc[0].to_dict(),
            "imputed_features": imputed_fields,
        }
    except Exception as e:
        logger.error(f"Error in predict_urgency_lgbm_with_details: {e}")
        return {
            "ml_model_triage": None,
            "predicted_tier": None,
            "predicted_class": None,
            "model_status": "error",
            "probabilities": {},
            "feature_vector": {},
            "imputed_features": [],
            "error": str(e),
        }


def explain_triage_decision(
    slm_output: Optional[Union[Dict[str, Any], str]] = None,
    lab_vitals_dict: Optional[Dict[str, Any]] = None,
    rule_triage: Optional[str] = "P3",
    ml_triage: Optional[str] = "P3",
    probabilities: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Generates a multi-dimensional clinical explanation explaining:
    1. Primary organ system affected.
    2. Vital sign deviations from normal physiological baselines.
    3. Red-flag symptoms and acuity markers.
    4. Model probability breakdown and dual-signal consensus/divergence rationale.
    """
    slm_data: Dict[str, Any] = {}
    if isinstance(slm_output, str):
        try:
            slm_data = json.loads(slm_output)
        except Exception:
            slm_data = {}
    elif isinstance(slm_output, dict):
        slm_data = slm_output

    vitals = extract_vitals_from_dict_or_text(lab_vitals_dict)
    
    # Check vital sign deviations
    vital_deviations = []
    hr = vitals.get("heart_rate", 88)
    if hr > 120:
        vital_deviations.append(f"Severe Tachycardia: Heart Rate {hr} bpm (Normal: 60-100 bpm)")
    elif hr > 100:
        vital_deviations.append(f"Tachycardia: Heart Rate {hr} bpm (Normal: 60-100 bpm)")
    elif hr < 50:
        vital_deviations.append(f"Critical Bradycardia: Heart Rate {hr} bpm (Normal: 60-100 bpm)")

    bp = vitals.get("systolic_bp", 125)
    if bp < 90:
        vital_deviations.append(f"Hypotension / Shock Risk: Systolic BP {bp} mmHg (Normal: 100-130 mmHg)")
    elif bp > 180:
        vital_deviations.append(f"Hypertensive Crisis Risk: Systolic BP {bp} mmHg (Normal: 100-130 mmHg)")
    elif bp > 140:
        vital_deviations.append(f"Elevated Systolic Blood Pressure: {bp} mmHg")

    spo2 = vitals.get("spo2_pct", 96)
    if spo2 < 90:
        vital_deviations.append(f"Severe Hypoxemia: SpO2 {spo2}% (Normal: >= 95%) - High Risk")
    elif spo2 < 94:
        vital_deviations.append(f"Borderline Hypoxia: SpO2 {spo2}% (Normal: >= 95%)")

    temp = vitals.get("temp_f", 98.6)
    if temp >= 101.5:
        vital_deviations.append(f"High Fever / Hyperthermia: Temp {temp} °F (Normal: 97.5-99.5 °F)")
    elif temp > 99.8:
        vital_deviations.append(f"Low-Grade Fever: Temp {temp} °F")
    elif temp < 96.0:
        vital_deviations.append(f"Hypothermia: Temp {temp} °F")

    # Primary system identification
    sys_flag = slm_data.get("primary_system_flag", "sys_systemic")
    system_display_names = {
        "sys_cardiopulmonary": "Cardiopulmonary (Heart / Lungs)",
        "sys_neurological": "Neurological (Brain / Nervous System)",
        "sys_gastrointestinal": "Gastrointestinal (Digestive System)",
        "sys_urogenital": "Urogenital (Renal / Urinary)",
        "sys_musculoskeletal": "Musculoskeletal (Bones / Joints / Spine)",
        "sys_dermatological": "Dermatological (Skin / Integumentary)",
        "sys_systemic": "Systemic / Infectious",
        "sys_eent": "Eye, Ear, Nose, & Throat (EENT)",
    }
    primary_sys_name = system_display_names.get(sys_flag, sys_flag.replace("sys_", "").title())

    # Red flag extraction
    red_flags = []
    if slm_data.get("red_flag_severe_pain"):
        red_flags.append("Patient reported severe, intolerable pain intensity")
    if slm_data.get("red_flag_acute_onset"):
        red_flags.append("Acute, sudden symptom onset (< 24-48 hours)")

    # Dual-signal consensus analysis
    r_prio = rule_triage or "P3"
    m_prio = ml_triage or "P3"
    
    if r_prio == m_prio:
        consensus_text = f"Concordance: Both deterministic protocol rules and statistical LightGBM classifier agree on acuity tier {r_prio}."
        is_divergent = False
    else:
        is_divergent = True
        if r_prio in ("P0", "P1") and m_prio in ("P2", "P3"):
            consensus_text = (
                f"Safety Escalation: Rule-based safety guardrail escalated case to {r_prio} "
                f"due to clinical red-flag presence, while ML model predicted {m_prio} based on physiological vitals."
            )
        else:
            consensus_text = (
                f"Divergence: Rule-based triage classified as {r_prio}, while ML model predicted {m_prio}."
            )

    return {
        "primary_system": primary_sys_name,
        "primary_system_flag": sys_flag,
        "vital_deviations": vital_deviations,
        "has_abnormal_vitals": len(vital_deviations) > 0,
        "red_flags": red_flags,
        "has_red_flags": len(red_flags) > 0,
        "rule_based_triage": r_prio,
        "ml_model_triage": m_prio,
        "consensus_summary": consensus_text,
        "is_divergent": is_divergent,
        "probabilities": probabilities or {},
    }

# Upstream compatibility exports
COLUMNS = FEATURE_COLUMNS
CLASS_TO_TIER = TARGET_PRIORITY_MAP
MLTriageResult = dict

def construct_tabular_features(evidence_state: Dict[str, Any], lab_vitals: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Compatibility bridge for tabular feature construction."""
    return _map_slm_and_vitals_to_dataframe(evidence_state, lab_vitals)

def get_model():
    """Compatibility bridge for get_lgbm_model."""
    return get_lgbm_model()
