import pytest
import numpy as np
import pandas as pd
from backend.ml_triage_service import (
    FEATURE_COLUMNS,
    DEFAULT_VITALS,
    _map_slm_and_vitals_to_dataframe,
    predict_urgency_lgbm,
    predict_urgency_lgbm_with_details,
    extract_vitals_from_dict_or_text,
)


def test_failure_point_1_json_key_mapping():
    """
    Failure Point 1: Verify exact keys (primary_system_flag, red_flag_severe_pain, red_flag_acute_onset)
    are extracted without KeyErrors across all possible input configurations.
    """
    # 1. Standard SLM output dictionary
    slm_standard = {
        "primary_system_flag": "sys_cardiopulmonary",
        "red_flag_severe_pain": True,
        "red_flag_acute_onset": False,
        "target_triage_boundary": "P1",
        "next_question_en": "Are you feeling chest tightness?",
        "next_question_hi": "Kya aapko seene me jakdan hai?",
    }
    df1 = _map_slm_and_vitals_to_dataframe(slm_standard)
    assert df1["sys_cardiopulmonary"].iloc[0] == 1
    assert df1["pain_severe"].iloc[0] == 1
    assert df1["onset_acute"].iloc[0] == 0

    # 2. Empty dictionary (keys missing entirely - must not raise KeyError)
    df2 = _map_slm_and_vitals_to_dataframe({})
    assert isinstance(df2, pd.DataFrame)
    assert list(df2.columns) == FEATURE_COLUMNS
    # Systemic fallback when no system flag is provided
    assert df2["sys_systemic"].iloc[0] == 1

    # 3. None input (must not raise KeyError)
    df3 = _map_slm_and_vitals_to_dataframe(None)
    assert isinstance(df3, pd.DataFrame)
    assert list(df3.columns) == FEATURE_COLUMNS

    # 4. JSON string format
    json_str = '{"primary_system_flag": "sys_neurological", "red_flag_severe_pain": true, "red_flag_acute_onset": true}'
    df4 = _map_slm_and_vitals_to_dataframe(json_str)
    assert df4["sys_neurological"].iloc[0] == 1
    assert df4["pain_severe"].iloc[0] == 1
    assert df4["onset_acute"].iloc[0] == 1

    # 5. Nested in evidence_state (as passed from triage_rules.py)
    evidence_state = {
        "facts": {"chief_complaint": "Severe chest pressure"},
        "vitals": {"heart_rate": 115, "systolic_bp": 160},
        "slm_extraction": {
            "primary_system_flag": "sys_cardiopulmonary",
            "red_flag_severe_pain": True,
            "red_flag_acute_onset": True,
        }
    }
    df5 = _map_slm_and_vitals_to_dataframe(evidence_state)
    assert df5["sys_cardiopulmonary"].iloc[0] == 1
    assert df5["pain_severe"].iloc[0] == 1
    assert df5["onset_acute"].iloc[0] == 1
    assert df5["heart_rate"].iloc[0] == 115
    assert df5["systolic_bp"].iloc[0] == 160


def test_failure_point_2_type_casting():
    """
    Failure Point 2: Verify boolean red flags are properly converted into int (0/1)
    and match the numeric dtypes expected by LightGBM.
    """
    # Test boolean True/False
    slm_bools = {
        "primary_system_flag": "sys_gastrointestinal",
        "red_flag_severe_pain": False,
        "red_flag_acute_onset": True,
    }
    df = _map_slm_and_vitals_to_dataframe(slm_bools)
    assert df["pain_severe"].iloc[0] == 0
    assert df["onset_acute"].iloc[0] == 1
    assert pd.api.types.is_integer_dtype(df["pain_severe"])
    assert pd.api.types.is_integer_dtype(df["onset_acute"])

    # Test string representations ("true", "false", "0", "1")
    slm_str_bools = {
        "primary_system_flag": "sys_urogenital",
        "red_flag_severe_pain": "true",
        "red_flag_acute_onset": "false",
    }
    df_str = _map_slm_and_vitals_to_dataframe(slm_str_bools)
    assert df_str["pain_severe"].iloc[0] == 1
    assert df_str["onset_acute"].iloc[0] == 0

    # Ensure all columns in DataFrame have numeric dtypes
    for col in FEATURE_COLUMNS:
        assert pd.api.types.is_numeric_dtype(df_str[col]), f"Column {col} is not numeric!"


def test_failure_point_3_one_hot_encoding_system_flags():
    """
    Failure Point 3: Verify primary_system_flag is correctly mapped to exactly one
    of the 8 organ system indicator columns expected by LightGBM.
    """
    system_columns = [c for c in FEATURE_COLUMNS if c.startswith("sys_")]
    assert len(system_columns) == 8

    # 1. Test canonical sys_* flags
    for sys_col in system_columns:
        slm = {"primary_system_flag": sys_col}
        df = _map_slm_and_vitals_to_dataframe(slm)
        assert df[sys_col].iloc[0] == 1, f"Failed to activate {sys_col}"
        # All other 7 must be 0
        other_cols = [c for c in system_columns if c != sys_col]
        assert df[other_cols].sum(axis=1).iloc[0] == 0, f"Other system columns were active for {sys_col}"

    # 2. Test without 'sys_' prefix (e.g. 'cardiopulmonary')
    slm_no_prefix = {"primary_system_flag": "cardiopulmonary"}
    df_np = _map_slm_and_vitals_to_dataframe(slm_no_prefix)
    assert df_np["sys_cardiopulmonary"].iloc[0] == 1

    # 3. Test clinical aliases ('cardio', 'neuro', 'gi', 'derm', 'ent', 'msk')
    aliases = {
        "cardio": "sys_cardiopulmonary",
        "neuro": "sys_neurological",
        "gi": "sys_gastrointestinal",
        "derm": "sys_dermatological",
        "ent": "sys_eent",
        "msk": "sys_musculoskeletal",
        "uro": "sys_urogenital",
    }
    for alias, canonical in aliases.items():
        df_alias = _map_slm_and_vitals_to_dataframe({"primary_system_flag": alias})
        assert df_alias[canonical].iloc[0] == 1, f"Alias {alias} did not map to {canonical}"


def test_failure_point_4_missing_vitals_fallback():
    """
    Failure Point 4: Verify that if OCR vitals are missing, None, NaN, or malformed,
    LightGBM handles them gracefully with population median defaults without crashing.
    """
    # 1. Missing vitals completely (None, empty dict)
    pred1 = predict_urgency_lgbm(None, None)
    assert pred1 in ("P0", "P1", "P2", "P3")

    pred2 = predict_urgency_lgbm({}, {})
    assert pred2 in ("P0", "P1", "P2", "P3")

    # 2. Vitals dictionary with NaNs, Nones, and malformed strings
    nasty_vitals = {
        "age": np.nan,
        "heart_rate": None,
        "systolic_bp": "N/A",
        "spo2_pct": float("nan"),
        "temp_f": "unknown"
    }
    extracted = extract_vitals_from_dict_or_text(nasty_vitals)
    assert extracted["age"] == DEFAULT_VITALS["age"]
    assert extracted["heart_rate"] == DEFAULT_VITALS["heart_rate"]
    assert extracted["systolic_bp"] == DEFAULT_VITALS["systolic_bp"]
    assert extracted["spo2_pct"] == DEFAULT_VITALS["spo2_pct"]
    assert extracted["temp_f"] == DEFAULT_VITALS["temp_f"]

    # Predict must not crash and must return valid tier
    pred3 = predict_urgency_lgbm(None, nasty_vitals)
    assert pred3 in ("P0", "P1", "P2", "P3")

    # 3. Partial vitals (e.g. OCR only found HR and BP)
    partial_vitals = {"heart_rate": 135, "systolic_bp": 85}
    details = predict_urgency_lgbm_with_details(
        {"primary_system_flag": "sys_cardiopulmonary", "red_flag_severe_pain": True, "red_flag_acute_onset": True},
        partial_vitals
    )
    assert details["predicted_tier"] in ("P0", "P1", "P2", "P3")
    assert details["feature_vector"]["heart_rate"] == 135
    assert details["feature_vector"]["systolic_bp"] == 85
    # Missing SpO2 and Temp filled from defaults
    assert details["feature_vector"]["spo2_pct"] == DEFAULT_VITALS["spo2_pct"]
    assert details["feature_vector"]["temp_f"] == DEFAULT_VITALS["temp_f"]
    assert len(details["probabilities"]) > 0
