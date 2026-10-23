"""
Stand-alone LightGBM training script for CareFlow Urgency Classifier.
Trains an LGBMClassifier on careflow_lightgbm_training_dataset.csv and saves
the model artifact to models/lightgbm_triage.pkl.
"""

import os
from pathlib import Path
import pandas as pd
import lightgbm as lgb
import joblib

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

def train_and_save_model(dataset_path: str = "careflow_lightgbm_training_dataset.csv", output_path: str = "models/lightgbm_triage.pkl"):
    base_dir = Path(__file__).resolve().parent.parent
    csv_file = Path(dataset_path)
    if not csv_file.is_absolute():
        csv_file = base_dir / dataset_path

    if not csv_file.exists():
        raise FileNotFoundError(f"Training dataset not found at: {csv_file}")

    print(f"Loading training data from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    # 2. Map gender to binary ('M': 0.0, 'F': 1.0)
    if 'gender' in df.columns:
        if df['gender'].dtype == object or df['gender'].dtype == str:
            df['gender'] = df['gender'].astype(str).str.strip().str.upper().map({'M': 0.0, 'F': 1.0}).fillna(0.0)
        else:
            df['gender'] = df['gender'].astype(float)

    # 3. Target column is target_priority (0, 1, 2, 3)
    target_col = 'target_priority'
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset columns.")

    X = df[COLUMNS].astype(float)
    y = df[target_col].astype(int)

    print(f"Training features shape: {X.shape}, target classes: {sorted(y.unique())}")

    # 4. Fit LightGBM Classifier
    print("Fitting lightgbm.LGBMClassifier...")
    clf = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=4,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X, y)

    # 5. Save model artifact
    save_file = Path(output_path)
    if not save_file.is_absolute():
        save_file = base_dir / output_path
    save_file.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, save_file)
    print(f"Successfully trained and saved LightGBM model to: {save_file} (size: {save_file.stat().st_size} bytes)")
    return clf

if __name__ == "__main__":
    train_and_save_model()
