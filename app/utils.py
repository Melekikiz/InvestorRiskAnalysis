# app/utils.py
import os
import pandas as pd
import numpy as np
import joblib
import shap
from scipy.spatial.distance import cdist

# === FEATURE_COLUMNS ===
FEATURE_COLUMNS = [
    "age",
    "income",
    "experience_years",
    "risk_survey_score",
    "total_transactions",
    "avg_amount",
    "active_days",
    "monthly_freq",
    "tx_count_30d",
    "pct_crypto",
    "pct_etf",
    "pct_forex",
    "pct_other",
    "pct_stocks",
    "pct_bond",
    "volatility_mean_30",
    "volatility_mean_60",
    "volatility_std_30"
]

# -------------------------------
def _base_dir():
    return os.path.dirname(os.path.abspath(__file__))

# -------------------------------
def load_model_objects(model_path=None, scaler_path=None, le_path=None):
    base = _base_dir()
    if model_path is None:
        model_path = os.path.join(base, "..", "models", "risk_model.pkl")
    if scaler_path is None:
        scaler_path = os.path.join(base, "..", "models", "scaler.pkl")
    if le_path is None:
        le_path = os.path.join(base, "..", "models", "label_encoder.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)

    # scaler'dan feature isimlerini alma
    if hasattr(scaler, 'feature_names_in_'):
        feature_columns = list(scaler.feature_names_in_)
    else:
        feature_columns = FEATURE_COLUMNS

    return model, scaler, le, feature_columns

# -------------------------------
def preprocess_input(input_df, scaler, feature_columns):
    df = input_df.copy()
    # Eksik kolonları 0 ile doldur
    for c in feature_columns:
        if c not in df.columns:
            df[c] = 0.0
    df = df[feature_columns]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_scaled = scaler.transform(df)
    return X_scaled

# -------------------------------
def predict_risk(input_df, model, scaler, le, feature_columns):
    X_scaled = preprocess_input(input_df, scaler, feature_columns)
    pred_encoded = model.predict(X_scaled)
    pred_label = le.inverse_transform(pred_encoded)
    return pred_label[0]

# -------------------------------
def shap_local_explanation(input_df, model, scaler, feature_columns, class_index=1):
    """
    SHAP local explanation for a single input.
    Works for:
      - binary classification
      - multi-class classification (3D output)
    Returns:
      - shap_df: pd.DataFrame (1 x num_features)
      - expected_value: float
    """
    X_scaled = preprocess_input(input_df, scaler, feature_columns)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled, check_additivity=False)

    # SHAP multi-class 3D array kontrolü
    if isinstance(shap_values, np.ndarray):
        # shape = (num_samples, num_features, num_classes)
        if shap_values.ndim == 3:
            # Tek satır input için: shap_values[0, :, class_index]
            shap_array = shap_values[0, :, class_index]
            shap_df = pd.DataFrame([shap_array], columns=feature_columns)
            expected_value = explainer.expected_value[class_index]
        else:
            shap_df = pd.DataFrame(shap_values, columns=feature_columns)
            expected_value = explainer.expected_value
    elif isinstance(shap_values, list):
        # Liste halinde multi-class (önceki versiyon)
        shap_array = shap_values[class_index][0]
        shap_df = pd.DataFrame([shap_array], columns=feature_columns)
        expected_value = explainer.expected_value[class_index]
    else:
        raise ValueError("Unknown SHAP output type.")

    return shap_df, expected_value


def load_cluster_profiles(profile_path=None):
    base = _base_dir()
    if profile_path is None:
        profile_path = os.path.join(base, "..", "intermediate", "user_features_clustered.csv")

    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"Cluster profile CSV not found at: {profile_path}")

    df = pd.read_csv(profile_path)

    if "cluster_label" not in df.columns:
        raise KeyError("Column 'cluster_label' missing in cluster profile CSV")

    # convert cluster_label to int
    df["cluster_label"] = df["cluster_label"].astype(int)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "cluster_label"]

    cluster_profiles = df.groupby("cluster_label")[numeric_cols].mean().reset_index()

    # eksik FEATURE_COLUMNS kolonlarını 0 ile doldur
    for c in FEATURE_COLUMNS:
        if c not in cluster_profiles.columns:
            cluster_profiles[c] = 0.0

    # index olarak cluster_label
    cluster_profiles = cluster_profiles.set_index("cluster_label")
    return cluster_profiles

# -------------------------------
def assign_cluster(input_df, cluster_profiles):
    df = input_df.copy()
    centroid_cols = [c for c in cluster_profiles.columns if c not in ["cluster_label"]]
    for c in centroid_cols:
        if c not in df.columns:
            df[c] = 0.0
    df = df[centroid_cols]

    centroids = cluster_profiles[centroid_cols]

    distances = cdist(df.values, centroids.values, metric="euclidean")
    cluster_idx = np.argmin(distances, axis=1)

    return centroids.index[cluster_idx[0]]
