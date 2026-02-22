"""
allergy_model_comparison.py
Aligned with the current live-feature pipeline:
- Uses the same LIVE_FEATURES set as final_allergy_model.py (no lag/rolling features).
- Trains baseline models and compares metrics.
- Generates SHAP explainability for the LightGBM model and saves artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from proxy_labeling import compute_proxy_score, compute_proxy_label, ensure_derived_features
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

BASE = Path(__file__).resolve().parent

LIVE_FEATURES = [
    "temperature",
    "humidity",
    "dewpoint",
    "precipitation",
    "windspeed",
    "winddirection",
    "pressure",
    "uv_index",
    "pm2_5",
    "pm10",
    "co",
    "no2",
    "so2",
    "o3",
    "european_aqi",
    "alder_pollen",
    "birch_pollen",
    "grass_pollen",
    "mugwort_pollen",
    "olive_pollen",
    "ragweed_pollen",
    "month",
    "day_of_week",
    "hour",
    "pollen_score",
    "pollution_score",
    "weather_risk",
]

print("\n=== Allergy Risk Model Comparison (Aligned with Live Features) ===")
df = pd.read_csv(BASE / "allergy_cleaned.csv").dropna().reset_index(drop=True)
print(f"✅ Loaded dataset with {len(df)} rows and {df.shape[1]} columns")

missing = [c for c in LIVE_FEATURES if c not in df.columns]
if missing:
    print(f"⚠ Missing columns in training data, filling with 0: {missing}")
    for c in missing:
        df[c] = 0

df = ensure_derived_features(df)
proxy_score = compute_proxy_score(df)
proxy_label = compute_proxy_label(proxy_score, threshold=0.5)

X = df[LIVE_FEATURES].copy()
y = proxy_label.astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # scaler was fit on DataFrame -> feature_names_in_ available

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

def evaluate_model(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, probs),
    }

results = []

print("\n--- Logistic Regression ---")
logreg = LogisticRegression(max_iter=2000)
logreg.fit(X_train, y_train)
results.append(evaluate_model("Logistic Regression", logreg, X_test, y_test))

print("\n--- XGBoost ---")
xgb = XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42)
xgb.fit(X_train, y_train)
results.append(evaluate_model("XGBoost", xgb, X_test, y_test))

print("\n--- LightGBM (candidate) ---")
lgbm = LGBMClassifier(random_state=42, n_estimators=400)
lgbm.fit(X_train, y_train)
results.append(evaluate_model("LightGBM", lgbm, X_test, y_test))

# Save LGBM + scaler (for Streamlit)
joblib.dump(lgbm, BASE / "allergy_lgbm_model.pkl")
joblib.dump(scaler, BASE / "allergy_scaler.pkl")
print("💾 Saved LightGBM model and scaler.")

# SHAP explainability for LGBM (on scaled feature space; consistent with training)
X_test_df = pd.DataFrame(X_test, columns=list(getattr(scaler, "feature_names_in_", LIVE_FEATURES)))
try:
    explainer = shap.TreeExplainer(lgbm)
    sv = explainer.shap_values(X_test_df)
    # Binary classifiers can return list[class] in some SHAP versions
    if isinstance(sv, list):
        sv = sv[1]

    shap.summary_plot(sv, X_test_df, show=False)
    plt.title("SHAP Summary - LightGBM (Scaled Features)")
    plt.tight_layout()
    plt.savefig(BASE / "shap_summary_lgbm.png")
    plt.close()
    print("📊 SHAP summary saved as shap_summary_lgbm.png")

    # Save top drivers as CSV
    mean_abs = np.abs(sv).mean(axis=0)
    shap_imp = (
        pd.DataFrame({"feature": X_test_df.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    shap_imp.to_csv(BASE / "shap_importance_lgbm.csv", index=False)
    print("📊 SHAP importance saved as shap_importance_lgbm.csv")
except Exception as e:
    print("⚠ SHAP failed:", e)

print("\n--- Stacking Ensemble ---")
estimators = [
    ("lr", LogisticRegression(max_iter=2000)),
    ("xgb", XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42)),
    ("lgbm", LGBMClassifier(random_state=42, n_estimators=300)),
]
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=2000),
    cv=5,
    n_jobs=-1,
    passthrough=False,
)
stack.fit(X_train, y_train)
results.append(evaluate_model("Stacking Ensemble", stack, X_test, y_test))
joblib.dump(stack, BASE / "allergy_stacked_model.pkl")
print("💾 Saved stacking model as allergy_stacked_model.pkl")

results_df = pd.DataFrame(results).set_index("Model")
print("\nModel Comparison Table")
print(results_df.round(4))

results_df.to_csv(BASE / "model_comparison_results.csv")
print("💾 Saved model comparison results to model_comparison_results.csv")

print("\n🎉 Done. Artifacts: model_comparison_results.csv, shap_summary_lgbm.png, shap_importance_lgbm.csv")
