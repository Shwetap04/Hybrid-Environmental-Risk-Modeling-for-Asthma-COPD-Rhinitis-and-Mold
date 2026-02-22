"""
final_allergy_model.py
Train final LightGBM + Stacking on the cleaned dataset and save artifacts.
Use this when you want to re-train final models after dataset updates.
"""
import os
import pickle
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, r2_score
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from imblearn.over_sampling import SMOTE
from proxy_labeling import compute_proxy_score, compute_proxy_label, ensure_derived_features

LIVE_FEATURES = [
    "temperature", "humidity", "dewpoint", "precipitation",
    "windspeed", "winddirection", "pressure", "uv_index",
    "pm2_5", "pm10", "co", "no2", "so2", "o3", "european_aqi",
    "alder_pollen", "birch_pollen", "grass_pollen", "mugwort_pollen",
    "olive_pollen", "ragweed_pollen",
    "month", "day_of_week", "hour",
    "pollen_score", "pollution_score", "weather_risk",
]

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

BASE = Path(__file__).resolve().parent

def safe_dump(obj, path: Path) -> None:
    path = Path(path)
    try:
        joblib.dump(obj, path)
    except OSError as e:
        print(f"[WARN] joblib.dump failed for {path} ({e}); falling back to pickle.")
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

print("=== Final model training ===")
df = pd.read_csv(BASE / "allergy_cleaned.csv")
df = df.dropna().reset_index(drop=True)

# Ensure live features exist (fill missing with 0s for safety)
missing = [c for c in LIVE_FEATURES if c not in df.columns]
if missing:
    print(f"[WARN] Missing columns in training data, filling with 0: {missing}")
    for c in missing:
        df[c] = 0

df = ensure_derived_features(df)
if "allergy_risk_score" in df.columns:
    proxy_score = pd.to_numeric(df["allergy_risk_score"], errors="coerce").fillna(0.0).clip(0, 1)
else:
    proxy_score = compute_proxy_score(df)

if "allergy_label" in df.columns:
    proxy_label = pd.to_numeric(df["allergy_label"], errors="coerce").fillna(0).astype(int).clip(0, 1)
else:
    proxy_label = compute_proxy_label(proxy_score, threshold=0.5)

print("[INFO] Label distribution:")
print(proxy_label.value_counts(dropna=False).sort_index().to_string())

X = df[LIVE_FEATURES].copy()
y = proxy_label
y_reg = proxy_score

# Split into train/val/test before SMOTE (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)
y_reg_train = y_reg.loc[X_train.index]
y_reg_val = y_reg.loc[X_val.index]
y_reg_test = y_reg.loc[X_test.index]

# Fit scaler on training DataFrame (so feature_names_in_ saved)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# SMOTE on TRAIN only
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_s, y_train)

# Final LightGBM
lgbm = LGBMClassifier(n_estimators=400, random_state=42)
lgbm.fit(X_train_res, y_train_res)

# Final stacking
estimators = [
    ("xgb", XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42)),
    ("lgbm_small", LGBMClassifier(n_estimators=200, random_state=42))
]
stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=2000), cv=5, n_jobs=-1)
stack.fit(X_train_res, y_train_res)

# Calibrate base model on validation set
calibrator = CalibratedClassifierCV(lgbm, method="isotonic", cv="prefit")
calibrator.fit(X_val_s, y_val)

# ---------------- Regression model ----------------
reg = LGBMRegressor(n_estimators=400, random_state=42)
reg.fit(X_train_s, y_reg_train)
reg_val_pred = reg.predict(X_val_s)
reg_test_pred = reg.predict(X_test_s)

# ---------------- Metrics ----------------
def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
    return metrics

metrics = [
    evaluate("LightGBM", lgbm, X_test_s, y_test),
    evaluate("LightGBM_Calibrated", calibrator, X_test_s, y_test),
    evaluate("Stacking", stack, X_test_s, y_test),
]
metrics_df = pd.DataFrame(metrics)
metrics_path = BASE / "final_model_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)
print(f"[INFO] Saved metrics to {metrics_path}")

# Regression metrics
reg_metrics = {
    "model": "LightGBM_Regressor",
    "rmse": float(np.sqrt(mean_squared_error(y_reg_test, reg_test_pred))),
    "r2": r2_score(y_reg_test, reg_test_pred),
}
reg_metrics_path = BASE / "final_model_metrics_regression.csv"
pd.DataFrame([reg_metrics]).to_csv(reg_metrics_path, index=False)
print(f"[INFO] Saved regression metrics to {reg_metrics_path}")

# Save artifacts
safe_dump(lgbm, BASE / "allergy_lgbm_model.pkl")
safe_dump(calibrator, BASE / "allergy_lgbm_calibrated.pkl")
safe_dump(stack, BASE / "allergy_stacked_model.pkl")
safe_dump(scaler, BASE / "allergy_scaler.pkl")
safe_dump(reg, BASE / "allergy_lgbm_regressor.pkl")
print("[INFO] Saved: allergy_lgbm_model.pkl, allergy_stacked_model.pkl, allergy_scaler.pkl")
