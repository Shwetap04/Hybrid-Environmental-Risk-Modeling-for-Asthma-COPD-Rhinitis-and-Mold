"""
predict_allergy_risk.py
Loads scaler + model (LightGBM or stacked) -> predicts on CSV/JSON row -> returns probabilities, label, SHAP explanation
Also includes a small 'agentic' advisor: rule-based + SHAP-driven feature drivers -> actionable advice.
Usage examples:
    python predict_allergy_risk.py --model allergy_lgbm_model.pkl --csv new_input.csv --out preds.csv
    python predict_allergy_risk.py --model allergy_stacked_model.pkl --json '{"temperature":...}' 
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import shap
import textwrap

META_COLS = ["patient_id","hospital_id","timestamp","city"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="allergy_lgbm_model.pkl", help="Model .pkl to load")
    p.add_argument("--scaler", type=str, default="allergy_scaler.pkl", help="Scaler .pkl to load")
    p.add_argument("--csv", type=str, help="CSV input path")
    p.add_argument("--json", type=str, help="Inline JSON string for one row")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", type=str, default="allergy_predictions.csv")
    args, _ = p.parse_known_args()
    return args

def load_input(args):
    if args.csv and Path(args.csv).exists():
        df = pd.read_csv(args.csv)
        return df
    if args.json:
        rec = json.loads(args.json)
        return pd.DataFrame([rec])
    # fallback example
    return pd.DataFrame([{
        "temperature": 28.5,"humidity":55,"dewpoint":17.0,"precipitation":0.0,"windspeed":3.2,
        "winddirection":120,"pressure":1012,"uv_index":5,"pm2_5":18.3,"pm10":28.7,"co":0.4,"no2":15,
        "so2":4,"o3":30,"european_aqi":45,"alder_pollen":0,"birch_pollen":5,"grass_pollen":25,"mugwort_pollen":0,
        "olive_pollen":0,"ragweed_pollen":2,"month":6,"day_of_week":2,"hour":14,"pollen_score":30,
        "pollution_score":35,"weather_risk":8,"allergy_risk_score":0.45,
        "patient_id":"demo-001","hospital_id":"HOS-11","timestamp":"2025-09-01T03:30:00+05:30","city":"Berlin"
    }])

def align_to_scaler_schema(df, scaler):
    feat_names = list(getattr(scaler, "feature_names_in_", []))
    if not feat_names:
        # if scaler was fit on numpy, try to infer by removing meta
        feat_names = [c for c in df.columns if c not in META_COLS]
        print("[WARN] scaler has no feature_names_in_; inferring from input columns (may be risky).")

    meta = df[[c for c in df.columns if c in META_COLS]].copy() if any(c in df.columns for c in META_COLS) else pd.DataFrame()
    # drop extras
    extras = [c for c in df.columns if c not in feat_names and c not in META_COLS]
    if extras:
        print(f"[INFO] Dropping unknown columns: {extras}")
        df = df.drop(columns=extras)
    # add missing
    missing = [c for c in feat_names if c not in df.columns]
    if missing:
        print(f"[INFO] Adding missing features with 0: {missing}")
        for c in missing:
            df[c] = 0
    X = df[feat_names].copy()
    # coerce to numeric
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0.0)
    return X, meta

def risk_band(prob):
    if prob < 0.33: return "Low"
    if prob < 0.66: return "Moderate"
    return "High"

def agentic_advice(row, shap_vals, feature_names):
    """
    Small rule-based agent augmented by SHAP drivers.
    - shap_vals: 1D array of shap values for positive class for this sample
    - returns short textual advice
    """
    # top drivers
    s = pd.Series(shap_vals, index=feature_names).abs().sort_values(ascending=False)
    top = s.head(5).index.tolist()
    advice = []
    # rules
    if row.get("pollen_score", 0) > 40 or row.get("grass_pollen",0) > 50 or row.get("birch_pollen",0)>30:
        advice.append("High pollen levels detected — consider staying indoors during peak hours, showering after being outside, and using antihistamines if prescribed.")
    if row.get("pm2_5",0) > 35 or row.get("european_aqi",0) > 75:
        advice.append("Air pollution is elevated — recommend wearing an N95/FFP2 mask outdoors and avoiding strenuous outdoor exercise.")
    if row.get("humidity",50) < 30:
        advice.append("Low humidity can irritate airways — use a humidifier and keep skin/moisturizer for nasal passages.")
    if not advice:
        advice.append("Conditions look moderate/low. Continue routine medication and monitor local forecasts.")
    # add SHAP-driven quick reason
    reason = f"Top drivers: {', '.join(top)} (most influential features for this prediction)."
    return " ".join(advice) + " " + reason

def main():
    args = parse_args()
    scaler = joblib.load(args.scaler)
    model = joblib.load(args.model)
    print("[OK] Model & Scaler loaded:", args.model, args.scaler)

    raw = load_input(args)
    X, meta = align_to_scaler_schema(raw.copy(), scaler)

    Xs = scaler.transform(X)  # scaler was fit on DataFrame -> feature order preserved
    probs = None
    try:
        probs = model.predict_proba(Xs)[:,1]
    except Exception:
        # if model doesn't have predict_proba (rare), use decision_function
        dec = model.decision_function(Xs)
        probs = 1/(1+np.exp(-dec))

    labels = (probs >= args.threshold).astype(int)
    out = raw.copy()
    out["predicted_allergy_label"] = labels
    out["allergy_risk_probability"] = probs
    out["risk_band"] = [risk_band(p) for p in probs]
    out["decision_threshold"] = args.threshold

    # SHAP explainability (TreeExplainer for LGBM/XGB/RF, Kernel for others)
    try:
        if hasattr(model, "predict_proba") and ("LGBM" in type(model).__name__ or "XGB" in type(model).__name__ or "RandomForest" in type(model).__name__):
            expl = shap.TreeExplainer(model)
            shap_vals = expl.shap_values(X)  # tree explainer returns list for binary in some versions
            # normalize into shape (n_samples, n_features) for positive class
            if isinstance(shap_vals, list):
                shap_pos = shap_vals[1]
            else:
                if shap_vals.ndim == 3:
                    shap_pos = shap_vals[:,:,1]
                else:
                    shap_pos = shap_vals
        else:
            expl = shap.KernelExplainer(model.predict_proba, X.iloc[:50,:])
            shap_pos = expl.shap_values(X)[1]
    except Exception as e:
        print("[WARN] SHAP failed:", e)
        shap_pos = None

    # Agentic advice for each row
    advices = []
    if shap_pos is not None:
        feat_names = X.columns.tolist()
        for i in range(X.shape[0]):
            advice = agentic_advice(X.iloc[i].to_dict(), shap_pos[i], feat_names)
            advices.append(advice)
    else:
        for i in range(X.shape[0]):
            advices.append("No explainability available. Follow standard precautions.")

    out["agentic_advice"] = advices

    # Print summary
    cols = (META_COLS if any(c in out.columns for c in META_COLS) else []) + ["predicted_allergy_label","allergy_risk_probability","risk_band"]
    print("\n=== Allergy Risk Predictions ===")
    print(out[cols].to_string(index=False))

    out.to_csv(args.out, index=False)
    print(f"[OK] Saved predictions to {args.out}")

if __name__ == "__main__":
    main()
