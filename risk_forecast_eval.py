"""
risk_forecast_eval.py
Train and evaluate a 7-day internal (risk-only) forecaster using dataset cities.

What we compare:
1) Env-driven "forecast" (Open-Meteo style): base model applied to FUTURE environmental features.
   - In offline evaluation we use the dataset's future environment rows (upper-bound / oracle).
2) Internal risk-only forecast: a separate time-series model that predicts future risk from past risk + time features.

Ground truth for evaluation:
- allergy_label in allergy_cleaned.csv (proxy label).

Outputs (written to this folder):
- risk_forecaster.pkl (global fallback)
- risk_forecaster_<city>.pkl + risk_forecaster_<city>_metadata.json
- risk_forecast_metrics.csv  (per-city metrics)
- risk_forecast_plot_<city>.png  (one example city plot)
"""

from __future__ import annotations

import os
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import re

from lightgbm import LGBMRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)
from proxy_labeling import compute_proxy_score, compute_proxy_label, ensure_derived_features

BASE = Path(__file__).resolve().parent
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))


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


def risk_band(prob: float) -> int:
    if prob < 0.33:
        return 0
    if prob < 0.66:
        return 1
    return 2


def slugify_city(city: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", city).strip("_")
    return s.lower()


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    # AUC requires both classes present.
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def compute_env_risk_probs(df_city: pd.DataFrame, model, scaler) -> np.ndarray:
    X = df_city[LIVE_FEATURES].copy()
    Xs = scaler.transform(X)
    Xs_df = pd.DataFrame(Xs, columns=X.columns)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(Xs_df)[:, 1]
    else:
        probs = model.predict(Xs_df)
    return np.clip(probs, 0, 1)


def make_risk_forecast_features(
    history_risk: list[float],
    future_ts: pd.Timestamp,
) -> dict:
    """
    Build one-step features from past risk values + calendar time.
    history_risk must contain the most recent values with history_risk[-1] being last observed/predicted.
    """
    feats: dict[str, float] = {}
    # Lags (1..24)
    for lag in range(1, 25):
        feats[f"risk_lag{lag}"] = float(history_risk[-lag])

    # Rolling means
    for w in (3, 6, 12, 24):
        feats[f"risk_roll{w}"] = float(np.mean(history_risk[-w:]))

    feats["hour"] = int(future_ts.hour)
    feats["day_of_week"] = int(future_ts.dayofweek)
    feats["month"] = int(future_ts.month)
    return feats


def build_forecaster_training_frame(df: pd.DataFrame, env_risk_prob_col: str = "env_risk_prob") -> pd.DataFrame:
    """
    Build supervised learning rows for 1-step ahead risk prediction.
    """
    rows = []
    for city, g in df.groupby("city"):
        g = g.sort_values("datetime").reset_index(drop=True)
        r = g[env_risk_prob_col].astype(float).to_numpy()
        ts = pd.to_datetime(g["datetime"])
        if len(g) < 30:
            continue
        # build features at t predicting r[t+1], using last 24 risk values up to t
        for t in range(24, len(g) - 1):
            hist = r[: t + 1].tolist()
            feats = make_risk_forecast_features(hist, ts.iloc[t + 1])
            feats["city"] = city
            feats["y"] = float(r[t + 1])
            rows.append(feats)
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("Not enough data to build training frame.")
    out = pd.get_dummies(out, columns=["city"], drop_first=False)
    return out


def train_risk_forecaster(train_frame: pd.DataFrame) -> LGBMRegressor:
    y = train_frame["y"].astype(float)
    X = train_frame.drop(columns=["y"])
    model = LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        random_state=42,
        subsample=0.9,
        colsample_bytree=0.9,
    )
    model.fit(X, y)
    return model


def forecast_risk_7d(
    model: LGBMRegressor,
    city: str,
    last_ts: pd.Timestamp,
    last_24_risk: list[float],
    city_dummy_cols: list[str],
    horizon_hours: int = 24 * 7,
) -> pd.DataFrame:
    hist = last_24_risk[:]  # copy
    out_rows = []
    for h in range(1, horizon_hours + 1):
        ts = last_ts + pd.Timedelta(hours=h)
        feats = make_risk_forecast_features(hist, ts)
        # one-hot city
        row = {**feats}
        for c in city_dummy_cols:
            row[c] = 0
        city_col = f"city_{city}"
        if city_col in row:
            row[city_col] = 1
        # predict
        X_row = pd.DataFrame([row])
        yhat = float(model.predict(X_row)[0])
        # keep in [0,1]
        yhat = max(0.0, min(1.0, yhat))
        hist.append(yhat)
        out_rows.append({"timestamp": ts, "internal_risk_prob": yhat})
    return pd.DataFrame(out_rows)


def eval_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_auc(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def main():
    horizon_hours = 24 * 7
    df = pd.read_csv(BASE / "allergy_cleaned.csv")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).reset_index(drop=True)
    df = ensure_derived_features(df)
    # Recompute proxy label using pollution-weighted score for alignment
    proxy_score = compute_proxy_score(df)
    df["allergy_label"] = compute_proxy_label(proxy_score, threshold=0.5)

    # Load base model + scaler (env-driven risk)
    reg_path = BASE / "allergy_lgbm_regressor.pkl"
    calib_path = BASE / "allergy_lgbm_calibrated.pkl"
    if reg_path.exists():
        base_model = joblib.load(reg_path)
        base_model_type = "regressor"
        base_model_file = reg_path.name
    else:
        base_model = joblib.load(calib_path if calib_path.exists() else BASE / "allergy_lgbm_model.pkl")
        base_model_type = "classifier"
        base_model_file = (calib_path if calib_path.exists() else BASE / "allergy_lgbm_model.pkl").name
    scaler = joblib.load(BASE / "allergy_scaler.pkl")

    # ensure required columns exist
    missing = [c for c in LIVE_FEATURES + ["allergy_label", "city", "datetime"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in allergy_cleaned.csv: {missing}")

    # Compute env-driven risk probability for all rows (this is what Open-Meteo style would produce if env were known)
    df = df.sort_values(["city", "datetime"]).reset_index(drop=True)
    df["env_risk_prob"] = 0.0
    for city, idx in df.groupby("city").groups.items():
        g = df.loc[idx].copy()
        probs = compute_env_risk_probs(g, base_model, scaler)
        df.loc[idx, "env_risk_prob"] = probs

    # Backtest split: last 7 days per city as test window (needs at least 7 days of hourly data)
    metrics_rows = []
    plot_city = None

    # Train forecaster on all cities, but only using data strictly before each city's test cutoff
    # Build a global training frame from pre-cutoff slices and fit one model.
    train_slices = []
    cutoffs = {}
    for city, g in df.groupby("city"):
        g = g.sort_values("datetime")
        if len(g) < (24 * 10):
            continue
        cutoff = g["datetime"].max() - pd.Timedelta(hours=horizon_hours)
        cutoffs[city] = cutoff
        train_slices.append(g[g["datetime"] < cutoff])
        if plot_city is None:
            plot_city = city

    if not train_slices:
        raise ValueError("Not enough per-city data to create a 7-day backtest.")

    train_df = pd.concat(train_slices, axis=0).reset_index(drop=True)
    train_frame = build_forecaster_training_frame(train_df, env_risk_prob_col="env_risk_prob")
    city_dummy_cols = [c for c in train_frame.columns if c.startswith("city_")]
    forecaster = train_risk_forecaster(train_frame)

    # Save forecaster for later use in UI
    joblib.dump(forecaster, BASE / "risk_forecaster.pkl")
    print(f"[INFO] Saved risk forecaster to {BASE / 'risk_forecaster.pkl'}")
    # Save metadata so UI can build the same feature columns
    meta = {
        "feature_columns": train_frame.drop(columns=["y"]).columns.tolist(),
        "city_dummy_columns": [c for c in train_frame.columns if c.startswith("city_")],
        "base_model_type": base_model_type,
        "base_model_file": base_model_file,
    }
    (BASE / "risk_forecaster_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[INFO] Saved metadata to {BASE / 'risk_forecaster_metadata.json'}")

    # Evaluate per city (and train city-specific forecasters)
    for city, cutoff in cutoffs.items():
        g = df[df["city"] == city].sort_values("datetime").reset_index(drop=True)
        test = g[g["datetime"] >= cutoff].copy()
        hist = g[g["datetime"] < cutoff].copy()
        if len(test) < horizon_hours or len(hist) < 48:
            continue

        # Train a city-specific forecaster on this city's history only
        city_forecaster = None
        city_dummy_cols_local = city_dummy_cols
        try:
            city_frame = build_forecaster_training_frame(hist, env_risk_prob_col="env_risk_prob")
            city_forecaster = train_risk_forecaster(city_frame)
            city_feature_cols = city_frame.drop(columns=["y"]).columns.tolist()
            city_dummy_cols_local = [c for c in city_feature_cols if c.startswith("city_")]

            slug = slugify_city(city)
            joblib.dump(city_forecaster, BASE / f"risk_forecaster_{slug}.pkl")
            meta = {
                "feature_columns": city_feature_cols,
                "city_dummy_columns": city_dummy_cols_local,
                "base_model_type": base_model_type,
                "base_model_file": base_model_file,
            }
            (BASE / f"risk_forecaster_{slug}_metadata.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            print(f"[INFO] Saved city forecaster: risk_forecaster_{slug}.pkl")
        except Exception as e:
            city_forecaster = None
            print(f"[WARN] City forecaster training failed for {city}: {e}")

        # Use last 24 env-risk values at cutoff as the "observed" risk history for the internal forecaster.
        hist_risk = hist["env_risk_prob"].astype(float).to_list()
        last_24 = hist_risk[-24:]
        last_ts = pd.to_datetime(hist["datetime"].iloc[-1])

        internal_fc = forecast_risk_7d(
            city_forecaster if city_forecaster is not None else forecaster,
            city=city,
            last_ts=last_ts,
            last_24_risk=last_24,
            city_dummy_cols=city_dummy_cols_local,
            horizon_hours=horizon_hours,
        )

        # Align with test timestamps
        test = test.copy()
        test = test.rename(columns={"datetime": "timestamp"})
        merged = pd.merge(test[["timestamp", "allergy_label", "env_risk_prob"]], internal_fc, on="timestamp", how="inner")
        if merged.empty:
            continue

        y_true = merged["allergy_label"].astype(int).to_numpy()
        env_prob = merged["env_risk_prob"].astype(float).to_numpy()
        int_prob = merged["internal_risk_prob"].astype(float).to_numpy()

        env_m = eval_probs(y_true, env_prob)
        int_m = eval_probs(y_true, int_prob)

        metrics_rows.append(
            {
                "city": city,
                "n_hours": int(len(merged)),
                "env_accuracy": env_m["accuracy"],
                "env_f1": env_m["f1"],
                "env_roc_auc": env_m["roc_auc"],
                "env_brier": env_m["brier"],
                "internal_accuracy": int_m["accuracy"],
                "internal_f1": int_m["f1"],
                "internal_roc_auc": int_m["roc_auc"],
                "internal_brier": int_m["brier"],
            }
        )

        # One example plot
        if plot_city == city:
            fig = plt.figure(figsize=(12, 5))
            ax = plt.gca()
            ax.plot(merged["timestamp"], merged["env_risk_prob"], label="Env-driven risk (oracle)", linewidth=2)
            ax.plot(merged["timestamp"], merged["internal_risk_prob"], label="Internal risk-only forecast", linewidth=2)
            # ground truth as points
            ax.scatter(
                merged["timestamp"],
                merged["allergy_label"].astype(int),
                label="Proxy ground truth (allergy_label)",
                s=10,
                alpha=0.6,
            )
            ax.set_title(f"7-Day Forecast Backtest — {city}")
            ax.set_ylabel("Risk probability / label")
            ax.set_xlabel("Time")
            ax.legend()
            plt.tight_layout()
            out_path = BASE / f"risk_forecast_plot_{city.replace(' ', '_')}.png"
            plt.savefig(out_path)
            plt.close(fig)
            print(f"[INFO] Saved plot to {out_path}")

    metrics_df = pd.DataFrame(metrics_rows).sort_values("internal_f1", ascending=False)
    out_csv = BASE / "risk_forecast_metrics.csv"
    metrics_df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved metrics to {out_csv}")

    if metrics_df.empty:
        print("[WARN] No metrics produced. Check that allergy_cleaned.csv has at least 7 days of hourly data per city.")
    else:
        print("\nTop rows:")
        print(metrics_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
