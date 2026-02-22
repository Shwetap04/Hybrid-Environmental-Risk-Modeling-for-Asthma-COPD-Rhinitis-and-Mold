"""
pacf_risk.py
Compute and save PACF plots for the model-implied risk time series.

Usage:
  python pacf_risk.py
  python pacf_risk.py --city Berlin --max_lag 72
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import pacf
except Exception as e:
    raise SystemExit(
        "statsmodels is required for PACF. Install with: pip install statsmodels\n"
        f"Original error: {e}"
    )

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--city", type=str, default="", help="City name in allergy_cleaned.csv")
    p.add_argument("--max_lag", type=int, default=72, help="Max PACF lag (hours)")
    return p.parse_args()


def compute_env_risk(df: pd.DataFrame, model, scaler) -> np.ndarray:
    X = df[LIVE_FEATURES].copy()
    Xs = scaler.transform(X)
    Xs_df = pd.DataFrame(Xs, columns=X.columns)
    return model.predict_proba(Xs_df)[:, 1]


def plot_pacf(series: np.ndarray, max_lag: int, title: str, out_path: Path):
    n = len(series)
    max_lag = min(max_lag, n // 2)
    vals = pacf(series, nlags=max_lag, method="ywm")
    conf = 1.96 / math.sqrt(n)

    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.bar(range(len(vals)), vals, width=0.8, color="#4B8A8D")
    ax.axhline(conf, color="red", linestyle="--", linewidth=1)
    ax.axhline(-conf, color="red", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("PACF")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

    # Save values as CSV too
    csv_path = out_path.with_suffix(".csv")
    pd.DataFrame({"lag": range(len(vals)), "pacf": vals}).to_csv(csv_path, index=False)


def main():
    args = parse_args()

    df = pd.read_csv(BASE / "allergy_cleaned.csv")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values(["city", "datetime"]).reset_index(drop=True)

    model = joblib.load(BASE / "allergy_lgbm_model.pkl")
    scaler = joblib.load(BASE / "allergy_scaler.pkl")

    cities = sorted(df["city"].dropna().unique().tolist())
    if args.city:
        if args.city not in cities:
            raise SystemExit(f"City not found: {args.city}")
        cities = [args.city]

    out_dir = BASE / "pacf_outputs"
    out_dir.mkdir(exist_ok=True)

    for city in cities:
        g = df[df["city"] == city].sort_values("datetime").reset_index(drop=True)
        if len(g) < 100:
            print(f"⚠ Skipping {city} (not enough rows: {len(g)})")
            continue
        g = g.copy()
        g["env_risk_prob"] = compute_env_risk(g, model, scaler)
        series = g["env_risk_prob"].astype(float).to_numpy()
        title = f"PACF of Risk (Env-driven) — {city}"
        out_path = out_dir / f"pacf_{city.replace(' ', '_')}.png"
        plot_pacf(series, args.max_lag, title, out_path)
        print(f"✅ Saved: {out_path}")

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()

