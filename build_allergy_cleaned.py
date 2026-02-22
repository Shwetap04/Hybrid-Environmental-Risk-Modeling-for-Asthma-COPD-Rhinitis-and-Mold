"""
Build allergy_cleaned-style dataset from merged Open-Meteo historical data.

Example:
python build_allergy_cleaned.py ^
  --input openmeteo_2023_2025_merged_full.csv ^
  --output allergy_cleaned.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from proxy_labeling import compute_proxy_label, compute_proxy_score, ensure_derived_features


OUTPUT_COLS = [
    "datetime",
    "city",
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
    "allergy_risk_score",
    "allergy_label",
    "allergy_severity",
]

NUMERIC_BASE_COLS = [
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
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build allergy_cleaned.csv from Open-Meteo merged data")
    p.add_argument("--input", default="openmeteo_2023_2025_merged_full.csv", help="Input merged CSV")
    p.add_argument("--output", default="allergy_cleaned.csv", help="Output cleaned CSV")
    p.add_argument("--threshold", type=float, default=0.5, help="Proxy binary label threshold")
    p.add_argument(
        "--threshold-quantile",
        type=float,
        default=None,
        help="If set (0..1), use this quantile of allergy_risk_score as label threshold",
    )
    return p.parse_args()


def _severity_from_score(score: pd.Series) -> pd.Series:
    q1 = score.quantile(0.33)
    q2 = score.quantile(0.66)
    return pd.Series(
        np.where(score <= q1, 0, np.where(score <= q2, 1, 2)),
        index=score.index,
        dtype="int64",
    )


def build_clean(df: pd.DataFrame, threshold: float, threshold_quantile: float | None = None) -> pd.DataFrame:
    out = df.copy()

    if "datetime" not in out.columns:
        if "timestamp" in out.columns:
            out = out.rename(columns={"timestamp": "datetime"})
        elif "time" in out.columns:
            out = out.rename(columns={"time": "datetime"})
        else:
            raise ValueError("Input must contain one of: datetime, timestamp, time")

    if "city" not in out.columns:
        raise ValueError("Input must contain city column")

    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.dropna(subset=["datetime", "city"]).copy()

    for c in NUMERIC_BASE_COLS:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Fill sensor gaps. Keeping zeros is consistent with current pipeline assumptions.
    out[NUMERIC_BASE_COLS] = out[NUMERIC_BASE_COLS].fillna(0.0)
    out["humidity"] = out["humidity"].clip(0, 100)

    # Always rebuild time-derived fields from datetime.
    out["month"] = out["datetime"].dt.month
    out["day_of_week"] = out["datetime"].dt.dayofweek
    out["hour"] = out["datetime"].dt.hour

    # Recompute derived features from base signals.
    for c in ["pollen_score", "pollution_score", "weather_risk"]:
        if c in out.columns:
            out = out.drop(columns=[c])
    out = ensure_derived_features(out)

    score = compute_proxy_score(out)
    if threshold_quantile is not None:
        if not (0 <= threshold_quantile <= 1):
            raise ValueError("threshold-quantile must be between 0 and 1")
        threshold = float(score.quantile(threshold_quantile))
    out["allergy_risk_score"] = score
    out["allergy_label"] = compute_proxy_label(score, threshold=threshold).astype("int64")
    out["allergy_severity"] = _severity_from_score(score)

    out = out.drop_duplicates(subset=["city", "datetime"], keep="last")
    out = out.sort_values(["city", "datetime"]).reset_index(drop=True)

    for c in OUTPUT_COLS:
        if c not in out.columns:
            out[c] = 0
    out = out[OUTPUT_COLS]

    return out


def main() -> None:
    args = parse_args()
    inp = Path(args.input)
    out_path = Path(args.output)

    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")

    df = pd.read_csv(inp)
    cleaned = build_clean(df, threshold=args.threshold, threshold_quantile=args.threshold_quantile)
    cleaned.to_csv(out_path, index=False)

    print(f"[DONE] wrote {out_path.resolve()}")
    print(f"[INFO] shape={cleaned.shape}")
    print(f"[INFO] threshold_used={args.threshold if args.threshold_quantile is None else 'quantile_' + str(args.threshold_quantile)}")
    print(
        "[INFO] label_counts="
        + cleaned["allergy_label"].value_counts(dropna=False).sort_index().to_string()
    )


if __name__ == "__main__":
    main()
