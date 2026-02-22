"""
proxy_labeling.py
Shared utilities for computing a pollution-weighted proxy allergy risk score + label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POLLEN_COLS = [
    "alder_pollen",
    "birch_pollen",
    "grass_pollen",
    "mugwort_pollen",
    "olive_pollen",
    "ragweed_pollen",
]
POLLUTION_COLS = ["pm2_5", "pm10", "no2", "o3", "so2", "co"]


def ensure_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pollen_score" not in df.columns:
        cols = [c for c in POLLEN_COLS if c in df.columns]
        if cols:
            df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
            df["pollen_score"] = df[cols].fillna(0).sum(axis=1)
        else:
            df["pollen_score"] = 0

    if "pollution_score" not in df.columns:
        cols = [c for c in POLLUTION_COLS if c in df.columns]
        if cols:
            df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
            df["pollution_score"] = df[cols].fillna(0).mean(axis=1)
        else:
            df["pollution_score"] = 0

    if "weather_risk" not in df.columns:
        if "humidity" in df.columns and "temperature" in df.columns:
            hum = pd.to_numeric(df["humidity"], errors="coerce").clip(0, 100).fillna(50)
            temp = pd.to_numeric(df["temperature"], errors="coerce").fillna(22)
            df["weather_risk"] = (100 - hum) * 0.3 + (temp.sub(22).abs() * 0.2)
        else:
            df["weather_risk"] = 0

    return df


def _norm(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    rng = s.max() - s.min()
    if rng == 0:
        return s * 0
    return (s - s.min()) / (rng + 1e-9)


def compute_proxy_score(
    df: pd.DataFrame,
    pollen_w: float = 0.35,
    pollution_w: float = 0.55,
    weather_w: float = 0.10,
) -> pd.Series:
    df = ensure_derived_features(df)
    pollen = _norm(df["pollen_score"])
    pollution = _norm(df["pollution_score"])
    weather = _norm(df["weather_risk"])
    score = pollen_w * pollen + pollution_w * pollution + weather_w * weather
    return score.clip(0, 1)


def compute_proxy_label(score: pd.Series, threshold: float = 0.5) -> pd.Series:
    return (score >= threshold).astype(int)

