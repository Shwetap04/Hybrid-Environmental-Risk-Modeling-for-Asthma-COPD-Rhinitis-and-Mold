from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent

ENV_PATH = BASE / "openmeteo_2023_2025_merged.csv"
OUTCOME_RAW_PATH = OUT_DIR / "outcomes_master_by_disease_raw.csv"


ENV_FEATURES = [
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
    "pollen_score",
    "pollution_score",
    "weather_risk",
]


def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    m = s.mean()
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - m) / sd


def load_environment_aggregates(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    env = pd.read_csv(path)
    if "datetime" not in env.columns or "city" not in env.columns:
        raise ValueError("Environment file must contain datetime and city columns.")

    env["datetime"] = pd.to_datetime(env["datetime"], errors="coerce")
    env = env.dropna(subset=["datetime", "city"]).copy()
    env["year"] = env["datetime"].dt.year
    env["month"] = env["datetime"].dt.month

    for c in ENV_FEATURES:
        if c not in env.columns:
            env[c] = 0.0
        env[c] = pd.to_numeric(env[c], errors="coerce")

    env["pollution_hour_flag"] = (
        (env["pm2_5"].fillna(0) >= 35) | (env["european_aqi"].fillna(0) >= 100)
    ).astype(int)

    agg_map = {**{c: "mean" for c in ENV_FEATURES}, "pollution_hour_flag": "sum"}

    env_yearly = (
        env.groupby(["city", "year"], as_index=False)
        .agg(agg_map)
        .rename(columns={"pollution_hour_flag": "hours_pollution_flag"})
    )
    env_monthly = (
        env.groupby(["city", "year", "month"], as_index=False)
        .agg(agg_map)
        .rename(columns={"pollution_hour_flag": "hours_pollution_flag"})
    )

    for d in (env_yearly, env_monthly):
        num_cols = [c for c in d.columns if c not in {"city", "year", "month"}]
        d[num_cols] = d[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return env_yearly, env_monthly


def load_outcomes_mixed(path: Path) -> pd.DataFrame:
    out = pd.read_csv(path)
    needed = {"city", "date", "disease", "outcome_value", "outcome_unit", "source", "outcome_type"}
    missing = needed - set(out.columns)
    if missing:
        raise ValueError(f"Outcome file missing columns: {sorted(missing)}")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["city", "date", "disease", "outcome_value"]).copy()
    out["outcome_value"] = pd.to_numeric(out["outcome_value"], errors="coerce")
    out = out.dropna(subset=["outcome_value"])

    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month

    unit_lower = out["outcome_unit"].astype(str).str.lower()
    source_lower = out["source"].astype(str).str.lower()
    out["is_count"] = unit_lower.str.contains("count", na=False)
    # Use monthly alignment where dates are daily event records (ISSSTE counts).
    out["period_level"] = np.where(
        out["is_count"] | source_lower.str.contains("issste", na=False),
        "month",
        "year",
    )

    keys = ["city", "year", "month", "period_level", "disease", "source", "outcome_unit", "outcome_type"]

    counts = (
        out[out["is_count"]]
        .groupby(keys, as_index=False)["outcome_value"]
        .sum()
    )
    non_counts = (
        out[~out["is_count"]]
        .groupby(keys, as_index=False)["outcome_value"]
        .mean()
    )
    mixed = pd.concat([counts, non_counts], ignore_index=True)

    # Collapse year-level outcomes to one row per city-year-disease-source.
    year_rows = mixed[mixed["period_level"] == "year"].copy()
    if not year_rows.empty:
        year_keys = ["city", "year", "period_level", "disease", "source", "outcome_unit", "outcome_type"]
        year_rows = year_rows.groupby(year_keys, as_index=False)["outcome_value"].mean()
        year_rows["month"] = 1

    month_rows = mixed[mixed["period_level"] == "month"].copy()

    out_final = pd.concat([year_rows, month_rows], ignore_index=True)
    out_final["period_start"] = pd.to_datetime(
        out_final["year"].astype("Int64").astype(str)
        + "-"
        + out_final["month"].astype("Int64").astype(str).str.zfill(2)
        + "-01",
        errors="coerce",
    )
    out_final = out_final.dropna(subset=["period_start"])

    out_final["outcome_value_z"] = (
        out_final.groupby(["source", "disease"], group_keys=False)["outcome_value"].apply(_zscore)
    )
    out_final["outcome_value_city_z"] = (
        out_final.groupby(["city", "disease"], group_keys=False)["outcome_value"].apply(_zscore)
    )
    return out_final


def build_aligned_sets(
    env_yearly: pd.DataFrame,
    env_monthly: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_year = outcomes[outcomes["period_level"] == "year"].copy()
    out_month = outcomes[outcomes["period_level"] == "month"].copy()

    merged_parts: list[pd.DataFrame] = []

    if not out_year.empty:
        m_year = out_year.merge(env_yearly, on=["city", "year"], how="inner")
        merged_parts.append(m_year)
    if not out_month.empty:
        m_month = out_month.merge(env_monthly, on=["city", "year", "month"], how="inner")
        merged_parts.append(m_month)

    if not merged_parts:
        merged_long = pd.DataFrame()
    else:
        merged_long = pd.concat(merged_parts, ignore_index=True)
        merged_long = merged_long.sort_values(
            ["city", "period_start", "disease", "source"]
        ).reset_index(drop=True)

    env_cols = [c for c in ENV_FEATURES + ["hours_pollution_flag"] if c in merged_long.columns]
    if merged_long.empty:
        wide = pd.DataFrame(columns=["city", "period_start", "year", "month", *env_cols])
        return merged_long, wide

    wide_targets = (
        merged_long.pivot_table(
            index=["city", "period_start", "year", "month", *env_cols],
            columns="disease",
            values="outcome_value_city_z",
            aggfunc="mean",
        )
        .reset_index()
    )
    wide_targets.columns.name = None
    return merged_long, wide_targets


def metrics_from_aligned(merged_long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for disease, d in merged_long.groupby("disease"):
        n = len(d)
        sp_pollution = np.nan
        sp_pollen = np.nan
        sp_weather = np.nan
        if n >= 3:
            sp_pollution = d["outcome_value_city_z"].corr(d["pollution_score"], method="spearman")
            sp_pollen = d["outcome_value_city_z"].corr(d["pollen_score"], method="spearman")
            sp_weather = d["outcome_value_city_z"].corr(d["weather_risk"], method="spearman")
        rows.append(
            {
                "disease": disease,
                "n_rows": int(n),
                "n_city_period": int(d[["city", "period_start"]].drop_duplicates().shape[0]),
                "n_sources": int(d["source"].nunique()),
                "spearman_outcome_vs_pollution": sp_pollution,
                "spearman_outcome_vs_pollen": sp_pollen,
                "spearman_outcome_vs_weather_risk": sp_weather,
            }
        )
    return pd.DataFrame(rows).sort_values("disease").reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env_yearly, env_monthly = load_environment_aggregates(ENV_PATH)
    outcomes = load_outcomes_mixed(OUTCOME_RAW_PATH)
    aligned_long, aligned_wide = build_aligned_sets(env_yearly, env_monthly, outcomes)
    metrics = metrics_from_aligned(aligned_long) if not aligned_long.empty else pd.DataFrame()

    long_path = OUT_DIR / "aligned_env_outcomes_long.csv"
    wide_path = OUT_DIR / "aligned_env_outcomes_wide.csv"
    metrics_path = OUT_DIR / "aligned_env_outcomes_metrics.csv"
    summary_path = OUT_DIR / "aligned_env_outcomes_summary.json"

    aligned_long.to_csv(long_path, index=False)
    aligned_wide.to_csv(wide_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    summary = {
        "environment_file": str(ENV_PATH),
        "outcome_file": str(OUTCOME_RAW_PATH),
        "aligned_long_shape": list(aligned_long.shape),
        "aligned_wide_shape": list(aligned_wide.shape),
        "cities_in_aligned": sorted(aligned_long["city"].dropna().unique().tolist()) if not aligned_long.empty else [],
        "diseases_in_aligned": sorted(aligned_long["disease"].dropna().unique().tolist()) if not aligned_long.empty else [],
        "period_range_in_aligned": [
            str(aligned_long["period_start"].min().date()) if not aligned_long.empty else None,
            str(aligned_long["period_start"].max().date()) if not aligned_long.empty else None,
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[DONE] {long_path} shape={aligned_long.shape}")
    print(f"[DONE] {wide_path} shape={aligned_wide.shape}")
    print(f"[DONE] {metrics_path} shape={metrics.shape}")
    print(f"[DONE] {summary_path}")
    if not metrics.empty:
        print("\nPer-disease overlap:")
        print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
