from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

EXPECTED_DISEASES = {"asthma", "copd", "rhinitis", "mold"}

CAPITALS = {
    "DE": ("Berlin", "Germany", "DEU"),
    "FR": ("Paris", "France", "FRA"),
    "IT": ("Rome", "Italy", "ITA"),
    "ES": ("Madrid", "Spain", "ESP"),
    "PL": ("Warsaw", "Poland", "POL"),
    "AT": ("Vienna", "Austria", "AUT"),
    "HU": ("Budapest", "Hungary", "HUN"),
    "EL": ("Athens", "Greece", "GRC"),
    "GR": ("Athens", "Greece", "GRC"),
}

OUT_COLS = [
    "city",
    "country",
    "iso3",
    "date",
    "disease",
    "indicator",
    "outcome_value",
    "outcome_unit",
    "outcome_type",
    "geo_level",
    "source",
]

CA_ASTHMA_URL = (
    "https://data.chhs.ca.gov/dataset/28698f95-0637-44f0-9072-a405d90f3f83/resource/"
    "781708cb-7b25-4967-b760-54b2a4b8cfed/download/asthma-ed-visit-rates_2019.csv"
)


def fetch_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), low_memory=False)


def resolve_ckan_resource_url(package_id: str, preferred_terms: list[str]) -> str:
    api = f"https://data.chhs.ca.gov/api/3/action/package_show?id={package_id}"
    r = requests.get(api, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    j = r.json()
    if not j.get("success"):
        raise RuntimeError(f"CKAN package_show failed for {package_id}")

    resources = j["result"].get("resources", [])
    scored = []
    for res in resources:
        name = str(res.get("name", "")).lower()
        desc = str(res.get("description", "")).lower()
        url = str(res.get("url", "")).strip()
        if not url:
            continue
        score = sum(1 for t in preferred_terms if t.lower() in name or t.lower() in desc or t.lower() in url.lower())
        scored.append((score, url))

    if not scored:
        raise RuntimeError(f"No resources found for package {package_id}")

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def find_col(df: pd.DataFrame, keywords: list[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for kw in keywords:
        for k, v in lower.items():
            if kw in k:
                return v
    return None


def to_year_date(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return pd.to_datetime(x.astype("Int64").astype(str) + "-01-01", errors="coerce")


def keep_total_if_present(df: pd.DataFrame, col: Optional[str]) -> pd.DataFrame:
    if not col or col not in df.columns:
        return df
    s = df[col].astype(str).str.upper()
    for tag in ["TOTAL", "TOT", "ALL", "ALL AGES", "ALL PERSONS", "T"]:
        m = s == tag
        if m.any():
            return df[m].copy()
    m = s.str.contains("TOTAL|TOT|ALL", na=False)
    return df[m].copy() if m.any() else df


def map_disease(icd_or_indicator: str) -> Optional[str]:
    t = str(icd_or_indicator).upper()

    if re.search(r"(^|[^0-9])J45|(^|[^0-9])J46|ASTHMA", t):
        return "asthma"
    if re.search(r"(^|[^0-9])J40|(^|[^0-9])J41|(^|[^0-9])J42|(^|[^0-9])J43|(^|[^0-9])J44|COPD|CHRONIC OBSTRUCTIVE", t):
        return "copd"
    if re.search(r"(^|[^0-9])J30|(^|[^0-9])J31|RHINITIS", t):
        return "rhinitis"
    if re.search(r"(^|[^0-9])J67|(^|[^0-9])B44|MOLD|FUNGAL|ASPERGILL", t):
        return "mold"
    return None


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in OUT_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["outcome_value"] = pd.to_numeric(out["outcome_value"], errors="coerce")
    return out[OUT_COLS].dropna(subset=["date", "outcome_value", "disease"])


def fetch_ca_asthma() -> pd.DataFrame:
    df = fetch_csv(CA_ASTHMA_URL)

    county_col = find_col(df, ["county", "area"])
    year_col = find_col(df, ["year", "time"])
    rate_col = find_col(df, ["rate"])
    count_col = find_col(df, ["count", "number"])
    sex_col = find_col(df, ["sex", "gender"])
    age_col = find_col(df, ["age"])

    if county_col:
        df = df[df[county_col].astype(str).str.contains("Los Angeles", case=False, na=False)]
    df = keep_total_if_present(df, sex_col)
    df = keep_total_if_present(df, age_col)

    val_col = rate_col or count_col
    out = pd.DataFrame({
        "city": "Los Angeles",
        "country": "United States",
        "iso3": "USA",
        "date": to_year_date(df[year_col]) if year_col else pd.NaT,
        "disease": "asthma",
        "indicator": "asthma_ed_visit_rate",
        "outcome_value": pd.to_numeric(df[val_col], errors="coerce") if val_col else pd.NA,
        "outcome_unit": "rate" if rate_col else "count",
        "outcome_type": "ED visit",
        "geo_level": "county",
        "source": "CA Open Data - Asthma ED",
    })
    return standardize(out)


def fetch_ca_preventable() -> pd.DataFrame:
    package_id = "f2b33545-db0a-4a53-a611-41de532e7c53"
    url = resolve_ckan_resource_url(
        package_id,
        preferred_terms=["preventable", "hospital", "county", "asthma", "copd"],
    )
    df = fetch_csv(url)

    county_col = find_col(df, ["county", "area"])
    year_col = find_col(df, ["year", "time"])
    # This table uses PQI/PQIDescription instead of generic condition names.
    cond_col = find_col(df, ["pqidescription", "pqi", "description", "condition", "indicator", "measure"])
    rate_col = find_col(df, ["riskadjrate_icd10", "obsrate_icd10", "rate"])
    count_col = find_col(df, ["count_icd10", "count", "number"])
    sex_col = find_col(df, ["sex", "gender"])
    age_col = find_col(df, ["age"])

    if county_col:
        df = df[df[county_col].astype(str).str.contains("Los Angeles", case=False, na=False)]
    df = keep_total_if_present(df, sex_col)
    df = keep_total_if_present(df, age_col)

    if cond_col:
        keep = df[cond_col].astype(str).str.contains("asthma|copd", case=False, na=False)
        df = df[keep]

    # Prefer ICD10 fields first to avoid mixed ICD9/ICD10 signal.
    val_col = rate_col or count_col
    cond_series = df[cond_col].astype(str) if cond_col else pd.Series("respiratory", index=df.index)
    base = pd.DataFrame({
        "city": "Los Angeles",
        "country": "United States",
        "iso3": "USA",
        "date": to_year_date(df[year_col]) if year_col else pd.NaT,
        "indicator": cond_series.values,
        "outcome_value": pd.to_numeric(df[val_col], errors="coerce") if val_col else pd.NA,
        "outcome_unit": "rate" if rate_col else "count",
        "outcome_type": "preventable hospitalization",
        "geo_level": "county",
        "source": "CA Open Data - Preventable Hosp",
    })

    # Expand mixed indicators like "COPD or Asthma in Older Adults" into both labels.
    cond_l = cond_series.str.lower()
    parts: list[pd.DataFrame] = []

    m_asthma = cond_l.str.contains("asthma", na=False)
    if m_asthma.any():
        d_asthma = base[m_asthma].copy()
        d_asthma["disease"] = "asthma"
        parts.append(d_asthma)

    m_copd = cond_l.str.contains("copd|chronic obstructive", na=False)
    if m_copd.any():
        d_copd = base[m_copd].copy()
        d_copd["disease"] = "copd"
        parts.append(d_copd)

    if not parts:
        fallback = base.copy()
        fallback["disease"] = cond_series.map(map_disease)
        parts = [fallback]

    out = pd.concat(parts, ignore_index=True)
    return standardize(out)


def choose_fingertips_cols(d: pd.DataFrame):
    cols = {c.lower(): c for c in d.columns}
    area_col = cols.get("areaname") or cols.get("area name") or find_col(d, ["area", "name"])
    date_col = cols.get("timeperiod") or cols.get("time period") or find_col(d, ["time", "year", "period"])
    val_col = cols.get("value") or find_col(d, ["value", "rate", "count"])
    return area_col, date_col, val_col


def fetch_london_fingertips() -> pd.DataFrame:
    import fingertips_py as ftp

    profile = ftp.get_profile_by_name("Respiratory disease")
    meta = ftp.get_metadata_for_profile_as_dataframe(profile["Id"])

    mask = meta["Indicator"].astype(str).str.contains(
        "Emergency hospital admissions for asthma|chronic obstructive|COPD",
        case=False,
        na=False,
    )
    target = meta.loc[mask, ["Indicator ID", "Indicator"]].drop_duplicates()

    rows = []
    for _, r in target.iterrows():
        ind_id = int(r["Indicator ID"])
        ind_name = str(r["Indicator"])
        disease = map_disease(ind_name)
        if disease not in {"asthma", "copd"}:
            continue

        d = ftp.get_data_for_indicator_at_all_available_geographies(ind_id)
        if d is None or len(d) == 0:
            continue

        area_col, date_col, val_col = choose_fingertips_cols(d)
        if not area_col or not date_col or not val_col:
            continue

        d = d[d[area_col].astype(str).str.contains("London", case=False, na=False)]
        if d.empty:
            continue

        rows.append(pd.DataFrame({
            "city": "London",
            "country": "United Kingdom",
            "iso3": "GBR",
            "date": pd.to_datetime(d[date_col].astype(str).str[:4] + "-01-01", errors="coerce"),
            "disease": disease,
            "indicator": ind_name,
            "outcome_value": pd.to_numeric(d[val_col], errors="coerce"),
            "outcome_unit": "rate",
            "outcome_type": "emergency admission",
            "geo_level": "region",
            "source": "UK Fingertips Inhale",
        }))

    if not rows:
        return pd.DataFrame(columns=OUT_COLS)
    return standardize(pd.concat(rows, ignore_index=True))


def fetch_eurostat() -> pd.DataFrame:
    import eurostat

    df = eurostat.get_data_df("hlth_co_disch2")
    if df is None or df.empty:
        return pd.DataFrame(columns=OUT_COLS)

    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
    id_cols = [c for c in df.columns if c not in year_cols]
    long_df = df.melt(id_vars=id_cols, value_vars=year_cols, var_name="year", value_name="outcome_value")

    geo_col = find_col(long_df, ["geo"])
    diag_col = find_col(long_df, ["icd10", "diag", "disease"])
    unit_col = find_col(long_df, ["unit"])
    sex_col = find_col(long_df, ["sex"])
    age_col = find_col(long_df, ["age"])

    if geo_col:
        long_df = long_df[long_df[geo_col].isin(CAPITALS.keys())]
    if unit_col and (long_df[unit_col].astype(str) == "P_HTHAB").any():
        long_df = long_df[long_df[unit_col].astype(str) == "P_HTHAB"]

    long_df = keep_total_if_present(long_df, sex_col)
    long_df = keep_total_if_present(long_df, age_col)

    code_col = diag_col if diag_col else find_col(long_df, ["diagnosis", "icd", "diag"])
    if not code_col:
        return pd.DataFrame(columns=OUT_COLS)

    codes = long_df[code_col].astype(str)
    disease = codes.map(map_disease)
    long_df = long_df[disease.notna()].copy()
    if long_df.empty:
        return pd.DataFrame(columns=OUT_COLS)

    rows = []
    for idx, r in long_df.iterrows():
        geo = r[geo_col] if geo_col else None
        if geo not in CAPITALS:
            continue
        city, country, iso3 = CAPITALS[geo]
        code = str(r[code_col])
        rows.append({
            "city": city,
            "country": country,
            "iso3": iso3,
            "date": pd.to_datetime(f"{int(r['year'])}-01-01", errors="coerce"),
            "disease": map_disease(code),
            "indicator": code,
            "outcome_value": pd.to_numeric(r["outcome_value"], errors="coerce"),
            "outcome_unit": str(r[unit_col]) if unit_col else "unknown",
            "outcome_type": "hospital discharge",
            "geo_level": "country_proxy",
            "source": "Eurostat hlth_co_disch2",
        })
    return standardize(pd.DataFrame(rows))


def fetch_mexico_issste_icd() -> pd.DataFrame:
    url = "https://repodatos.atdt.gob.mx/api_update/issste/datos_egresos_hospitalarios/egresos_hospitalarios_issste2024.csv"
    df = fetch_csv(url)

    city_col = find_col(df, ["entidad", "estado", "city"])
    code_col = find_col(df, ["diagnostico_principal_cie10", "cie10", "icd"])
    date_col = find_col(df, ["fecha_egreso", "fecha", "date"])

    if not city_col or not code_col:
        return pd.DataFrame(columns=OUT_COLS)

    city_mask = (
        df[city_col].astype(str).str.contains("Ciudad", case=False, na=False)
        & df[city_col].astype(str).str.contains("Mexico", case=False, na=False)
    )
    df = df[city_mask].copy()

    codes = df[code_col].astype(str).str.upper()
    keep = (
        codes.str.startswith("J30")
        | codes.str.startswith("J31")
        | codes.str.startswith("J67")
        | codes.str.startswith("B44")
    )
    df = df[keep].copy()
    if df.empty:
        return pd.DataFrame(columns=OUT_COLS)

    out = pd.DataFrame({
        "city": "Mexico City",
        "country": "Mexico",
        "iso3": "MEX",
        "date": pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.to_datetime("2024-01-01"),
        "disease": codes[keep].map(map_disease).values,
        "indicator": codes[keep].values,
        "outcome_value": 1.0,
        "outcome_unit": "count",
        "outcome_type": "hospital discharge",
        "geo_level": "state_proxy",
        "source": "Mexico ISSSTE Egresos",
    })
    return standardize(out)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    gcols = [
        "city", "country", "iso3", "date", "disease",
        "source", "outcome_type", "outcome_unit", "geo_level",
    ]
    out = (
        df.groupby(gcols, as_index=False)
        .agg(
            outcome_value=("outcome_value", "mean"),
            indicator=("indicator", lambda x: " | ".join(sorted(set(str(v) for v in x if pd.notna(v))))[:500]),
        )
        .sort_values(["city", "date", "disease", "source"])
        .reset_index(drop=True)
    )
    return out[OUT_COLS]


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    for name, fn in [
        ("ca_asthma", fetch_ca_asthma),
        ("ca_preventable", fetch_ca_preventable),
        ("uk_fingertips", fetch_london_fingertips),
        ("eurostat_hlth_co_disch2", fetch_eurostat),
        ("mexico_issste_icd", fetch_mexico_issste_icd),
    ]:
        try:
            d = fn()
            if d is not None and not d.empty:
                d.to_csv(raw_dir / f"{name}.csv", index=False)
                parts.append(d)
                print(f"[OK] {name}: {len(d)} rows")
            else:
                print(f"[WARN] {name}: no rows")
        except Exception as e:
            print(f"[WARN] {name}: {type(e).__name__}: {e}")

    if not parts:
        raise RuntimeError("No outcomes pulled from internet sources.")

    raw = pd.concat(parts, ignore_index=True).drop_duplicates().reset_index(drop=True)
    raw = standardize(raw)
    agg = aggregate(raw)

    raw_path = out_dir / "outcomes_master_by_disease_raw.csv"
    agg_path = out_dir / "outcomes_master_by_disease.csv"
    raw.to_csv(raw_path, index=False)
    agg.to_csv(agg_path, index=False)

    got = set(agg["disease"].dropna().unique().tolist())
    missing = EXPECTED_DISEASES - got

    print(f"[DONE] raw: {raw_path} shape={raw.shape}")
    print(f"[DONE] agg: {agg_path} shape={agg.shape}")
    print("[INFO] diseases_found:", sorted(got))
    print("[INFO] cities_found:", sorted(agg["city"].dropna().unique().tolist()))

    if missing:
        raise RuntimeError(
            f"Missing diseases from external internet pulls: {sorted(missing)}. "
            "Need additional source(s) for those labels."
        )


if __name__ == "__main__":
    main()
