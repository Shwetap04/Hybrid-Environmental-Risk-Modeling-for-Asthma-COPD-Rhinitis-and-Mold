# clean_hospital.py
"""
Clean the WHO 'HFA_402' hospital discharges file into a simple table:
country_iso3, year, admissions, city
Saves hospital_clean.csv
"""

import re
import pandas as pd
from pathlib import Path

RAW = "hospital_visits.csv"   # the file you uploaded
OUT = "hospital_clean.csv"

# map iso3 to country name / city (only five countries needed)
iso_map = {
    "DEU": ("Germany", "Berlin"),
    "GBR": ("United Kingdom", "London"),
    "FRA": ("France", "Paris"),
    "ESP": ("Spain", "Madrid"),
    "ITA": ("Italy", "Rome")
}

def parse_raw_lines(path):
    """Return list of tuples (iso3, year, value) from messy file."""
    records = []
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    # Look for lines with pattern: ISO  <stuff>  YEAR  VALUE
    # Examples in your snippet: DEU    ALL    1993    993981
    pattern = re.compile(r"\b([A-Z]{3})\b.*?\bALL\b.*?(\d{4})\D+([0-9,]+)")
    for ln in lines:
        m = pattern.search(ln)
        if m:
            iso = m.group(1)
            year = int(m.group(2))
            val = int(m.group(3).replace(",", ""))
            records.append((iso, year, val))
    return records

if __name__ == "__main__":
    if not Path(RAW).exists():
        raise FileNotFoundError(f"{RAW} not found. Place the WHO file in the project folder.")

    recs = parse_raw_lines(RAW)
    if not recs:
        raise RuntimeError("No records parsed from hospital file. Check file format. Parsing looks for lines like 'DEU  ALL  1993  993981'.")

    df = pd.DataFrame(recs, columns=["iso3", "year", "admissions"])
    # keep only our five countries
    df = df[df["iso3"].isin(iso_map.keys())].copy()
    df["country"] = df["iso3"].map(lambda c: iso_map[c][0])
    df["city"] = df["iso3"].map(lambda c: iso_map[c][1])
    df = df[["country", "iso3", "city", "year", "admissions"]].sort_values(["city","year"]).reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f"✅ Clean hospital data saved to {OUT} (rows: {len(df)})")
