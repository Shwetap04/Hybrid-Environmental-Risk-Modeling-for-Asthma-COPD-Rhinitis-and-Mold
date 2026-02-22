# clean_hospital_csv.py
import pandas as pd

hosp_file = "hospital_visits.csv"
output_file = "hospital_visits_clean.csv"

# Read raw file
with open(hosp_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find first row that starts with 'country'
start_row = None
for i, line in enumerate(lines):
    if line.lower().startswith("country"):
        start_row = i
        break

if start_row is None:
    raise ValueError("❌ Could not find header row with 'country' in hospital file.")

# Load from correct header
hosp = pd.read_csv(hosp_file, skiprows=start_row)

# Ensure correct column names
expected_cols = ["country", "iso3", "city", "year", "admissions"]
if list(hosp.columns[:5]) != expected_cols:
    hosp.columns = expected_cols

# Keep only useful columns
hosp = hosp[["country", "city", "year", "admissions"]]

# Drop missing or invalid years
hosp["year"] = pd.to_numeric(hosp["year"], errors="coerce")
hosp = hosp.dropna(subset=["year", "admissions"]).reset_index(drop=True)
hosp["year"] = hosp["year"].astype(int)

# Save cleaned version
hosp.to_csv(output_file, index=False)

print(f"✅ Cleaned hospital file saved: {output_file}")
print(hosp.head(10))
