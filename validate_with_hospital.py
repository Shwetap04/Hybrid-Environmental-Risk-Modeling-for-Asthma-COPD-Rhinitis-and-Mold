import pandas as pd

# === Load predictions ===
pred_file = "allergy_predictions.csv"
hosp_file = "hospital_clean.csv"   # ✅ your cleaned version

pred = pd.read_csv(pred_file, parse_dates=["timestamp"])
print(f"✅ Predictions loaded: {pred.shape}")

# === Load hospital visits ===
hosp = pd.read_csv(hosp_file)

print(f"✅ Hospital shape: {hosp.shape}")
print("📝 First few rows:\n", hosp.head())

# === Keep relevant cols ===
hosp = hosp[["city", "year", "admissions"]].copy()
hosp["year"] = pd.to_numeric(hosp["year"], errors="coerce")
hosp["admissions"] = pd.to_numeric(hosp["admissions"], errors="coerce")

print(f"✅ Cleaned hospital shape: {hosp.shape}")
print(hosp.head())

# === Aggregate predictions to yearly per city ===
pred["year"] = pred["timestamp"].dt.year
pred_grouped = (
    pred.groupby(["city", "year"])["allergy_risk_probability"]
    .mean()
    .reset_index(name="avg_predicted_risk")
)

print("✅ Aggregated predictions:\n", pred_grouped.head())

# === Merge predictions with hospital visits ===
merged = pd.merge(pred_grouped, hosp, on=["city", "year"], how="inner")

print("📊 Merged Data:")
print(merged.head())

# === Correlation check ===
if not merged.empty:
    corr = merged["avg_predicted_risk"].corr(merged["admissions"])
    print(f"\n🔗 Correlation between predicted allergy risk and hospital admissions: {corr:.3f}")
else:    
    print("⚠ No overlapping years found between predictions and hospital visits.")
