# test_predict_manual.py
import pandas as pd
import joblib
from predict_allergy_risk import align_to_scaler_schema, risk_band

# ------------------ MANUAL TEST INPUT ------------------ #
# One row each for Low, Moderate, High risk expectation
data = [
    # LOW risk: low pollen + low pollution + mild weather
    {
        "temperature": 20, "humidity": 60, "dewpoint": 10,
        "precipitation": 0.0, "windspeed": 2, "winddirection": 90,
        "pressure": 1015, "uv_index": 2,
        "pm2_5": 5, "pm10": 10, "co": 0.2, "no2": 5, "so2": 2, "o3": 20,
        "european_aqi": 10,
        "alder_pollen": 0, "birch_pollen": 0, "grass_pollen": 1,
        "mugwort_pollen": 0, "olive_pollen": 0, "ragweed_pollen": 0,
        "month": 1, "day_of_week": 2, "hour": 9,
        "pollen_score": 1, "pollution_score": 8, "weather_risk": 2,
        "allergy_risk_score": 0.1,
        "city": "London"
    },
    # MODERATE risk: medium pollen + medium pollution
    {
        "temperature": 28, "humidity": 45, "dewpoint": 16,
        "precipitation": 0.0, "windspeed": 5, "winddirection": 150,
        "pressure": 1010, "uv_index": 5,
        "pm2_5": 20, "pm10": 35, "co": 0.5, "no2": 20, "so2": 5, "o3": 40,
        "european_aqi": 60,
        "alder_pollen": 10, "birch_pollen": 15, "grass_pollen": 30,
        "mugwort_pollen": 5, "olive_pollen": 0, "ragweed_pollen": 2,
        "month": 5, "day_of_week": 3, "hour": 14,
        "pollen_score": 62, "pollution_score": 24, "weather_risk": 10,
        "allergy_risk_score": 0.5,
        "city": "London"
    },
    # HIGH risk: high pollen + high pollution + unfavorable weather
    {
        "temperature": 32, "humidity": 30, "dewpoint": 22,
        "precipitation": 0.0, "windspeed": 1, "winddirection": 200,
        "pressure": 1005, "uv_index": 8,
        "pm2_5": 60, "pm10": 90, "co": 1.2, "no2": 80, "so2": 20, "o3": 90,
        "european_aqi": 150,
        "alder_pollen": 50, "birch_pollen": 70, "grass_pollen": 90,
        "mugwort_pollen": 20, "olive_pollen": 10, "ragweed_pollen": 15,
        "month": 6, "day_of_week": 4, "hour": 16,
        "pollen_score": 255, "pollution_score": 68, "weather_risk": 25,
        "allergy_risk_score": 0.9,
        "city": "London"
    }
]

df = pd.DataFrame(data)

# ------------------ LOAD MODEL ------------------ #
scaler = joblib.load("allergy_scaler.pkl")
model = joblib.load("allergy_lgbm_model.pkl")

# ------------------ ALIGN & PREDICT ------------------ #
X, meta = align_to_scaler_schema(df, scaler)
Xs = scaler.transform(X)
probs = model.predict_proba(Xs)[:, 1]
labels = (probs >= 0.5).astype(int)

df["predicted_allergy_label"] = labels
df["allergy_risk_probability"] = probs
df["risk_band"] = [risk_band(p) for p in probs]

print("\n=== Manual Test Predictions ===")
print(df[["city", "temperature", "pm2_5", "grass_pollen",
          "allergy_risk_probability", "risk_band"]])
