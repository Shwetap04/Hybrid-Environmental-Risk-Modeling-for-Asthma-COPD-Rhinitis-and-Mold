
```md
# AI-Powered Environmental Allergy Risk Assessment

A practical ML system that estimates environmental allergy risk from weather + air-quality signals, supports city-level forecasting, and adds disease-specific signals (asthma, COPD, rhinitis, mold) using external outcome datasets.

---

## Why This Project

Allergy and respiratory flare-ups are strongly influenced by environment.  
This project combines:

- Real-time environmental inputs (Open-Meteo)
- Machine learning risk prediction
- Forecasting and explainability
- External outcome alignment for disease-specific validation

---

## Current Status (Important)

This repository currently has **two modeling tracks**:

- **Base allergy risk model** (`allergy_lgbm_model.pkl`, `allergy_lgbm_regressor.pkl`): trained on proxy-style target columns in `allergy_cleaned.csv`.
- **Disease-specific live signals** (`external_outcomes/models_by_disease/*.pkl`): trained from external outcome datasets aligned with environmental data.

So the system is **hybrid today**: not fully proxy-free yet, but already partially outcome-grounded.

---

## Key Features

- Streamlit UI for:
  - Manual entry
  - Open-Meteo live prediction
  - CSV upload prediction
- Hourly forecast plot and daily summary
- Internal risk forecaster comparison
- SHAP-based explainability
- Disease-specific live signals:
  - Asthma
  - COPD
  - Rhinitis
  - Mold
- Local precautions chatbot with optional Ollama (`llama3`) context-aware responses

---

## Tech Stack

- Python, Streamlit
- LightGBM, XGBoost, scikit-learn
- SHAP
- pandas, numpy, joblib
- Open-Meteo APIs
- Optional: Ollama (local LLM)

---

## Project Structure

- `allergy_ui.py` — main Streamlit app
- `final_allergy_model.py` — base model training (proxy-based target)
- `build_allergy_cleaned.py` — builds `allergy_cleaned.csv` from Open-Meteo merged data
- `fetch_openmeteo_historical.py` — fetches historical Open-Meteo data
- `risk_forecast_eval.py` — internal forecaster training/evaluation
- `external_outcomes/build_outcomes_by_disease.py` — builds disease outcome master datasets from external sources
- `external_outcomes/align_env_outcomes.py` — aligns outcomes with environment aggregates
- `external_outcomes/train_outcome_models.py` — trains disease-specific outcome models
- `validate_with_hospital.py` — correlation check with hospital admissions

---

## Quick Start

### 1) Environment setup

```bash
pip install pandas numpy scikit-learn lightgbm xgboost imbalanced-learn shap streamlit requests joblib
```

Optional chatbot backend:
```bash
ollama run llama3
```

### 2) Launch app

From `final project`:

```bash
streamlit run allergy_ui.py
```

---

## Reproducible Pipeline

From `final project`:

### A) Build/refresh environmental dataset
```bash
python fetch_openmeteo_historical.py --start-date 2023-01-01 --end-date 2025-12-31 --out openmeteo_2023_2025_merged_full.csv
```

### B) Build cleaned schema dataset
```bash
python build_allergy_cleaned.py --input openmeteo_2023_2025_merged_full.csv --output allergy_cleaned.csv
```

### C) Train base allergy models
```bash
python final_allergy_model.py
```

### D) Build disease outcomes + align + train disease models
```bash
python -u external_outcomes/build_outcomes_by_disease.py
python -u external_outcomes/align_env_outcomes.py
python -u external_outcomes/train_outcome_models.py
```

### E) Forecast evaluation
```bash
python risk_forecast_eval.py
```

---

## Data Sources

- Open-Meteo (historical + live weather/air quality)
- California asthma ED outcomes
- Eurostat respiratory discharge indicators
- Mexico ISSSTE discharge records (ICD-coded mappings used for rhinitis/mold)
- Additional UK/other connectors where available

---

## Evaluation Artifacts

- `final_model_metrics.csv` — base classification metrics
- `final_model_metrics_regression.csv` — base regressor metrics
- `model_comparison_results.csv` — baseline model comparison
- `shap_summary_lgbm.png` and `shap_importance_lgbm.csv` — explainability
- `risk_forecast_metrics.csv` — forecast metrics
- `external_outcomes/outcome_model_metrics.csv` — disease model LOOCV metrics
- `external_outcomes/aligned_env_outcomes_metrics.csv` — env-outcome alignment statistics

---

## Limitations

- Base risk model is still trained on proxy-style labels.
- Outcome-aligned disease data overlap is currently limited in city-time coverage.
- This is a decision-support prototype, not a medical diagnostic tool.

---

## Roadmap (Near-Term)

- Replace base proxy target with stronger external outcome labels where overlap allows.
- Increase city-time overlap for rhinitis and mold.
- Move from mixed hybrid reporting to fully outcome-grounded reporting.
- Add stricter temporal validation (rolling-origin backtests by city).

---

## Medical Disclaimer

This project provides predictive risk guidance for research/education.  
It is **not** medical advice. For symptoms or treatment, consult a licensed clinician.

---

## Author

Shweta  
Final Year Project — AI & Data Science Engineering
```
