# AI Coding Agent Instructions for Allergy Risk Assessment Model

## Project Overview
This is an ML pipeline for predicting environmental allergy risk using weather, air quality, and pollen data. The system fetches data from Open-Meteo APIs, preprocesses it with proxy labels, trains ensemble models (LightGBM primary), and provides predictions via CLI, Streamlit UI, and real-time APIs.

## Architecture
- **Data Flow**: `allergy_data_fetcher.py` → `allergy_data_preprocessing.py` → model training → prediction scripts
- **Models**: Binary classification (allergy risk: 0/1) with probability output, banded into Low (<0.33), Moderate (0.33-0.66), High (>0.66)
- **Features**: Environmental (temperature, humidity, pm2_5, pm10, o3, pollen counts, uv_index, pressure) + derived (month, day_of_week, hour) + engineered (lags, rolling averages)
- **Preprocessing**: StandardScaler on DataFrame (preserves feature_names_in_), SMOTE for imbalance, interpolation by city
- **Artifacts**: Models/scalers saved as .pkl with joblib; predictions include SHAP explanations and agentic advice

## Key Patterns
- **Feature Engineering**: Add lags (1-3 hours) and rolling means (3-hour) for time-series features like temperature, humidity, pm2_5, grass_pollen [see `allergy_model_comparison.py` lines 25-35]
- **Proxy Labels**: Combine pollen_score (sum of pollen cols), pollution_score (mean of pollutants), weather_risk (humidity + temp deviation) into allergy_risk_score percentile-based binary label [see `allergy_data_preprocessing.py` lines 25-50]
- **Model Loading**: Always load scaler first, align input features to scaler.feature_names_in_, transform, then predict_proba [see `predict_allergy_risk.py` lines 45-70]
- **SHAP Integration**: Use TreeExplainer for tree models, KernelExplainer fallback; shap_values for positive class [see `predict_allergy_risk.py` lines 120-140]
- **API Fetching**: Open-Meteo endpoints for weather/air-quality; merge on datetime, add city column [see `allergy_data_fetcher.py` lines 40-80]

## Workflows
- **Train Models**: Run `python final_allergy_model.py` for production LightGBM + stacking; uses `allergy_cleaned.csv`
- **Compare Models**: `python allergy_model_comparison.py` evaluates LogisticRegression, XGBoost, LightGBM, Stacking with metrics + SHAP plot
- **Predict**: `python predict_allergy_risk.py --csv input.csv --out preds.csv` for batch; includes agentic advice based on rules + SHAP drivers
- **Real-time**: `python realtime_predict.py` fetches live London data and predicts
- **UI**: `streamlit run allergy_ui.py` for interactive predictions (manual entry or CSV upload)
- **Validate**: `python validate_with_hospital.py` correlates predictions with hospital admissions data

## Conventions
- Drop meta columns (datetime, city) before training/prediction; keep for output
- Use joblib for all model/scaler I/O
- Handle missing features by adding 0s, unknown by dropping [see align_to_scaler_schema in `predict_allergy_risk.py`]
- Agentic advice: Rule-based (e.g., high pollen → stay indoors) augmented by top 5 SHAP features
- Time-series grouping: Always groupby city for interpolation/lags/rolling

## Dependencies
- Core: pandas, numpy, scikit-learn, lightgbm, xgboost, imbalanced-learn, shap
- UI: streamlit
- APIs: requests

Reference: `allergy_cleaned.csv` for feature schema, `final_allergy_model.py` for training example, `predict_allergy_risk.py` for prediction pipeline.