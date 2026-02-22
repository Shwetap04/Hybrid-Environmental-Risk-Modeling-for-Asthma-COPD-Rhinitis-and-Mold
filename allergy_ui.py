# allergy_ui.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import requests
from collections import OrderedDict
import re
from predict_allergy_risk import align_to_scaler_schema

# ------------------- PAGE SETUP -------------------
st.set_page_config(page_title="🌿 Allergy Risk Predictor", page_icon="🤧", layout="centered")

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to bottom, #f0f8ff, #e6f3ef, #f7f9fc);
    }
    h1, p { text-align: center; }
    div[data-testid="stChatMessage"] {
        border-radius: 18px;
        padding: 12px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------- HEADER -------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#4B8A8D;'>🌿 AI-Powered Allergy Risk Predictor</h1>
    <p style='text-align:center; color:#666;'>Get real-time predictions of allergy flare-up risk using environment data.</p>
    """,
    unsafe_allow_html=True,
)

# ------------------- LOAD MODEL -------------------
@st.cache_resource
def load_model():
    base = Path(__file__).resolve().parent
    model = joblib.load(base / "allergy_lgbm_model.pkl")
    scaler = joblib.load(base / "allergy_scaler.pkl")
    return model, scaler

@st.cache_resource
def load_calibrated_model():
    base = Path(__file__).resolve().parent
    p = base / "allergy_lgbm_calibrated.pkl"
    if p.exists():
        return joblib.load(p)
    return None

@st.cache_resource
def load_regressor_model():
    base = Path(__file__).resolve().parent
    p = base / "allergy_lgbm_regressor.pkl"
    if p.exists():
        return joblib.load(p)
    return None

@st.cache_resource
def load_multi_model():
    base = Path(__file__).resolve().parent
    model = joblib.load(base / "multioutput_model.pkl")
    scaler = joblib.load(base / "multioutput_scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()
    calibrated_model = load_calibrated_model()
    reg_model = load_regressor_model()
    st.success("✅ Model & Scaler loaded successfully")
except Exception:
    st.error("⚠ Could not load model/scaler. Please ensure `allergy_lgbm_model.pkl` and `allergy_scaler.pkl` exist.")
    st.stop()

prob_model = calibrated_model if calibrated_model is not None else model

def predict_risk_probs(Xs_df: pd.DataFrame, prefer: str = "classifier") -> np.ndarray:
    if prefer == "regressor" and reg_model is not None:
        preds = reg_model.predict(Xs_df)
        return np.clip(preds, 0, 1)
    # default: classifier probabilities
    return prob_model.predict_proba(Xs_df)[:, 1]

try:
    multi_model, multi_scaler = load_multi_model()
    st.success("✅ Multi-output condition model loaded successfully")
except Exception:
    multi_model, multi_scaler = None, None
    st.info("ℹ Multi-output model not found. Run `train_multioutput.py` to enable multi-condition outputs.")


# ------------------- HELPER -------------------
def risk_band(prob: float) -> str:
    if prob < 0.33:
        return "Low"
    elif prob < 0.66:
        return "Moderate"
    else:
        return "High"

def severity_label(val) -> str:
    try:
        v = int(val)
    except Exception:
        return "Unknown"
    if v == 0:
        return "Low"
    if v == 1:
        return "Moderate"
    if v == 2:
        return "High"
    return "Unknown"

def make_risk_forecast_features(history_risk: list[float], future_ts: pd.Timestamp) -> dict:
    feats: dict[str, float] = {}
    for lag in range(1, 25):
        feats[f"risk_lag{lag}"] = float(history_risk[-lag])
    for w in (3, 6, 12, 24):
        feats[f"risk_roll{w}"] = float(np.mean(history_risk[-w:]))
    feats["hour"] = int(future_ts.hour)
    feats["day_of_week"] = int(future_ts.dayofweek)
    feats["month"] = int(future_ts.month)
    return feats

MULTI_TARGETS = [
    ("rhinitis_severity", "Allergic Rhinitis"),
    ("asthma_severity", "Asthma Exacerbation"),
    ("copd_severity", "COPD Irritation"),
    ("mold_severity", "Mold-Related Allergy"),
]

def render_multi_outputs(df: pd.DataFrame) -> None:
    if multi_model is None or multi_scaler is None:
        return
    X, _ = align_to_scaler_schema(df, multi_scaler)
    Xs = multi_scaler.transform(X)
    Xs_df = pd.DataFrame(Xs, columns=X.columns)
    preds = multi_model.predict(Xs_df)
    out = pd.DataFrame(preds, columns=[k for k, _ in MULTI_TARGETS])
    for k, label in MULTI_TARGETS:
        out[k] = out[k].map(severity_label)
        out = out.rename(columns={k: label})
    st.subheader("Multi-Condition Risk (Condition-Specific Proxy Outputs)")
    st.dataframe(out, use_container_width=True)
    st.session_state["latest_multi_condition_outputs"] = out.to_dict("records")


DISEASE_LIVE_FEATURES = [
    "pollution_score",
    "pollen_score",
    "weather_risk",
    "pm2_5",
    "european_aqi",
    "temperature",
    "humidity",
    "windspeed",
]

DISEASE_ORDER = ["asthma", "copd", "rhinitis", "mold"]


@st.cache_resource
def load_disease_live_models() -> dict:
    base = Path(__file__).resolve().parent / "external_outcomes" / "models_by_disease"
    models = {}
    for disease in DISEASE_ORDER:
        p = base / f"{disease}_linear_model.pkl"
        if p.exists():
            try:
                models[disease] = joblib.load(p)
            except Exception:
                pass
    return models


def _score_band(x: float) -> str:
    if x < 0.33:
        return "Low"
    if x < 0.66:
        return "Moderate"
    return "High"


def render_disease_live_outputs(df: pd.DataFrame) -> None:
    models = load_disease_live_models()
    if not models:
        return

    tmp = df.copy()
    tmp = add_derived_features(tmp)
    for c in DISEASE_LIVE_FEATURES:
        if c not in tmp.columns:
            tmp[c] = 0
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0)

    X_live = tmp[DISEASE_LIVE_FEATURES]
    latest_idx = len(X_live) - 1
    scores = {}
    rows = []
    for disease in DISEASE_ORDER:
        model_d = models.get(disease)
        if model_d is None:
            continue
        raw = float(model_d.predict(X_live)[latest_idx])
        # Map z-score-like model output to 0..1 so UI is comparable across diseases.
        score01 = float(1.0 / (1.0 + np.exp(-raw)))
        scores[disease] = score01
        rows.append(
            {
                "disease": disease,
                "live_signal": score01,
                "band": _score_band(score01),
            }
        )

    if not rows:
        return

    st.subheader("Disease-Specific Live Signals")
    st.caption("Derived from current live environmental inputs using per-disease external-outcome models.")
    cols = st.columns(len(rows))
    for i, r in enumerate(rows):
        cols[i].metric(r["disease"].upper(), f"{r['live_signal']:.1%}", r["band"])
    st.session_state["latest_disease_live_scores"] = rows

def explain_shap_for_sample(model, Xs_df: pd.DataFrame, top_k: int = 10):
    """
    SHAP explanation for the *base* allergy probability model.
    Note: The base LGBM model is trained on scaled features, so SHAP is computed
    in the scaled feature space (still useful for relative drivers).
    """
    try:
        import shap  # lazy import; keeps startup lighter if SHAP isn't needed
    except Exception as e:
        st.warning(f"SHAP not available: {e}")
        return

    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xs_df)
        if isinstance(sv, list):
            sv = sv[1]
        vals = sv[0]
        s = pd.Series(vals, index=Xs_df.columns).sort_values(key=lambda x: x.abs(), ascending=False)
        top = s.head(top_k).reset_index()
        top.columns = ["feature", "shap_value"]
        st.subheader("Explainability (SHAP Drivers)")
        st.caption("Top features by absolute SHAP contribution (scaled feature space).")
        st.dataframe(top, use_container_width=True)
        st.bar_chart(top.set_index("feature")["shap_value"])
    except Exception as e:
        st.warning(f"SHAP explanation failed: {e}")

def _num(val, default=0.0) -> float:
    try:
        if val is None:
            return default
        if isinstance(val, str) and not val.strip():
            return default
        if pd.isna(val):
            return default
        return float(val)
    except Exception:
        return default

def health_note(row: dict) -> str:
    notes = []
    pm25 = _num(row.get("pm2_5", 0))
    aqi = _num(row.get("european_aqi", 0))
    grass = _num(row.get("grass_pollen", 0))
    birch = _num(row.get("birch_pollen", 0))
    ragweed = _num(row.get("ragweed_pollen", 0))
    humidity = _num(row.get("humidity", 50), default=50)

    if pm25 >= 35 or aqi >= 100:
        notes.append("Respiratory irritation risk may be elevated due to air pollution.")
    if grass >= 50 or birch >= 30 or ragweed >= 20:
        notes.append("Allergen exposure looks elevated (pollen).")
    if humidity < 30:
        notes.append("Low humidity can irritate airways.")
    return " ".join(notes) if notes else "No additional respiratory warning based on current inputs."

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        df["month"] = ts.dt.month
        df["day_of_week"] = ts.dt.dayofweek
        df["hour"] = ts.dt.hour
    pollen_cols = [c for c in ["alder_pollen","birch_pollen","grass_pollen","mugwort_pollen","olive_pollen","ragweed_pollen"] if c in df.columns]
    if pollen_cols:
        df[pollen_cols] = df[pollen_cols].apply(pd.to_numeric, errors="coerce")
        df["pollen_score"] = df[pollen_cols].fillna(0).sum(axis=1)
    if any(c in df.columns for c in ["pm2_5","pm10","no2","o3","so2","co"]):
        cols = [c for c in ["pm2_5","pm10","no2","o3","so2","co"] if c in df.columns]
        df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
        df["pollution_score"] = df[cols].fillna(0).mean(axis=1)
    if "humidity" in df.columns and "temperature" in df.columns:
        hum = df["humidity"].clip(0, 100).fillna(50)
        temp = df["temperature"].fillna(22)
        df["weather_risk"] = (100 - hum) * 0.3 + (temp.sub(22).abs() * 0.2)
    return df

# ------------------- CHATBOT (LIGHTWEIGHT AI) -------------------
OLLAMA_URL = "http://localhost:11434/api/chat"

def _build_kb_docs() -> list[dict]:
    return [
        {
            "title": "Project Summary",
            "text": (
                "This system predicts allergy and respiratory risk using real-time weather, "
                "air quality, and pollen signals. It outputs a risk probability, risk band, "
                "and optional multi-condition severity outputs."
            ),
        },
        {
            "title": "Models Used",
            "text": (
                "The base predictor is a LightGBM classifier. A calibrated variant improves "
                "probability calibration, and a LightGBM regressor is trained on a proxy risk score "
                "to provide a smooth risk scale."
            ),
        },
        {
            "title": "Forecasting Modes",
            "text": (
                "Open-Meteo Forecast uses future weather/air/pollen inputs from the API. "
                "Internal Forecast uses a risk-only time-series model built from the past 24 hours "
                "of model risk with lag and rolling features."
            ),
        },
        {
            "title": "Validation Approach",
            "text": (
                "Validation includes model comparison metrics and correlation against hospital "
                "admissions aggregated by city and year. Results are framed as population-level "
                "risk indicators, not clinical diagnosis."
            ),
        },
        {
            "title": "Explainability",
            "text": (
                "SHAP explanations highlight the top drivers of the base model's predictions. "
                "Because the model is trained on scaled features, SHAP values are shown in the "
                "scaled feature space and are used for relative importance."
            ),
        },
        {
            "title": "Limitations",
            "text": (
                "Proxy labels are derived from environmental exposure signals and do not represent "
                "patient-level ground truth. Outputs are advisory and should not be used as medical diagnosis."
            ),
        },
        {
            "title": "General Precautions",
            "text": (
                "General precautions include reducing outdoor exposure during high pollution or pollen, "
                "keeping windows closed, using air filtration, and limiting strenuous outdoor activity. "
                "If you have a clinical plan or prescribed medication, follow your clinician's guidance."
            ),
        },
    ]

@st.cache_resource
def _build_chat_index():
    docs = _build_kb_docs()
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = [f"{d['title']}. {d['text']}" for d in docs]
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(texts)
        return vectorizer, X, docs
    except Exception:
        return None, None, docs

@st.cache_data
def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        return None
    return None

def _summarize_results_context(base: Path, max_features: int = 8) -> str:
    parts: list[str] = []

    # Model comparison
    comp = _read_csv_if_exists(base / "model_comparison_results.csv")
    if comp is not None and not comp.empty:
        parts.append("Model comparison (higher is better):")
        try:
            comp_small = comp.copy()
            if "Model" in comp_small.columns:
                comp_small = comp_small.set_index("Model")
            parts.append(comp_small.round(4).to_string())
        except Exception:
            pass

    # Final model metrics (classifier + regressor)
    fm = _read_csv_if_exists(base / "final_model_metrics.csv")
    if fm is not None and not fm.empty:
        parts.append("Final model metrics (classifier):")
        parts.append(fm.round(4).to_string(index=False))

    fm_reg = _read_csv_if_exists(base / "final_model_metrics_regression.csv")
    if fm_reg is not None and not fm_reg.empty:
        parts.append("Final model metrics (regressor):")
        parts.append(fm_reg.round(4).to_string(index=False))

    # Multi-output metrics
    mo = _read_csv_if_exists(base / "multioutput_metrics.csv")
    if mo is not None and not mo.empty:
        parts.append("Multi-output severity metrics:")
        parts.append(mo.round(4).to_string(index=False))

    # Forecast metrics
    rf = _read_csv_if_exists(base / "risk_forecast_metrics.csv")
    if rf is not None and not rf.empty:
        parts.append("Risk forecaster metrics:")
        parts.append(rf.round(4).to_string(index=False))

    # SHAP top drivers
    shap_imp = _read_csv_if_exists(base / "shap_importance_lgbm.csv")
    if shap_imp is not None and not shap_imp.empty:
        parts.append("Top SHAP drivers (LightGBM):")
        top = shap_imp.head(max_features)
        parts.append(top.to_string(index=False))

    return "\n\n".join(parts).strip()

def answer_chatbot(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return "Ask a question about the project, model, validation, or precautions."

    # Optional local LLM (Ollama)
    use_ollama = bool(st.session_state.get("use_ollama"))
    ollama_model = st.session_state.get("ollama_model", "llama3")
    if use_ollama:
        try:
            last_prob = st.session_state.get("last_risk_prob")
            last_band = st.session_state.get("last_risk_band")
            last_row = st.session_state.get("last_risk_row", {})
            ctx_lines = []
            if last_prob is not None and last_band:
                ctx_lines.append(f"Latest risk: {last_band} ({float(last_prob):.1%}).")
            if isinstance(last_row, dict) and last_row:
                # include key exposure signals if available
                for k in ("pm2_5", "european_aqi", "grass_pollen", "birch_pollen", "ragweed_pollen", "humidity"):
                    if k in last_row:
                        ctx_lines.append(f"{k}: {last_row.get(k)}")
            context = "\n".join(ctx_lines) if ctx_lines else "No recent prediction available."

            base = Path(__file__).resolve().parent
            results_ctx = _summarize_results_context(base)
            if results_ctx:
                context = f"{context}\n\nProject results:\n{results_ctx}"

            system = (
                "You are a safety-focused precautions assistant for an allergy and respiratory risk app. "
                "Give concise, practical precautions based on the provided context. "
                "Do NOT provide medical diagnosis. Include a brief disclaimer to follow clinical guidance."
            )
            user_msg = f"Question: {q}\n\nContext:\n{context}\n\nAnswer in bullet points."

            payload = {
                "model": ollama_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                "stream": False,
            }
            resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "").strip()
            if content:
                return content
        except Exception:
            # Fall back to local rules + KB if Ollama isn't reachable
            pass

    # Precautions intent (simple keyword match)
    ql = q.lower()
    precaution_kw = [
        "precaution", "precautions", "what should i do", "what can i do", "avoid",
        "protect", "safety", "safe", "care", "symptom", "allergy", "asthma", "mask",
    ]
    if any(k in ql for k in precaution_kw):
        # Use latest prediction context if available
        last_prob = st.session_state.get("last_risk_prob")
        last_band = st.session_state.get("last_risk_band")
        last_row = st.session_state.get("last_risk_row", {})

        tips = []
        pm25 = _num(last_row.get("pm2_5", 0))
        aqi = _num(last_row.get("european_aqi", 0))
        grass = _num(last_row.get("grass_pollen", 0))
        birch = _num(last_row.get("birch_pollen", 0))
        ragweed = _num(last_row.get("ragweed_pollen", 0))
        humidity = _num(last_row.get("humidity", 50), default=50)

        if pm25 >= 35 or aqi >= 100:
            tips.extend([
                "Limit outdoor time; avoid heavy outdoor exercise.",
                "Wear a well-fitting mask (N95/KN95) if you must go out.",
                "Keep windows closed; use air filtration if available.",
            ])
        if grass >= 50 or birch >= 30 or ragweed >= 20:
            tips.extend([
                "Reduce outdoor exposure during peak pollen hours.",
                "Shower and change clothes after being outdoors.",
                "Keep windows closed to reduce indoor pollen.",
            ])
        if humidity < 30:
            tips.append("Consider a humidifier; dry air can irritate airways.")

        if last_band in ("Moderate", "High"):
            tips.append("Have your personal action plan ready if you have one.")
        if not tips:
            tips.append("Conditions look low-to-moderate. Basic precautions: stay hydrated and ventilate indoor air responsibly.")

        risk_line = ""
        if last_prob is not None and last_band:
            risk_line = f"Latest risk: {last_band} ({float(last_prob):.1%}).\n\n"

        tips_text = "\n".join(f"- {t}" for t in tips)
        return (
            f"{risk_line}Precautions:\n{tips_text}\n\n"
            "Note: This is general guidance, not medical advice. If you have a clinical plan, follow it."
        )

    vectorizer, X, docs = _build_chat_index()
    if vectorizer is None or X is None:
        # Fallback keyword match
        for d in docs:
            if any(k in ql for k in d["title"].lower().split()):
                return f"{d['text']}\n\nSource: {d['title']}"
        return "I don't have a confident answer. Try asking about the model, validation, or forecasting."

    qv = vectorizer.transform([q])
    sims = (X @ qv.T).toarray().ravel()
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    best = docs[best_idx]

    if best_score < 0.10:
        return "I’m not confident on that. Try asking about the model, data, validation, or forecasting."
    return f"{best['text']}\n\nSource: {best['title']}"

@st.cache_data(ttl=1800)
def fetch_open_meteo_forecast(lat: float, lon: float, city: str, forecast_days: int = 3) -> pd.DataFrame:
    """
    Fetch hourly forecast from Open-Meteo weather + air-quality endpoints and map to the model schema.
    Returns a dataframe with one row per hour (timestamp).
    """
    weather_url = "https://api.open-meteo.com/v1/forecast"
    air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    forecast_days = int(max(1, min(7, forecast_days)))

    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,pressure_msl,uv_index,wind_speed_10m,wind_direction_10m",
        "forecast_days": forecast_days,
        "timezone": "auto",
    }
    air_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,european_aqi,alder_pollen,birch_pollen,grass_pollen,mugwort_pollen,olive_pollen,ragweed_pollen",
        "forecast_days": forecast_days,
        "timezone": "auto",
    }

    w = requests.get(weather_url, params=weather_params, timeout=30).json()
    a = requests.get(air_url, params=air_params, timeout=30).json()

    if "hourly" not in w or "hourly" not in a:
        raise ValueError("Open-Meteo API did not return hourly data.")

    wdf = pd.DataFrame(w["hourly"])
    adf = pd.DataFrame(a["hourly"])

    if "time" not in wdf.columns or "time" not in adf.columns:
        raise ValueError("Open-Meteo hourly payload missing 'time'.")

    merged = pd.merge(wdf, adf, on="time", how="inner")
    if merged.empty:
        raise ValueError("No overlapping hourly records returned from Open-Meteo.")

    col_map = {
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "dew_point_2m": "dewpoint",
        "precipitation": "precipitation",
        "pressure_msl": "pressure",
        "uv_index": "uv_index",
        "wind_speed_10m": "windspeed",
        "wind_direction_10m": "winddirection",
        "pm2_5": "pm2_5",
        "pm10": "pm10",
        "carbon_monoxide": "co",
        "nitrogen_dioxide": "no2",
        "sulphur_dioxide": "so2",
        "ozone": "o3",
        "european_aqi": "european_aqi",
        "alder_pollen": "alder_pollen",
        "birch_pollen": "birch_pollen",
        "grass_pollen": "grass_pollen",
        "mugwort_pollen": "mugwort_pollen",
        "olive_pollen": "olive_pollen",
        "ragweed_pollen": "ragweed_pollen",
    }

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(merged["time"], errors="coerce")
    for src, dst in col_map.items():
        if src in merged.columns:
            out[dst] = merged[src]
        else:
            out[dst] = 0
    out["city"] = city

    # numeric coercion + fill
    for col in out.columns:
        if col not in ["timestamp", "city"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.fillna(0)

    out = add_derived_features(out)
    return out

def load_dataset_cities() -> list[str]:
    try:
        df = pd.read_csv(Path(__file__).resolve().parent / "allergy_cleaned.csv", usecols=["city"])
        cities = sorted(df["city"].dropna().unique().tolist())
        return cities
    except Exception:
        return []

def load_dataset_city_coords() -> tuple[list[str], dict[str, tuple[float, float]]]:
    """
    Try to load city list (and optional coordinates) from allergy_dataset.csv or allergy_cleaned.csv.
    Returns (city_list, coords_map). coords_map may be empty if lat/lon not found.
    """
    base = Path(__file__).resolve().parent
    for fname in ["allergy_dataset.csv", "allergy_cleaned.csv"]:
        p = base / fname
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        city_col = cols.get("city")
        lat_col = cols.get("lat") or cols.get("latitude")
        lon_col = cols.get("lon") or cols.get("lng") or cols.get("longitude") or cols.get("long")
        if not city_col:
            continue
        cities = sorted(df[city_col].dropna().unique().tolist())
        coords = {}
        if lat_col and lon_col:
            tmp = df[[city_col, lat_col, lon_col]].dropna()
            grouped = tmp.groupby(city_col)[[lat_col, lon_col]].mean().reset_index()
            for _, row in grouped.iterrows():
                coords[str(row[city_col])] = (float(row[lat_col]), float(row[lon_col]))
        return cities, coords
    return [], {}

def slugify_city(city: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", city).strip("_")
    return s.lower()

def load_risk_forecaster(city: str | None = None):
    base = Path(__file__).resolve().parent
    if city:
        slug = slugify_city(city)
        p = base / f"risk_forecaster_{slug}.pkl"
        if p.exists():
            try:
                return joblib.load(p)
            except Exception:
                pass
    p = base / "risk_forecaster.pkl"
    if p.exists():
        try:
            return joblib.load(p)
        except Exception:
            return None
    return None

def load_risk_forecaster_meta(city: str | None = None):
    base = Path(__file__).resolve().parent
    if city:
        slug = slugify_city(city)
        p = base / f"risk_forecaster_{slug}_metadata.json"
        if p.exists():
            try:
                import json
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
    p = base / "risk_forecaster_metadata.json"
    if not p.exists():
        return None
    try:
        import json
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def fetch_open_meteo(lat: float, lon: float, city: str) -> pd.DataFrame:
    weather_url = "https://api.open-meteo.com/v1/forecast"
    air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,pressure_msl,uv_index,wind_speed_10m,wind_direction_10m",
        "timezone": "auto",
    }
    air_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,european_aqi,alder_pollen,birch_pollen,grass_pollen,mugwort_pollen,olive_pollen,ragweed_pollen",
        "timezone": "auto",
    }

    w = requests.get(weather_url, params=weather_params, timeout=30).json()
    a = requests.get(air_url, params=air_params, timeout=30).json()

    if "hourly" not in w or "hourly" not in a:
        raise ValueError("Open-Meteo API did not return hourly data.")

    wdf = pd.DataFrame(w["hourly"])
    adf = pd.DataFrame(a["hourly"])
    wdf["timestamp"] = pd.to_datetime(wdf["time"])
    adf["timestamp"] = pd.to_datetime(adf["time"])
    merged = pd.merge(wdf, adf, on="timestamp", how="inner")
    if merged.empty:
        raise ValueError("No overlapping hourly records returned from Open-Meteo.")

    latest = merged.iloc[-1].to_dict()
    row = {
        "timestamp": latest.get("timestamp"),
        "temperature": latest.get("temperature_2m"),
        "humidity": latest.get("relative_humidity_2m"),
        "dewpoint": latest.get("dew_point_2m"),
        "precipitation": latest.get("precipitation"),
        "pressure": latest.get("pressure_msl"),
        "uv_index": latest.get("uv_index"),
        "windspeed": latest.get("wind_speed_10m"),
        "winddirection": latest.get("wind_direction_10m"),
        "pm2_5": latest.get("pm2_5"),
        "pm10": latest.get("pm10"),
        "co": latest.get("carbon_monoxide"),
        "no2": latest.get("nitrogen_dioxide"),
        "so2": latest.get("sulphur_dioxide"),
        "o3": latest.get("ozone"),
        "european_aqi": latest.get("european_aqi"),
        "alder_pollen": latest.get("alder_pollen"),
        "birch_pollen": latest.get("birch_pollen"),
        "grass_pollen": latest.get("grass_pollen"),
        "mugwort_pollen": latest.get("mugwort_pollen"),
        "olive_pollen": latest.get("olive_pollen"),
        "ragweed_pollen": latest.get("ragweed_pollen"),
        "city": city,
    }
    df = pd.DataFrame([row])
    for col in df.columns:
        if col not in ["timestamp", "city"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(0)
    df = add_derived_features(df)
    return df

# ------------------- INPUT MODE -------------------
mode = st.radio("Choose Input Mode:", ["Manual Entry", "Open-Meteo Live", "Upload CSV"])
show_shap = st.checkbox("Show SHAP explainability for base allergy prediction", value=False)

def preferred_model_for_mode(current_mode: str) -> str:
    if current_mode == "Open-Meteo Live":
        return "regressor" if reg_model is not None else "classifier"
    return "classifier"

if mode == "Manual Entry":
    col1, col2 = st.columns(2)
    with col1:
        temp = st.number_input("🌡 Temperature (°C)", 0, 50, 25)
        humidity = st.number_input("💧 Humidity (%)", 0, 100, 50)
        pm25 = st.number_input("PM2.5 (µg/m³)", 0, 500, 20)
        pm10 = st.number_input("PM10 (µg/m³)", 0, 500, 30)
        o3 = st.number_input("Ozone (µg/m³)", 0, 500, 40)
    with col2:
        grass_pollen = st.number_input("🌱 Grass Pollen Count", 0, 300, 15)
        birch_pollen = st.number_input("🌳 Birch Pollen Count", 0, 300, 5)
        ragweed_pollen = st.number_input("🍂 Ragweed Pollen Count", 0, 300, 2)
        uv_index = st.number_input("☀ UV Index", 0, 12, 4)
        pressure = st.number_input("Pressure (hPa)", 900, 1100, 1012)

    if st.button("🔮 Predict Allergy Risk"):
        row = {
            "temperature": temp,
            "humidity": humidity,
            "pm2_5": pm25,
            "pm10": pm10,
            "o3": o3,
            "grass_pollen": grass_pollen,
            "birch_pollen": birch_pollen,
            "ragweed_pollen": ragweed_pollen,
            "uv_index": uv_index,
            "pressure": pressure,
        }
        df = pd.DataFrame([row])
        df = add_derived_features(df)

        X, _ = align_to_scaler_schema(df, scaler)
        Xs = scaler.transform(X)
        Xs_df = pd.DataFrame(Xs, columns=X.columns)

        prefer = preferred_model_for_mode(mode)
        prob = predict_risk_probs(Xs_df, prefer=prefer)[0]
        st.session_state["last_risk_prob"] = float(prob)
        st.session_state["last_risk_band"] = risk_band(prob)
        st.session_state["last_risk_row"] = row
        st.session_state["latest_mode"] = "Manual Entry"
        st.subheader(f"Predicted Risk: **{risk_band(prob)}** 🌿")
        st.metric("Allergy Risk Probability", f"{prob:.2%}")
        st.caption(health_note(row))
        if show_shap:
            explain_shap_for_sample(model, Xs_df)
        render_multi_outputs(df)
        render_disease_live_outputs(df)

elif mode == "Open-Meteo Live":
    st.markdown("Use live weather, air quality, and pollen signals from Open-Meteo.")
    default_coords = {
        # Common dataset cities (fallback if dataset doesn't provide coords)
        "Berlin": (52.5200, 13.4050),
        "London": (51.5072, -0.1276),
        "Paris": (48.8566, 2.3522),
        "Rome": (41.9028, 12.4964),
        "Madrid": (40.4168, -3.7038),
        # Extra polluted/benchmark cities
        "Delhi, India (polluted)": (28.6139, 77.2090),
        "Lahore, Pakistan (polluted)": (31.5204, 74.3587),
        "Beijing, China (polluted)": (39.9042, 116.4074),
        "Mexico City, Mexico": (19.4326, -99.1332),
        "Los Angeles, USA": (34.0522, -118.2437),
    }

    dataset_cities, dataset_coords = load_dataset_city_coords()
    presets = OrderedDict()

    # Priority: cities from dataset first
    for city in dataset_cities:
        if city in dataset_coords:
            presets[city] = dataset_coords[city]
        elif city in default_coords:
            presets[city] = default_coords[city]

    # Then add remaining defaults
    for city, coords in default_coords.items():
        if city not in presets:
            presets[city] = coords

    if not presets:
        presets = default_coords

    choice = st.selectbox("Preset location", list(presets.keys()))
    lat, lon = presets[choice]
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", value=float(lat), format="%.4f")
    with col2:
        lon = st.number_input("Longitude", value=float(lon), format="%.4f")
    city = st.text_input("City label", value=choice.split(" (")[0])
    forecast_days = st.slider("Forecast horizon (days)", min_value=1, max_value=7, value=3, step=1)

    compare_internal = st.checkbox("Compare with internal risk-only forecast", value=False)
    if compare_internal:
        if city not in load_dataset_cities():
            st.info("Internal forecast comparison is only available for dataset cities.")
        elif load_risk_forecaster(city) is None or load_risk_forecaster_meta(city) is None:
            st.info("Internal forecaster not found for this city. Run risk_forecast_eval.py to enable comparison.")

    if st.button("🌐 Fetch Live Data & Predict"):
        try:
            df = fetch_open_meteo(lat, lon, city)
            st.write("📍 Latest record:")
            st.json(df.iloc[0].to_dict())
            X, _ = align_to_scaler_schema(df, scaler)
            Xs = scaler.transform(X)
            Xs_df = pd.DataFrame(Xs, columns=X.columns)
            prefer = preferred_model_for_mode(mode)
            prob = predict_risk_probs(Xs_df, prefer=prefer)[0]
            st.session_state["last_risk_prob"] = float(prob)
            st.session_state["last_risk_band"] = risk_band(prob)
            st.session_state["last_risk_row"] = df.iloc[0].to_dict()
            st.session_state["latest_mode"] = "Open-Meteo Live"
            st.session_state["latest_city"] = city
            st.subheader(f"Predicted Risk: **{risk_band(prob)}** 🌿")
            st.metric("Allergy Risk Probability", f"{prob:.2%}")
            st.caption(health_note(df.iloc[0].to_dict()))
            if show_shap:
                explain_shap_for_sample(model, Xs_df)
            render_multi_outputs(df)
            render_disease_live_outputs(df)
        except Exception as e:
            st.error(f"Live fetch failed: {type(e).__name__}: {e}")
            st.stop()

        st.divider()
        st.subheader("Forecast (Hourly)")
        try:
            fc = fetch_open_meteo_forecast(lat, lon, city, forecast_days=forecast_days)
            fc["timestamp"] = pd.to_datetime(fc["timestamp"], errors="coerce")
            fc = fc.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            Xf, _ = align_to_scaler_schema(fc, scaler)
            Xfs = scaler.transform(Xf)
            Xfs_df = pd.DataFrame(Xfs, columns=Xf.columns)
            prefer = preferred_model_for_mode(mode)
            fc["allergy_risk_probability"] = predict_risk_probs(Xfs_df, prefer=prefer)
            fc["risk_band"] = fc["allergy_risk_probability"].map(risk_band)
            fc["respiratory_pollution_flag"] = (fc["pm2_5"] >= 35) | (fc["european_aqi"] >= 100)

            chart_df = fc[["timestamp", "allergy_risk_probability"]].set_index("timestamp")

            # Optional comparison with internal risk-only forecast
            if compare_internal and city in load_dataset_cities():
                forecaster = load_risk_forecaster(city)
                meta = load_risk_forecaster_meta(city)
                if forecaster is not None and meta is not None:
                    base_type = meta.get("base_model_type", "classifier")
                    # Build last-24 env-risk from the same city history in allergy_cleaned.csv
                    base = Path(__file__).resolve().parent
                    hist_df = pd.read_csv(base / "allergy_cleaned.csv")
                    hist_df["datetime"] = pd.to_datetime(hist_df["datetime"], errors="coerce")
                    hist_df = hist_df.dropna(subset=["datetime"])
                    hist_df = hist_df[hist_df["city"] == city].sort_values("datetime").reset_index(drop=True)
                    if len(hist_df) >= 48:
                        Xh = hist_df.copy()
                        Xh_aligned, _ = align_to_scaler_schema(Xh, scaler)
                        Xh_s = scaler.transform(Xh_aligned)
                        Xh_df = pd.DataFrame(Xh_s, columns=Xh_aligned.columns)
                        hist_df["env_risk_prob"] = predict_risk_probs(Xh_df, prefer=base_type)
                        last_24 = hist_df["env_risk_prob"].astype(float).to_list()[-24:]

                        feature_cols = meta.get("feature_columns", [])
                        city_cols = meta.get("city_dummy_columns", [])
                        horizon_hours = len(fc)
                        hist = last_24[:]
                        internal_rows = []
                        for h in range(1, horizon_hours + 1):
                            ts = pd.to_datetime(fc["timestamp"].iloc[h - 1])
                            feats = make_risk_forecast_features(hist, ts)
                            row = {**feats}
                            for c in city_cols:
                                row[c] = 0
                            cc = f"city_{city}"
                            if cc in row:
                                row[cc] = 1
                            X_row = pd.DataFrame([row])
                            for c in feature_cols:
                                if c not in X_row.columns:
                                    X_row[c] = 0
                            X_row = X_row[feature_cols]
                            yhat = float(forecaster.predict(X_row)[0])
                            yhat = max(0.0, min(1.0, yhat))
                            hist.append(yhat)
                            internal_rows.append({"timestamp": ts, "internal_risk_prob": yhat})
                        internal_fc = pd.DataFrame(internal_rows).set_index("timestamp")
                        chart_df = chart_df.join(internal_fc, how="left")
                    else:
                        st.info("Not enough dataset history to compute internal forecast for comparison.")
                else:
                    st.info("Internal forecaster not available.")

            st.line_chart(chart_df)
            if compare_internal and "internal_risk_prob" in chart_df.columns:
                valid = chart_df.dropna()
                if not valid.empty:
                    diff = (valid["allergy_risk_probability"] - valid["internal_risk_prob"]).abs().mean()
                    st.caption(f"Mean absolute difference (env-driven vs internal): {diff:.4f}")
                    st.session_state["latest_env_internal_mad"] = float(diff)

            daily = fc.copy()
            daily["date"] = pd.to_datetime(daily["timestamp"]).dt.date
            daily_summary = (
                daily.groupby("date")
                .agg(
                    avg_allergy_risk=("allergy_risk_probability", "mean"),
                    max_allergy_risk=("allergy_risk_probability", "max"),
                    max_pm2_5=("pm2_5", "max"),
                    max_aqi=("european_aqi", "max"),
                    hours_pollution_flag=("respiratory_pollution_flag", "sum"),
                )
                .reset_index()
            )
            st.caption("Daily summary (derived from hourly forecasts).")
            st.dataframe(daily_summary, use_container_width=True)
            forecast_cols = ["timestamp", "allergy_risk_probability"]
            if "internal_risk_prob" in chart_df.columns:
                forecast_cols.append("internal_risk_prob")
            fc_ctx = chart_df.reset_index()
            st.session_state["latest_forecast_hourly"] = fc_ctx[forecast_cols].tail(24 * int(forecast_days)).to_dict("records")
            st.session_state["latest_forecast_daily_summary"] = daily_summary.to_dict("records")
        except Exception as e:
            st.error(f"Forecast failed: {type(e).__name__}: {e}")

elif mode == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV with feature columns", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("📂 Uploaded Data Preview:", df.head())
        df = add_derived_features(df)
        X, _ = align_to_scaler_schema(df, scaler)
        Xs = scaler.transform(X)
        Xs_df = pd.DataFrame(Xs, columns=X.columns)
        prefer = preferred_model_for_mode(mode)
        probs = predict_risk_probs(Xs_df, prefer=prefer)
        df["allergy_risk_probability"] = probs
        df["risk_band"] = [risk_band(p) for p in probs]
        if len(df) > 0:
            st.session_state["last_risk_prob"] = float(df["allergy_risk_probability"].iloc[0])
            st.session_state["last_risk_band"] = risk_band(df["allergy_risk_probability"].iloc[0])
            st.session_state["last_risk_row"] = df.iloc[0].to_dict()
            st.session_state["latest_mode"] = "Upload CSV"
        st.write("✅ Predictions:", df[["allergy_risk_probability", "risk_band"]])
        st.download_button("⬇ Download Predictions", df.to_csv(index=False), "predictions.csv")
        if show_shap and len(df) > 0:
            explain_shap_for_sample(model, Xs_df.iloc[:1])
        render_multi_outputs(df)
        render_disease_live_outputs(df)

# ------------------- PRECAUTIONS CHATBOT -------------------
def build_assistant_context() -> str:
    parts: list[str] = []
    mode = st.session_state.get("latest_mode")
    city = st.session_state.get("latest_city")
    if mode:
        parts.append(f"Current mode: {mode}")
    if city:
        parts.append(f"City: {city}")

    prob = st.session_state.get("last_risk_prob")
    band = st.session_state.get("last_risk_band")
    if prob is not None and band is not None:
        parts.append(f"Latest allergy risk: {band} ({float(prob):.2%})")

    row = st.session_state.get("last_risk_row")
    if isinstance(row, dict) and row:
        keys = [
            "temperature", "humidity", "pm2_5", "pm10", "european_aqi",
            "grass_pollen", "birch_pollen", "ragweed_pollen", "pollution_score", "weather_risk"
        ]
        vals = [f"{k}={row[k]}" for k in keys if k in row]
        if vals:
            parts.append("Latest input signals: " + ", ".join(vals))

    daily = st.session_state.get("latest_forecast_daily_summary")
    if isinstance(daily, list) and daily:
        parts.append("Forecast daily summary:")
        for d in daily[:7]:
            date = d.get("date")
            avg_risk = d.get("avg_allergy_risk")
            max_pm = d.get("max_pm2_5")
            max_aqi = d.get("max_aqi")
            parts.append(f"- {date}: avg_risk={avg_risk}, max_pm2_5={max_pm}, max_aqi={max_aqi}")

    mad = st.session_state.get("latest_env_internal_mad")
    if mad is not None:
        parts.append(f"Forecast comparison MAD (env vs internal): {float(mad):.4f}")

    multi = st.session_state.get("latest_multi_condition_outputs")
    if isinstance(multi, list) and multi:
        parts.append("Latest multi-condition outputs:")
        for k, v in multi[0].items():
            parts.append(f"- {k}: {v}")

    disease_live = st.session_state.get("latest_disease_live_scores")
    if isinstance(disease_live, list) and disease_live:
        parts.append("Latest disease-specific live signals:")
        for r in disease_live:
            parts.append(f"- {r.get('disease')}: score={r.get('live_signal')}, band={r.get('band')}")

    return "\n".join(parts).strip() or "No latest UI outputs available."


def rule_based_precautions_reply(prompt: str) -> str | None:
    text = (prompt or "").lower().strip()
    if text in {"hi", "hello", "hey", "hii", "good morning", "good evening"}:
        return "Hello. Ask about precautions, forecast interpretation, or model outputs."
    if "how are you" in text:
        return "Ready. Ask your question on allergy risk, forecast, or precautions."
    if "precaution" in text or "what should i do" in text or "what can i do" in text:
        row = st.session_state.get("last_risk_row", {}) or {}
        pm25 = _num(row.get("pm2_5", 0))
        aqi = _num(row.get("european_aqi", 0))
        grass = _num(row.get("grass_pollen", 0))
        birch = _num(row.get("birch_pollen", 0))
        ragweed = _num(row.get("ragweed_pollen", 0))
        tips: list[str] = []
        if pm25 >= 35 or aqi >= 100:
            tips.extend([
                "Limit outdoor exposure and heavy outdoor exercise.",
                "Use a well-fitted mask (N95/KN95) outdoors.",
                "Keep windows closed and run indoor filtration if available.",
            ])
        if grass >= 50 or birch >= 30 or ragweed >= 20:
            tips.extend([
                "Avoid peak pollen hours when possible.",
                "Shower and change clothes after outdoor exposure.",
                "Dry clothes indoors during high pollen hours.",
            ])
        if not tips:
            tips.append("Current signals are relatively mild; maintain routine precautions and hydration.")
        return "Precautions:\n" + "\n".join(f"- {t}" for t in tips)
    if "mean absolute difference" in text or "mad" in text:
        mad = st.session_state.get("latest_env_internal_mad")
        if mad is None:
            return "No forecast comparison value is available yet. Run Open-Meteo Live with comparison enabled."
        return f"Current env-vs-internal mean absolute difference is {float(mad):.4f}. Lower is better."
    return None


def stream_ollama_reply(messages: list[dict], model_name: str) -> str:
    try:
        import ollama  # type: ignore
        stream = ollama.chat(model=model_name, messages=messages, stream=True)
        full = ""
        container = st.chat_message("assistant")
        placeholder = container.empty()
        for chunk in stream:
            token = chunk.get("message", {}).get("content", "")
            full += token
            placeholder.markdown(full)
        return full.strip()
    except Exception:
        payload = {"model": model_name, "messages": messages, "stream": False}
        resp = requests.post(OLLAMA_URL, json=payload, timeout=45)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()


st.divider()
st.subheader("Precautions Assistant (AI Chatbot)")
st.caption("Ask precautions, forecast interpretation, or model-output questions. The assistant uses the latest UI outputs as context.")

with st.expander("Ollama Settings (optional)"):
    st.session_state["use_ollama"] = st.checkbox("Use local Ollama", value=True)
    st.session_state["ollama_model"] = st.text_input("Ollama model name", value="llama3")

if "assistant_messages" not in st.session_state:
    st.session_state["assistant_messages"] = [
        {"role": "assistant", "content": "Hello. Ask about precautions, forecast, or your latest risk output."}
    ]

if st.button("Clear chat"):
    st.session_state["assistant_messages"] = [
        {"role": "assistant", "content": "Hello. Ask about precautions, forecast, or your latest risk output."}
    ]

for msg in st.session_state["assistant_messages"]:
    st.chat_message(msg["role"]).markdown(msg["content"])

if prompt := st.chat_input("Ask about precautions, forecast, or model outputs..."):
    st.session_state["assistant_messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    reply = rule_based_precautions_reply(prompt)
    if reply is None:
        context = build_assistant_context()
        system_prompt = (
            "You are an allergy-risk precautions assistant. "
            "Use only the provided context from the app outputs. "
            "Be concise, practical, and avoid diagnosis. "
            "If data is insufficient, say so clearly."
        )
        messages = [{"role": "system", "content": system_prompt}]
        for m in st.session_state["assistant_messages"][-8:]:
            if m.get("role") in {"user", "assistant"}:
                messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{prompt}"})

        if st.session_state.get("use_ollama"):
            try:
                reply = stream_ollama_reply(messages, st.session_state.get("ollama_model", "llama3"))
            except Exception as e:
                reply = f"Ollama error: {type(e).__name__}: {e}"
                st.chat_message("assistant").markdown(reply)
        else:
            reply = (
                "Rule-based answer was not matched and Ollama is disabled.\n"
                "Enable Ollama or ask a precautions/forecast-specific question."
            )
            st.chat_message("assistant").markdown(reply)
    else:
        st.chat_message("assistant").markdown(reply)

    st.session_state["assistant_messages"].append({"role": "assistant", "content": reply})

# ------------------- FOOTER -------------------
st.markdown(
    "<p style='text-align:center; color:#999; font-size:0.8em;'>⚕ This is a predictive model. For medical concerns, consult a doctor.</p>",
    unsafe_allow_html=True,
)
