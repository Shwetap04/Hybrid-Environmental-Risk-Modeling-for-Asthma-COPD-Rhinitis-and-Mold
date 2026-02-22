from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


OUT_DIR = Path(__file__).resolve().parent
ALIGNED_PATH = OUT_DIR / "aligned_env_outcomes_long.csv"
MODELS_DIR = OUT_DIR / "models_by_disease"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "pollution_score",
    "pollen_score",
    "weather_risk",
    "pm2_5",
    "european_aqi",
    "temperature",
    "humidity",
    "windspeed",
]

TARGET = "outcome_value_z"


def run_loocv(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    loo = LeaveOneOut()
    preds = np.full(len(y), np.nan, dtype=float)
    for train_idx, test_idx in loo.split(X):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        model.fit(X_train, y_train)
        preds[test_idx[0]] = float(model.predict(X_test)[0])
    return preds


def main() -> None:
    if not ALIGNED_PATH.exists():
        raise FileNotFoundError(f"Missing aligned dataset: {ALIGNED_PATH}")

    df = pd.read_csv(ALIGNED_PATH)
    for c in FEATURES + [TARGET]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

    rows: list[dict] = []
    pred_rows: list[pd.DataFrame] = []

    for disease, d in df.groupby("disease"):
        n = len(d)
        result = {"disease": disease, "n_rows": int(n)}

        if n < 3:
            result.update(
                {
                    "status": "insufficient_data",
                    "mae_loocv": np.nan,
                    "rmse_loocv": np.nan,
                }
            )
            rows.append(result)
            continue

        X = d[FEATURES].copy()
        y = d[TARGET].copy()
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("reg", LinearRegression()),
            ]
        )

        loocv_pred = run_loocv(model, X, y)
        mae = mean_absolute_error(y, loocv_pred)
        rmse = float(np.sqrt(mean_squared_error(y, loocv_pred)))

        model.fit(X, y)
        model_path = MODELS_DIR / f"{disease}_linear_model.pkl"
        joblib.dump(model, model_path)

        result.update(
            {
                "status": "trained",
                "mae_loocv": float(mae),
                "rmse_loocv": rmse,
                "model_path": str(model_path),
            }
        )
        rows.append(result)

        tmp = d[["city", "year", "disease", TARGET]].copy()
        tmp["pred_loocv"] = loocv_pred
        pred_rows.append(tmp)

    metrics = pd.DataFrame(rows).sort_values("disease").reset_index(drop=True)
    metrics_path = OUT_DIR / "outcome_model_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    if pred_rows:
        preds = pd.concat(pred_rows, ignore_index=True)
    else:
        preds = pd.DataFrame(columns=["city", "year", "disease", TARGET, "pred_loocv"])
    preds_path = OUT_DIR / "outcome_model_loocv_predictions.csv"
    preds.to_csv(preds_path, index=False)

    print(f"[DONE] {metrics_path} shape={metrics.shape}")
    print(f"[DONE] {preds_path} shape={preds.shape}")
    print("\nPer-disease training status:")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
