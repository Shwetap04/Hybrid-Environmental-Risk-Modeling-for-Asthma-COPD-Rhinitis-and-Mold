"""
Fetch historical hourly weather + air quality data from Open-Meteo for selected cities.

Output columns are aligned to the current project schema where possible:
datetime, city, temperature, humidity, dewpoint, precipitation, windspeed,
winddirection, pressure, uv_index, pm2_5, pm10, co, no2, so2, o3,
european_aqi, alder_pollen, birch_pollen, grass_pollen, mugwort_pollen,
olive_pollen, ragweed_pollen, month, day_of_week, hour, pollen_score,
pollution_score, weather_risk

Notes:
- Open-Meteo free tier usually does not require an API key.
- Some historical domains may not provide pollen and/or uv_index. Missing fields
  are left as NaN (or 0 if --fill-missing-zero is used).
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_HOURLY = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "pressure_msl",
    # Not guaranteed for all historical models; included when available.
    "uv_index",
]

AIR_HOURLY_CORE = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "european_aqi",
]

AIR_HOURLY_POLLEN = [
    "alder_pollen",
    "birch_pollen",
    "grass_pollen",
    "mugwort_pollen",
    "olive_pollen",
    "ragweed_pollen",
]

POLLEN_COLS = [
    "alder_pollen",
    "birch_pollen",
    "grass_pollen",
    "mugwort_pollen",
    "olive_pollen",
    "ragweed_pollen",
]

OUTPUT_COLS = [
    "datetime",
    "city",
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
    "month",
    "day_of_week",
    "hour",
    "pollen_score",
    "pollution_score",
    "weather_risk",
]


@dataclass(frozen=True)
class City:
    name: str
    lat: float
    lon: float
    cohort: str


DEFAULT_CITIES = [
    City("Berlin", 52.5200, 13.4050, "europe"),
    City("London", 51.5072, -0.1276, "europe"),
    City("Paris", 48.8566, 2.3522, "europe"),
    City("Rome", 41.9028, 12.4964, "europe"),
    City("Madrid", 40.4168, -3.7038, "europe"),
    City("Milan", 45.4642, 9.1900, "europe"),
    City("Warsaw", 52.2297, 21.0122, "europe"),
    City("Vienna", 48.2082, 16.3738, "europe"),
    City("Budapest", 47.4979, 19.0402, "europe"),
    City("Athens", 37.9838, 23.7275, "europe"),
    City("Delhi", 28.6139, 77.2090, "respiratory"),
    City("Lahore", 31.5204, 74.3587, "respiratory"),
    City("Beijing", 39.9042, 116.4074, "respiratory"),
    City("Mexico City", 19.4326, -99.1332, "respiratory"),
    City("Los Angeles", 34.0522, -118.2437, "respiratory"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch Open-Meteo historical data for a date range")
    p.add_argument("--start-date", default="2025-01-01", help="YYYY-MM-DD")
    p.add_argument("--end-date", default="2025-12-31", help="YYYY-MM-DD")
    p.add_argument(
        "--cohort",
        choices=["all", "europe", "respiratory"],
        default="all",
        help="City cohort to fetch",
    )
    p.add_argument("--timezone", default="UTC", help="Timezone for API response")
    p.add_argument("--api-key", default=os.getenv("OPENMETEO_API_KEY", ""), help="Optional Open-Meteo API key")
    p.add_argument("--cities-file", default="", help="Optional CSV with columns: city,lat,lon,cohort")
    p.add_argument("--out", default="openmeteo_historical_merged.csv", help="Output CSV path")
    p.add_argument("--sleep-seconds", type=float, default=0.2, help="Pause between city requests")
    p.add_argument("--fill-missing-zero", action="store_true", help="Fill missing numeric fields with 0")
    return p.parse_args()


def validate_date(d: str) -> str:
    datetime.strptime(d, "%Y-%m-%d")
    return d


def load_cities(cities_file: str, cohort: str) -> list[City]:
    if cities_file:
        df = pd.read_csv(cities_file)
        needed = {"city", "lat", "lon"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"cities-file missing columns: {sorted(missing)}")
        if "cohort" not in df.columns:
            df["cohort"] = "custom"
        cities = [
            City(str(r["city"]), float(r["lat"]), float(r["lon"]), str(r["cohort"]))
            for _, r in df.iterrows()
        ]
    else:
        cities = DEFAULT_CITIES

    if cohort == "all":
        return cities
    return [c for c in cities if c.cohort == cohort]


def request_json(url: str, params: dict) -> dict:
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        reason = data.get("reason", data["error"])
        raise RuntimeError(f"API error: {reason}")
    return data


def fetch_weather(city: City, start_date: str, end_date: str, timezone: str, api_key: str) -> pd.DataFrame:
    params = {
        "latitude": city.lat,
        "longitude": city.lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(WEATHER_HOURLY),
        "timezone": timezone,
    }
    if api_key:
        params["apikey"] = api_key
    data = request_json(WEATHER_URL, params)
    if "hourly" not in data:
        raise RuntimeError("Weather API response missing 'hourly'")
    w = pd.DataFrame(data["hourly"])
    if "time" not in w.columns:
        raise RuntimeError("Weather API response missing 'time'")
    w = w.rename(columns={"time": "datetime"})
    w["datetime"] = pd.to_datetime(w["datetime"], errors="coerce")
    return w


def fetch_air(city: City, start_date: str, end_date: str, timezone: str, api_key: str) -> pd.DataFrame:
    params = {
        "latitude": city.lat,
        "longitude": city.lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(AIR_HOURLY_CORE + AIR_HOURLY_POLLEN),
        "timezone": timezone,
    }
    if api_key:
        params["apikey"] = api_key

    try:
        data = request_json(AIR_URL, params)
    except Exception:
        # Fallback when pollen is unavailable for historical region/model.
        params["hourly"] = ",".join(AIR_HOURLY_CORE)
        data = request_json(AIR_URL, params)

    if "hourly" not in data:
        raise RuntimeError("Air quality API response missing 'hourly'")
    a = pd.DataFrame(data["hourly"])
    if "time" not in a.columns:
        raise RuntimeError("Air quality API response missing 'time'")
    a = a.rename(columns={"time": "datetime"})
    a["datetime"] = pd.to_datetime(a["datetime"], errors="coerce")
    for c in AIR_HOURLY_POLLEN:
        if c not in a.columns:
            a[c] = pd.NA
    return a


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = out["datetime"].dt.month
    out["day_of_week"] = out["datetime"].dt.dayofweek
    out["hour"] = out["datetime"].dt.hour

    for c in POLLEN_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    out["pollen_score"] = out[POLLEN_COLS].sum(axis=1, min_count=1)

    pollutant_cols = ["pm2_5", "pm10", "co", "no2", "so2", "o3"]
    for c in pollutant_cols:
        if c not in out.columns:
            out[c] = pd.NA
    out["pollution_score"] = out[pollutant_cols].mean(axis=1)

    h = pd.to_numeric(out["humidity"], errors="coerce")
    t = pd.to_numeric(out["temperature"], errors="coerce")
    out["weather_risk"] = (100 - h.clip(0, 100)) * 0.3 + (t.sub(22).abs() * 0.2)
    return out


def map_schema(df: pd.DataFrame, city_name: str) -> pd.DataFrame:
    col_map = {
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "dew_point_2m": "dewpoint",
        "wind_speed_10m": "windspeed",
        "wind_direction_10m": "winddirection",
        "pressure_msl": "pressure",
        "carbon_monoxide": "co",
        "nitrogen_dioxide": "no2",
        "sulphur_dioxide": "so2",
        "ozone": "o3",
    }
    out = df.rename(columns=col_map).copy()
    out["city"] = city_name
    for c in OUTPUT_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[OUTPUT_COLS].copy()
    return out


def fetch_city(city: City, start_date: str, end_date: str, timezone: str, api_key: str) -> pd.DataFrame:
    w = fetch_weather(city, start_date, end_date, timezone, api_key)
    a = fetch_air(city, start_date, end_date, timezone, api_key)
    merged = pd.merge(w, a, on="datetime", how="inner")
    merged = map_schema(merged, city.name)
    merged = add_derived_features(merged)
    return merged


def coerce_numeric(df: pd.DataFrame, exclude: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    ex = set(exclude)
    for c in out.columns:
        if c not in ex:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def main() -> None:
    args = parse_args()
    start_date = validate_date(args.start_date)
    end_date = validate_date(args.end_date)
    if end_date < start_date:
        raise ValueError("end-date must be >= start-date")

    cities = load_cities(args.cities_file, args.cohort)
    if not cities:
        raise ValueError("No cities selected")

    frames: list[pd.DataFrame] = []
    print(f"[INFO] Fetching {len(cities)} cities from {start_date} to {end_date}")
    for i, city in enumerate(cities, start=1):
        print(f"[INFO] ({i}/{len(cities)}) {city.name} ({city.lat}, {city.lon})")
        try:
            df_city = fetch_city(city, start_date, end_date, args.timezone, args.api_key)
            frames.append(df_city)
            print(f"[OK] Rows: {len(df_city)}")
        except Exception as e:
            print(f"[WARN] Skipped {city.name}: {type(e).__name__}: {e}")
        time.sleep(max(0.0, args.sleep_seconds))

    if not frames:
        raise RuntimeError("No city data fetched successfully")

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["city", "datetime"]).reset_index(drop=True)
    out = coerce_numeric(out, exclude=["datetime", "city"])
    if args.fill_missing_zero:
        out = out.fillna(0)

    out_path = Path(args.out)
    out.to_csv(out_path, index=False)
    print(f"[OK] Saved merged dataset: {out_path.resolve()}")
    print(f"[INFO] Shape: {out.shape}")
    print(f"[INFO] Columns: {list(out.columns)}")


if __name__ == "__main__":
    main()
