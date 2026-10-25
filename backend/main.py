"""FastAPI app for the Long-Lead Wildfire Outlook."""
from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import data_ingestion
import ml_pipeline
from grid import ALL_STATES, STATE_BBOX, STATE_NAMES, forecast_areas_for_state

app = FastAPI(title="Long-Lead Wildfire Outlook API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache: state -> {"model": ..., "metrics": ..., "feature_importance": ...,
# "features": DataFrame, "source": "real"/"synthetic"}
_STATE_CACHE: dict[str, dict] = {}

FEATURE_LABELS = {
    "t2m_max_4wk_avg": "recent max temperatures",
    "precip_4wk_sum": "recent rainfall",
    "gwetroot_latest": "root-zone soil wetness",
    "gwetroot_trend": "soil wetness trend",
    "cpc_temp_prob": "NOAA's temperature outlook",
    "cpc_temp_cat_num": "NOAA's temperature outlook category",
    "cpc_precip_prob": "NOAA's precipitation outlook",
    "cpc_precip_cat_num": "NOAA's precipitation outlook category",
    "week_of_year_sin": "time of year",
    "week_of_year_cos": "time of year",
    "area_climatology": "this area's fire history for this time of year",
}

# For area-specific reasoning: (plain label, unit, value*scale=display units,
# True if a LOWER value raises fire risk).
REASON_FEATURES = {
    "gwetroot_latest": ("root-zone soil moisture", "%", 100, True),
    "t2m_max_4wk_avg": ("recent max temperatures", "°C", 1, False),
    "precip_4wk_sum": ("rainfall over the last month", "mm", 1, True),
}


def _region_name(area, state: str) -> str:
    """Plain-language location within the state, e.g. "Northeast Texas",
    since area ids like "TX-5" mean nothing to a general reader."""
    lat_min, lat_max, lon_min, lon_max = STATE_BBOX[state]
    lat_mid, lon_mid = (lat_min + lat_max) / 2, (lon_min + lon_max) / 2
    lat_band, lon_band = (lat_max - lat_min) / 6, (lon_max - lon_min) / 6
    ns = "North" if area.lat > lat_mid + lat_band else "South" if area.lat < lat_mid - lat_band else ""
    ew = "East" if area.lon > lon_mid + lon_band else "West" if area.lon < lon_mid - lon_band else ""
    direction = f"{ns}{ew.lower()}" if ns and ew else (ns or ew or "Central")
    return f"{direction} {STATE_NAMES[state]}"


def _area_reasons(area_history: pd.DataFrame, latest_row, climatology: float) -> list[str]:
    """Compares this area's current conditions to its OWN historical typical
    values (not a global/other-area comparison), so the explanation is
    honestly specific to this area rather than a generic restatement of
    which features matter state-wide."""
    reasons = []
    for feature, (label, unit, scale, lower_is_riskier) in REASON_FEATURES.items():
        series = area_history[feature].dropna()
        if series.empty:
            continue
        typical = float(series.mean())
        current = float(latest_row[feature])
        if abs(typical) < 1e-6 or abs(current - typical) / (abs(typical) + 1e-6) < 0.12:
            continue  # not meaningfully different from this area's norm
        lower = current < typical
        effect = "raising" if lower == lower_is_riskier else "lowering"
        reasons.append(
            f"{label.capitalize()}, {'lower' if lower else 'higher'} than usual here: "
            f"{current * scale:.0f}{unit} vs. a typical {typical * scale:.0f}{unit} — {effect} fire risk."
        )
    if climatology > 0.15:
        reasons.append(
            f"This time of year is historically one of the more active stretches for fire in this area "
            f"(elevated conditions in roughly {round(climatology * 100)}% of years)."
        )
    return reasons[:3]


def _get_state_bundle(state: str) -> dict:
    state = state.upper()
    if state not in ALL_STATES:
        raise HTTPException(status_code=404, detail=f"Unknown state: {state}")
    if state in _STATE_CACHE:
        return _STATE_CACHE[state]

    df, source = data_ingestion.load_state_data(state)
    features = ml_pipeline.build_features(df)
    result = ml_pipeline.train_model(features)

    bundle = {
        "model": result["model"],
        "metrics": result["metrics"],
        "feature_importance": result["feature_importance"],
        "features": result["features"],
        "raw": df,
        "source": source,
    }
    _STATE_CACHE[state] = bundle
    return bundle


@app.get("/api/states")
def list_states():
    out = []
    for code in ALL_STATES:
        areas = forecast_areas_for_state(code)
        centroid_lat = sum(a.lat for a in areas) / len(areas)
        centroid_lon = sum(a.lon for a in areas) / len(areas)
        out.append(
            {
                "code": code,
                "name": STATE_NAMES[code],
                "area_count": len(areas),
                "lat": round(centroid_lat, 3),
                "lon": round(centroid_lon, 3),
                "has_real_data": data_ingestion.has_real_data(code),
            }
        )
    return out


@app.get("/api/data-status")
def data_status():
    return data_ingestion.data_status(ALL_STATES)


@app.get("/api/30-day-outlook")
def thirty_day_outlook(state: str):
    state = state.upper()
    bundle = _get_state_bundle(state)
    latest = ml_pipeline.predict_latest(bundle["model"], bundle["features"])
    areas = {a.area_id: a for a in forecast_areas_for_state(state)}
    features = bundle["features"]

    all_driver_labels = []
    for f in bundle["feature_importance"]:
        label = FEATURE_LABELS.get(f["feature"], f["feature"])
        if label not in all_driver_labels:  # week_of_year_sin/cos both mean "time of year"
            all_driver_labels.append(label)
    drivers = all_driver_labels[:3]

    out = []
    for _, row in latest.iterrows():
        area = areas.get(row["area_id"])
        area_history = features[features["area_id"] == row["area_id"]]
        out.append(
            {
                "id": row["area_id"],
                "region": _region_name(area, state) if area else row["area_id"],
                "lat": area.lat if area else None,
                "lon": area.lon if area else None,
                "bbox": [area.lat_min, area.lon_min, area.lat_max, area.lon_max] if area else None,
                "as_of": row["week_start"].date().isoformat(),
                "danger_prob": round(float(row["danger_prob"]), 3),
                "danger_class": row["danger_class"],
                "root_soil_wetness": round(float(row["gwetroot_latest"]), 3),
                "drivers": drivers,
                "reasons": _area_reasons(area_history, row, float(row["area_climatology"])),
            }
        )
    return out


@app.get("/api/model-metrics")
def model_metrics(state: str):
    bundle = _get_state_bundle(state)
    return {"metrics": bundle["metrics"], "feature_importance": bundle["feature_importance"], "source": bundle["source"]}


@app.get("/api/area-history")
def area_history(area_id: str, days: int = 120):
    state = area_id.split("-")[0]
    bundle = _get_state_bundle(state)
    features = bundle["features"]
    area_features = features[features["area_id"] == area_id].sort_values("week_start")
    raw = bundle["raw"]
    area_raw = raw[raw["area_id"] == area_id].sort_values("week_start")

    cutoff = area_raw["week_start"].max() - pd.Timedelta(days=days)
    area_raw = area_raw[area_raw["week_start"] >= cutoff]
    area_features = area_features[area_features["week_start"] >= cutoff]

    probs = bundle["model"].predict_proba(area_features[ml_pipeline.FEATURE_COLUMNS])[:, 1] if not area_features.empty else []

    return {
        "area_id": area_id,
        "dates": [d.date().isoformat() for d in area_raw["week_start"]],
        "t30": area_raw["cpc_temp_prob"].tolist(),
        "p30": area_raw["cpc_precip_prob"].tolist(),
        "temp_cat": area_raw["cpc_temp_cat"].tolist(),
        "precip_cat": area_raw["cpc_precip_cat"].tolist(),
        "fire": area_raw["fire_detections"].tolist(),
        "dc": [round(float(p), 3) for p in probs],
    }
