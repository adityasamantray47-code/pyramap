"""FastAPI app for the Long-Lead Wildfire Outlook."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import data_ingestion
import ml_pipeline
from grid import ALL_STATES, STATE_NAMES, forecast_areas_for_state

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


def _get_state_bundle(state: str) -> dict:
    state = state.upper()
    if state not in ALL_STATES:
        raise HTTPException(status_code=404, detail=f"Unknown state: {state}")
    if state in _STATE_CACHE:
        return _STATE_CACHE[state]

    df, source = data_ingestion.load_state_data(state)
    features = ml_pipeline.build_features(df)
    try:
        result = ml_pipeline.train_model(features)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"Could not train model for {state}: {exc}")

    bundle = {
        "model": result["model"],
        "metrics": result["metrics"],
        "feature_importance": result["feature_importance"],
        "features": features,
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
    bundle = _get_state_bundle(state)
    latest = ml_pipeline.predict_latest(bundle["model"], bundle["features"])
    areas = {a.area_id: a for a in forecast_areas_for_state(state.upper())}

    out = []
    for _, row in latest.iterrows():
        area = areas.get(row["area_id"])
        drivers = [FEATURE_LABELS.get(f["feature"], f["feature"]) for f in bundle["feature_importance"][:3]]
        out.append(
            {
                "id": row["area_id"],
                "lat": area.lat if area else None,
                "lon": area.lon if area else None,
                "as_of": row["week_start"].date().isoformat(),
                "danger_prob": round(float(row["danger_prob"]), 3),
                "danger_class": row["danger_class"],
                "root_soil_wetness": round(float(row["gwetroot_latest"]), 3),
                "drivers": drivers,
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

    cutoff = area_raw["week_start"].max() - pd_timedelta(days)
    area_raw = area_raw[area_raw["week_start"] >= cutoff]
    area_features = area_features[area_features["week_start"] >= cutoff]

    probs = bundle["model"].predict_proba(area_features[ml_pipeline.FEATURE_COLUMNS])[:, 1] if not area_features.empty else []

    return {
        "area_id": area_id,
        "dates": [d.date().isoformat() for d in area_raw["week_start"]],
        "t30": area_raw["cpc_temp_prob"].tolist(),
        "p30": area_raw["cpc_precip_prob"].tolist(),
        "fire": area_raw["fire_detections"].tolist(),
        "dc": [round(float(p), 3) for p in probs],
    }


def pd_timedelta(days: int):
    import pandas as pd

    return pd.Timedelta(days=days)
