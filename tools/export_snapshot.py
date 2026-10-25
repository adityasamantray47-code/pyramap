"""
Builds all_states.json: one static snapshot of the whole app's data, for a
demo page that can't call the live FastAPI backend.

Run from backend/ so its imports resolve:
  cd backend && ../venv/Scripts/python.exe ../tools/export_snapshot.py
(or: backend/venv/Scripts/python.exe tools/export_snapshot.py, run from repo root)
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

BACKEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
sys.path.insert(0, BACKEND_DIR)

import data_ingestion  # noqa: E402
import ml_pipeline  # noqa: E402
from grid import ALL_STATES, forecast_areas_for_state  # noqa: E402

HISTORY_DAYS = 120
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "all_states.json")


def build_state_entry(state: str, state_meta: dict) -> dict | None:
    df, source = data_ingestion.load_state_data(state)
    features = ml_pipeline.build_features(df)
    try:
        result = ml_pipeline.train_model(features)
    except ValueError as exc:
        print(f"  {state}: skipping, could not train model ({exc})")
        return None

    model = result["model"]
    features = result["features"]
    latest = ml_pipeline.predict_latest(model, features)
    areas = {a.area_id: a for a in forecast_areas_for_state(state)}

    outlook = []
    for _, row in latest.iterrows():
        area = areas.get(row["area_id"])
        outlook.append(
            {
                "id": row["area_id"],
                "lat": area.lat if area else None,
                "lon": area.lon if area else None,
                "danger_prob": round(float(row["danger_prob"]), 3),
                "danger_class": row["danger_class"],
                "root_soil_wetness": round(float(row["gwetroot_latest"]), 3),
            }
        )

    history = {}
    cutoff = df["week_start"].max() - dt.timedelta(days=HISTORY_DAYS)
    for area_id in areas:
        area_raw = df[(df["area_id"] == area_id) & (df["week_start"] >= cutoff)].sort_values("week_start")
        area_feat = features[(features["area_id"] == area_id) & (features["week_start"] >= cutoff)].sort_values("week_start")
        probs = (
            model.predict_proba(area_feat[ml_pipeline.FEATURE_COLUMNS])[:, 1].round(3).tolist()
            if not area_feat.empty
            else []
        )
        history[area_id] = {
            "dc": probs,
            "t30": area_raw["cpc_temp_prob"].tolist(),
            "p30": area_raw["cpc_precip_prob"].tolist(),
            "fire": area_raw["fire_detections"].tolist(),
        }

    return {
        "meta": state_meta,
        "outlook": outlook,
        "metrics": result["metrics"],
        "importance": result["feature_importance"][:5],
        "history": history,
    }


def main():
    status = data_ingestion.data_status(ALL_STATES)
    snapshot = {"as_of": dt.date.today().isoformat(), "data_status": status, "states": {}}

    for i, state in enumerate(ALL_STATES, start=1):
        print(f"[{i}/{len(ALL_STATES)}] {state}")
        areas = forecast_areas_for_state(state)
        state_meta = {
            "code": state,
            "area_count": len(areas),
            "has_real_data": data_ingestion.has_real_data(state),
        }
        entry = build_state_entry(state, state_meta)
        if entry:
            snapshot["states"][state] = entry

    with open(OUTPUT_PATH, "w") as f:
        json.dump(snapshot, f)
    print(f"\nWrote {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
