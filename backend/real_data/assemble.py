"""
Joins NASA POWER + NASA FIRMS + NOAA CPC into the same weekly per-area schema
that synthetic_data.py produces, and writes it to
backend/data_cache/states/<STATE>.csv.gz.

Columns: area_id, week_start, t2m_max, precip, gwetroot,
         cpc_temp_prob, cpc_temp_cat, cpc_precip_prob, cpc_precip_cat,
         fire_detections
"""
from __future__ import annotations

import datetime as dt
import os

import pandas as pd

from grid import forecast_areas_for_state
from real_data import cpc, firms, power

DATA_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_cache", "states")

FIRMS_SOURCE = os.environ.get("FIRMS_SOURCE", "VIIRS_SNPP_NRT")
HISTORY_WEEKS = int(os.environ.get("HISTORY_WEEKS", "104"))  # 2 years of history


def _weekly(dates: pd.Series) -> pd.Series:
    return dates - pd.to_timedelta(dates.dt.weekday, unit="D")


def _weekly_power(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["week_start", "t2m_max", "precip", "gwetroot"])
    df = df.copy()
    df["week_start"] = _weekly(df["date"])
    agg = df.groupby("week_start").agg(t2m_max=("t2m_max", "mean"), precip=("precip", "sum"), gwetroot=("gwetroot", "mean"))
    agg = agg.reset_index()
    agg["week_start"] = agg["week_start"].dt.date.astype(str)
    return agg


def _cpc_monthly_outlook(lat: float, lon: float, issue_date: dt.date) -> dict:
    """Fetches the latest CPC monthly outlook zip (falls back a month if the
    current month's zip isn't published yet) and samples it at (lat, lon)."""
    for back in range(3):
        y, m = issue_date.year, issue_date.month - back
        while m <= 0:
            m += 12
            y -= 1
        try:
            temp_zip = cpc.download_zip(y, m, "seastemp")
            precip_zip = cpc.download_zip(y, m, "seasprcp")
        except Exception:
            continue
        temp_shapes = cpc.list_shapefiles(temp_zip)
        precip_shapes = cpc.list_shapefiles(precip_zip)
        temp_name = cpc.pick_shortest_lead(temp_shapes)
        precip_name = cpc.pick_shortest_lead(precip_shapes)
        if not temp_name or not precip_name:
            continue
        temp_recs = cpc.read_shape_records(temp_zip, temp_name)
        precip_recs = cpc.read_shape_records(precip_zip, precip_name)
        temp_rec = cpc.nearest_record(temp_recs, lat, lon)
        precip_rec = cpc.nearest_record(precip_recs, lat, lon)
        if temp_rec and precip_rec:
            return {
                "cpc_temp_prob": temp_rec["prob"],
                "cpc_temp_cat": temp_rec["cat"],
                "cpc_precip_prob": precip_rec["prob"],
                "cpc_precip_cat": precip_rec["cat"],
            }
    return {"cpc_temp_prob": None, "cpc_temp_cat": "normal", "cpc_precip_prob": None, "cpc_precip_cat": "normal"}


def assemble_area(area_id: str, lat: float, lon: float, map_key: str) -> pd.DataFrame:
    end = dt.date.today()
    start = end - dt.timedelta(weeks=HISTORY_WEEKS)

    power_df = power.fetch_power_daily(lat, lon, start, end)
    weekly = _weekly_power(power_df)

    pad = 0.05  # degrees, keeps the FIRMS query tight around the area's centroid
    bbox = (lon - pad, lat - pad, lon + pad, lat + pad)
    fire_df = firms.fetch_area_csv_range(map_key, FIRMS_SOURCE, bbox, start, end)
    fire_weekly = firms.weekly_detection_counts(fire_df)

    outlook = _cpc_monthly_outlook(lat, lon, end)

    out = weekly.merge(fire_weekly, on="week_start", how="left")
    out["fire_detections"] = out["fire_detections"].fillna(0).astype(int)
    out["area_id"] = area_id
    for k, v in outlook.items():
        out[k] = v
    cols = [
        "area_id", "week_start", "t2m_max", "precip", "gwetroot",
        "cpc_temp_prob", "cpc_temp_cat", "cpc_precip_prob", "cpc_precip_cat",
        "fire_detections",
    ]
    return out[cols]


def assemble_state(state: str, map_key: str, verbose: bool = True) -> pd.DataFrame:
    areas = forecast_areas_for_state(state)
    frames = []
    for area in areas:
        if verbose:
            print(f"  [{state}] fetching {area.area_id} ({area.lat:.2f}, {area.lon:.2f})")
        try:
            frames.append(assemble_area(area.area_id, area.lat, area.lon, map_key))
        except Exception as exc:
            print(f"  [{state}] FAILED {area.area_id}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_state(state: str, df: pd.DataFrame) -> str:
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    path = os.path.join(DATA_CACHE_DIR, f"{state.upper()}.csv.gz")
    df.to_csv(path, index=False, compression="gzip")
    return path
