"""
Joins NASA POWER + NASA FIRMS + NOAA CPC into the same weekly per-area schema
that synthetic_data.py produces, and writes it to
backend/data_cache/states/<STATE>.csv.gz.

Columns: area_id, week_start, t2m_max, precip, gwetroot,
         cpc_temp_prob, cpc_temp_cat, cpc_precip_prob, cpc_precip_cat,
         fire_detections

Request budget per state (confirmed live): 1 FIRMS query sequence for the
whole state (not per area - fire detections are fetched once for the state's
full bounding box and bucketed into areas locally, see firms.assign_to_areas),
one POWER point query per area, and CPC zips are downloaded at most once per
(year, month) for the whole 50-state run (cached in real_data/cpc.py). Calling
FIRMS once per area instead (what an earlier version of this did) was enough
to trip the API's rate limit over a full 50-state history pull.
"""
from __future__ import annotations

import datetime as dt
import os

import pandas as pd

from grid import STATE_BBOX, forecast_areas_for_state
from real_data import cpc, firms, power

DATA_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_cache", "states")

FIRMS_BASE_SOURCE = os.environ.get("FIRMS_SOURCE", "VIIRS_SNPP")  # NRT/SP suffix picked automatically per date
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


def _add_cpc_outlook(weekly: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """Attaches the CPC outlook that was actually valid for each week's
    calendar month - not today's outlook stamped across all of history (an
    earlier version of this did that: it called the CPC fetch once per area
    using today's date, so every one of the ~100 historical weeks in the
    cache ended up with the exact same cpc_temp_prob/cat, which is wrong -
    the whole point is that the outlook changes month to month)."""
    if weekly.empty:
        for col in ("cpc_temp_prob", "cpc_temp_cat", "cpc_precip_prob", "cpc_precip_cat"):
            weekly[col] = None
        return weekly
    weekly = weekly.copy()
    months = pd.to_datetime(weekly["week_start"])
    outlooks = [cpc.outlook_for_month(lat, lon, d.year, d.month) for d in months]
    outlook_df = pd.DataFrame(outlooks, index=weekly.index)
    return pd.concat([weekly, outlook_df], axis=1)


def assemble_state(state: str, map_key: str, verbose: bool = True) -> pd.DataFrame:
    state = state.upper()
    areas = forecast_areas_for_state(state)
    end = dt.date.today()
    start = end - dt.timedelta(weeks=HISTORY_WEEKS)

    lat_min, lat_max, lon_min, lon_max = STATE_BBOX[state]
    state_bbox = (lon_min, lat_min, lon_max, lat_max)  # (west, south, east, north)

    if verbose:
        print(f"  [{state}] fetching fire detections for whole state...")
    try:
        fire_raw = firms.fetch_area_csv_range_smart(map_key, FIRMS_BASE_SOURCE, state_bbox, start, end)
        fire_by_area = firms.assign_to_areas(fire_raw, areas)
    except Exception as exc:
        print(f"  [{state}] FIRMS FAILED: {exc}")
        fire_by_area = pd.DataFrame(columns=["area_id", "week_start", "fire_detections"])

    frames = []
    for area in areas:
        if verbose:
            print(f"  [{state}] fetching {area.area_id} ({area.lat:.2f}, {area.lon:.2f})")
        try:
            power_df = power.fetch_power_daily(area.lat, area.lon, start, end)
            weekly = _weekly_power(power_df)
            weekly = _add_cpc_outlook(weekly, area.lat, area.lon)

            area_fire = fire_by_area[fire_by_area["area_id"] == area.area_id][["week_start", "fire_detections"]]
            out = weekly.merge(area_fire, on="week_start", how="left")
            out["fire_detections"] = out["fire_detections"].fillna(0).astype(int)
            out["area_id"] = area.area_id
            frames.append(out)
        except Exception as exc:
            print(f"  [{state}] FAILED {area.area_id}: {exc}")

    if not frames:
        return pd.DataFrame()

    cols = [
        "area_id", "week_start", "t2m_max", "precip", "gwetroot",
        "cpc_temp_prob", "cpc_temp_cat", "cpc_precip_prob", "cpc_precip_cat",
        "fire_detections",
    ]
    return pd.concat(frames, ignore_index=True)[cols]


def save_state(state: str, df: pd.DataFrame) -> str:
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    path = os.path.join(DATA_CACHE_DIR, f"{state.upper()}.csv.gz")
    df.to_csv(path, index=False, compression="gzip")
    return path
