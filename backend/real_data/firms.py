"""
NASA FIRMS (Fire Information for Resource Management System) API.

Docs: https://firms.modaps.eosdis.nasa.gov/api/
Two endpoints used here:
  - data_availability: tells us the date range each sensor/source actually covers.
  - area/csv: fire detections (lat/lon/confidence/etc.) inside a bounding box,
    for up to `day_range` days ending on `date`.

MODIS_SP / MODIS_NRT stopped being updated when MODIS Aqua/Terra fire
detections were deprecated by NASA in favor of VIIRS; if data_availability
shows MODIS's max_date is old, switch SOURCE to a VIIRS product (VIIRS_SNPP_NRT
for recent data, or VIIRS_SNPP_SP / VIIRS_1 for the full historical archive
back to ~2012).
"""
from __future__ import annotations

import datetime as dt
import io

import pandas as pd
import requests

BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api"


def data_availability(map_key: str, source: str = "all", timeout: int = 30) -> pd.DataFrame:
    url = f"{BASE_URL}/data_availability/csv/{map_key}/{source}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def fetch_area_csv(
    map_key: str,
    source: str,
    bbox: tuple[float, float, float, float],
    date: dt.date,
    day_range: int = 10,
    timeout: int = 60,
) -> pd.DataFrame:
    """bbox is (west, south, east, north) in degrees, as FIRMS expects.
    day_range is capped at 10 by the FIRMS area API for most sources."""
    west, south, east, north = bbox
    coords = f"{west},{south},{east},{north}"
    url = f"{BASE_URL}/area/csv/{map_key}/{source}/{coords}/{day_range}/{date.strftime('%Y-%m-%d')}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    text = resp.text.strip()
    if not text or text.lower().startswith("invalid") or text.lower().startswith("error"):
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text))
    return df


def fetch_area_csv_range(
    map_key: str,
    source: str,
    bbox: tuple[float, float, float, float],
    start: dt.date,
    end: dt.date,
    timeout: int = 60,
) -> pd.DataFrame:
    """Walks backward from `end` in 10-day windows to cover [start, end],
    since the area API caps day_range at 10."""
    frames = []
    window_end = end
    while window_end >= start:
        window_days = min(10, (window_end - start).days + 1)
        df = fetch_area_csv(map_key, source, bbox, window_end, day_range=window_days, timeout=timeout)
        if not df.empty:
            frames.append(df)
        window_end -= dt.timedelta(days=window_days)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates()


def normalize_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """MODIS confidence is numeric (0-100); VIIRS confidence is often the
    strings l/n/h (low/nominal/high). Normalize both to a 0-100 numeric scale
    so downstream code doesn't need to know which sensor it came from."""
    if "confidence" not in df.columns or df.empty:
        return df
    df = df.copy()
    if df["confidence"].dtype == object:
        mapping = {"l": 25.0, "n": 60.0, "h": 90.0}
        df["confidence"] = df["confidence"].str.lower().map(mapping).fillna(60.0)
    else:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(60.0)
    return df


def weekly_detection_counts(df: pd.DataFrame, min_confidence: float = 0.0) -> pd.DataFrame:
    """Returns columns: week_start, fire_detections. `df` must have acq_date
    and confidence columns (as returned by the area API)."""
    if df.empty:
        return pd.DataFrame(columns=["week_start", "fire_detections"])
    df = normalize_confidence(df)
    df = df[df["confidence"] >= min_confidence]
    dates = pd.to_datetime(df["acq_date"])
    week_start = dates - pd.to_timedelta(dates.dt.weekday, unit="D")
    counts = week_start.value_counts().sort_index()
    out = counts.rename_axis("week_start").reset_index(name="fire_detections")
    out["week_start"] = out["week_start"].dt.date.astype(str)
    return out
