"""
NASA FIRMS (Fire Information for Resource Management System) API.
Docs: https://firms.modaps.eosdis.nasa.gov/api/

- data_availability: date range each sensor/source actually covers.
- area/csv: fire detections in a bounding box, up to `day_range` days
  ending on `date`. This key's tier caps day_range at 5 (not 10, despite
  some FIRMS docs), and can 429/403 under heavy request volume or 400 in
  short throttled bursts - every request here retries with backoff.
"""
from __future__ import annotations

import datetime as dt
import functools
import io
import time

import pandas as pd
import requests

BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api"
MAX_RETRIES = 6
MAX_DAY_RANGE = 5


def _get_with_retry(url: str, timeout: int) -> requests.Response:
    delay = 3.0
    resp = None
    for _ in range(MAX_RETRIES):
        resp = requests.get(url, timeout=timeout)
        if resp.status_code not in (429, 403):
            return resp
        time.sleep(delay)
        delay *= 2
    return resp  # give up, let caller's raise_for_status report it


@functools.lru_cache(maxsize=8)
def data_availability(map_key: str, source: str = "all", timeout: int = 30) -> pd.DataFrame:
    """Depends only on the account, not location/date - cached so a 50-state
    run calls this once instead of once per state."""
    url = f"{BASE_URL}/data_availability/csv/{map_key}/{source}"
    resp = _get_with_retry(url, timeout)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def fetch_area_csv(
    map_key: str,
    source: str,
    bbox: tuple[float, float, float, float],
    date: dt.date,
    day_range: int = MAX_DAY_RANGE,
    timeout: int = 60,
) -> pd.DataFrame:
    """bbox is (west, south, east, north) in degrees, as FIRMS expects."""
    west, south, east, north = bbox
    coords = f"{west},{south},{east},{north}"
    url = f"{BASE_URL}/area/csv/{map_key}/{source}/{coords}/{day_range}/{date.strftime('%Y-%m-%d')}"
    resp = _get_with_retry(url, timeout)
    resp.raise_for_status()
    text = resp.text.strip()
    if not text or text.lower().startswith(("invalid", "error")):
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(text))


def source_coverage(map_key: str, base_source: str, timeout: int = 30) -> dict[str, tuple[dt.date, dt.date]]:
    """{"NRT": (min_date, max_date), "SP": (min_date, max_date)} for a base
    sensor name (e.g. "VIIRS_SNPP"). A missing suffix is omitted."""
    avail = data_availability(map_key, "all", timeout=timeout)
    out = {}
    for suffix in ("NRT", "SP"):
        row = avail[avail["data_id"] == f"{base_source}_{suffix}"]
        if not row.empty:
            out[suffix] = (
                dt.datetime.strptime(str(row.iloc[0]["min_date"]), "%Y-%m-%d").date(),
                dt.datetime.strptime(str(row.iloc[0]["max_date"]), "%Y-%m-%d").date(),
            )
    return out


def fetch_area_csv_range_smart(
    map_key: str,
    base_source: str,
    bbox: tuple[float, float, float, float],
    start: dt.date,
    end: dt.date,
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetches [start, end] using whichever NRT/SP product actually covers
    each part of the window (NRT only keeps the last couple of months)."""
    coverage = source_coverage(map_key, base_source, timeout=timeout)
    frames = []
    for suffix, (cov_min, cov_max) in coverage.items():
        window_start = max(start, cov_min)
        window_end = min(end, cov_max)
        if window_start > window_end:
            continue
        df = fetch_area_csv_range(map_key, f"{base_source}_{suffix}", bbox, window_start, window_end, timeout=timeout)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def fetch_area_csv_range(
    map_key: str,
    source: str,
    bbox: tuple[float, float, float, float],
    start: dt.date,
    end: dt.date,
    timeout: int = 60,
) -> pd.DataFrame:
    """Walks backward from `end` in MAX_DAY_RANGE-day windows to cover [start, end]."""
    frames = []
    window_end = end
    first = True
    while window_end >= start:
        if not first:
            time.sleep(0.1)  # stay polite across hundreds of windows
        first = False
        window_days = min(MAX_DAY_RANGE, (window_end - start).days + 1)

        df = pd.DataFrame()
        last_exc = None
        for backoff in (0, 5.0, 15.0):  # some windows 400 in sustained bursts; retry rides it out
            if backoff:
                time.sleep(backoff)
            try:
                df = fetch_area_csv(map_key, source, bbox, window_end, day_range=window_days, timeout=timeout)
                last_exc = None
                break
            except requests.exceptions.HTTPError as exc:
                last_exc = exc
        if last_exc is not None:
            print(f"    [firms] skipping window ending {window_end} after retries failed: {last_exc}")
        if not df.empty:
            frames.append(df)
        window_end -= dt.timedelta(days=window_days)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def normalize_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """MODIS confidence is numeric; VIIRS is l/n/h strings. Normalize both
    to a 0-100 scale."""
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
    """Returns columns: week_start, fire_detections."""
    if df.empty:
        return pd.DataFrame(columns=["week_start", "fire_detections"])
    df = normalize_confidence(df)
    df = df[df["confidence"] >= min_confidence]
    dates = pd.to_datetime(df["acq_date"])
    week_start = dates - pd.to_timedelta(dates.dt.weekday, unit="D")
    out = week_start.value_counts().sort_index().rename_axis("week_start").reset_index(name="fire_detections")
    out["week_start"] = out["week_start"].dt.date.astype(str)
    return out


def assign_to_areas(df: pd.DataFrame, areas: list) -> pd.DataFrame:
    """Buckets one state-wide detections DataFrame into forecast areas by
    bounding-box containment, then returns weekly counts per area - one
    state-wide query instead of one per area."""
    if df.empty:
        return pd.DataFrame(columns=["area_id", "week_start", "fire_detections"])
    df = normalize_confidence(df)
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")

    area_id = pd.Series([None] * len(df), index=df.index, dtype=object)
    for area in areas:
        mask = area_id.isna() & lat.between(area.lat_min, area.lat_max) & lon.between(area.lon_min, area.lon_max)
        area_id[mask] = area.area_id

    out = df.copy()
    out["area_id"] = area_id
    out = out.dropna(subset=["area_id"])
    dates = pd.to_datetime(out["acq_date"])
    out["week_start"] = (dates - pd.to_timedelta(dates.dt.weekday, unit="D")).dt.date.astype(str)
    return out.groupby(["area_id", "week_start"]).size().rename("fire_detections").reset_index()
