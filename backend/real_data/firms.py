"""
NASA FIRMS (Fire Information for Resource Management System) API.

Docs: https://firms.modaps.eosdis.nasa.gov/api/
Two endpoints used here:
  - data_availability: tells us the date range each sensor/source actually covers.
  - area/csv: fire detections (lat/lon/confidence/etc.) inside a bounding box,
    for up to `day_range` days ending on `date`.

Checked live (2026-10-24): MODIS_NRT/MODIS_SP are both still updated through
today, contrary to what we expected going in - no need to fall back to VIIRS.
We default to VIIRS_SNPP anyway (slightly better spatial resolution), but
source_coverage() below checks live availability either way instead of
assuming.

Also checked live: every source/tier combination caps the area/csv
`day_range` at 5 days, not the 10 some FIRMS docs mention for other tiers -
see max_day_range_for_source(). And the API will 429 if hit too fast across
the thousands of requests a full 50-state pull makes, so every request here
retries with backoff.
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


def _get_with_retry(url: str, timeout: int) -> requests.Response:
    delay = 3.0
    resp = None
    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, timeout=timeout)
        if resp.status_code not in (429, 403):
            return resp
        time.sleep(delay)
        delay *= 2
    return resp  # give up after MAX_RETRIES, let caller's raise_for_status report it


@functools.lru_cache(maxsize=8)
def data_availability(map_key: str, source: str = "all", timeout: int = 30) -> pd.DataFrame:
    """Depends only on the account/key, not on location or date - cached so a
    50-state run calls this once instead of once per area (which is what was
    tripping the API's rate limit: data_availability doesn't need to be
    called more than once per process run)."""
    url = f"{BASE_URL}/data_availability/csv/{map_key}/{source}"
    resp = _get_with_retry(url, timeout)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def max_day_range_for_source(source: str) -> int:
    """Confirmed live against all four combinations (MODIS/VIIRS_SNPP x
    NRT/SP): this API key's tier caps every source's area/csv day_range at 5
    ("Invalid day range. Expects [1..5]."), not the 10 the FIRMS docs mention
    for some tiers."""
    return 5


def fetch_area_csv(
    map_key: str,
    source: str,
    bbox: tuple[float, float, float, float],
    date: dt.date,
    day_range: int | None = None,
    timeout: int = 60,
) -> pd.DataFrame:
    """bbox is (west, south, east, north) in degrees, as FIRMS expects."""
    if day_range is None:
        day_range = max_day_range_for_source(source)
    west, south, east, north = bbox
    coords = f"{west},{south},{east},{north}"
    url = f"{BASE_URL}/area/csv/{map_key}/{source}/{coords}/{day_range}/{date.strftime('%Y-%m-%d')}"
    resp = _get_with_retry(url, timeout)
    resp.raise_for_status()
    text = resp.text.strip()
    if not text or text.lower().startswith("invalid") or text.lower().startswith("error"):
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text))
    return df


def source_coverage(map_key: str, base_source: str, timeout: int = 30) -> dict[str, tuple[dt.date, dt.date]]:
    """Returns {"NRT": (min_date, max_date), "SP": (min_date, max_date)} for
    the given base sensor name (e.g. "VIIRS_SNPP", "MODIS"), parsed from the
    data_availability endpoint. A suffix missing from the response is omitted."""
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
    """Fetches [start, end] using whichever of the base sensor's NRT/SP
    products actually covers each part of the window (NRT only keeps the last
    couple of months; SP is the historical archive). See source_coverage()."""
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
    """Walks backward from `end` in windows sized to what this source's API
    allows, to cover [start, end]."""
    max_window = max_day_range_for_source(source)
    frames = []
    window_end = end
    first = True
    while window_end >= start:
        if not first:
            time.sleep(0.3)  # stay well under the API's rate limit across hundreds of windows
        first = False
        window_days = min(max_window, (window_end - start).days + 1)
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


def assign_to_areas(df: pd.DataFrame, areas: list) -> pd.DataFrame:
    """Buckets one state-wide detections DataFrame into forecast areas by
    bounding-box containment, then returns weekly counts per area. One state-
    wide FIRMS query + local bucketing replaces one query per area - this is
    what made a 50-state pull practical instead of hitting the API's rate
    limit (confirmed live: ~290 per-area queries over 2 years of history
    triggered 403s; one query per state does not)."""
    if df.empty:
        return pd.DataFrame(columns=["area_id", "week_start", "fire_detections"])
    df = normalize_confidence(df)
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")

    area_id = pd.Series([None] * len(df), index=df.index, dtype=object)
    for area in areas:
        mask = (
            area_id.isna()
            & (lat >= area.lat_min) & (lat <= area.lat_max)
            & (lon >= area.lon_min) & (lon <= area.lon_max)
        )
        area_id[mask] = area.area_id

    out = df.copy()
    out["area_id"] = area_id
    out = out.dropna(subset=["area_id"])
    dates = pd.to_datetime(out["acq_date"])
    out["week_start"] = (dates - pd.to_timedelta(dates.dt.weekday, unit="D")).dt.date.astype(str)
    counts = out.groupby(["area_id", "week_start"]).size().rename("fire_detections").reset_index()
    return counts
