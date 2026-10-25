"""
NOAA CPC (Climate Prediction Center) monthly temperature/precipitation outlook.

Published as zipped shapefiles, one per issue month, containing several lead
times. We want the shortest lead (the "next month" outlook) - confirmed
against a real download that NOAA names it "lead14". Fields are "Prob"
(33-100) and "Cat" (category, text or small int depending on the file -
see normalize_category).
"""
from __future__ import annotations

import functools
import io
import os
import zipfile

import requests
import shapefile  # pyshp

CPC_BASE_URL = "https://ftp.cpc.ncep.noaa.gov/GIS/us_tempprcpfcst"

# Each zip is ~50MB; cached to disk so re-running prepare_data.py after an
# interruption doesn't re-download the full set.
CPC_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_cache", "cpc_raw")

# Substrings (lowercased) for "shortest lead / next month" shapefiles, checked
# in order. "lead14" is confirmed; the rest are fallbacks if NOAA renames.
LEAD_NAME_HINTS = ["lead14", "mon1", "m01", "_01_", "lead1"]

CATEGORY_MAP = {
    "b": "below", "below": "below", "-1": "below",
    "n": "normal", "normal": "normal", "0": "normal", "e": "normal", "ec": "normal",
    "a": "above", "above": "above", "1": "above",
}


@functools.lru_cache(maxsize=32)
def _download_zip_bytes(year: int, month: int, kind: str, timeout: int) -> bytes:
    cache_path = os.path.join(CPC_CACHE_DIR, f"{kind}_{year:04d}{month:02d}.zip")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return f.read()

    url = f"{CPC_BASE_URL}/{kind}_{year:04d}{month:02d}.zip"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    content = resp.content

    os.makedirs(CPC_CACHE_DIR, exist_ok=True)
    with open(cache_path, "wb") as f:
        f.write(content)
    return content


def download_zip(year: int, month: int, kind: str = "seastemp", timeout: int = 30) -> zipfile.ZipFile:
    """kind is 'seastemp' (temperature) or 'seasprcp' (precipitation)."""
    content = _download_zip_bytes(year, month, kind, timeout)
    return zipfile.ZipFile(io.BytesIO(content))


def list_shapefiles(zf: zipfile.ZipFile) -> list[str]:
    return sorted({n[:-4] for n in zf.namelist() if n.lower().endswith(".shp")})


def pick_shortest_lead(shapefile_names: list[str]) -> str | None:
    lowered = [(n, n.lower()) for n in shapefile_names]
    for hint in LEAD_NAME_HINTS:
        for name, low in lowered:
            if hint in low:
                return name
    return shapefile_names[0] if shapefile_names else None


def normalize_category(raw) -> str:
    key = str(raw).strip().lower()
    return CATEGORY_MAP.get(key, "normal")


def read_shape_records(zf: zipfile.ZipFile, shape_name: str) -> list[dict]:
    """Reads one shapefile (by base name, no extension) out of the zip and
    returns a list of {lon, lat, prob, cat} dicts, one per polygon centroid
    (good enough resolution for sampling at a forecast-area's point)."""

    def member(ext: str) -> io.BytesIO:
        return io.BytesIO(zf.read(f"{shape_name}{ext}"))

    sf = shapefile.Reader(shp=member(".shp"), dbf=member(".dbf"), shx=member(".shx"))
    field_names = [f[0] for f in sf.fields[1:]]
    prob_field = next((f for f in field_names if f.lower() == "prob"), None)
    cat_field = next((f for f in field_names if f.lower() == "cat"), None)

    records = []
    for shape_rec in sf.iterShapeRecords():
        points = shape_rec.shape.points
        if not points:
            continue
        lon = sum(p[0] for p in points) / len(points)
        lat = sum(p[1] for p in points) / len(points)
        rec = shape_rec.record.as_dict()
        prob = rec.get(prob_field) if prob_field else None
        cat = normalize_category(rec.get(cat_field)) if cat_field else "normal"
        records.append({"lon": lon, "lat": lat, "prob": prob, "cat": cat})
    return records


def nearest_record(records: list[dict], lat: float, lon: float) -> dict | None:
    if not records:
        return None
    return min(records, key=lambda r: (r["lat"] - lat) ** 2 + (r["lon"] - lon) ** 2)


@functools.lru_cache(maxsize=64)
def records_for_issue(year: int, month: int, kind: str) -> tuple[dict, ...] | None:
    """Parsed records for one issue's shortest-lead shapefile, cached by
    (year, month, kind) so sampling many areas only parses it once."""
    try:
        zf = download_zip(year, month, kind)
    except Exception:
        return None
    shapes = list_shapefiles(zf)
    name = pick_shortest_lead(shapes)
    if not name:
        return None
    return tuple(read_shape_records(zf, name))


def outlook_for_month(lat: float, lon: float, target_year: int, target_month: int) -> dict:
    """The outlook valid for (target_year, target_month): the next-month
    shape from the zip issued the month before, falling back to earlier
    issues if that one isn't published."""
    issue_year, issue_month = target_year, target_month - 1
    if issue_month <= 0:
        issue_month += 12
        issue_year -= 1

    for back in range(4):
        y, m = issue_year, issue_month - back
        while m <= 0:
            m += 12
            y -= 1
        temp_recs = records_for_issue(y, m, "seastemp")
        precip_recs = records_for_issue(y, m, "seasprcp")
        if not temp_recs or not precip_recs:
            continue
        temp_rec = nearest_record(list(temp_recs), lat, lon)
        precip_rec = nearest_record(list(precip_recs), lat, lon)
        if temp_rec and precip_rec:
            return {
                "cpc_temp_prob": temp_rec["prob"],
                "cpc_temp_cat": temp_rec["cat"],
                "cpc_precip_prob": precip_rec["prob"],
                "cpc_precip_cat": precip_rec["cat"],
            }
    return {"cpc_temp_prob": None, "cpc_temp_cat": "normal", "cpc_precip_prob": None, "cpc_precip_cat": "normal"}
