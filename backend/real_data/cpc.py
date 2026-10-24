"""
NOAA CPC (Climate Prediction Center) monthly temperature/precipitation outlook.

These are published as zipped shapefiles, one zip per issue month, containing
several lead times (how many months ahead each shape forecasts). We want the
shortest lead available (the "next month" outlook). The exact shapefile naming
and field layout is confirmed by inspecting a real downloaded zip (done once,
by hand, during setup) rather than guessed here; LEAD_NAME_HINTS below is the
first thing to update if NOAA changes their naming.

Expected per the task brief: field names "Prob" (probability, 33-100) and
"Cat" (category). Cat might be encoded as text (B/N/A), or as small integers
mapped the same way (-1 below, 0 normal, 1 above) depending on the file -
`normalize_category` below handles both and documents the mapping it used.
"""
from __future__ import annotations

import functools
import io
import zipfile

import requests
import shapefile  # pyshp

CPC_BASE_URL = "https://ftp.cpc.ncep.noaa.gov/GIS/us_tempprcpfcst"

# Candidate substrings (lowercased) for "shortest lead / next month" shapefiles,
# checked in order. CPC's naming has used "lead14" (half-month lead counted in
# half-months: lead 0.5 month) in some releases, and "mon1"/"01" in others.
LEAD_NAME_HINTS = ["lead14", "mon1", "m01", "_01_", "lead1"]

CATEGORY_MAP = {
    "b": "below", "below": "below", "-1": "below",
    "n": "normal", "normal": "normal", "0": "normal", "e": "normal", "ec": "normal",
    "a": "above", "above": "above", "1": "above",
}


@functools.lru_cache(maxsize=32)
def _download_zip_bytes(year: int, month: int, kind: str, timeout: int) -> bytes:
    url = f"{CPC_BASE_URL}/{kind}_{year:04d}{month:02d}.zip"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def download_zip(year: int, month: int, kind: str = "seastemp", timeout: int = 30) -> zipfile.ZipFile:
    """kind is 'seastemp' (temperature) or 'seasprcp' (precipitation).
    The outlook only depends on (year, month, kind) - not on location - so
    every forecast area in every state reuses the same cached download
    instead of re-fetching this ~50MB zip per area."""
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
