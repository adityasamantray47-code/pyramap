"""
NASA POWER (Prediction Of Worldwide Energy Resources) daily point API.

Docs: https://power.larc.nasa.gov/docs/services/api/temporal/daily/
Gives daily weather/soil data for a single lat/lon point, going back decades,
usually a few days behind real time (varies by parameter; GWETROOT tends to
lag more than T2M_MAX/PRECTOTCORR because it comes from a reanalysis model).

Missing-value marker: NASA POWER uses -999 (sometimes -999.0) for days it
has no data for yet.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import requests

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = ["T2M_MAX", "PRECTOTCORR", "GWETROOT"]
MISSING_VALUE = -999.0


def fetch_power_daily(lat: float, lon: float, start: dt.date, end: dt.date, timeout: int = 30) -> pd.DataFrame:
    """Returns a DataFrame with columns: date, t2m_max, precip, gwetroot.
    Rows where NASA POWER reported the missing marker are dropped."""
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }
    resp = requests.get(POWER_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    return parse_power_response(payload)


def parse_power_response(payload: dict) -> pd.DataFrame:
    param_data = payload["properties"]["parameter"]
    t2m_max = param_data.get("T2M_MAX", {})
    precip = param_data.get("PRECTOTCORR", {})
    gwetroot = param_data.get("GWETROOT", {})

    dates = sorted(t2m_max.keys())
    rows = []
    for d in dates:
        t = t2m_max.get(d)
        p = precip.get(d)
        g = gwetroot.get(d)
        if t is None or t == MISSING_VALUE:
            continue
        rows.append(
            {
                "date": dt.datetime.strptime(d, "%Y%m%d").date().isoformat(),
                "t2m_max": None if t == MISSING_VALUE else t,
                "precip": None if p == MISSING_VALUE else p,
                "gwetroot": None if g == MISSING_VALUE else g,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def days_behind_today(df: pd.DataFrame, column: str) -> int | None:
    """How many days behind today the most recent non-null value of `column` is."""
    if df.empty or column not in df.columns:
        return None
    valid = df.dropna(subset=[column])
    if valid.empty:
        return None
    last_date = valid["date"].max().date()
    return (dt.date.today() - last_date).days
