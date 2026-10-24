"""
Synthetic ("practice") data generator.

This produces data in the exact same weekly schema that real_data/assemble.py
produces from NASA POWER + NASA FIRMS + NOAA CPC. It's used whenever real data
for a state hasn't been downloaded yet (or a real source fails), so the rest
of the app (features, model, API, frontend) always has something to run on.

Schema (one row per area per ISO week), columns:
  area_id, week_start (date), t2m_max, precip, gwetroot,
  cpc_temp_prob, cpc_temp_cat, cpc_precip_prob, cpc_precip_cat,
  fire_detections

cpc_*_cat is one of "below", "normal", "above" (next-month outlook category).
fire_detections is the count of satellite fire detections that week in the area.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from grid import all_forecast_areas

WEEKS_OF_HISTORY = 160  # ~3 years


def _seed_for(area_id: str) -> int:
    return int(hashlib.sha256(area_id.encode()).hexdigest()[:8], 16)


def _fire_season_weight(week_of_year: int, lat: float) -> float:
    """Northern-hemisphere fire season bias: summer peak, stronger at drier mid
    latitudes (rough proxy for the western US fire season)."""
    season = np.cos((week_of_year - 30) / 52 * 2 * np.pi)  # peaks ~week 30 (late July)
    season = max(season, -0.3)
    lat_factor = 1.0 if 30 <= lat <= 45 else 0.6
    return season * lat_factor


def _aridity_factor(lon: float) -> float:
    """Rough proxy for the west-of-the-100th-meridian aridity gradient in the
    US: the west is drier and far more fire-prone on average than the humid
    east, which an earlier version of this generator ignored (giving every
    state, including wet eastern ones, an unrealistically high fire-week
    rate). 0.15 (humid east) to 1.0 (arid west); not meant to be precise,
    just enough to keep practice data roughly regionally plausible."""
    return float(np.clip((-100 - lon) / 15, 0.15, 1.0))


def generate_area_series(area_id: str, lat: float, lon: float, weeks: int = WEEKS_OF_HISTORY) -> pd.DataFrame:
    rng = np.random.default_rng(_seed_for(area_id))
    end = pd.Timestamp.today().normalize()
    end -= pd.Timedelta(days=end.weekday())  # align to Monday
    dates = pd.date_range(end=end, periods=weeks, freq="W-MON")

    aridity = _aridity_factor(lon)
    rows = []
    soil = 0.35 + rng.normal(0, 0.03)
    for i, week_start in enumerate(dates):
        woy = week_start.isocalendar().week
        season = _fire_season_weight(woy, lat)

        base_temp = 15 + 13 * np.sin((woy - 13) / 52 * 2 * np.pi) + (35 - abs(lat - 35)) * 0.15
        t2m_max = base_temp + rng.normal(0, 3)

        precip_scale = 3.0 * (1.0 - 0.5 * max(season, 0)) * (1.3 - 0.6 * aridity)
        precip = max(0.0, rng.gamma(2.0, precip_scale))

        soil = np.clip(soil + 0.015 * (precip - 8) / 8 - 0.01 * max(season, 0) + rng.normal(0, 0.01), 0.05, 0.6)

        dryness = (1 - soil / 0.6) * 0.6 + max(season, 0) * 0.4
        fire_rate = max(0.0, (dryness - 0.5)) * 10 * (0.25 + 0.75 * aridity)
        fire_detections = int(rng.poisson(max(fire_rate, 0.003)))

        # CPC-style monthly outlook, imperfectly anticipating the next month's
        # actual anomaly (gives the model *some* real lead-time signal).
        future_idx = min(i + 4, weeks - 1)
        temp_signal = rng.normal(0, 1) * 0.5 + (1 if season > 0.3 else -1 if season < -0.3 else 0) * 0.5
        precip_signal = -temp_signal + rng.normal(0, 0.6)
        cpc_temp_cat = "above" if temp_signal > 0.3 else "below" if temp_signal < -0.3 else "normal"
        cpc_precip_cat = "above" if precip_signal > 0.3 else "below" if precip_signal < -0.3 else "normal"
        cpc_temp_prob = float(np.clip(40 + abs(temp_signal) * 18 + rng.normal(0, 5), 33, 70))
        cpc_precip_prob = float(np.clip(40 + abs(precip_signal) * 18 + rng.normal(0, 5), 33, 70))

        rows.append(
            {
                "area_id": area_id,
                "week_start": week_start.date().isoformat(),
                "t2m_max": round(float(t2m_max), 2),
                "precip": round(float(precip), 2),
                "gwetroot": round(float(soil), 4),
                "cpc_temp_prob": round(cpc_temp_prob, 1),
                "cpc_temp_cat": cpc_temp_cat,
                "cpc_precip_prob": round(cpc_precip_prob, 1),
                "cpc_precip_cat": cpc_precip_cat,
                "fire_detections": fire_detections,
            }
        )
    return pd.DataFrame(rows)


def generate_state_dataframe(state: str) -> pd.DataFrame:
    areas = [a for a in all_forecast_areas() if a.state == state.upper()]
    frames = [generate_area_series(a.area_id, a.lat, a.lon) for a in areas]
    return pd.concat(frames, ignore_index=True)
