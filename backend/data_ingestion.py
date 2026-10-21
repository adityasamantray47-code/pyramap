"""
Loads weekly per-area data for a state: real data from
data_cache/states/<STATE>.csv.gz if it has been downloaded (see
prepare_data.py), otherwise synthetic practice data so the app always has
something to run on.
"""
from __future__ import annotations

import os

import pandas as pd

import synthetic_data

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache", "states")

REQUIRED_COLUMNS = [
    "area_id", "week_start", "t2m_max", "precip", "gwetroot",
    "cpc_temp_prob", "cpc_temp_cat", "cpc_precip_prob", "cpc_precip_cat",
    "fire_detections",
]


def cache_path(state: str) -> str:
    return os.path.join(DATA_CACHE_DIR, f"{state.upper()}.csv.gz")


def has_real_data(state: str) -> bool:
    return os.path.exists(cache_path(state))


def load_state_data(state: str) -> tuple[pd.DataFrame, str]:
    """Returns (dataframe, source) where source is "real" or "synthetic"."""
    path = cache_path(state)
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["week_start"])
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if not missing:
            return df, "real"
    df = synthetic_data.generate_state_dataframe(state)
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df, "synthetic"


def data_status(states: list[str]) -> dict:
    real_states = [s for s in states if has_real_data(s)]
    as_of = None
    for s in real_states:
        df, _ = load_state_data(s)
        if not df.empty:
            latest = df["week_start"].max()
            as_of = latest if as_of is None else max(as_of, latest)
    return {
        "states_with_real_data": sorted(real_states),
        "states_with_synthetic_data": sorted(set(states) - set(real_states)),
        "data_through": as_of.date().isoformat() if as_of is not None else None,
    }
