"""
Forecast areas ("grid") for the Long-Lead Wildfire Outlook.

Each state is split into a small number of rectangular lat/lon cells based
on its bounding box, not its true polygon boundary (avoids needing
geopandas/fiona). Bigger states get more cells. Each area has an id like
"CO-1", a centroid lat/lon, and a bounding box.
"""
from __future__ import annotations

from dataclasses import dataclass

# Approximate state bounding boxes: (lat_min, lat_max, lon_min, lon_max).
# Good enough for splitting into forecast-area cells; not survey-accurate borders.
STATE_BBOX = {
    "AL": (30.2, 35.0, -88.5, -84.9), "AK": (51.2, 71.4, -179.1, -129.9),
    "AZ": (31.3, 37.0, -114.8, -109.0), "AR": (33.0, 36.5, -94.6, -89.6),
    "CA": (32.5, 42.0, -124.4, -114.1), "CO": (37.0, 41.0, -109.1, -102.0),
    "CT": (41.0, 42.1, -73.7, -71.8), "DE": (38.4, 39.8, -75.8, -75.0),
    "FL": (24.5, 31.0, -87.6, -80.0), "GA": (30.4, 35.0, -85.6, -80.8),
    "HI": (18.9, 22.3, -160.3, -154.8), "ID": (42.0, 49.0, -117.3, -111.0),
    "IL": (37.0, 42.5, -91.5, -87.0), "IN": (37.8, 41.8, -88.1, -84.8),
    "IA": (40.4, 43.5, -96.6, -90.1), "KS": (37.0, 40.0, -102.1, -94.6),
    "KY": (36.5, 39.1, -89.6, -81.9), "LA": (29.0, 33.0, -94.0, -89.0),
    "ME": (43.0, 47.5, -71.1, -66.9), "MD": (37.9, 39.7, -79.5, -75.0),
    "MA": (41.2, 42.9, -73.5, -69.9), "MI": (41.7, 48.2, -90.4, -82.4),
    "MN": (43.5, 49.4, -97.2, -89.5), "MS": (30.2, 35.0, -91.7, -88.1),
    "MO": (36.0, 40.6, -95.8, -89.1), "MT": (44.4, 49.0, -116.1, -104.0),
    "NE": (40.0, 43.0, -104.1, -95.3), "NV": (35.0, 42.0, -120.0, -114.0),
    "NH": (42.7, 45.3, -72.6, -70.6), "NJ": (38.9, 41.4, -75.6, -73.9),
    "NM": (31.3, 37.0, -109.1, -103.0), "NY": (40.5, 45.0, -79.8, -71.8),
    "NC": (33.8, 36.6, -84.3, -75.5), "ND": (45.9, 49.0, -104.1, -96.6),
    "OH": (38.4, 42.0, -84.8, -80.5), "OK": (33.6, 37.0, -103.0, -94.4),
    "OR": (42.0, 46.3, -124.6, -116.5), "PA": (39.7, 42.3, -80.5, -74.7),
    "RI": (41.1, 42.0, -71.9, -71.1), "SC": (32.0, 35.2, -83.4, -78.5),
    "SD": (42.5, 45.9, -104.1, -96.4), "TN": (35.0, 36.7, -90.3, -81.6),
    "TX": (25.8, 36.5, -106.6, -93.5), "UT": (37.0, 42.0, -114.1, -109.0),
    "VT": (42.7, 45.0, -73.4, -71.5), "VA": (36.5, 39.5, -83.7, -75.2),
    "WA": (45.5, 49.0, -124.8, -116.9), "WV": (37.2, 40.6, -82.7, -77.7),
    "WI": (42.5, 47.1, -92.9, -86.8), "WY": (41.0, 45.0, -111.1, -104.0),
}

STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming",
}

ALL_STATES = sorted(STATE_BBOX.keys())

# Target cell size in degrees. Smaller states get 1 cell; larger states get more,
# split roughly evenly across lat/lon so cells stay close to square.
CELL_SIZE_DEG = 2.2
MAX_CELLS_PER_STATE = 9


@dataclass(frozen=True)
class ForecastArea:
    area_id: str
    state: str
    state_name: str
    index: int
    lat: float  # centroid
    lon: float  # centroid
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def to_dict(self) -> dict:
        return {
            "id": self.area_id,
            "state": self.state,
            "state_name": self.state_name,
            "lat": round(self.lat, 4),
            "lon": round(self.lon, 4),
            "bbox": [
                round(self.lat_min, 4),
                round(self.lon_min, 4),
                round(self.lat_max, 4),
                round(self.lon_max, 4),
            ],
        }


def _grid_dims(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> tuple[int, int]:
    n_lat = max(1, round((lat_max - lat_min) / CELL_SIZE_DEG))
    n_lon = max(1, round((lon_max - lon_min) / CELL_SIZE_DEG))
    while n_lat * n_lon > MAX_CELLS_PER_STATE:
        if n_lat >= n_lon and n_lat > 1:
            n_lat -= 1
        elif n_lon > 1:
            n_lon -= 1
        else:
            break
    return n_lat, n_lon


def forecast_areas_for_state(state: str) -> list[ForecastArea]:
    state = state.upper()
    if state not in STATE_BBOX:
        raise ValueError(f"Unknown state code: {state}")
    lat_min, lat_max, lon_min, lon_max = STATE_BBOX[state]
    n_lat, n_lon = _grid_dims(lat_min, lat_max, lon_min, lon_max)
    lat_step = (lat_max - lat_min) / n_lat
    lon_step = (lon_max - lon_min) / n_lon

    areas = []
    idx = 1
    for i in range(n_lat):
        for j in range(n_lon):
            cell_lat_min = lat_min + i * lat_step
            cell_lat_max = lat_min + (i + 1) * lat_step
            cell_lon_min = lon_min + j * lon_step
            cell_lon_max = lon_min + (j + 1) * lon_step
            areas.append(
                ForecastArea(
                    area_id=f"{state}-{idx}",
                    state=state,
                    state_name=STATE_NAMES[state],
                    index=idx,
                    lat=(cell_lat_min + cell_lat_max) / 2,
                    lon=(cell_lon_min + cell_lon_max) / 2,
                    lat_min=cell_lat_min,
                    lat_max=cell_lat_max,
                    lon_min=cell_lon_min,
                    lon_max=cell_lon_max,
                )
            )
            idx += 1
    return areas


def all_forecast_areas(states: list[str] | None = None) -> list[ForecastArea]:
    states = states or ALL_STATES
    areas = []
    for state in states:
        areas.extend(forecast_areas_for_state(state))
    return areas


def area_lookup(states: list[str] | None = None) -> dict[str, ForecastArea]:
    return {a.area_id: a for a in all_forecast_areas(states)}
