"""
Offline tests for the real_data/ pipeline. These never touch the network -
every requests call is monkeypatched with a fake response built to look like
what the real NASA/NOAA endpoints return. This is meant to run before (and
after) hitting the real APIs, so we always know the parsing/joining logic
itself still works even if a source is temporarily unreachable.

Run with:  python -m tests.test_real_data_offline
       or: pytest tests/test_real_data_offline.py
"""
from __future__ import annotations

import datetime as dt
import io
import sys
import zipfile

import pandas as pd
import shapefile

sys.path.insert(0, ".")

from real_data import assemble, firms, power, cpc  # noqa: E402


class FakeResponse:
    def __init__(self, text=None, content=None, json_data=None, status=200):
        self.text = text or ""
        self.content = content or b""
        self._json = json_data
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json


def fake_power_json(start: dt.date, end: dt.date) -> dict:
    t2m, precip, gwet = {}, {}, {}
    d = start
    while d <= end:
        key = d.strftime("%Y%m%d")
        t2m[key] = 22.5
        precip[key] = 1.2
        gwet[key] = 0.31
        d += dt.timedelta(days=1)
    # last day missing, like a real feed that lags behind today
    last_key = end.strftime("%Y%m%d")
    t2m[last_key] = power.MISSING_VALUE
    return {"properties": {"parameter": {"T2M_MAX": t2m, "PRECTOTCORR": precip, "GWETROOT": gwet}}}


def fake_firms_area_csv() -> str:
    rows = [
        "latitude,longitude,confidence,acq_date,acq_time",
        "40.60,-105.58,85,2024-08-01,1200",
        "40.61,-105.57,n,2024-08-01,1340",
        "40.62,-105.56,70,2024-08-02,0130",
    ]
    return "\n".join(rows)


def fake_cpc_zip(prob_value=55, cat_value="A") -> bytes:
    buf = io.BytesIO()
    writer = shapefile.Writer(shx=io.BytesIO(), shp=io.BytesIO(), dbf=io.BytesIO())
    writer.field("Prob", "N")
    writer.field("Cat", "C")
    writer.poly([[(-105.6, 40.5), (-105.4, 40.5), (-105.4, 40.7), (-105.6, 40.7), (-105.6, 40.5)]])
    writer.record(Prob=prob_value, Cat=cat_value)
    writer.close()

    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("seastemp_202409_lead14.shp", writer.shp.getbuffer().tobytes())
        zf.writestr("seastemp_202409_lead14.shx", writer.shx.getbuffer().tobytes())
        zf.writestr("seastemp_202409_lead14.dbf", writer.dbf.getbuffer().tobytes())
    return buf.getvalue()


def test_power_parses_and_drops_missing():
    start, end = dt.date(2024, 8, 1), dt.date(2024, 8, 3)
    payload = fake_power_json(start, end)
    df = power.parse_power_response(payload)
    assert len(df) == 2  # the last (missing) day is dropped
    assert set(df.columns) == {"date", "t2m_max", "precip", "gwetroot"}
    assert df["t2m_max"].iloc[0] == 22.5


def test_firms_weekly_counts_normalizes_confidence():
    df = pd.read_csv(io.StringIO(fake_firms_area_csv()))
    weekly = firms.weekly_detection_counts(df)
    assert "fire_detections" in weekly.columns
    assert weekly["fire_detections"].sum() == 3


def test_cpc_reads_shapefile_from_zip():
    zf = zipfile.ZipFile(io.BytesIO(fake_cpc_zip(prob_value=55, cat_value="A")))
    names = cpc.list_shapefiles(zf)
    assert names, "expected at least one shapefile in the fake zip"
    chosen = cpc.pick_shortest_lead(names)
    assert chosen is not None
    records = cpc.read_shape_records(zf, chosen)
    assert records
    rec = cpc.nearest_record(records, lat=40.6, lon=-105.5)
    assert rec["cat"] == "above"
    assert rec["prob"] == 55


def test_assemble_area_joins_sources(monkeypatch):
    start = dt.date.today() - dt.timedelta(weeks=4)
    end = dt.date.today()

    def fake_get(url, params=None, timeout=None):
        if "power.larc.nasa.gov" in url:
            return FakeResponse(json_data=fake_power_json(start, end))
        if "firms" in url:
            return FakeResponse(text=fake_firms_area_csv())
        raise AssertionError(f"unexpected URL in test: {url}")

    monkeypatch.setattr("real_data.power.requests.get", fake_get)
    monkeypatch.setattr("real_data.firms.requests.get", fake_get)
    monkeypatch.setattr("real_data.cpc.download_zip", lambda y, m, kind: zipfile.ZipFile(io.BytesIO(fake_cpc_zip())))

    df = assemble.assemble_area("CO-1", lat=40.6, lon=-105.5, map_key="FAKEKEY")
    assert not df.empty
    for col in [
        "area_id", "week_start", "t2m_max", "precip", "gwetroot",
        "cpc_temp_prob", "cpc_temp_cat", "cpc_precip_prob", "cpc_precip_cat",
        "fire_detections",
    ]:
        assert col in df.columns
    assert (df["area_id"] == "CO-1").all()


def test_synthetic_fallback_matches_schema():
    import data_ingestion

    df, source = data_ingestion.load_state_data("DE")  # no real cache for DE -> synthetic
    assert source == "synthetic"
    for col in data_ingestion.REQUIRED_COLUMNS:
        assert col in df.columns


if __name__ == "__main__":
    tests = [
        test_power_parses_and_drops_missing,
        test_firms_weekly_counts_normalizes_confidence,
        test_cpc_reads_shapefile_from_zip,
        test_synthetic_fallback_matches_schema,
    ]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")

    # test_assemble_area_joins_sources needs a monkeypatch fixture; run it by hand.
    class _MP:
        def setattr(self, target, value):
            import importlib

            parts = target.split(".")
            # Walk from the longest importable prefix down, then setattr the rest.
            for split in range(len(parts) - 1, 0, -1):
                mod_path = ".".join(parts[:split])
                try:
                    obj = importlib.import_module(mod_path)
                except ImportError:
                    continue
                for attr in parts[split:-1]:
                    obj = getattr(obj, attr)
                setattr(obj, parts[-1], value)
                return
            raise ImportError(f"could not resolve {target}")

    test_assemble_area_joins_sources(_MP())
    print("PASS test_assemble_area_joins_sources")
    print("\nAll offline tests passed.")
