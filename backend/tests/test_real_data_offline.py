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

from grid import forecast_areas_for_state  # noqa: E402
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


def fake_firms_data_availability(today: dt.date) -> str:
    # Mirrors the real API: NRT only covers recent months, SP covers the
    # historical archive, meeting with no gap.
    nrt_min = today - dt.timedelta(days=60)
    rows = [
        "data_id,min_date,max_date",
        f"VIIRS_SNPP_NRT,{nrt_min.isoformat()},{today.isoformat()}",
        f"VIIRS_SNPP_SP,2012-01-20,{(nrt_min - dt.timedelta(days=1)).isoformat()}",
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


def test_assemble_state_joins_sources(monkeypatch):
    start = dt.date.today() - dt.timedelta(weeks=4)
    end = dt.date.today()
    data_availability_calls = []
    area_csv_calls = []

    def fake_get(url, params=None, timeout=None):
        if "power.larc.nasa.gov" in url:
            return FakeResponse(json_data=fake_power_json(start, end))
        if "data_availability" in url:
            data_availability_calls.append(url)
            return FakeResponse(text=fake_firms_data_availability(end))
        if "firms" in url:
            area_csv_calls.append(url)
            return FakeResponse(text=fake_firms_area_csv())
        raise AssertionError(f"unexpected URL in test: {url}")

    monkeypatch.setattr("real_data.power.requests.get", fake_get)
    monkeypatch.setattr("real_data.firms.requests.get", fake_get)
    monkeypatch.setattr("real_data.cpc.download_zip", lambda y, m, kind: zipfile.ZipFile(io.BytesIO(fake_cpc_zip())))
    monkeypatch.setattr("real_data.assemble.HISTORY_WEEKS", 4)  # keep the fake window-walk short
    monkeypatch.setattr("real_data.firms.time.sleep", lambda s: None)  # skip the real politeness delay
    firms.data_availability.cache_clear()
    cpc.records_for_issue.cache_clear()

    df = assemble.assemble_state("CO", map_key="FAKEKEY", verbose=False)
    assert not df.empty
    for col in [
        "area_id", "week_start", "t2m_max", "precip", "gwetroot",
        "cpc_temp_prob", "cpc_temp_cat", "cpc_precip_prob", "cpc_precip_cat",
        "fire_detections",
    ]:
        assert col in df.columns
    assert set(df["area_id"].unique()) == {a.area_id for a in forecast_areas_for_state("CO")}
    # The key efficiency property this test guards: FIRMS is queried once for
    # the whole state's bbox, not once per area, and data_availability is
    # cached (called once, not once per area) - this is what avoided the
    # live 403 rate-limit seen when both were done per-area.
    assert len(data_availability_calls) == 1
    # With HISTORY_WEEKS=4 this is a handful of 5-day windows for ONE state-wide
    # query; if FIRMS were (incorrectly) queried per area again, CO's 6 areas
    # would multiply this by 6.
    assert 1 <= len(area_csv_calls) <= 10


def test_synthetic_fallback_matches_schema(monkeypatch, tmp_path):
    import data_ingestion

    # Point the cache dir somewhere empty so this test doesn't depend on
    # whether DE's real data has been downloaded in this environment.
    monkeypatch.setattr(data_ingestion, "DATA_CACHE_DIR", str(tmp_path))

    df, source = data_ingestion.load_state_data("DE")
    assert source == "synthetic"
    for col in data_ingestion.REQUIRED_COLUMNS:
        assert col in df.columns


if __name__ == "__main__":
    tests = [
        test_power_parses_and_drops_missing,
        test_firms_weekly_counts_normalizes_confidence,
        test_cpc_reads_shapefile_from_zip,
    ]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")

    # test_assemble_state_joins_sources needs a monkeypatch fixture; run it by hand.
    class _MP:
        def setattr(self, *args):
            if len(args) == 3:
                # pytest-style setattr(obj, "attr_name", value)
                obj, attr_name, value = args
                setattr(obj, attr_name, value)
                return
            target, value = args
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

    test_assemble_state_joins_sources(_MP())
    print("PASS test_assemble_state_joins_sources")

    import tempfile

    test_synthetic_fallback_matches_schema(_MP(), tempfile.mkdtemp())
    print("PASS test_synthetic_fallback_matches_schema")
    print("\nAll offline tests passed.")
