import datetime as dt

import numpy as np
import pandas as pd
import pytest

from geowatch.proximity import KM_PER_DEG, neighbours_at, parking_slots
from geowatch.spacetrack_io import download_geo_window

T0 = pd.Timestamp("2024-01-01")


def drifting_then_parked() -> pd.DataFrame:
    """Parked at 10°E for 40 days (with two station-keeping burns), drifts east at
    0.5 deg/day for 20 days, then parks at 20°E for 30 days."""
    t = np.arange(0, 90, 0.5)
    drift = np.where((t >= 40) & (t < 60), 0.5, 0.0)
    drift[np.isin(t, [15.0, 30.0])] = 0.05  # short station-keeping blips
    lon = 10.0 + np.cumsum(drift) * 0.5
    return pd.DataFrame(
        {
            "epoch": T0 + pd.to_timedelta(t, unit="D"),
            "n_rev_day": 1.0027,
            "lon_deg": lon,
            "drift_deg_day": drift,
        }
    )


def test_parking_slots_finds_two_stays_and_merges_station_keeping_blips():
    slots = parking_slots(drifting_then_parked())
    assert len(slots) == 2
    assert slots.loc[0, "lon_deg"] == pytest.approx(10.0, abs=0.1)
    assert slots.loc[1, "lon_deg"] == pytest.approx(20.0, abs=0.1)
    assert slots.loc[0, "days"] > 35  # the blips at day 15 and 30 did not split the stay


def test_parking_slots_ignores_short_stops():
    el = drifting_then_parked()
    assert parking_slots(el.iloc[:6]).empty  # 3 days parked: shorter than min_days


def belt_snapshot() -> pd.DataFrame:
    rows = [
        (1, "TARGET A", 179.95),  # just across the antimeridian from the inspector
        (2, "TARGET B", 179.70),
        (3, "FAR AWAY", 100.00),
        (4, "INSPECTOR", -179.90),
    ]
    return pd.DataFrame(
        [
            {"norad_id": n, "object_name": name, "epoch": T0 + pd.Timedelta(hours=h),
             "n_rev_day": 1.0027, "lon_deg": lon, "drift_deg_day": 0.0, "inc_deg": 0.05}
            for n, name, lon in rows
            for h in (0, 12)
        ]
    )  # fmt: skip


def test_neighbours_handle_the_antimeridian_and_sort_by_distance():
    near = neighbours_at(belt_snapshot(), lon_deg=-179.90, when=T0, exclude=(4,))
    assert list(near["norad_id"]) == [1, 2]
    assert near.loc[0, "lon_offset_deg"] == pytest.approx(-0.15)
    assert near.loc[0, "arc_km"] == pytest.approx(0.15 * KM_PER_DEG)


def test_neighbours_respect_the_offset_limit():
    near = neighbours_at(belt_snapshot(), lon_deg=-179.90, when=T0, max_offset_deg=0.2)
    assert set(near["norad_id"]) == {1, 4}


class FakeSpaceTrack:
    def __init__(self):
        self.calls = []

    def gp_history(self, **query):
        self.calls.append(query)
        return [
            {"NORAD_CAT_ID": "1", "OBJECT_NAME": "X", "OBJECT_TYPE": "PAYLOAD",
             "EPOCH": "2024-01-01T00:00:00", "GP_ID": "1", "TLE_LINE1": "1", "TLE_LINE2": "2"}
        ]  # fmt: skip


def test_geo_window_is_cached_and_asks_only_for_tle_fields(tmp_path):
    client = FakeSpaceTrack()
    day = dt.date(2024, 1, 1)
    first = download_geo_window(client, day, directory=tmp_path)
    second = download_geo_window(client, day, directory=tmp_path)
    assert first == second and first.exists()
    assert len(client.calls) == 1  # the second call used the cached file
    query = client.calls[0]
    assert "TLE_LINE1" in query["predicates"] and "MEAN_MOTION" not in query["predicates"]
    assert query["epoch"].startswith("2023-12-30") and "2024-01-04" in query["epoch"]
