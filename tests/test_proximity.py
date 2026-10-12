import datetime as dt

import numpy as np
import pandas as pd
import pytest
from sgp4 import exporter
from sgp4.api import WGS72, Satrec

from geowatch.ephemeris import ephemeris
from geowatch.proximity import (
    KM_PER_DEG,
    neighbours_at,
    parking_slots,
    relative_motion,
    ric_basis,
    summarize_pair,
    to_ric,
)
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


# --- Phase 3B: RIC frame ------------------------------------------------------------------

R_GEO, V_GEO = 42164.0, 3.0747  # km, km/s


def test_ric_basis_of_a_simple_orbit_is_the_identity():
    np.testing.assert_allclose(ric_basis([R_GEO, 0, 0], [0, V_GEO, 0]), np.eye(3), atol=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_ric_basis_is_a_rotation(seed):
    rng = np.random.default_rng(seed)
    r = rng.normal(size=3) * R_GEO
    v = rng.normal(size=3) * V_GEO
    M = ric_basis(r, v)
    np.testing.assert_allclose(M @ M.T, np.eye(3), atol=1e-12)  # orthonormal
    assert np.linalg.det(M) == pytest.approx(1.0)  # right-handed, not a reflection


def test_ric_basis_works_on_arrays_of_vectors():
    rng = np.random.default_rng(7)
    r, v = rng.normal(size=(5, 3)) * R_GEO, rng.normal(size=(5, 3)) * V_GEO
    stacked = ric_basis(r, v)
    assert stacked.shape == (5, 3, 3)
    for k in range(5):
        np.testing.assert_allclose(stacked[k], ric_basis(r[k], v[k]))


@pytest.mark.parametrize(
    ("chaser", "expected"),
    [
        ([R_GEO + 10, 0, 0], [10, 0, 0]),  # higher: radial
        ([R_GEO, 5, 0], [0, 5, 0]),  # further along the velocity: in-track
        ([R_GEO, 0, -3], [0, 0, -3]),  # below the orbit plane: cross-track
    ],
)
def test_to_ric_components(chaser, expected):
    ric = to_ric(chaser, [R_GEO, 0, 0], [0, V_GEO, 0])
    np.testing.assert_allclose(ric, expected, atol=1e-9)


def test_to_ric_follows_the_target_frame():
    # Target a quarter of an orbit later: at +y, moving towards -x
    target_r, target_v = [0, R_GEO, 0], [-V_GEO, 0, 0]
    np.testing.assert_allclose(to_ric([-5, R_GEO, 0], target_r, target_v), [0, 5, 0], atol=1e-9)
    np.testing.assert_allclose(to_ric([0, R_GEO + 2, 0], target_r, target_v), [2, 0, 0], atol=1e-9)


def test_to_ric_preserves_distance_and_works_on_arrays():
    rng = np.random.default_rng(11)
    r_t, v_t = rng.normal(size=(6, 3)) * R_GEO, rng.normal(size=(6, 3)) * V_GEO
    r_c = r_t + rng.normal(size=(6, 3)) * 50
    ric = to_ric(r_c, r_t, v_t)
    assert ric.shape == (6, 3)
    np.testing.assert_allclose(np.linalg.norm(ric, axis=1), np.linalg.norm(r_c - r_t, axis=1))


# --- Phase 3B: ephemerides and relative motion -----------------------------------------


def _tle_record(gp_id, jd, mean_anomaly_deg, norad=90001):
    sat = Satrec()
    sat.sgp4init(WGS72, "i", norad, jd - 2433281.5, 0.0, 0.0, 0.0, 0.0002, 0.0, 0.0005,
                 np.radians(mean_anomaly_deg), 2 * np.pi * 1.00273791 / 1440.0, 0.0)  # fmt: skip
    l1, l2 = exporter.export_tle(sat)
    epoch = pd.Timestamp("1970-01-01") + pd.to_timedelta((jd - 2440587.5) * 86400, unit="s")
    return {"GP_ID": str(gp_id), "EPOCH": epoch.isoformat(), "TLE_LINE1": l1, "TLE_LINE2": l2}


JD0 = 2460676.5  # 2025-01-01 00:00 UTC


def test_ephemeris_switches_to_the_latest_element_set():
    raw = pd.DataFrame([_tle_record(1, JD0, 10.0), _tle_record(2, JD0 + 1, 50.0)])
    times = pd.to_datetime(["2025-01-01 12:00", "2025-01-02 06:00"])
    r, _, age = ephemeris(raw, times)
    np.testing.assert_allclose(age, [0.5, 0.25])
    second_only = ephemeris(raw.iloc[[1]], times[[1]])[0]
    np.testing.assert_allclose(r[1], second_only[0])  # second time uses the second set


def test_relative_motion_of_a_satellite_parked_behind_another():
    """Same orbit, 0.05 deg behind in mean anomaly: about 37 km behind in-track."""
    target = pd.DataFrame([_tle_record(1, JD0, 100.0, norad=90001)])
    chaser = pd.DataFrame([_tle_record(1, JD0, 99.95, norad=90002)])
    rel = relative_motion(chaser, target, "2025-01-01", "2025-01-03", step_hours=1)
    summary = summarize_pair(rel)
    expected = R_GEO * np.radians(0.05)
    assert summary["median_range_km"] == pytest.approx(expected, rel=0.05)
    assert summary["mean_I_km"] == pytest.approx(-expected, rel=0.05)
    assert abs(summary["mean_R_km"]) < 1.0 and abs(summary["mean_C_km"]) < 1.0
