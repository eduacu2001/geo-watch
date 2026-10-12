import json

import numpy as np
import pandas as pd
import pytest

from geowatch.webdata import (
    GEO_SMA_KM,
    check_no_raw_data,
    daily_range,
    daily_series,
    daily_track,
    decimation_step,
    display_name,
    manoeuvre_code,
    relative_series,
    rounded,
    unix_days,
)


def test_unix_days_matches_javascript_dates():
    assert unix_days("1970-01-02")[0] == pytest.approx(1.0)
    assert unix_days(pd.Timestamp("2025-01-01 12:00"))[0] == pytest.approx(20089.5)


def test_rounded_writes_nan_as_null_and_survives_json():
    values = rounded([1.234, np.nan, -0.0049, np.inf], 2)
    assert values == [1.23, None, -0.0, None]
    json.dumps(values, allow_nan=False)  # no NaN or Infinity in the output


def elements(epochs, lon, n_rev_day=1.0027):
    return pd.DataFrame(
        {
            "epoch": pd.to_datetime(epochs),
            "n_rev_day": n_rev_day,
            "lon_deg": lon,
            "a_km": GEO_SMA_KM + 10.0,
            "inc_deg": 0.05,
            "drift_deg_day": 0.0,
        }
    )


def test_daily_track_keeps_the_last_set_of_each_day_and_marks_gaps():
    el = elements(["2025-01-01 03:00", "2025-01-01 20:00", "2025-01-03 10:00"], [10.0, 11.0, 12.0])
    track = daily_track(el)
    assert track["d0"] == int(unix_days("2025-01-01")[0])
    assert track["lon"] == [11.0, None, 12.0]  # 2 January has no element set
    assert track["dh"] == [10, None, 10]


def test_daily_track_ignores_transfer_orbit_sets():
    el = elements(["2025-01-01", "2025-01-02"], [10.0, 11.0])
    el.loc[0, "n_rev_day"] = 2.3  # still in the transfer orbit
    assert daily_track(el)["lon"] == [11.0]


def test_daily_series_takes_the_median_per_day():
    times = pd.to_datetime(["2025-01-01 01:00", "2025-01-01 05:00", "2025-01-01 09:00"])
    series = daily_series(times, [1.0, 5.0, 2.0], 1)
    assert series["v"] == [2.0]


@pytest.mark.parametrize(
    ("n", "max_points", "step"), [(100, 600, 1), (601, 600, 2), (2448, 800, 4), (6000, 600, 12)]
)
def test_decimation_step_divides_the_day(n, max_points, step):
    assert decimation_step(n, max_points) == step
    assert 24 % step == 0
    assert len(range(0, n, step)) <= max_points


def test_decimation_step_of_very_long_series_keeps_two_samples_a_day():
    assert decimation_step(24 * 1000, 800) == 12


def test_relative_series_is_decimated_and_keeps_the_time_step():
    times = pd.date_range("2025-01-01", periods=1200, freq="h")  # 50 days, hourly
    rel = pd.DataFrame({"time": times, "R_km": 0.5, "I_km": -40.0, "C_km": 1.0, "range_km": 40.0})
    path = relative_series(rel, max_points=600)
    assert len(path["I"]) == 600
    assert path["dt_h"] == pytest.approx(2.0)
    assert path["t0"] == pytest.approx(unix_days("2025-01-01")[0])


def test_daily_range_summarises_each_day():
    times = pd.date_range("2025-01-01", periods=48, freq="h")
    rng = np.where(np.arange(48) < 24, 30.0 + np.arange(48) % 24, 50.0)
    rel = pd.DataFrame({"time": times, "range_km": rng})
    daily = daily_range(rel)
    assert daily["lo"] == [30.0, 50.0] and daily["hi"] == [53.0, 50.0]
    assert daily["mean"][0] == pytest.approx(41.5)


@pytest.mark.parametrize(
    ("kind", "code"),
    [("E-W station-keeping", "sk"), ("drift start + N-S", "start+ns"), ("N-S (inclination)", "ns")],
)
def test_manoeuvre_codes(kind, code):
    assert manoeuvre_code(kind) == code


@pytest.mark.parametrize(
    ("raw", "shown"),
    [
        ("INTELSAT 33E", "Intelsat 33E"),
        ("SESAT 2 (EXPRESS AM-22)", "SESAT 2 (Express AM-22)"),
        ("EUTE KONNECT VHTS", "Eutelsat Konnect VHTS"),
        ("AZERSPACE 2/INTELSAT 38", "Azerspace 2/Intelsat 38"),
        ("MEV-2", "MEV-2"),
    ],
)
def test_display_name(raw, shown):
    assert display_name(raw) == shown


def test_raw_tle_lines_are_never_exported():
    line1 = "1 40258U 14059A   25280.50000000 -.00000123  00000-0  00000-0 0  9991"
    with pytest.raises(ValueError):
        check_no_raw_data({"objects": [{"name": "Luch", "note": line1}]})
    with pytest.raises(ValueError):
        check_no_raw_data({"TLE_LINE1": "x"})
    check_no_raw_data({"objects": [{"name": "1 satellite", "lon": [1.0, None]}]})
