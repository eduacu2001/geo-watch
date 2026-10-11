import math

import numpy as np
import pytest
from sgp4 import exporter
from sgp4.api import WGS72, Satrec

from geowatch.elements import (
    _true_longitude_deg,
    elements_from_satrecs,
    elements_from_tle,
    gmst_deg,
    read_tle_text,
    wrap180,
)

JD_EPOCH = 2461300.5  # 2026-09-22 00:00 UTC
SIDEREAL_REV_DAY = 1.00273790935


def make_geo_satrec(n_rev_day: float = SIDEREAL_REV_DAY, lon_offset_deg: float = 0.0) -> Satrec:
    """Synthetic near-GEO satellite built directly from mean elements."""
    sat = Satrec()
    sat.sgp4init(
        WGS72,
        "i",
        99999,
        JD_EPOCH - 2433281.5,
        0.0,  # bstar
        0.0,  # ndot
        0.0,  # nddot
        0.0003,  # eccentricity
        math.radians(10.0),  # argument of perigee
        math.radians(0.05),  # inclination
        math.radians(20.0 + lon_offset_deg),  # mean anomaly
        n_rev_day * 2 * math.pi / 1440.0,  # mean motion [rad/min]
        math.radians(30.0),  # RAAN
    )
    return sat


def test_gmst_matches_vallado_example_3_5():
    # Vallado, Fundamentals of Astrodynamics: 20 Aug 1992 12:14 UT1 -> 152.578787886 deg
    assert gmst_deg(2448855.009722) == pytest.approx(152.578787886, abs=1e-4)


def test_wrap180():
    assert wrap180(190.0) == pytest.approx(-170.0)
    assert wrap180(-190.0) == pytest.approx(170.0)
    assert wrap180(45.0) == pytest.approx(45.0)


def test_geo_semimajor_axis():
    df = elements_from_satrecs([make_geo_satrec()])
    assert df.loc[0, "a_km"] == pytest.approx(42164.0, abs=20.0)
    assert df.loc[0, "n_rev_day"] == pytest.approx(SIDEREAL_REV_DAY, rel=1e-9)


def test_mean_longitude_close_to_propagated_longitude():
    sat = make_geo_satrec()
    df = elements_from_satrecs([sat])
    true_lon = _true_longitude_deg(sat, sat.jdsatepoch, sat.jdsatepochF)
    assert wrap180(df.loc[0, "lon_deg"] - true_lon) == pytest.approx(0.0, abs=0.1)


@pytest.mark.parametrize("n_rev_day", [SIDEREAL_REV_DAY, 1.0010, 1.0050])
def test_drift_rate_matches_longitude_change_over_ten_days(n_rev_day):
    sat = make_geo_satrec(n_rev_day)
    drift = elements_from_satrecs([sat]).loc[0, "drift_deg_day"]
    days = 10 * 360.0 / 360.98564736629  # whole sidereal days, so daily wobble cancels
    lon0 = _true_longitude_deg(sat, sat.jdsatepoch, sat.jdsatepochF)
    lon1 = _true_longitude_deg(sat, sat.jdsatepoch, sat.jdsatepochF + days)
    observed = wrap180(lon1 - lon0) / days
    assert drift == pytest.approx(observed, abs=0.01)


def test_faster_eastward_drift_for_lower_orbit():
    # A satellite below GEO (higher mean motion) moves east relative to the ground
    slow = elements_from_satrecs([make_geo_satrec(SIDEREAL_REV_DAY)]).loc[0, "drift_deg_day"]
    fast = elements_from_satrecs([make_geo_satrec(1.0050)]).loc[0, "drift_deg_day"]
    assert fast - slow == pytest.approx((1.0050 - SIDEREAL_REV_DAY) * 360.0, rel=0.05)


def test_read_tle_text_handles_two_and_three_line_formats():
    l1, l2 = exporter.export_tle(make_geo_satrec())
    two_line = f"{l1}\n{l2}\n{l1}\n{l2}\n"
    three_line = f"SAT A\n{l1}\n{l2}\nSAT B\n{l1}\n{l2}\n"
    assert read_tle_text(two_line) == [(l1, l2), (l1, l2)]
    assert read_tle_text(three_line) == [(l1, l2), (l1, l2)]


def test_elements_from_tle_round_trip():
    sat = make_geo_satrec(lon_offset_deg=5.0)
    l1, l2 = exporter.export_tle(sat)
    df = elements_from_tle([l1], [l2])
    assert df.loc[0, "norad_id"] == 99999
    assert df.loc[0, "inc_deg"] == pytest.approx(0.05, abs=1e-4)
    assert df.loc[0, "ecc"] == pytest.approx(0.0003, abs=1e-7)
    assert np.isfinite(df.loc[0, "drift_deg_day"])
