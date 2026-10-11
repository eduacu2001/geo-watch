"""Tests for phase 2b (geowatch.kalman).

The first blocks test the five building blocks one by one; the last block runs the whole
detector on a synthetic satellite whose manoeuvres are known.
Run only these with:  pytest tests/test_kalman.py -v
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from geowatch.detection import detect_manoeuvres
from geowatch.kalman import (
    K_J22_DEG_DAY2,
    LON_STABLE_DEG,
    KalmanConfig,
    detect_manoeuvres_kf,
    ew_predict,
    kf_update,
    longitude_acceleration,
    longitude_acceleration_gradient,
    nis,
)

NO_PROCESS_NOISE = dataclasses.replace(KalmanConfig(), q_drift=0.0, q_bias=0.0)


def integrate(x0, days, step=0.001):
    """Reference solution of the longitude dynamics with a fine RK4 integration."""

    def f(x):
        return np.array([x[1], longitude_acceleration(x[0]) + x[2], 0.0])

    x = np.array(x0, dtype=float)
    n = int(round(days / step))
    for _ in range(n):
        k1 = f(x)
        k2 = f(x + 0.5 * step * k1)
        k3 = f(x + 0.5 * step * k2)
        k4 = f(x + step * k3)
        x = x + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


# --- 1. longitude_acceleration -------------------------------------------------------


@pytest.mark.parametrize("lon", [LON_STABLE_DEG, LON_STABLE_DEG - 180, LON_STABLE_DEG + 90])
def test_acceleration_is_zero_at_equilibrium_points(lon):
    assert longitude_acceleration(lon) == pytest.approx(0.0, abs=1e-12)


def test_acceleration_pulls_towards_the_stable_longitudes():
    assert longitude_acceleration(112.0) < 0  # east of 75°E -> pushed west
    assert longitude_acceleration(40.0) > 0  # west of 75°E -> pushed east
    assert longitude_acceleration(-60.0) < 0  # east of 105°W -> pushed west


def test_acceleration_amplitude():
    assert longitude_acceleration(LON_STABLE_DEG + 45) == pytest.approx(-K_J22_DEG_DAY2)
    assert longitude_acceleration(LON_STABLE_DEG - 45) == pytest.approx(K_J22_DEG_DAY2)


# --- 2. longitude_acceleration_gradient ---------------------------------------------


@pytest.mark.parametrize("lon", [-170.0, -60.0, 0.0, 30.0, 75.07, 112.0, 160.0])
def test_gradient_matches_finite_differences(lon):
    h = 1e-4
    numeric = (longitude_acceleration(lon + h) - longitude_acceleration(lon - h)) / (2 * h)
    assert longitude_acceleration_gradient(lon) == pytest.approx(numeric, rel=1e-6, abs=1e-12)


# --- 3. ew_predict ---------------------------------------------------------------------


@pytest.mark.parametrize("dt", [0.5, 3.0, 17.0])
def test_prediction_matches_fine_integration(dt):
    x0 = np.array([112.0, 0.04, -0.0004])
    x, _ = ew_predict(x0, np.diag([1e-4, 1e-6, 1e-8]), dt, KalmanConfig())
    reference = integrate(x0, dt)
    assert x[0] == pytest.approx(reference[0], abs=1e-4)
    assert x[1] == pytest.approx(reference[1], abs=1e-5)
    assert x[2] == pytest.approx(x0[2])


def test_prediction_wraps_longitude():
    x, _ = ew_predict(np.array([179.9, 0.5, 0.0]), np.eye(3) * 1e-6, 1.0, KalmanConfig())
    assert -180.0 <= x[0] < 180.0
    assert x[0] == pytest.approx(-179.6, abs=0.01)


def test_covariance_follows_the_jacobian():
    """Without process noise, P must evolve as F P F^T with F the numerical Jacobian."""
    x0, dt = np.array([112.0, 0.04, -0.0004]), 2.0
    P0 = np.diag([4e-4, 1e-6, 1e-8])
    _, P = ew_predict(x0, P0, dt, NO_PROCESS_NOISE)
    eps = np.array([1e-4, 1e-6, 1e-7])
    F = np.zeros((3, 3))
    for j in range(3):
        dx = np.zeros(3)
        dx[j] = eps[j]
        plus, _ = ew_predict(x0 + dx, P0, dt, NO_PROCESS_NOISE)
        minus, _ = ew_predict(x0 - dx, P0, dt, NO_PROCESS_NOISE)
        F[:, j] = (plus - minus) / (2 * eps[j])
    np.testing.assert_allclose(P, F @ P0 @ F.T, rtol=1e-3, atol=1e-14)


def test_process_noise_makes_uncertainty_grow():
    P0 = np.diag([1e-4, 1e-6, 1e-8])
    _, P = ew_predict(np.array([112.0, 0.04, 0.0]), P0, 5.0, KalmanConfig())
    np.testing.assert_allclose(P, P.T)
    assert np.all(np.linalg.eigvalsh(P) > 0)
    assert P[0, 0] > P0[0, 0] and P[1, 1] > P0[1, 1]


# --- 4. kf_update --------------------------------------------------------------------


def test_update_scalar_example():
    # Prior 0 +- 2, measurement 5 +- 1  ->  gain 0.8, estimate 4, variance 0.8
    x, P = kf_update(np.array([0.0]), np.array([[4.0]]), np.array([5.0]), np.eye(1), np.eye(1))
    assert x[0] == pytest.approx(4.0)
    assert P[0, 0] == pytest.approx(0.8)


def test_update_reduces_uncertainty_and_keeps_covariance_symmetric():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 3))
    P0 = A @ A.T + np.eye(3)
    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    R = np.diag([0.5, 0.2])
    _, P = kf_update(np.zeros(3), P0, np.array([1.0, -1.0]), H, R)
    np.testing.assert_allclose(P, P.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(P0 - P) > -1e-12)  # never more uncertain than before


# --- 5. nis --------------------------------------------------------------------------


def test_nis_known_values():
    assert nis(np.array([3.0, 4.0]), np.diag([9.0, 16.0])) == pytest.approx(2.0)
    S = np.array([[2.0, 0.5], [0.5, 1.0]])
    nu = np.array([1.0, -2.0])
    assert nis(nu, S) == pytest.approx(nu @ np.linalg.inv(S) @ nu)


# --- 6. the whole detector -------------------------------------------------------------

BURNS = {40.3: 0.09, 95.7: -0.08}  # day -> drift change [deg/day]
NS_BURN_DAY, NS_BURN = 70.2, -0.3  # day, change of ix [deg]
BAD_TLE_DAY = 60.0


def synthetic_satellite(seed: int = 3) -> pd.DataFrame:
    """150 days of a GEO satellite at 112°E following the J22 dynamics, with noisy TLEs."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, 150, 0.5) + rng.uniform(-0.1, 0.1, 300)
    x, last, lon, drift = np.array([112.3, -0.03, -0.0004]), 0.0, [], []
    for tk in t:
        for day, jump in BURNS.items():
            if last < day <= tk:
                x = integrate(x, day - last, step=0.05)
                x[1] += jump
                last = day
        x = integrate(x, tk - last, step=0.05)
        last = tk
        lon.append(x[0])
        drift.append(x[1])
    lon = np.array(lon) + rng.normal(0, 0.003, t.size)
    drift = np.array(drift) + rng.normal(0, 0.0008, t.size)
    drift[np.argmin(np.abs(t - BAD_TLE_DAY))] += 0.04  # one bad element set
    ix = 0.1 + 0.002 * t + rng.normal(0, 0.0005, t.size) + np.where(t > NS_BURN_DAY, NS_BURN, 0)
    iy = 0.05 + 0.001 * t + rng.normal(0, 0.0005, t.size)
    return pd.DataFrame(
        {
            "epoch": pd.Timestamp("2024-01-01") + pd.to_timedelta(t, unit="D"),
            "n_rev_day": 1.0027,
            "lon_deg": lon,
            "drift_deg_day": drift,
            "ix_deg": ix,
            "iy_deg": iy,
        }
    )


def _near(det: pd.DataFrame, day: float) -> pd.DataFrame:
    when = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)
    return det[(det["t_est"] - when).abs() <= pd.Timedelta(days=1.5)]


def test_detector_finds_the_three_burns_and_ignores_the_bad_tle():
    det = detect_manoeuvres_kf(synthetic_satellite())
    for day in [*BURNS, NS_BURN_DAY]:
        assert len(_near(det, day)) == 1, f"no detection near day {day}"
    assert _near(det, BAD_TLE_DAY).empty
    assert len(det) == 3


def test_detector_estimates_the_drift_change():
    det = detect_manoeuvres_kf(synthetic_satellite())
    for day, jump in BURNS.items():
        assert _near(det, day).iloc[0]["d_drift_deg_day"] == pytest.approx(jump, abs=0.01)


def test_filter_does_not_diverge():
    """A consistent filter has mean NIS near 2 (2-dof chi-square). The noise floors make the
    default filter conservative on this clean synthetic data (mean NIS well below 2), so
    only the upper side is checked: tens or hundreds mean the prediction or update is wrong."""
    _, nis_values = detect_manoeuvres_kf(synthetic_satellite(), return_nis=True)
    assert 0.0 < np.mean(nis_values["ew"]) < 5.0
    assert 0.0 < np.mean(nis_values["ns"]) < 5.0


def test_same_output_columns_as_phase_2a():
    el = synthetic_satellite()
    assert list(detect_manoeuvres_kf(el).columns) == list(detect_manoeuvres(el).columns)
