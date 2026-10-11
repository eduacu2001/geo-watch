import numpy as np
import pandas as pd
import pytest

from geowatch.detection import DV_PER_DRIFT_MS, GEO_SPEED_MS, detect_manoeuvres, find_spikes
from geowatch.validation import match, merge_close_times

T0 = pd.Timestamp("2024-01-01")
EW_TIMES = [60.3, 130.7]  # days
EW_JUMPS = [0.09, -0.05]  # deg/day
NS_TIME, NS_JUMP = 100.2, -0.3  # days, deg (applied to ix)
SPIKE_TIME = 80.0


def synthetic_geo(seed: int = 1) -> pd.DataFrame:
    """200 days of a station-kept GEO satellite: two E-W burns, one N-S burn, one bad TLE."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, 200, 0.5) + rng.uniform(-0.1, 0.1, 400)
    drift = -0.045 - 0.0005 * (t - 60.3) + rng.normal(0, 0.0008, t.size)
    for when, jump in zip(EW_TIMES, EW_JUMPS, strict=True):
        drift += np.where(t > when, jump, 0.0)
    ix = 0.05 + 0.002 * t + rng.normal(0, 0.0005, t.size) + np.where(t > NS_TIME, NS_JUMP, 0.0)
    iy = 0.02 + 0.001 * t + rng.normal(0, 0.0005, t.size)
    spike = np.argmin(np.abs(t - SPIKE_TIME))
    drift[spike] += 0.05  # an isolated bad element set
    return pd.DataFrame(
        {
            "epoch": T0 + pd.to_timedelta(t, unit="D"),
            "n_rev_day": 1.0027,
            "drift_deg_day": drift,
            "ix_deg": ix,
            "iy_deg": iy,
        }
    )


def _near(det: pd.DataFrame, day: float, tol: float = 1.0) -> pd.DataFrame:
    when = T0 + pd.Timedelta(days=day)
    return det[(det["t_est"] - when).abs() <= pd.Timedelta(days=tol)]


def test_detects_all_three_manoeuvres_and_nothing_else():
    det = detect_manoeuvres(synthetic_geo())
    assert len(det) == 3
    for day in [*EW_TIMES, NS_TIME]:
        assert len(_near(det, day)) == 1


def test_spike_is_rejected_not_detected():
    el = synthetic_geo()
    spikes = find_spikes(el)
    assert spikes.sum() == 1
    assert abs((el["epoch"][spikes].iloc[0] - T0).total_seconds() / 86400 - SPIKE_TIME) < 0.3
    assert _near(detect_manoeuvres(el), SPIKE_TIME, tol=3).empty


def test_delta_v_and_type_estimates():
    det = detect_manoeuvres(synthetic_geo())
    ew = _near(det, EW_TIMES[0]).iloc[0]
    assert ew["d_drift_deg_day"] == pytest.approx(EW_JUMPS[0], rel=0.1)
    assert ew["dv_ew_ms"] == pytest.approx(DV_PER_DRIFT_MS * EW_JUMPS[0], rel=0.1)
    assert ew["type"] == "E-W station-keeping"
    ns = _near(det, NS_TIME).iloc[0]
    assert ns["dv_ns_ms"] == pytest.approx(GEO_SPEED_MS * np.radians(abs(NS_JUMP)), rel=0.1)
    assert ns["type"].startswith("N-S")


def test_short_series_returns_empty_table():
    det = detect_manoeuvres(synthetic_geo().head(8))
    assert det.empty
    assert "t_est" in det.columns


def test_merge_close_times():
    times = pd.DatetimeIndex(["2020-03-18 17:30", "2020-03-19 17:16", "2020-05-01"])
    assert list(merge_close_times(times)) == [times[0], times[2]]


def test_match_window_is_asymmetric():
    truth = pd.DatetimeIndex(["2024-01-10 12:00"])

    def det_at(day: str) -> pd.DataFrame:
        t = pd.Timestamp(day)
        return pd.DataFrame({"t_before": [t], "t_after": [t], "t_est": [t]})

    late = match(det_at("2024-01-12 12:00"), truth)  # 2 days after the burn: TLE lag
    early = match(det_at("2024-01-08 12:00"), truth)  # 2 days before the burn: impossible
    assert (late.true_positives, late.false_positives) == (1, 0)
    assert (early.true_positives, early.false_positives, early.false_negatives) == (0, 1, 1)
    assert late.lags_days[0] == pytest.approx(2.0)
