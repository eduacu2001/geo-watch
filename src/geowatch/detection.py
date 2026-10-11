"""Manoeuvre detection from a series of TLE mean elements (GEO).

Method (level 1 of the roadmap: robust change detection, no orbit model needed):

1. Signals. East-west manoeuvres change the semi-major axis, which shows up as a step in the
   longitude drift rate. North-south manoeuvres change the inclination vector (ix, iy).
2. Spike rejection. Single element sets that disagree with both neighbours, while the
   neighbours agree with each other, are bad TLEs, not manoeuvres. They are removed first.
3. Two-window jump. At every gap between consecutive element sets, a straight line is fitted
   to the ``window`` sets before and the ``window`` sets after; the jump is the difference of
   the two lines at the middle of the gap. Natural evolution (the slow drift acceleration
   caused by Earth's ellipticity, the luni-solar inclination growth) is linear over a few days
   and cancels out; a manoeuvre does not.
4. Robust significance. Each jump is divided by the local noise of the jump series itself
   (rolling median absolute deviation over ~120 element sets). Manoeuvres are rare, so the
   MAD measures the noise and not the manoeuvres.
5. A detection needs ``score > threshold`` and a physically meaningful size. Detections closer
   than ``window`` element sets or ``min_separation_days`` are merged, keeping the strongest.

Validated against ground-truth manoeuvres of five Fengyun satellites (scripts/validate_detector.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# GEO constants for converting element jumps into delta-v
GEO_SPEED_MS = 3074.66  # circular orbital speed at GEO [m/s]
GEO_MEAN_MOTION_DEG_DAY = 360.9856  # [deg/day]
# Tangential burn: d(drift)/dt = -(3/2) n da/a  and  dv = v da / (2a)  ->  dv = v / (3n) |d drift|
DV_PER_DRIFT_MS = GEO_SPEED_MS / (3.0 * GEO_MEAN_MOTION_DEG_DAY)  # ~2.84 m/s per deg/day

SIGNALS = ["drift_deg_day", "ix_deg", "iy_deg"]


@dataclass(frozen=True)
class DetectorConfig:
    window: int = 6  # element sets on each side of a candidate gap
    threshold: float = 8.0  # jump significance, in robust standard deviations
    min_drift_jump: float = 0.01  # deg/day (~3 cm/s): smaller E-W jumps are ignored
    min_inc_jump: float = 0.03  # deg (~1.6 m/s): smaller N-S jumps are ignored
    min_separation_days: float = 3.0
    noise_window: int = 121  # element sets used to estimate the local noise
    spike_threshold: float = 6.0
    drifting_deg_day: float = 0.1  # above this the satellite is drifting, not station-kept
    mean_motion_range: tuple[float, float] = (0.9, 1.1)  # rev/day: GEO regime only


def _line_fits(t: np.ndarray, y: np.ndarray, k: int):
    """Least-squares lines on every run of k consecutive samples: slope, mean t, mean y."""
    tw, yw = sliding_window_view(t, k), sliding_window_view(y, k)
    t_mean, y_mean = tw.mean(axis=1), yw.mean(axis=1)
    dt = tw - t_mean[:, None]
    slope = (dt * (yw - y_mean[:, None])).sum(axis=1) / np.maximum((dt**2).sum(axis=1), 1e-12)
    return slope, t_mean, y_mean


def _before_after(t, y, k, t_eval, skip_after=0):
    """Line predictions at t_eval[i] from the k samples before i and the k samples from i+skip."""
    n = len(t)
    before, after = np.full(n, np.nan), np.full(n, np.nan)
    if n < k + skip_after + 1:
        return before, after
    slope, t_mean, y_mean = _line_fits(t, y, k)
    i = np.arange(k, n)
    before[i] = y_mean[i - k] + slope[i - k] * (t_eval[i] - t_mean[i - k])
    j = np.arange(0, n - k + 1 - skip_after)
    after[j] = y_mean[j + skip_after] + slope[j + skip_after] * (t_eval[j] - t_mean[j + skip_after])
    return before, after


def rolling_sigma(x: np.ndarray, window: int) -> np.ndarray:
    """Robust local standard deviation: 1.4826 x rolling median absolute deviation."""
    s = pd.Series(x)
    med = s.rolling(window, center=True, min_periods=15).median()
    sigma = (s - med).abs().rolling(window, center=True, min_periods=15).median() * 1.4826
    fallback = 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x))) if np.isfinite(x).any() else 0
    return sigma.fillna(fallback).to_numpy()


def find_spikes(elements: pd.DataFrame, config: DetectorConfig | None = None) -> np.ndarray:
    """Boolean mask of isolated bad element sets."""
    config = config or DetectorConfig()
    t = _days(elements)
    bad = np.zeros(len(t), dtype=bool)
    k = max(config.window - 2, 3)
    for col in SIGNALS:
        y = elements[col].to_numpy(dtype=float)
        before, after = _before_after(t, y, k, t, skip_after=1)  # neighbours, excluding i itself
        residual = y - 0.5 * (before + after)
        step = after - before
        sigma = rolling_sigma(residual, config.noise_window)
        with np.errstate(invalid="ignore"):
            bad |= (np.abs(residual) > config.spike_threshold * sigma) & (
                np.abs(residual) > 2 * np.abs(step)
            )
    return bad


def _days(elements: pd.DataFrame) -> np.ndarray:
    return (elements["epoch"] - elements["epoch"].iloc[0]).dt.total_seconds().to_numpy() / 86400


def prepare(elements: pd.DataFrame, config: DetectorConfig | None = None) -> pd.DataFrame:
    """Element sets the detector works on: GEO regime, finite signals, spikes removed."""
    config = config or DetectorConfig()
    lo, hi = config.mean_motion_range
    el = elements[(elements["n_rev_day"] > lo) & (elements["n_rev_day"] < hi)]
    el = el.dropna(subset=SIGNALS).sort_values("epoch").reset_index(drop=True)
    if len(el) < 2 * config.window + 1:
        return el
    return el[~find_spikes(el, config)].reset_index(drop=True)


def _classify(drift_before: float, drift_after: float, ew: bool, ns: bool, cfg) -> str:
    if ns and not ew:
        return "N-S (inclination)"
    lim = cfg.drifting_deg_day
    if abs(drift_before) < lim and abs(drift_after) < lim:
        kind = "E-W station-keeping"
    elif abs(drift_before) < lim:
        kind = "drift start"
    elif abs(drift_after) < lim:
        kind = "drift stop"
    else:
        kind = "drift change"
    return kind + (" + N-S" if ns else "")


def detect_manoeuvres(elements: pd.DataFrame, config: DetectorConfig | None = None) -> pd.DataFrame:
    """Detect manoeuvres in a table produced by geowatch.elements.

    Returns one row per manoeuvre. The manoeuvre happened between ``t_before`` and
    ``t_after`` (the bracketing element-set epochs); ``t_est`` is their midpoint. TLEs react
    to a manoeuvre with a lag (median ~0.7 days on the validation set), so the true time is
    usually at or slightly before ``t_before``.
    """
    columns = [
        "t_before", "t_after", "t_est", "type", "drift_before", "drift_after",
        "d_drift_deg_day", "d_inc_deg", "dv_ew_ms", "dv_ns_ms", "dv_total_ms", "score",
    ]  # fmt: skip
    config = config or DetectorConfig()
    el = prepare(elements, config)
    k = config.window
    if len(el) < 2 * k + 1:
        return pd.DataFrame(columns=columns)

    t = _days(el)
    t_mid = np.full(len(t), np.nan)
    t_mid[1:] = 0.5 * (t[1:] + t[:-1])  # middle of the gap between i-1 and i

    jumps, levels = {}, {}
    for col in SIGNALS:
        before, after = _before_after(t, el[col].to_numpy(dtype=float), k, t_mid)
        jumps[col], levels[col] = after - before, (before, after)

    d_drift = jumps["drift_deg_day"]
    d_inc = np.hypot(jumps["ix_deg"], jumps["iy_deg"])
    z_drift = np.abs(d_drift) / np.maximum(rolling_sigma(d_drift, config.noise_window), 1e-9)
    sigma_inc = np.hypot(
        rolling_sigma(jumps["ix_deg"], config.noise_window),
        rolling_sigma(jumps["iy_deg"], config.noise_window),
    ) / np.sqrt(2)
    z_inc = d_inc / np.maximum(sigma_inc, 1e-9) / np.sqrt(2)

    with np.errstate(invalid="ignore"):
        ew = (z_drift > config.threshold) & (np.abs(d_drift) > config.min_drift_jump)
        ns = (z_inc > config.threshold) & (d_inc > config.min_inc_jump)
    score = np.fmax(np.where(ew, z_drift, 0.0), np.where(ns, z_inc, 0.0))

    # Keep the strongest candidate in each neighbourhood
    candidates = np.flatnonzero(ew | ns)
    kept: list[int] = []
    for i in candidates[np.argsort(-score[candidates])]:
        if all(abs(i - j) > k and abs(t[i] - t[j]) > config.min_separation_days for j in kept):
            kept.append(i)

    rows = []
    for i in sorted(kept):
        drift_before, drift_after = (lvl[i] for lvl in levels["drift_deg_day"])
        dv_ew = DV_PER_DRIFT_MS * abs(d_drift[i]) if ew[i] else 0.0
        dv_ns = 2 * GEO_SPEED_MS * np.sin(np.radians(d_inc[i]) / 2) if ns[i] else 0.0
        t_before, t_after = el["epoch"].iloc[i - 1], el["epoch"].iloc[i]
        rows.append(
            {
                "t_before": t_before,
                "t_after": t_after,
                "t_est": t_before + (t_after - t_before) / 2,
                "type": _classify(drift_before, drift_after, ew[i], ns[i], config),
                "drift_before": drift_before,
                "drift_after": drift_after,
                "d_drift_deg_day": d_drift[i],
                "d_inc_deg": d_inc[i],
                "dv_ew_ms": dv_ew,
                "dv_ns_ms": dv_ns,
                "dv_total_ms": float(np.hypot(dv_ew, dv_ns)),
                "score": score[i],
            }
        )
    return pd.DataFrame(rows, columns=columns)
