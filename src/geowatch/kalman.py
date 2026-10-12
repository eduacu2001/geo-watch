"""Model-based manoeuvre detection with Kalman filters (phase 2b).

Two filters run over the element sets of one object:

* **East-west: extended Kalman filter** on the longitude dynamics. State
  x = [lon (deg), drift (deg/day), bias (deg/day^2)]. The drift changes because of the J22
  tesseral harmonic (Earth's equator is slightly elliptical), which pulls GEO satellites
  towards the stable longitudes 75.07°E and 104.93°W:

        d(lon)/dt   = drift
        d(drift)/dt = a(lon) + bias,     a(lon) = -K sin(2 (lon - LON_STABLE))
        d(bias)/dt  = 0 (+ small random walk)

  ``bias`` absorbs what the J22 term misses (higher harmonics, solar pressure). The
  dynamics are non-linear in ``lon``, hence the *extended* filter. Measurements: the TLE's
  longitude and drift rate.
* **North-south: linear Kalman filter** on the inclination vector, which drifts almost
  linearly under the Sun and Moon. State [ix, iy, d(ix)/dt, d(iy)/dt]; measurements [ix, iy].

Manoeuvre test: every new element set is compared with the prediction through the
normalised innovation squared, NIS = nu^T S^-1 nu, which follows a chi-square distribution
with 2 degrees of freedom when the filter model is right. One element set above the gate
may be a bad TLE; two in a row mean the orbit really changed: a manoeuvre is declared and
the filter is restarted from the new measurement.

The measurement noise R is estimated from the data itself (rolling robust spread), because
TLE quality differs a lot between objects: a fixed R tuned on the Fengyun satellites makes
the filter fire constantly on noisier objects such as Luch/Olymp-K.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from geowatch.detection import DV_PER_DRIFT_MS, GEO_SPEED_MS, DetectorConfig, _classify
from geowatch.elements import wrap180

K_J22_DEG_DAY2 = 0.00168  # amplitude of the J22 longitude acceleration [deg/day^2]
LON_STABLE_DEG = 75.07  # stable equilibrium longitudes: 75.07°E and 75.07 - 180 = 104.93°W


@dataclass(frozen=True)
class KalmanConfig:
    # Process noise
    q_drift: float = 1e-7  # white-noise acceleration PSD on the E-W state [deg^2/day^3]
    q_bias: float = 1e-10  # random walk of the bias acceleration [deg^2/day^5]
    sigma_bias0: float = 1e-4  # prior on the bias acceleration [deg/day^2]
    q_inc: float = 1e-9  # white-noise acceleration PSD on the inclination vector [deg^2/day^3]
    max_step_days: float = 0.5  # longest integration sub-step in the E-W prediction
    # Measurement noise floors (the noise itself is estimated from the data)
    min_sigma_lon: float = 0.01  # deg
    min_sigma_drift: float = 0.002  # deg/day
    min_sigma_inc: float = 0.0015  # deg
    noise_window: int = 121  # element sets in the rolling noise estimate
    # Detection
    gate: float = 27.63  # chi-square, 2 dof, false-alarm probability 1e-6 per element set
    burn_in: int = 10  # updates before detections are allowed
    min_drift_jump: float = 0.01  # deg/day, same physical floor as phase 2a
    min_inc_jump: float = 0.03  # deg, same physical floor as phase 2a
    min_separation_days: float = 3.0
    mean_motion_range: tuple[float, float] = (0.9, 1.1)


# --------------------------------------------------------------------------------------
# Filter building blocks
# --------------------------------------------------------------------------------------


def longitude_acceleration(lon_deg):
    """J22 longitude acceleration a(lon) = -K sin(2 (lon - LON_STABLE)) [deg/day^2].

    Works for a float or a numpy array of longitudes in degrees.
    """
    # np.sin works in radians
    arg = 2.0 * np.deg2rad(lon_deg - LON_STABLE_DEG)
    return -K_J22_DEG_DAY2 * np.sin(arg)


def longitude_acceleration_gradient(lon_deg):
    """Derivative d a / d lon [1/day^2], used in the EKF Jacobian.

    The sine's argument is in degrees, so the chain rule brings a factor 2 pi/180.
    """
    arg = 2.0 * np.deg2rad(lon_deg - LON_STABLE_DEG)
    factor = 2.0 * np.pi / 180.0
    return -K_J22_DEG_DAY2 * np.cos(arg) * factor


def ew_predict(x, P, dt, cfg: KalmanConfig):
    """EKF prediction of the E-W state x = [lon, drift, bias] and its covariance over dt days.

    Dynamics: lon' = drift, drift' = a(lon) + bias, bias' = 0. Integrated in sub-steps of at
    most cfg.max_step_days (TLE gaps reach 17 days) with a second-order Taylor step; the
    covariance is propagated with the Jacobian of that step plus white-noise-acceleration
    process noise. Returns new arrays and leaves the inputs untouched.
    """
    x = np.array(x, dtype=float, copy=True)
    P = np.array(P, dtype=float, copy=True)

    n = max(1, int(np.ceil(dt / cfg.max_step_days)))
    h = dt / n
    q = cfg.q_drift
    q_bias = cfg.q_bias

    for _ in range(n):
        lon, drift, bias = x
        acc = longitude_acceleration(lon) + bias
        g = longitude_acceleration_gradient(lon)

        # 1. State: second-order Taylor step
        lon_new = lon + drift * h + acc * h * h / 2.0
        drift_new = drift + acc * h
        x = np.array([lon_new, drift_new, bias])

        # 2. Jacobian of the step, F[i, j] = d(new x_i) / d(old x_j)
        F = np.array(
            [
                [1.0 + g * h * h / 2.0, h, h * h / 2.0],
                [g * h, 1.0, h],
                [0.0, 0.0, 1.0],
            ]
        )

        # 3. Covariance with white-noise-acceleration process noise
        Q = np.array(
            [
                [q * h**3 / 3.0, q * h**2 / 2.0, 0.0],
                [q * h**2 / 2.0, q * h, 0.0],
                [0.0, 0.0, q_bias * h],
            ]
        )
        P = F @ P @ F.T + Q

    x[0] = wrap180(x[0])
    return x, P


def ns_predict(x, P, dt, cfg: KalmanConfig):
    """Propagate the N-S state (constant-velocity model, linear) by dt days."""
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    q = cfg.q_inc
    block = np.array([[q * dt**3 / 3, q * dt**2 / 2], [q * dt**2 / 2, q * dt]])
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = block
    Q[np.ix_([1, 3], [1, 3])] = block
    return F @ np.asarray(x, dtype=float), F @ np.asarray(P, dtype=float) @ F.T + Q


def kf_update(x, P, nu, H, R):
    """Kalman measurement update with a precomputed innovation nu = z - H x.

    The caller computes nu because the longitude difference has to be wrapped. The
    covariance uses the Joseph form, which stays symmetric and positive definite with
    rounding errors. Returns new arrays.
    """
    x = np.array(x, dtype=float, copy=True)
    P = np.array(P, dtype=float, copy=True)
    H = np.asarray(H, dtype=float)
    R = np.asarray(R, dtype=float)
    nu = np.asarray(nu, dtype=float)

    S = H @ P @ H.T + R
    # K = P H^T S^-1, solved as S^T K^T = (P H^T)^T instead of inverting S
    K = np.linalg.solve(S.T, (P @ H.T).T).T

    x = x + K @ nu
    i_kh = np.eye(P.shape[0]) - K @ H
    P = i_kh @ P @ i_kh.T + K @ R @ K.T
    return x, P


def nis(nu, S) -> float:
    """Normalised innovation squared nu^T S^-1 nu.

    Chi-square distributed with len(nu) degrees of freedom when the filter model is right.
    """
    nu = np.asarray(nu, dtype=float)
    S = np.asarray(S, dtype=float)
    return float(nu @ np.linalg.solve(S, nu))


# --------------------------------------------------------------------------------------
# Measurement noise from the data
# --------------------------------------------------------------------------------------


def _robust_sigma(x: np.ndarray, window: int, floor: float) -> np.ndarray:
    s = pd.Series(x)
    med = s.rolling(window, center=True, min_periods=15).median()
    sigma = (s - med).abs().rolling(window, center=True, min_periods=15).median() * 1.4826
    fallback = 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x)))
    return np.maximum(sigma.fillna(fallback).to_numpy(), floor)


def _interpolation_residuals(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """How far each sample is from the straight line through its two neighbours.

    Scaled so that, for white noise of standard deviation sigma on top of any locally linear
    trend, the result has standard deviation sigma.
    """
    w_prev = (t[2:] - t[1:-1]) / (t[2:] - t[:-2])
    w_next = 1.0 - w_prev
    residual = y[1:-1] - (w_prev * y[:-2] + w_next * y[2:])
    residual /= np.sqrt(1.0 + w_prev**2 + w_next**2)
    return np.concatenate([[residual[0]], residual, [residual[-1]]])


def measurement_noise(el: pd.DataFrame, cfg: KalmanConfig):
    """Per-element-set measurement noise of lon, drift and inclination, estimated robustly.

    Each sample is compared with the line through its neighbours (this removes trends), and
    the spread of those residuals is measured with a rolling median absolute deviation.
    """
    t = _days(el)
    if np.any(np.diff(t) <= 0):  # repeated epochs: nudge them apart to keep weights finite
        t = t + np.arange(len(t)) * 1e-9
    lon = np.degrees(np.unwrap(np.radians(el["lon_deg"].to_numpy())))
    w = cfg.noise_window

    def sigma(y, floor):
        return _robust_sigma(_interpolation_residuals(t, y), w, floor)

    s_lon = sigma(lon, cfg.min_sigma_lon)
    s_drift = sigma(el["drift_deg_day"].to_numpy(), cfg.min_sigma_drift)
    s_ix = sigma(el["ix_deg"].to_numpy(), cfg.min_sigma_inc)
    s_iy = sigma(el["iy_deg"].to_numpy(), cfg.min_sigma_inc)
    return s_lon, s_drift, np.hypot(s_ix, s_iy) / np.sqrt(2)


# --------------------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------------------


def _days(el: pd.DataFrame) -> np.ndarray:
    return (el["epoch"] - el["epoch"].iloc[0]).dt.total_seconds().to_numpy() / 86400


def run_filter(t, Z, H, R_seq, x0, P0, predict, restart, cfg, wrap_first=False):
    """Run one filter over a measurement series and return manoeuvre events and NIS values.

    For each new measurement: predict, compute the innovation and its NIS, then
      * NIS within the gate        -> normal update
      * first value above the gate -> keep it pending (could be a bad TLE), no update
      * second value in a row      -> manoeuvre at the pending set; restart the filter
    """
    x, P = np.asarray(x0, dtype=float), np.asarray(P0, dtype=float)
    events, nis_values = [], []
    pending, n_updates = None, 0
    for k in range(1, len(t)):
        x, P = predict(x, P, t[k] - t[k - 1], cfg)
        nu = Z[k] - H @ x
        if wrap_first:
            nu[0] = wrap180(nu[0])
        R = R_seq[k]
        S = H @ P @ H.T + R
        d2 = nis(nu, S)
        if d2 <= cfg.gate or n_updates < cfg.burn_in:
            pending = None
            x, P = kf_update(x, P, nu, H, R)
            n_updates += 1
            nis_values.append(d2)
        elif pending is None:
            pending = k
        else:
            events.append({"k": pending, "nu": nu.copy(), "predicted": H @ x, "nis": d2})
            x, P = restart(Z[k], x, P, R)
            pending = None
    return events, np.array(nis_values)


COLUMNS = [
    "t_before", "t_after", "t_est", "type", "drift_before", "drift_after",
    "d_drift_deg_day", "d_inc_deg", "dv_ew_ms", "dv_ns_ms", "dv_total_ms", "score",
]  # fmt: skip


def detect_manoeuvres_kf(
    elements: pd.DataFrame, config: KalmanConfig | None = None, return_nis: bool = False
):
    """Kalman-filter manoeuvre detection. Same output columns as detection.detect_manoeuvres.

    With ``return_nis=True`` also returns the NIS of every accepted update, per filter, to
    check the filter's consistency (for a 2-dof chi-square the mean should be close to 2).
    """
    cfg = config or KalmanConfig()
    lo, hi = cfg.mean_motion_range
    el = elements[(elements["n_rev_day"] > lo) & (elements["n_rev_day"] < hi)]
    el = el.dropna(subset=["lon_deg", "drift_deg_day", "ix_deg", "iy_deg"])
    el = el.sort_values("epoch").reset_index(drop=True)
    empty = pd.DataFrame(columns=COLUMNS)
    if len(el) < cfg.burn_in + 3:
        return (empty, {}) if return_nis else empty

    t = _days(el)
    s_lon, s_drift, s_inc = measurement_noise(el, cfg)

    # East-west EKF
    z_ew = el[["lon_deg", "drift_deg_day"]].to_numpy()
    H_ew = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    R_ew = np.array([np.diag([a**2, b**2]) for a, b in zip(s_lon, s_drift, strict=True)])
    x0_ew = np.array([z_ew[0, 0], z_ew[0, 1], 0.0])
    P0_ew = np.diag([s_lon[0] ** 2, s_drift[0] ** 2, cfg.sigma_bias0**2])

    def restart_ew(z, x, P, R):
        return np.array([z[0], z[1], x[2]]), np.diag([R[0, 0], R[1, 1], P[2, 2]])

    ew_events, nis_ew = run_filter(
        t, z_ew, H_ew, R_ew, x0_ew, P0_ew, ew_predict, restart_ew, cfg, wrap_first=True
    )

    # North-south linear KF
    z_ns = el[["ix_deg", "iy_deg"]].to_numpy()
    H_ns = np.hstack([np.eye(2), np.zeros((2, 2))])
    R_ns = np.array([np.eye(2) * s**2 for s in s_inc])
    x0_ns = np.array([z_ns[0, 0], z_ns[0, 1], 0.0, 0.0])
    P0_ns = np.diag([s_inc[0] ** 2] * 2 + [0.01**2] * 2)

    def restart_ns(z, x, P, R):
        return np.array([z[0], z[1], x[2], x[3]]), np.diag([R[0, 0], R[1, 1], P[2, 2], P[3, 3]])

    ns_events, nis_ns = run_filter(t, z_ns, H_ns, R_ns, x0_ns, P0_ns, ns_predict, restart_ns, cfg)

    table = _events_to_table(el, ew_events, ns_events, cfg)
    return (table, {"ew": nis_ew, "ns": nis_ns}) if return_nis else table


def _events_to_table(el, ew_events, ns_events, cfg) -> pd.DataFrame:
    raw = []
    for e in ew_events:
        if abs(e["nu"][1]) >= cfg.min_drift_jump:
            raw.append(
                {
                    "k": e["k"],
                    "ew": True,
                    "ns": False,
                    "d_drift": e["nu"][1],
                    "d_inc": 0.0,
                    "drift_before": e["predicted"][1],
                    "score": np.sqrt(e["nis"]),
                }  # fmt: skip
            )
    for e in ns_events:
        d_inc = float(np.hypot(e["nu"][0], e["nu"][1]))
        if d_inc >= cfg.min_inc_jump:
            raw.append(
                {
                    "k": e["k"],
                    "ew": False,
                    "ns": True,
                    "d_drift": 0.0,
                    "d_inc": d_inc,
                    "drift_before": np.nan,
                    "score": np.sqrt(e["nis"]),
                }  # fmt: skip
            )
    raw.sort(key=lambda r: r["k"])

    merged: list[dict] = []
    for r in raw:
        r["t_before"], r["t_after"] = el["epoch"].iloc[r["k"] - 1], el["epoch"].iloc[r["k"]]
        r["t_est"] = r["t_before"] + (r["t_after"] - r["t_before"]) / 2
        if merged:
            prev = merged[-1]
            gap_days = (r["t_est"] - prev["t_est"]).total_seconds() / 86400
            if gap_days <= cfg.min_separation_days:
                if r["ew"]:
                    prev.update(ew=True, d_drift=r["d_drift"], drift_before=r["drift_before"])
                if r["ns"]:
                    prev.update(ns=True, d_inc=r["d_inc"])
                prev["score"] = max(prev["score"], r["score"])
                continue
        merged.append(r)

    rows = []
    for r in merged:
        drift_before = r["drift_before"]
        if not np.isfinite(drift_before):
            drift_before = el["drift_deg_day"].iloc[r["k"] - 1]
        drift_after = drift_before + r["d_drift"]
        dv_ew = DV_PER_DRIFT_MS * abs(r["d_drift"]) if r["ew"] else 0.0
        dv_ns = 2 * GEO_SPEED_MS * np.sin(np.radians(r["d_inc"]) / 2) if r["ns"] else 0.0
        rows.append(
            {
                "t_before": r["t_before"],
                "t_after": r["t_after"],
                "t_est": r["t_est"],
                "type": _classify(drift_before, drift_after, r["ew"], r["ns"], DetectorConfig()),
                "drift_before": drift_before,
                "drift_after": drift_after,
                "d_drift_deg_day": r["d_drift"],
                "d_inc_deg": r["d_inc"],
                "dv_ew_ms": dv_ew,
                "dv_ns_ms": dv_ns,
                "dv_total_ms": float(np.hypot(dv_ew, dv_ns)),
                "score": r["score"],
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)
