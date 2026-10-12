"""Proximity operations in GEO (phase 3).

Phase 3A - where did an inspector park, and who was nearby?

1. ``parking_slots``: periods when the inspector is station-kept at one longitude.
2. ``neighbours_at``: objects within a longitude window of the inspector on a given date,
   using a snapshot of the whole GEO belt around that date (spacetrack_io.download_geo_window).

Phase 3B (next) will compute the actual distance and relative motion against each neighbour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from geowatch.elements import wrap180

GEO_RADIUS_KM = 42164.0
KM_PER_DEG = GEO_RADIUS_KM * np.pi / 180.0  # ~736 km of arc per degree of longitude


def _geo_regime(elements: pd.DataFrame) -> pd.DataFrame:
    el = elements[(elements["n_rev_day"] > 0.9) & (elements["n_rev_day"] < 1.1)]
    el = el.dropna(subset=["lon_deg", "drift_deg_day"])
    return el.sort_values("epoch").reset_index(drop=True)


def _circular_median_deg(lon: np.ndarray) -> float:
    unwrapped = np.degrees(np.unwrap(np.radians(lon)))
    return float(wrap180(np.median(unwrapped)))


def parking_slots(
    elements: pd.DataFrame,
    max_drift: float = 0.02,
    min_days: float = 5.0,
    max_gap_days: float = 3.0,
    merge_lon_deg: float = 0.3,
    merge_gap_days: float = 30.0,
) -> pd.DataFrame:
    """Periods when the object is parked (station-kept) at one longitude.

    1. An element set is "parked" when the drift rate, smoothed with a 5-set rolling median,
       is below ``max_drift`` deg/day. Consecutive parked sets form a run; runs break at TLE
       gaps longer than ``max_gap_days`` and are kept if they last ``min_days``.
    2. Station-keeping burns briefly raise the drift and split a stay into several runs, so
       runs less than ``merge_lon_deg`` apart in longitude and ``merge_gap_days`` apart in
       time are merged into one slot.
    """
    columns = ["start", "end", "days", "lon_deg", "lon_span_deg", "n_sets"]
    el = _geo_regime(elements)
    if el.empty:
        return pd.DataFrame(columns=columns)
    smooth = el["drift_deg_day"].rolling(5, center=True, min_periods=1).median().abs()
    parked = (smooth < max_drift).to_numpy()
    gap_after = np.append(el["epoch"].diff().dt.total_seconds().to_numpy()[1:] / 86400, np.inf)

    runs, start = [], None
    for i in range(len(el)):
        if parked[i]:
            if start is None:
                start = i
            if gap_after[i] > max_gap_days:  # the next set is too far away: close the run
                runs.append((start, i))
                start = None
        elif start is not None:
            runs.append((start, i - 1))
            start = None

    slots: list[dict] = []
    for a, b in runs:
        seg = el.iloc[a : b + 1]
        lon = _circular_median_deg(seg["lon_deg"].to_numpy())
        run = {"start": seg["epoch"].iloc[0], "end": seg["epoch"].iloc[-1], "lon_deg": lon,
               "idx": list(range(a, b + 1))}  # fmt: skip
        if slots:
            prev = slots[-1]
            close_in_lon = abs(wrap180(lon - prev["lon_deg"])) < merge_lon_deg
            close_in_time = (run["start"] - prev["end"]).total_seconds() / 86400 < merge_gap_days
            if close_in_lon and close_in_time:
                prev["end"] = run["end"]
                prev["idx"] += run["idx"]
                prev["lon_deg"] = _circular_median_deg(el["lon_deg"].iloc[prev["idx"]].to_numpy())
                continue
        slots.append(run)

    rows = []
    for s in slots:
        days = (s["end"] - s["start"]).total_seconds() / 86400
        if days < min_days:
            continue
        lon = np.degrees(np.unwrap(np.radians(el["lon_deg"].iloc[s["idx"]].to_numpy())))
        rows.append(
            {
                "start": s["start"],
                "end": s["end"],
                "days": days,
                "lon_deg": s["lon_deg"],
                "lon_span_deg": float(lon.max() - lon.min()),
                "n_sets": len(s["idx"]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def nearest_set(elements: pd.DataFrame, when: pd.Timestamp) -> pd.DataFrame:
    """For each object, the element set whose epoch is closest to ``when``."""
    el = elements.assign(_dt=(elements["epoch"] - when).abs())
    idx = el.groupby("norad_id")["_dt"].idxmin()
    return el.loc[idx].drop(columns="_dt").reset_index(drop=True)


def neighbours_at(
    belt: pd.DataFrame,
    lon_deg: float,
    when: pd.Timestamp,
    max_offset_deg: float = 0.5,
    exclude: tuple[int, ...] = (),
) -> pd.DataFrame:
    """Objects of a GEO-belt snapshot within ``max_offset_deg`` of ``lon_deg`` at ``when``.

    ``belt`` is a table from geowatch.elements (one or more element sets per object). The
    result has the longitude offset (east positive) and the equivalent arc length in km,
    nearest first.
    """
    sets = nearest_set(_geo_regime(belt), when)
    sets = sets[~sets["norad_id"].isin(exclude)]
    offset = wrap180(sets["lon_deg"].to_numpy() - lon_deg)
    near = sets.assign(lon_offset_deg=offset, arc_km=np.abs(offset) * KM_PER_DEG)
    near = near[np.abs(near["lon_offset_deg"]) <= max_offset_deg]
    cols = ["norad_id", "object_name", "epoch", "lon_deg", "lon_offset_deg", "arc_km",
            "drift_deg_day", "inc_deg"]  # fmt: skip
    cols = [c for c in cols if c in near.columns]
    return near.sort_values("arc_km")[cols].reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Phase 3B - distance and relative motion
# --------------------------------------------------------------------------------------


def ric_basis(r, v):
    """Rotation matrix from the inertial (TEME) frame to the RIC frame of an object.

    Rows: R = r/|r| (radial), I = C x R (in-track), C = (r x v)/|r x v| (cross-track), so
    that M @ vec gives a vector's radial, in-track and cross-track components. Works for
    single vectors (3,) -> (3, 3) and for arrays (n, 3) -> (n, 3, 3).
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)

    radial = r / np.linalg.norm(r, axis=-1, keepdims=True)
    h = np.cross(r, v)
    cross = h / np.linalg.norm(h, axis=-1, keepdims=True)
    in_track = np.cross(cross, radial)

    # Stack along the second-to-last axis: rows are R, I, C
    return np.stack([radial, in_track, cross], axis=-2)


def to_ric(r_chaser, r_target, v_target):
    """Position of the chaser relative to the target, in the target's RIC frame [km].

    Works for single vectors (3,) and for arrays (n, 3), each row with its own rotation.
    """
    r_chaser = np.asarray(r_chaser, dtype=float)
    r_target = np.asarray(r_target, dtype=float)
    v_target = np.asarray(v_target, dtype=float)

    M = ric_basis(r_target, v_target)  # (3, 3) or (n, 3, 3)
    rel = r_chaser - r_target  # (3,) or (n, 3)

    # Matrix-vector product that works for both shapes
    return np.einsum("...ij,...j->...i", M, rel)


def relative_motion(
    chaser_raw: pd.DataFrame,
    target_raw: pd.DataFrame,
    start,
    end,
    step_hours: float = 1.0,
) -> pd.DataFrame:
    """Chaser position relative to the target (target RIC frame) on a regular time grid.

    Both objects are propagated with SGP4 from their latest element set at each time.
    Columns: time, R_km, I_km, C_km, range_km, and the age of the element sets used.
    """
    from geowatch.ephemeris import ephemeris

    times = pd.date_range(
        pd.Timestamp(start), pd.Timestamp(end), freq=pd.Timedelta(hours=step_hours)
    )
    r_c, _, age_c = ephemeris(chaser_raw, times)
    r_t, v_t, age_t = ephemeris(target_raw, times)
    ric = to_ric(r_c, r_t, v_t)
    out = pd.DataFrame(
        {
            "time": times,
            "R_km": ric[:, 0],
            "I_km": ric[:, 1],
            "C_km": ric[:, 2],
            "range_km": np.linalg.norm(r_c - r_t, axis=1),
            "age_chaser_days": age_c,
            "age_target_days": age_t,
        }
    )
    return out.dropna(subset=["range_km"]).reset_index(drop=True)


def summarize_pair(rel: pd.DataFrame, sigma_km: float = np.nan) -> dict:
    """Headline numbers of one inspector-target encounter.

    ``sigma_km`` is the combined TLE position uncertainty of the two objects (see
    ephemeris.tle_consistency); distances are only meaningful down to roughly that level.
    """
    if rel.empty:
        return {}
    step_days = rel["time"].diff().dt.total_seconds().median() / 86400
    rng = rel["range_km"]
    return {
        "start": rel["time"].iloc[0],
        "end": rel["time"].iloc[-1],
        "min_range_km": float(rng.min()),
        "p10_range_km": float(rng.quantile(0.10)),
        "median_range_km": float(rng.median()),
        "max_range_km": float(rng.max()),
        "time_of_min": rel["time"].iloc[int(rng.to_numpy().argmin())],
        "mean_R_km": float(rel["R_km"].mean()),
        "mean_I_km": float(rel["I_km"].mean()),
        "mean_C_km": float(rel["C_km"].mean()),
        "days_within_50km": float((rng < 50).sum() * step_days),
        "days_within_100km": float((rng < 100).sum() * step_days),
        "sigma_km": float(sigma_km),
    }
