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
