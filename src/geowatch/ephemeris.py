"""Positions and velocities from a TLE history, and an estimate of how accurate they are."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sgp4.api import Satrec


def _prepare(raw: pd.DataFrame) -> tuple[pd.DatetimeIndex, list[Satrec]]:
    raw = raw.dropna(subset=["TLE_LINE1", "TLE_LINE2"])
    if "GP_ID" in raw.columns:
        raw = raw.assign(_gp=pd.to_numeric(raw["GP_ID"], errors="coerce")).sort_values("_gp")
    raw = raw.assign(_epoch=pd.to_datetime(raw["EPOCH"]))
    # keep the latest publication of each epoch, then sort by epoch
    raw = raw.drop_duplicates(subset="_epoch", keep="last").sort_values("_epoch")
    sats = [
        Satrec.twoline2rv(a, b) for a, b in zip(raw["TLE_LINE1"], raw["TLE_LINE2"], strict=True)
    ]
    return pd.DatetimeIndex(raw["_epoch"]), sats


def _julian(times: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    jd = times.to_julian_date().to_numpy()
    whole = np.floor(jd)
    return whole, jd - whole


def ephemeris(raw: pd.DataFrame, times) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TEME position [km] and velocity [km/s] at ``times`` from a raw gp_history table.

    Each time is propagated from the most recent element set published before it (the way an
    operator would have seen the object then); times before the first set use the first one.
    Also returns the age of the element set used, in days (negative before the first set).
    """
    times = pd.DatetimeIndex(times)
    epochs, sats = _prepare(raw)
    n = len(times)
    r, v, age = np.full((n, 3), np.nan), np.full((n, 3), np.nan), np.full(n, np.nan)
    if not sats:
        return r, v, age
    idx = np.clip(np.searchsorted(epochs.asi8, times.asi8, side="right") - 1, 0, len(sats) - 1)
    age = (times - epochs[idx]).total_seconds().to_numpy() / 86400
    whole, frac = _julian(times)
    for k in np.unique(idx):
        sel = idx == k
        err, rk, vk = sats[k].sgp4_array(whole[sel], frac[sel])
        rk[err != 0] = np.nan
        vk[err != 0] = np.nan
        r[sel], v[sel] = rk, vk
    return r, v, age


def tle_consistency(
    raw: pd.DataFrame, start=None, end=None, max_gap_days: float = 3.0
) -> pd.DataFrame:
    """How far each element set's prediction lands from the next one, in the next one's RIC frame.

    For consecutive sets i, i+1 (less than ``max_gap_days`` apart), set i is propagated to the
    epoch of set i+1 and compared with set i+1's own position there. The spread of these
    differences is a practical measure of TLE position uncertainty for that object.
    """
    from geowatch.proximity import ric_basis  # local import: proximity imports this module

    epochs, sats = _prepare(raw)
    rows = []
    for i in range(len(sats) - 1):
        t0, t1 = epochs[i], epochs[i + 1]
        if start is not None and t1 < pd.Timestamp(start):
            continue
        if end is not None and t1 > pd.Timestamp(end):
            break
        gap = (t1 - t0).total_seconds() / 86400
        if gap <= 0 or gap > max_gap_days:
            continue
        whole, frac = _julian(pd.DatetimeIndex([t1]))
        e0, r_pred, _ = sats[i].sgp4(whole[0], frac[0])
        e1, r_obs, v_obs = sats[i + 1].sgp4(whole[0], frac[0])
        if e0 or e1:
            continue
        d = ric_basis(np.array(r_obs), np.array(v_obs)) @ (np.array(r_pred) - np.array(r_obs))
        rows.append({"epoch": t1, "gap_days": gap, "dR_km": d[0], "dI_km": d[1], "dC_km": d[2],
                     "err_km": float(np.linalg.norm(d))})  # fmt: skip
    return pd.DataFrame(rows, columns=["epoch", "gap_days", "dR_km", "dI_km", "dC_km", "err_km"])
