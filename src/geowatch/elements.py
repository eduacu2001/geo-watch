"""Mean orbital elements and GEO-specific quantities from TLEs.

Following Shorten et al. (2023), orbit changes are tracked with the *mean* elements published
in each TLE rather than with SGP4-propagated osculating states, which add short-period terms
that look like noise (or like fake manoeuvres) when sampled once a day.

GEO quantities computed for every element set:

* ``lon_deg``   - mean geographic longitude at epoch, east positive, in [-180, 180).
* ``drift_deg_day`` - longitude drift rate (deg/day, east positive). Computed by propagating
  the element set half a sidereal day either side of its epoch with SGP4/SDP4 and differencing
  the true longitude. Spanning exactly one orbit cancels the once-per-orbit oscillation, and
  using SDP4 keeps the deep-space (resonance and luni-solar) terms. On real Fengyun-2F data
  this agrees with the drift observed between consecutive TLEs to ~0.002 deg/day, while the
  simpler J2-only estimate is biased by ~0.007 deg/day.
* ``ex, ey``    - eccentricity vector (e cos(RAAN+argp), e sin(RAAN+argp)).
* ``ix_deg, iy_deg`` - inclination vector (i cos RAAN, i sin RAAN).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sgp4.api import Satrec

J2000_JD = 2451545.0
UNIX_EPOCH_JD = 2440587.5
EARTH_ROTATION_DEG_PER_DAY = 360.98564736629  # GMST rate, degrees per UT1 day (IAU-82)
SIDEREAL_DAY = 360.0 / EARTH_ROTATION_DEG_PER_DAY  # in solar days (~0.99727)
GEO_MAX_MEAN_MOTION = 1.5  # rev/day; GEO quantities are left empty for faster orbits


def gmst_deg(jd_ut1: float | np.ndarray) -> float | np.ndarray:
    """Greenwich Mean Sidereal Time in degrees, IAU-82 model (Vallado, eq. 3-45).

    UTC is used in place of UT1 throughout the project; the difference (<0.9 s) is below
    0.004 deg of longitude, far smaller than TLE accuracy.
    """
    d = np.asarray(jd_ut1, dtype=float) - J2000_JD
    t = d / 36525.0
    gmst = 280.46061837 + EARTH_ROTATION_DEG_PER_DAY * d + 0.000387933 * t**2 - t**3 / 38710000.0
    return np.mod(gmst, 360.0)


def wrap180(deg: float | np.ndarray) -> float | np.ndarray:
    """Wrap angles to [-180, 180)."""
    return (np.asarray(deg) + 180.0) % 360.0 - 180.0


def read_tle_text(text: str) -> list[tuple[str, str]]:
    """Extract (line1, line2) pairs from 2-line or 3-line (named) TLE text."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(lines) - 1:
        if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            pairs.append((lines[i], lines[i + 1]))
            i += 2
        else:  # name line of a 3LE, or anything else
            i += 1
    return pairs


def _true_longitude_deg(sat: Satrec, jd: float, fr: float) -> float:
    """Geographic longitude of the SGP4 position (TEME rotated by GMST)."""
    err, r, _ = sat.sgp4(jd, fr)
    if err != 0:
        return np.nan
    return float(np.degrees(np.arctan2(r[1], r[0])) - gmst_deg(jd + fr))


def _drift_rate_deg_day(sat: Satrec) -> float:
    """Longitude drift rate from SGP4 positions half a sidereal day either side of epoch."""
    half = SIDEREAL_DAY / 2.0
    lon_before = _true_longitude_deg(sat, sat.jdsatepoch, sat.jdsatepochF - half)
    lon_after = _true_longitude_deg(sat, sat.jdsatepoch, sat.jdsatepochF + half)
    return float(wrap180(lon_after - lon_before) / SIDEREAL_DAY)


def elements_from_satrecs(sats: Iterable[Satrec], sort: bool = True) -> pd.DataFrame:
    """Build a table of mean elements and GEO quantities, one row per element set.

    Rows are sorted by epoch unless ``sort=False`` (then they keep the input order).
    """
    sats = list(sats)
    if not sats:
        return pd.DataFrame()

    jd_whole = np.array([s.jdsatepoch for s in sats])
    jd_frac = np.array([s.jdsatepochF for s in sats])
    raan = np.array([s.nodeo for s in sats])
    argp = np.array([s.argpo for s in sats])
    mean_anom = np.array([s.mo for s in sats])
    inc = np.array([s.inclo for s in sats])
    ecc = np.array([s.ecco for s in sats])
    n_rev_day = np.array([s.no_kozai for s in sats]) * 1440.0 / (2.0 * np.pi)

    seconds = ((jd_whole - UNIX_EPOCH_JD) + jd_frac) * 86400.0
    epoch = pd.to_datetime(seconds, unit="s").round("ms")

    is_geo_like = n_rev_day < GEO_MAX_MEAN_MOTION
    lon = wrap180(np.degrees(raan + argp + mean_anom) - gmst_deg(jd_whole + jd_frac))
    drift = np.array(
        [
            _drift_rate_deg_day(s) if geo else np.nan
            for s, geo in zip(sats, is_geo_like, strict=True)
        ]
    )

    df = pd.DataFrame(
        {
            "epoch": epoch,
            "norad_id": [int(s.satnum) for s in sats],
            "n_rev_day": n_rev_day,
            "a_km": [s.a * s.radiusearthkm for s in sats],
            "ecc": ecc,
            "inc_deg": np.degrees(inc),
            "raan_deg": np.degrees(raan),
            "argp_deg": np.degrees(argp),
            "mean_anomaly_deg": np.degrees(mean_anom),
            "bstar": [s.bstar for s in sats],
            "lon_deg": np.where(is_geo_like, lon, np.nan),
            "drift_deg_day": drift,
            "ex": ecc * np.cos(raan + argp),
            "ey": ecc * np.sin(raan + argp),
            "ix_deg": np.degrees(inc) * np.cos(raan),
            "iy_deg": np.degrees(inc) * np.sin(raan),
        }
    )
    if sort:
        df = df.sort_values("epoch", kind="stable").reset_index(drop=True)
    return df


def elements_from_tle(
    line1: Sequence[str], line2: Sequence[str], sort: bool = True
) -> pd.DataFrame:
    """Mean elements and GEO quantities from TLE line pairs."""
    sats = [Satrec.twoline2rv(l1, l2) for l1, l2 in zip(line1, line2, strict=True)]
    return elements_from_satrecs(sats, sort=sort)


def elements_from_history(raw: pd.DataFrame) -> pd.DataFrame:
    """Mean elements from a raw Space-Track ``gp_history`` table (as stored by spacetrack_io).

    When the same epoch was published more than once, the most recent publication is kept.
    """
    raw = raw.dropna(subset=["TLE_LINE1", "TLE_LINE2"])
    if "GP_ID" in raw.columns:
        raw = raw.assign(_gp=pd.to_numeric(raw["GP_ID"], errors="coerce")).sort_values("_gp")
    df = elements_from_tle(raw["TLE_LINE1"].tolist(), raw["TLE_LINE2"].tolist(), sort=False)
    df["object_name"] = raw["OBJECT_NAME"].to_numpy() if "OBJECT_NAME" in raw else ""
    # Stable sort keeps publication order within an epoch, so 'last' is the latest publication
    df = df.sort_values("epoch", kind="stable").drop_duplicates(subset="epoch", keep="last")
    return df.reset_index(drop=True)
