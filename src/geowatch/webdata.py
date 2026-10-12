"""Turn the pipeline's results into the compact JSON that the website reads (docs/data/).

Only derived quantities leave this module: longitudes, heights, distances and manoeuvre
times. No TLE lines or other raw Space-Track records are written, in line with the
Space-Track user agreement (``check_no_raw_data`` enforces it before anything is saved).

Times are written as days since 1970-01-01 (Unix days), which JavaScript turns into a date
with ``new Date(days * 864e5)``.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

GEO_SMA_KM = 42164.17  # semi-major axis of a geosynchronous orbit
UNIX_EPOCH = pd.Timestamp("1970-01-01")
_TLE_LINE = re.compile(r"^[12] [0-9 ]{5}[A-Z ] ")


_ACRONYMS = {"ABS", "AEHF", "AM", "AMC", "DEB", "DSCS", "GE", "GOES", "GSAT", "INSAT", "IRNSS",
             "JCSAT", "KA", "MEV", "MUOS", "NSS", "PAS", "SAT", "SBIRS", "SES", "SESAT", "SJ", "SL",
             "SMS", "TDRS", "TJS", "USA", "VHTS", "WGS", "R/B", "LINUSS", "COMSATBW"}  # fmt: skip
_DISPLAY_OVERRIDES = {
    "EUTE KA SAT 9A (KA SAT)": "Eutelsat KA-SAT 9A",
    "NIGCOMSAT 1R": "NigComSat-1R",
}


def display_name(name: str) -> str:
    """Catalogue name in reading case: 'INTELSAT 33E' -> 'Intelsat 33E'.

    Tokens with digits and known acronyms keep their capitals; 'EUTE' is spelled out.
    """
    name = str(name).strip()
    if name in _DISPLAY_OVERRIDES:
        return _DISPLAY_OVERRIDES[name]

    def word_case(word: str) -> str:
        core = word.strip("()")
        if not core or any(ch.isdigit() for ch in core) or core.split("-")[0] in _ACRONYMS:
            return word
        return word.replace(core, core.capitalize())

    words = name.replace("EUTE ", "EUTELSAT ").split()
    return " ".join("/".join(word_case(part) for part in w.split("/")) for w in words)


def unix_days(times) -> np.ndarray:
    """Days since 1970-01-01 for a timestamp or a sequence of timestamps."""
    times = pd.DatetimeIndex(np.atleast_1d(pd.to_datetime(times)))
    return (times - UNIX_EPOCH).total_seconds().to_numpy() / 86400.0


def rounded(values, digits: int) -> list:
    """Values rounded for JSON, with NaN written as null."""
    out = []
    for v in np.asarray(values, dtype=float):
        if not math.isfinite(v):
            out.append(None)
        elif digits == 0:
            out.append(int(round(v)))
        else:
            out.append(round(float(v), digits))
    return out


def daily_track(elements: pd.DataFrame) -> dict:
    """One sample per day (the day's last element set) of longitude, height and inclination.

    Only element sets in the GEO regime (0.9-1.1 rev/day) are used. Days without any element
    set are written as null, so the arrays form a regular daily grid starting at ``d0``.
    ``dh`` is the height of the orbit above the geosynchronous one, in km.
    """
    el = elements[(elements["n_rev_day"] > 0.9) & (elements["n_rev_day"] < 1.1)]
    el = el.dropna(subset=["lon_deg"]).sort_values("epoch")
    if el.empty:
        return {"d0": None, "lon": [], "dh": [], "inc": [], "drift": []}
    daily = el.groupby(el["epoch"].dt.floor("D")).last()
    days = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(days)
    return {
        "d0": int(unix_days(days[0])[0]),
        "lon": rounded(daily["lon_deg"], 2),
        "dh": rounded(daily["a_km"] - GEO_SMA_KM, 0),
        "inc": rounded(daily["inc_deg"], 2),
        "drift": rounded(daily["drift_deg_day"], 3),
    }


def daily_series(times, values, digits: int, how: str = "median") -> dict:
    """A value per day (median by default) on a regular daily grid starting at ``d0``."""
    s = pd.Series(np.asarray(values, dtype=float), index=pd.DatetimeIndex(times)).dropna()
    if s.empty:
        return {"d0": None, "v": []}
    daily = getattr(s.groupby(s.index.floor("D")), how)()
    days = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return {"d0": int(unix_days(days[0])[0]), "v": rounded(daily.reindex(days), digits)}


def decimation_step(n: int, max_points: int, steps_per_day: int = 24) -> int:
    """Stride that keeps about ``max_points`` of ``n`` samples taken ``steps_per_day`` a day.

    Strides that divide a day are preferred, so every day is sampled at the same hours and the
    daily relative-motion cycle does not alias into a false beat. Series too long for that
    keep two samples a day and may exceed ``max_points``.
    """
    need = max(1, math.ceil(n / max_points))
    divisors = [k for k in range(1, steps_per_day) if steps_per_day % k == 0]
    return next((k for k in divisors if k >= need), divisors[-1])


def relative_series(rel: pd.DataFrame, max_points: int = 800) -> dict:
    """Decimated RIC path of one encounter (output of proximity.relative_motion, hourly).

    Distances are kept to 10 m for close pairs (median range under 5 km) and to 100 m
    otherwise, far below the 1-5 km TLE uncertainty in both cases.
    """
    if rel.empty:
        return {"t0": None, "dt_h": None, "R": [], "I": [], "C": []}
    step_h0 = rel["time"].diff().dt.total_seconds().median() / 3600.0
    stride = decimation_step(len(rel), max_points, steps_per_day=max(1, round(24 / step_h0)))
    sub = rel.iloc[::stride]
    digits = 2 if rel["range_km"].median() < 5 else 1
    return {
        "t0": round(float(unix_days(sub["time"].iloc[0])[0]), 4),
        "dt_h": round(float(step_h0 * stride), 3),
        "R": rounded(sub["R_km"], digits),
        "I": rounded(sub["I_km"], digits),
        "C": rounded(sub["C_km"], digits),
    }


def daily_range(rel: pd.DataFrame) -> dict:
    """Daily minimum, mean and maximum distance of an encounter, from the full series."""
    if rel.empty:
        return {"d0": None, "lo": [], "mean": [], "hi": []}
    s = rel.set_index("time")["range_km"]
    by_day = s.groupby(s.index.floor("D"))
    days = pd.date_range(s.index.min().floor("D"), s.index.max().floor("D"), freq="D")
    digits = 2 if s.median() < 5 else 1
    return {
        "d0": int(unix_days(days[0])[0]),
        "lo": rounded(by_day.min().reindex(days), digits),
        "mean": rounded(by_day.mean().reindex(days), digits),
        "hi": rounded(by_day.max().reindex(days), digits),
    }


MANOEUVRE_CODES = {
    "E-W station-keeping": "sk",
    "drift start": "start",
    "drift stop": "stop",
    "drift change": "drift",
    "N-S (inclination)": "ns",
}


def manoeuvre_code(kind: str) -> str:
    """Short code of a detection.py manoeuvre type; '+ns' marks an added plane change."""
    base, _, extra = str(kind).partition(" + ")
    code = MANOEUVRE_CODES.get(base, "other")
    return code + "+ns" if extra == "N-S" else code


def manoeuvre_records(manoeuvres: pd.DataFrame) -> list[dict]:
    """Detected manoeuvres as {d: Unix day, k: type code, dv: total delta-v in m/s}."""
    if manoeuvres.empty:
        return []
    days = unix_days(manoeuvres["t_est"])
    return [
        {"d": round(float(d), 3), "k": manoeuvre_code(t), "dv": round(float(dv), 2)}
        for d, t, dv in zip(days, manoeuvres["type"], manoeuvres["dv_total_ms"], strict=True)
    ]


def slot_records(
    slots: pd.DataFrame, neighbours: pd.DataFrame, reported: dict[int, str], top: int = 6
) -> list[dict]:
    """Parking slots of one inspector with the closest objects found next to each.

    ``neighbours`` is the table written by scripts/find_neighbours.py (column ``slot`` is the
    slot index). Each neighbour keeps its smallest along-arc separation over the snapshots.
    """
    out = []
    for k, slot in enumerate(slots.itertuples()):
        near = neighbours[neighbours["slot"] == k] if len(neighbours) else neighbours
        rows = []
        if len(near):
            best = near.sort_values("arc_km").drop_duplicates("norad_id").head(top)
            for r in best.itertuples():
                item = {"id": int(r.norad_id), "name": str(r.object_name),
                        "arc": round(float(r.arc_km), 1)}  # fmt: skip
                if int(r.norad_id) in reported:
                    item["reported"] = True
                rows.append(item)
        out.append(
            {
                "k": k,
                "d0": round(float(unix_days(slot.start)[0]), 3),
                "d1": round(float(unix_days(slot.end)[0]), 3),
                "lon": round(float(slot.lon_deg), 2),
                "near": rows,
            }
        )
    return out


def check_no_raw_data(obj) -> None:
    """Raise if anything that looks like a TLE line (raw Space-Track data) is in ``obj``."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).upper().startswith("TLE"):
                raise ValueError(f"raw field {key!r} must not be exported")
            check_no_raw_data(value)
    elif isinstance(obj, list | tuple):
        for value in obj:
            check_no_raw_data(value)
    elif isinstance(obj, str) and _TLE_LINE.match(obj):
        raise ValueError(f"TLE line found in the export: {obj[:30]}...")
