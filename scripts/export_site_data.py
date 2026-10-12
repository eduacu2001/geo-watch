"""Export the derived data behind the website (docs/data/geowatch.json and encounters.json).

Reads what the other scripts produced - data/raw/gp_history, data/raw/geo_windows and
data/processed/*.csv - and writes only derived numbers (longitudes, heights, distances,
manoeuvre times), never TLEs: see src/geowatch/webdata.py.

Usage:
    python scripts/export_site_data.py
    python -m http.server -d docs      # then open http://localhost:8000
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from geowatch import benchmark
from geowatch.catalog import load_catalog
from geowatch.config import GEO_WINDOW_DIR, PROCESSED_DIR, PROJECT_ROOT
from geowatch.elements import elements_from_history, wrap180
from geowatch.ephemeris import ephemeris, tle_consistency
from geowatch.proximity import neighbours_at, parking_slots, relative_motion, summarize_pair
from geowatch.spacetrack_io import read_history
from geowatch.validation import match, merge_close_times
from geowatch.webdata import (
    GEO_SMA_KM,
    check_no_raw_data,
    daily_range,
    daily_series,
    daily_track,
    display_name,
    manoeuvre_records,
    relative_series,
    rounded,
    slot_records,
    unix_days,
)

SITE_DATA_DIR = PROJECT_ROOT / "docs" / "data"
V_GEO_MS = 3074.7  # orbital speed in GEO [m/s]

# Satellites that public analyses name as Luch/Olymp-K targets (Kayhan Space, 2025)
REPORTED_TARGETS = {
    25473: "Intelsat 7",
    26824: "Intelsat 901",
    27438: "Intelsat 905",
    20776: "Skynet 4C",
    39509: "Athena-Fidus",
    25071: "Astra 1G",
    40984: "Turksat 4B",
    28134: "SESAT 2",
}
REPORTED_SOURCE = (
    "https://kayhan.space/blog/tracking-the-tracker-is-russias-luch-olymp-approaching-the-"
    "end-of-its-mission/aid/34"
)
MAX_MEDIAN_KM = 150.0  # encounters shown on the site
# Encounters outside any parking slot, added by hand: (chaser, target, start, end, label)
EXTRA_ENCOUNTERS = [
    (49330, 34779, "2021-12-24", "2022-01-25", "SJ-21 docks with Beidou-2 G2 and tows it"),
]
SHORT_NAMES = {40258: "Luch/Olymp-K", 55841: "Luch/Olymp 2", 49330: "SJ-21", 62485: "SJ-25",
               34779: "Beidou-2 G2"}  # fmt: skip
TRACK_START = {34779: "2021-10-01"}  # companions: only from when their inspector flies
COUNTRY = {40258: "Russia", 55841: "Russia", 49330: "China", 62485: "China", 34779: "China"}


def median_tle_error(raw: pd.DataFrame, start, end) -> float:
    err = tle_consistency(raw, start, end)["err_km"]
    return float(err.median()) if len(err) else np.nan


# --------------------------------------------------------------------------------------
# Inspectors: daily tracks, parking slots, neighbours, manoeuvres
# --------------------------------------------------------------------------------------


def inspector_block(norad: int, name: str, role: str, note: str) -> tuple[dict, pd.DataFrame]:
    raw = read_history(norad)
    el = elements_from_history(raw)
    block = {
        "id": norad,
        "name": SHORT_NAMES.get(norad, name),
        "fullName": name,
        "role": role,
        "country": COUNTRY.get(norad, ""),
        "note": note,
        "track": daily_track(el[el["epoch"] >= pd.Timestamp(TRACK_START.get(norad, "1957"))]),
        "slots": [],
        "manoeuvres": [],
    }
    if role == "inspector":
        slots = parking_slots(el)
        path = PROCESSED_DIR / f"neighbours_{norad}.csv"
        neighbours = pd.read_csv(path) if path.exists() else pd.DataFrame()
        if len(neighbours):
            neighbours["object_name"] = neighbours["object_name"].map(display_name)
        block["slots"] = slot_records(slots, neighbours, REPORTED_TARGETS)
    path = PROCESSED_DIR / f"manoeuvres_{norad}.csv"
    if path.exists():
        man = pd.read_csv(path, parse_dates=["t_est"])
        block["manoeuvres"] = manoeuvre_records(man)
    return block, el


# --------------------------------------------------------------------------------------
# Encounters: relative motion in the target's RIC frame
# --------------------------------------------------------------------------------------


def encounter_list() -> pd.DataFrame:
    """Encounters to show: median distance under MAX_MEDIAN_KM, each pair and period once."""
    tables = [pd.read_csv(p, parse_dates=["start", "end"])
              for p in sorted(PROCESSED_DIR.glob("proximity_*.csv"))]  # fmt: skip
    if not tables:
        return pd.DataFrame()
    enc = pd.concat(tables, ignore_index=True)
    enc = enc[enc["median_range_km"] <= MAX_MEDIAN_KM]
    # SJ-21 and SJ-25 appear twice (each as the other's neighbour): keep one per period
    pair = (
        enc[["inspector", "target"]].min(axis=1).astype(str)
        + "-"
        + enc[["inspector", "target"]].max(axis=1).astype(str)
    )
    enc = enc.assign(_pair=pair, _month=enc["start"].dt.to_period("M"))
    enc = enc.sort_values("start").drop_duplicates(["_pair", "_month"], keep="last")
    return enc.drop(columns=["_pair", "_month"]).reset_index(drop=True)


def encounter_record(chaser: int, target: int, target_name: str, start, end, slot, label=""):
    chaser_raw, target_raw = read_history(chaser), read_history(target)
    rel = relative_motion(chaser_raw, target_raw, start, end, step_hours=1.0)
    if len(rel) < 24:
        return None
    sigma = math.hypot(
        median_tle_error(chaser_raw, start, end), median_tle_error(target_raw, start, end)
    )
    s = summarize_pair(rel, sigma)
    return {
        "id": f"{chaser}-{target}-{pd.Timestamp(start):%Y%m%d}",
        "chaser": chaser,
        "target": target,
        "targetName": display_name(target_name),
        "slot": None if slot is None else int(slot),
        "label": label,
        "d0": round(float(unix_days(s["start"])[0]), 3),
        "d1": round(float(unix_days(s["end"])[0]), 3),
        "median": round(s["median_range_km"], 2),
        "min": round(s["min_range_km"], 2),
        "p10": round(s["p10_range_km"], 2),
        "max": round(s["max_range_km"], 1),
        "dMin": round(float(unix_days(s["time_of_min"])[0]), 3),
        "meanR": round(s["mean_R_km"], 2),
        "meanI": round(s["mean_I_km"], 2),
        "meanC": round(s["mean_C_km"], 2),
        "days50": round(s["days_within_50km"], 1),
        "days100": round(s["days_within_100km"], 1),
        "sigma": None if not math.isfinite(sigma) else round(sigma, 2),
        "reported": target in REPORTED_TARGETS,
        "path": relative_series(rel),
        "daily": daily_range(rel),
    }


# --------------------------------------------------------------------------------------
# Detector validation (Fengyun ground truth)
# --------------------------------------------------------------------------------------


def validation_block() -> dict:
    summary = pd.read_csv(PROCESSED_DIR / "validation_summary.csv")
    detections = pd.read_csv(
        PROCESSED_DIR / "validation_detections.csv", parse_dates=["t_before", "t_after", "t_est"]
    )
    out = {"satellites": [], "summary": [], "comparison": []}
    for r in summary.itertuples():
        out["summary"].append({
            "name": r.satellite, "truth": int(r.true_manoeuvres), "found": int(r.TP),
            "detections": int(r.detections), "fp": int(r.FP), "fn": int(r.FN),
            "precision": round(r.precision, 3), "recall": round(r.recall, 3),
            "f1": round(r.F1, 3), "lag": round(r.median_lag_days, 2),
        })  # fmt: skip
    path = PROCESSED_DIR / "detector_comparison.csv"
    if path.exists():
        for r in pd.read_csv(path).itertuples():
            out["comparison"].append({
                "detector": r.detector, "found": int(r.TP), "fp": int(r.FP), "fn": int(r.FN),
                "precision": round(r.precision, 3), "recall": round(r.recall, 3),
                "f1": round(r.F1, 3), "lag": round(r.median_lag_days, 2),
            })  # fmt: skip

    for name, (norad, regime) in benchmark.SATELLITES.items():
        if regime != "GEO":
            continue
        det = detections[detections["satellite"] == name].reset_index(drop=True)
        truth = benchmark.load_manoeuvres(name)
        result = match(det, truth)
        merged = merge_close_times(truth)
        found = ~merged.isin(result.missed)
        try:
            el = elements_from_history(read_history(norad))
        except FileNotFoundError:
            el = benchmark.load_elements(name)
        start, end = merged.min() - pd.Timedelta(days=20), merged.max() + pd.Timedelta(days=20)
        el = el[(el["epoch"] >= start) & (el["epoch"] <= end)]
        el = el[(el["n_rev_day"] > 0.9) & (el["n_rev_day"] < 1.1)]
        out["satellites"].append(
            {
                "name": name,
                "drift": daily_series(el["epoch"], el["drift_deg_day"], 3),
                "truth": [
                    {"d": round(float(d), 3), "found": bool(f)}
                    for d, f in zip(unix_days(merged), found, strict=True)
                ],
                "detections": [
                    {"d": round(float(d), 3), "ok": bool(m), "k": t, "dv": round(float(v), 2)}
                    for d, m, t, v in zip(
                        unix_days(det["t_est"]),
                        det["matched"],
                        det["type"],
                        det["dv_total_ms"],
                        strict=True,
                    )
                ],
            }
        )
    return out


# --------------------------------------------------------------------------------------
# The GEO belt today, and where each inspector is now
# --------------------------------------------------------------------------------------


def latest_window():
    files = sorted(GEO_WINDOW_DIR.glob("geo_*.parquet"))
    if not files:
        return None, None
    path = files[-1]
    day = pd.Timestamp(datetime.strptime(path.stem[4:], "%Y%m%d")) + pd.Timedelta(hours=12)
    return pd.read_parquet(path), day


def belt_block(raw: pd.DataFrame, when: pd.Timestamp) -> dict:
    el = elements_from_history(raw)
    el = el[(el["n_rev_day"] > 0.9) & (el["n_rev_day"] < 1.1)].dropna(subset=["lon_deg"])
    el = el.assign(_dt=(el["epoch"] - when).abs()).sort_values("_dt")
    el = el.drop_duplicates("norad_id")
    # Propagate each object's nearest set to the same instant so the longitudes line up
    lon, dh = [], []
    for _, r in el.iterrows():
        lon.append(wrap180(r["lon_deg"] + r["drift_deg_day"] * (when - r["epoch"]).total_seconds()
                           / 86400))  # fmt: skip
        dh.append(r["a_km"] - GEO_SMA_KM)
    types = raw.drop_duplicates("NORAD_CAT_ID").set_index("NORAD_CAT_ID")["OBJECT_TYPE"]
    types.index = types.index.astype(int)
    kind = el["norad_id"].map(types).fillna("UNKNOWN").str[0].tolist()
    return {"d": round(float(unix_days(when)[0]), 3), "lon": rounded(lon, 2),
            "dh": rounded(dh, 0), "type": kind}  # fmt: skip


def now_block(raw_window: pd.DataFrame, when: pd.Timestamp, inspectors: dict) -> list[dict]:
    """Position of each inspector at ``when`` and the three closest catalogued objects."""
    belt = elements_from_history(raw_window)
    names = raw_window.drop_duplicates("NORAD_CAT_ID").set_index("NORAD_CAT_ID")["OBJECT_NAME"]
    names.index = names.index.astype(int)
    rows = []
    for norad, el in inspectors.items():
        own_raw = read_history(norad)
        r0, _, age = ephemeris(own_raw, [when])
        last = el[el["epoch"] <= when].iloc[-1]
        lon_now = wrap180(last["lon_deg"] + last["drift_deg_day"]
                          * (when - last["epoch"]).total_seconds() / 86400)  # fmt: skip
        near = neighbours_at(belt, float(lon_now), when, max_offset_deg=2.0, exclude=(norad,))
        closest = []
        for cand in near["norad_id"].head(25):
            cand_raw = raw_window[raw_window["NORAD_CAT_ID"].astype(int) == int(cand)]
            rc, _, _ = ephemeris(cand_raw, [when])
            dist = float(np.linalg.norm(rc[0] - r0[0]))
            if math.isfinite(dist):
                closest.append({"id": int(cand), "name": display_name(names.get(int(cand), "")),
                                "km": round(dist, 1)})  # fmt: skip
        closest = sorted(closest, key=lambda c: c["km"])[:3]
        rows.append({
            "id": norad, "d": round(float(unix_days(when)[0]), 3),
            "lon": round(float(lon_now), 2), "drift": round(float(last["drift_deg_day"]), 3),
            "dh": round(float(last["a_km"] - GEO_SMA_KM)), "inc": round(float(last["inc_deg"]), 2),
            "lastSet": round(float(unix_days(last["epoch"])[0]), 3), "closest": closest,
        })  # fmt: skip
    return rows


# --------------------------------------------------------------------------------------
# Headline numbers used in the text (refreshed with the data)
# --------------------------------------------------------------------------------------


def plane_change(el: pd.DataFrame, before: tuple[str, str], after: tuple[str, str]) -> dict:
    """Angle between the mean orbit planes of two periods, and the delta-v it takes."""

    def normal(period):
        sel = el[(el["epoch"] >= period[0]) & (el["epoch"] <= period[1])]
        i, node = np.radians(sel["inc_deg"].mean()), np.radians(sel["raan_deg"].mean())
        return np.array([np.sin(i) * np.sin(node), -np.sin(i) * np.cos(node), np.cos(i)]), sel

    n1, s1 = normal(before)
    n2, s2 = normal(after)
    angle = float(np.degrees(np.arccos(np.clip(n1 @ n2, -1, 1))))
    return {"incBefore": round(float(s1["inc_deg"].mean()), 1),
            "incAfter": round(float(s2["inc_deg"].mean()), 1),
            "angle": round(angle, 1),
            "dv": round(2 * V_GEO_MS * math.sin(math.radians(angle) / 2), -1)}  # fmt: skip


def facts_block(blocks: dict, elements: dict, encounters: list[dict], validation: dict) -> dict:
    facts = {}
    luch = blocks.get(40258)
    if luch:
        enc = [e for e in encounters if e["chaser"] == 40258 and e["median"] <= 100]
        facts["luchSlots"] = len(luch["slots"])
        facts["luchClose"] = len(enc)
        facts["luchTargets"] = len({e["target"] for e in enc})
        facts["luchStandoff"] = (
            round(float(np.median([e["median"] for e in enc])), 0) if enc else None
        )
        near_ids = {n["id"] for s in luch["slots"] for n in s["near"]}
        facts["reportedFound"] = len(set(REPORTED_TARGETS) & near_ids)
        facts["reportedTotal"] = len(REPORTED_TARGETS)
        ns = [m["d"] for m in luch["manoeuvres"] if "ns" in m["k"]]
        facts["luchLastNS"] = max(ns) if ns else None
    for e in encounters:
        if e["chaser"] == 55841 and e["target"] == 44476:
            facts["i39Days50"] = round(e["days50"])
            facts["i39Days"] = round(e["d1"] - e["d0"])
            facts["i39Median"] = round(e["median"])
            facts["i39Since"] = e["d0"]
    if 49330 in elements:
        facts["sjPlane"] = plane_change(
            elements[49330], ("2025-08-01", "2025-08-08"), ("2025-08-15", "2025-08-22")
        )
    docked = [e for e in encounters if {e["chaser"], e["target"]} == {49330, 62485}]
    if docked:
        facts["sjDockedMedian"] = min(e["median"] for e in docked)
    total = next((s for s in validation["summary"] if s["name"] == "TOTAL"), None)
    if total:
        facts["validation"] = total
    return facts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(SITE_DATA_DIR), help="output folder")
    args = parser.parse_args()

    catalog = load_catalog()
    tracked = catalog[catalog["role"].isin(["inspector"]) | (catalog["norad_id"] == 34779)]
    blocks, elements = {}, {}
    for r in tracked.itertuples():
        print(f"{r.name}: track, slots and manoeuvres")
        blocks[r.norad_id], elements[r.norad_id] = inspector_block(
            r.norad_id, r.name, r.role, r.notes
        )

    encounters = []
    for r in encounter_list().itertuples():
        print(f"encounter {SHORT_NAMES.get(r.inspector, r.inspector)} - {r.target_name}")
        rec = encounter_record(r.inspector, r.target, r.target_name, r.start, r.end, r.slot)
        if rec:
            encounters.append(rec)
    for chaser, target, start, end, label in EXTRA_ENCOUNTERS:
        print(f"encounter {label}")
        rec = encounter_record(chaser, target, SHORT_NAMES[target], start, end, None, label)
        if rec:
            encounters.append(rec)
    encounters.sort(key=lambda e: (e["chaser"], e["d0"]))

    print("validation")
    validation = validation_block()

    raw_window, when = latest_window()
    belt, now = None, []
    if raw_window is not None:
        print(f"GEO belt on {when:%Y-%m-%d}")
        belt = belt_block(raw_window, when)
        insp = {k: v for k, v in elements.items() if blocks[k]["role"] == "inspector"}
        now = now_block(raw_window, when, insp)

    last_epoch = max(el["epoch"].max() for el in elements.values())
    site = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "dataUntil": round(float(unix_days(last_epoch)[0]), 3),
        "objects": list(blocks.values()),
        "reported": {
            "source": REPORTED_SOURCE,
            "targets": [{"id": k, "name": v} for k, v in REPORTED_TARGETS.items()],
        },
        "belt": belt,
        "now": now,
        "validation": validation,
        "facts": facts_block(blocks, elements, encounters, validation),
    }
    paths = {"geowatch.json": site, "encounters.json": {"encounters": encounters}}
    out_dir = Path(args.out)
    out_dir = out_dir if out_dir.is_absolute() else PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in paths.items():
        check_no_raw_data(data)
        text = json.dumps(data, separators=(",", ":"), allow_nan=False)
        (out_dir / filename).write_text(text, encoding="utf-8")
        print(f"wrote {out_dir / filename} ({len(text) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
