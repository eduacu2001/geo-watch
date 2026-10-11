"""Compare the two manoeuvre detectors on the Fengyun ground truth and on the tracked objects.

    phase 2a: robust two-window change detection (geowatch.detection)
    phase 2b: Kalman filters with a NIS test       (geowatch.kalman)

Both are scored with exactly the same matching rules (geowatch.validation). The script also
prints the mean NIS of the Kalman filters: for a consistent filter it should be close to 2.

Usage:
    python scripts/compare_detectors.py
"""

import numpy as np
import pandas as pd

from geowatch import benchmark
from geowatch.catalog import load_catalog
from geowatch.config import DATA_DIR
from geowatch.detection import detect_manoeuvres
from geowatch.elements import elements_from_history
from geowatch.kalman import detect_manoeuvres_kf
from geowatch.spacetrack_io import read_history
from geowatch.validation import match

PADDING = pd.Timedelta(days=20)


def load_elements(norad_id: int, name: str | None = None) -> pd.DataFrame:
    try:
        return elements_from_history(read_history(norad_id))
    except FileNotFoundError:
        if name is None:
            raise
        return benchmark.load_elements(name)


def score(detector, with_nis: bool = False):
    totals = {"TP": 0, "FP": 0, "FN": 0}
    lags, nis_ew, nis_ns = [], [], []
    for name, (norad_id, regime) in benchmark.SATELLITES.items():
        if regime != "GEO":
            continue
        truth = benchmark.load_manoeuvres(name)
        start, end = truth.min() - PADDING, truth.max() + PADDING
        el = load_elements(norad_id, name)
        el = el[(el["epoch"] >= start - 3 * PADDING) & (el["epoch"] <= end + 3 * PADDING)]
        if with_nis:
            det, nis = detector(el, return_nis=True)
            nis_ew.append(nis["ew"])
            nis_ns.append(nis["ns"])
        else:
            det = detector(el)
        in_period = (det["t_after"] >= start) & (det["t_before"] <= end)
        result = match(det[in_period].reset_index(drop=True), truth)
        totals["TP"] += result.true_positives
        totals["FP"] += result.false_positives
        totals["FN"] += result.false_negatives
        lags.extend(result.lags_days)
    tp, fp, fn = totals["TP"], totals["FP"], totals["FN"]
    row = {
        **totals,
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "F1": 2 * tp / (2 * tp + fp + fn),
        "median_lag_days": float(np.median(lags)),
    }
    if with_nis:
        row["mean_NIS_ew"] = float(np.mean(np.concatenate(nis_ew)))
        row["mean_NIS_ns"] = float(np.mean(np.concatenate(nis_ns)))
    return row


def main() -> None:
    summary = pd.DataFrame(
        [
            {"detector": "2a two-window", **score(detect_manoeuvres)},
            {"detector": "2b Kalman", **score(detect_manoeuvres_kf, with_nis=True)},
        ]
    )
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print("Fengyun ground truth (same matching rules for both):")
        print(summary.to_string(index=False))

    rows = []
    for obj in load_catalog().itertuples():
        if obj.role == "validation":
            continue
        try:
            el = load_elements(obj.norad_id)
        except FileNotFoundError:
            continue
        a, b = detect_manoeuvres(el), detect_manoeuvres_kf(el)
        rows.append(
            {
                "object": obj.name,
                "2a manoeuvres": len(a),
                "2b manoeuvres": len(b),
                "2a dv [m/s]": a["dv_total_ms"].sum(),
                "2b dv [m/s]": b["dv_total_ms"].sum(),
            }
        )
    if rows:
        with pd.option_context("display.width", 160, "display.float_format", "{:.0f}".format):
            print("\nTracked objects (no ground truth - where do the detectors disagree?):")
            print(pd.DataFrame(rows).to_string(index=False))

    out = DATA_DIR / "processed"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "detector_comparison.csv", index=False)


if __name__ == "__main__":
    main()
