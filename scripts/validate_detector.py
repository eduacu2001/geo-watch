"""Score the manoeuvre detector against the Fengyun ground truth (Shorten et al., 2023).

Uses the Space-Track history downloaded for each satellite (falls back to the benchmark's own
TLE file), scores only the period covered by the ground truth, and saves:
    data/processed/validation_summary.csv     one row per satellite + total
    data/processed/validation_detections.csv  every detection, flagged matched or not
    figures/validation_<satellite>.png        ground truth vs detections for one satellite

Usage:
    python scripts/validate_detector.py
    python scripts/validate_detector.py --plot Fengyun-4A --start 2019-01-01 --end 2020-12-31
"""

import argparse

import numpy as np
import pandas as pd

from geowatch import benchmark
from geowatch.config import DATA_DIR, FIGURES_DIR
from geowatch.detection import DetectorConfig, detect_manoeuvres
from geowatch.elements import elements_from_history
from geowatch.plots import quicklook
from geowatch.spacetrack_io import read_history
from geowatch.validation import match, merge_close_times

PADDING = pd.Timedelta(days=20)  # scored margin around the ground-truth period


def load_elements(name: str, norad_id: int) -> pd.DataFrame:
    try:
        return elements_from_history(read_history(norad_id))
    except FileNotFoundError:
        return benchmark.load_elements(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", default="Fengyun-4A", help="satellite to plot")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2020-12-31")
    args = parser.parse_args()

    config = DetectorConfig()
    rows, all_detections, all_lags = [], [], []
    for name, (norad_id, regime) in benchmark.SATELLITES.items():
        if regime != "GEO":
            continue
        truth = benchmark.load_manoeuvres(name)
        start, end = truth.min() - PADDING, truth.max() + PADDING
        elements = load_elements(name, norad_id)
        elements = elements[
            (elements["epoch"] >= start - 3 * PADDING) & (elements["epoch"] <= end + 3 * PADDING)
        ]
        detections = detect_manoeuvres(elements, config)
        in_period = (detections["t_after"] >= start) & (detections["t_before"] <= end)
        scored = detections[in_period].reset_index(drop=True)
        result = match(scored, truth)
        all_lags.append(result.lags_days)

        flagged = scored.assign(satellite=name, matched=scored.index.isin(result.matched.index))
        all_detections.append(flagged)
        rows.append(
            {
                "satellite": name,
                "true_manoeuvres": len(merge_close_times(truth)),
                "detections": len(scored),
                "TP": result.true_positives,
                "FP": result.false_positives,
                "FN": result.false_negatives,
                "precision": result.precision,
                "recall": result.recall,
                "F1": result.f1,
                "median_lag_days": np.median(result.lags_days) if len(result.lags_days) else np.nan,
            }
        )

        if name == args.plot:
            fig = quicklook(
                elements,
                f"{name}: ground truth vs detections",
                manoeuvres=truth,
                detections=detections,
                start=args.start,
                end=args.end,
            )
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            fig.savefig(FIGURES_DIR / f"validation_{name.lower()}.png", dpi=150)

    summary = pd.DataFrame(rows)
    tp, fp, fn = summary["TP"].sum(), summary["FP"].sum(), summary["FN"].sum()
    lags = np.concatenate(all_lags)
    total = {
        "satellite": "TOTAL",
        "true_manoeuvres": summary["true_manoeuvres"].sum(),
        "detections": summary["detections"].sum(),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "F1": 2 * tp / (2 * tp + fp + fn),
        "median_lag_days": np.median(lags),
    }
    summary = pd.concat([summary, pd.DataFrame([total])], ignore_index=True)

    out_dir = DATA_DIR / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "validation_summary.csv", index=False)
    pd.concat(all_detections).to_csv(out_dir / "validation_detections.csv", index=False)

    with pd.option_context("display.width", 160, "display.float_format", "{:.2f}".format):
        print(summary.to_string(index=False))
    print(
        f"\nDetection lag (detection minus true time): median {np.median(lags):.2f} d, "
        f"90th percentile {np.percentile(lags, 90):.2f} d"
    )
    print(f"Saved {out_dir / 'validation_summary.csv'} and figures/validation_*.png")


if __name__ == "__main__":
    main()
