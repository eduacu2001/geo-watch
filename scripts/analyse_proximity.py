"""Distance and relative motion between each inspector and its neighbours (phase 3B).

Reads data/processed/neighbours_<NORAD>.csv (from find_neighbours.py) and, for every
neighbour that came within --max-km during a parking slot, propagates both objects with SGP4
over the slot and measures the distance and the inspector's position in the target's RIC
frame. Results:
    data/processed/proximity_<NORAD>.csv        one row per inspector-target-slot
    figures/proximity_<NORAD>_<TARGET>_<SLOT>.png  for the --plot-top closest encounters

Usage:
    python scripts/analyse_proximity.py --norad 40258
    python scripts/analyse_proximity.py --plot-top 8
"""

import argparse

import numpy as np
import pandas as pd

from geowatch.catalog import load_catalog, select
from geowatch.config import FIGURES_DIR, PROCESSED_DIR
from geowatch.ephemeris import tle_consistency
from geowatch.plots import relative_motion_figure
from geowatch.proximity import relative_motion, summarize_pair
from geowatch.spacetrack_io import read_history


def median_error(raw: pd.DataFrame, start, end) -> float:
    err = tle_consistency(raw, start, end)["err_km"]
    return float(err.median()) if len(err) else np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--norad", type=int, nargs="+", help="inspector NORAD IDs")
    parser.add_argument("--max-km", type=float, default=100.0, help="arc distance in the snapshot")
    parser.add_argument("--step-hours", type=float, default=1.0)
    parser.add_argument("--plot-top", type=int, default=5, help="figures for the N closest")
    args = parser.parse_args()

    inspectors = select(
        load_catalog(), norad_ids=args.norad, role=None if args.norad else "inspector"
    )
    for insp in inspectors.itertuples():
        path = PROCESSED_DIR / f"neighbours_{insp.norad_id}.csv"
        if not path.exists():
            print(f"{path} not found - run scripts/find_neighbours.py first")
            continue
        neighbours = pd.read_csv(path, parse_dates=["slot_start", "slot_end"])
        pairs = (
            neighbours[neighbours["arc_km"] <= args.max_km]
            .sort_values("arc_km")
            .drop_duplicates(["slot", "norad_id"])
        )
        chaser = read_history(insp.norad_id)
        rows, motions = [], {}
        for p in pairs.itertuples():
            try:
                target = read_history(p.norad_id)
            except FileNotFoundError:
                continue  # run find_neighbours.py --download-targets
            rel = relative_motion(chaser, target, p.slot_start, p.slot_end, args.step_hours)
            if len(rel) < 24:
                continue
            sigma = np.hypot(
                median_error(chaser, p.slot_start, p.slot_end),
                median_error(target, p.slot_start, p.slot_end),
            )
            summary = summarize_pair(rel, sigma)
            rows.append({"inspector": insp.norad_id, "slot": p.slot, "target": p.norad_id,
                         "target_name": p.object_name, **summary})  # fmt: skip
            motions[(p.slot, p.norad_id)] = (rel, sigma)

        if not rows:
            print(f"\n{insp.name}: no encounters to analyse")
            continue
        table = pd.DataFrame(rows).sort_values("median_range_km").reset_index(drop=True)
        table.to_csv(PROCESSED_DIR / f"proximity_{insp.norad_id}.csv", index=False)

        print(f"\n{insp.name} (NORAD {insp.norad_id}): {len(table)} encounters")
        cols = ["start", "end", "target_name", "median_range_km", "min_range_km",
                "mean_I_km", "days_within_50km", "sigma_km"]  # fmt: skip
        view = table[cols].assign(start=table["start"].dt.date, end=table["end"].dt.date)
        with pd.option_context("display.width", 160, "display.float_format", "{:.1f}".format):
            print(view.head(15).to_string(index=False))

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        for r in table.head(args.plot_top).itertuples():
            rel, sigma = motions[(r.slot, r.target)]
            title = f"{insp.name} around {r.target_name}, {r.start:%Y-%m-%d} to {r.end:%Y-%m-%d}"
            fig = relative_motion_figure(rel, title, sigma)
            fig.savefig(FIGURES_DIR / f"proximity_{insp.norad_id}_{r.target}_{r.slot}.png", dpi=150)


if __name__ == "__main__":
    main()
