"""Detect manoeuvres for the objects in config/objects.csv and save a catalogue per object.

Output: data/processed/manoeuvres_<NORAD>.csv (and figures/manoeuvres_<NORAD>.png with --plot)

Usage:
    python scripts/detect_manoeuvres.py                      # every object in the catalogue
    python scripts/detect_manoeuvres.py --norad 40258 --plot
    python scripts/detect_manoeuvres.py --role inspector
"""

import argparse

import pandas as pd

from geowatch.catalog import load_catalog, select
from geowatch.config import DATA_DIR, FIGURES_DIR
from geowatch.detection import DetectorConfig, detect_manoeuvres
from geowatch.elements import elements_from_history
from geowatch.plots import quicklook
from geowatch.spacetrack_io import read_history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--norad", type=int, nargs="+", help="NORAD IDs")
    parser.add_argument("--group", help="catalogue group (e.g. luch, sj)")
    parser.add_argument("--role", help="catalogue role (inspector, target, validation)")
    parser.add_argument("--plot", action="store_true", help="save a figure per object")
    parser.add_argument("--start", help="first date to plot")
    parser.add_argument("--end", help="last date to plot")
    args = parser.parse_args()

    objects = select(load_catalog(), norad_ids=args.norad, group=args.group, role=args.role)
    out_dir = DATA_DIR / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = DetectorConfig()

    for row in objects.itertuples():
        try:
            elements = elements_from_history(read_history(row.norad_id))
        except FileNotFoundError as err:
            print(err)
            continue
        man = detect_manoeuvres(elements, config)
        man.to_csv(out_dir / f"manoeuvres_{row.norad_id}.csv", index=False)

        counts = ", ".join(f"{n} {kind}" for kind, n in man["type"].value_counts().items())
        print(
            f"\n{row.norad_id} {row.name}: {len(man)} manoeuvres, "
            f"total dv >= {man['dv_total_ms'].sum():.1f} m/s"
        )
        if counts:
            print(f"  {counts}")
        if len(man):
            last = man.tail(5)[["t_est", "type", "drift_before", "drift_after", "dv_total_ms"]]
            with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
                print("  latest:\n" + last.to_string(index=False))

        if args.plot:
            fig = quicklook(
                elements,
                f"{row.name} (NORAD {row.norad_id}) - detected manoeuvres",
                detections=man,
                start=args.start,
                end=args.end,
            )
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            fig.savefig(FIGURES_DIR / f"manoeuvres_{row.norad_id}.png", dpi=150)


if __name__ == "__main__":
    main()
