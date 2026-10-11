"""Download (or update) the full TLE history of the objects in config/objects.csv.

The first run downloads everything; later runs only fetch element sets newer than the ones
already stored in data/raw/gp_history/.

Usage:
    python scripts/download_history.py                 # every object in the catalogue
    python scripts/download_history.py --norad 40258   # only Luch (Olymp-K)
    python scripts/download_history.py --group luch    # only one group
"""

import argparse
import logging

import pandas as pd

from geowatch.catalog import load_catalog, select
from geowatch.spacetrack_io import make_client, read_history, update_history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--norad", type=int, nargs="+", help="NORAD IDs to update")
    parser.add_argument("--group", help="catalogue group (e.g. luch, sj, benchmark)")
    parser.add_argument("--role", help="catalogue role (inspector, target, validation)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    objects = select(load_catalog(), norad_ids=args.norad, group=args.group, role=args.role)
    if objects.empty:
        print("No objects match the filters.")
        return

    with make_client() as st:
        for row in objects.itertuples():
            added = update_history(st, row.norad_id)
            try:
                raw = read_history(row.norad_id)
            except FileNotFoundError:
                print(f"{row.norad_id:>6}  {row.name:<20} no data returned")
                continue
            first, last = pd.Timestamp(raw["EPOCH"].min()), pd.Timestamp(raw["EPOCH"].max())
            print(
                f"{row.norad_id:>6}  {row.name:<20} +{added:<5} new   "
                f"{len(raw):>6} total   {first:%Y-%m-%d} -> {last:%Y-%m-%d}"
            )


if __name__ == "__main__":
    main()
