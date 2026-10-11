"""Search the Space-Track catalogue by name, to find NORAD IDs to add to config/objects.csv.

Usage:
    python scripts/find_object.py "INTELSAT 9"
    python scripts/find_object.py LUCH
"""

import argparse

import pandas as pd

from geowatch.spacetrack_io import make_client, search_satcat

COLUMNS = ["NORAD_CAT_ID", "SATNAME", "INTLDES", "COUNTRY", "LAUNCH", "DECAY", "PERIOD",
           "INCLINATION", "APOGEE", "PERIGEE"]  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="part of the object name (case-insensitive)")
    args = parser.parse_args()

    with make_client() as st:
        df = search_satcat(st, args.name)
    if df.empty:
        print("No matches.")
        return
    cols = [c for c in COLUMNS if c in df.columns]
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
