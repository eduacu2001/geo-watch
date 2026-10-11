"""Check every object in config/objects.csv against the official Space-Track catalogue.

Usage:
    python scripts/check_catalog.py
"""

import pandas as pd

from geowatch.catalog import load_catalog
from geowatch.spacetrack_io import fetch_satcat, make_client


def main() -> None:
    catalog = load_catalog()
    with make_client() as st:
        satcat = fetch_satcat(st, catalog["norad_id"])

    if satcat.empty:
        print("Space-Track returned no catalogue entries - check the IDs and your login.")
        return

    cols = ["NORAD_CAT_ID", "SATNAME", "INTLDES", "COUNTRY", "LAUNCH", "DECAY", "PERIOD"]
    cols = [c for c in cols if c in satcat.columns]
    merged = catalog[["norad_id", "name", "role"]].merge(
        satcat[cols], left_on="norad_id", right_on="NORAD_CAT_ID", how="left"
    )
    merged = merged.drop(columns="NORAD_CAT_ID").rename(columns={"SATNAME": "official_name"})

    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(merged.to_string(index=False))

    missing = merged[merged["official_name"].isna()]
    if len(missing):
        print("\nNot found in the catalogue:", ", ".join(map(str, missing["norad_id"])))
    else:
        print(f"\nAll {len(merged)} objects found. Compare 'name' and 'official_name' above.")


if __name__ == "__main__":
    main()
