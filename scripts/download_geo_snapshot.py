"""Save the latest element set of every object currently in the GEO regime.

Used later to find which satellites sit near an inspector's longitude.

Usage:
    python scripts/download_geo_snapshot.py
"""

import pandas as pd

from geowatch.elements import elements_from_history
from geowatch.spacetrack_io import download_geo_snapshot, make_client


def main() -> None:
    with make_client() as st:
        path = download_geo_snapshot(st)
    raw = pd.read_parquet(path)
    el = elements_from_history(raw)
    drifting = (el["drift_deg_day"].abs() > 0.05).sum()
    print(f"Saved {len(raw)} GEO-regime objects to {path}")
    print(f"  {drifting} of them drifting faster than 0.05 deg/day (not station-kept)")


if __name__ == "__main__":
    main()
