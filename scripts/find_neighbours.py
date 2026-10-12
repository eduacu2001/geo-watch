"""Find which satellites were parked next to each inspector (phase 3A).

For every parking slot of an inspector, downloads a snapshot of the whole GEO belt around the
slot's middle date (and around its start and end for stays longer than 180 days), and lists
the objects within --max-offset degrees of longitude. Results:
    data/processed/neighbours_<NORAD>.csv
With --download-targets, it also downloads the full TLE history of every neighbour that came
within --target-km, ready for the distance analysis of phase 3B.

Usage:
    python scripts/find_neighbours.py --norad 40258
    python scripts/find_neighbours.py --role inspector --download-targets
"""

import argparse
import logging

import pandas as pd

from geowatch.catalog import load_catalog, select
from geowatch.config import PROCESSED_DIR
from geowatch.elements import elements_from_history
from geowatch.proximity import nearest_set, neighbours_at, parking_slots
from geowatch.spacetrack_io import (
    download_geo_window,
    history_path,
    make_client,
    read_history,
    update_history,
)

LONG_STAY_DAYS = 180


def snapshot_dates(slot) -> list[pd.Timestamp]:
    middle = slot.start + (slot.end - slot.start) / 2
    if slot.days <= LONG_STAY_DAYS:
        return [middle]
    return [slot.start + pd.Timedelta(days=3), middle, slot.end - pd.Timedelta(days=3)]


def load_belt(path) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    if raw.empty:
        return pd.DataFrame()
    belt = elements_from_history(raw)
    types = raw.drop_duplicates("NORAD_CAT_ID").set_index("NORAD_CAT_ID")["OBJECT_TYPE"]
    types.index = types.index.astype(int)
    return belt.assign(object_type=belt["norad_id"].map(types))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--norad", type=int, nargs="+", help="inspector NORAD IDs")
    parser.add_argument("--role", default="inspector", help="catalogue role (default inspector)")
    parser.add_argument("--max-offset", type=float, default=0.5, help="deg of longitude")
    parser.add_argument("--download-targets", action="store_true")
    parser.add_argument("--target-km", type=float, default=100.0, help="for --download-targets")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    catalog = load_catalog()
    inspectors = select(catalog, norad_ids=args.norad, role=None if args.norad else args.role)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    targets: set[int] = set()

    with make_client() as st:
        for insp in inspectors.itertuples():
            own = elements_from_history(read_history(insp.norad_id))
            slots = parking_slots(own)
            print(f"\n{insp.name} (NORAD {insp.norad_id}): {len(slots)} parking slots")
            rows = []
            for k, slot in enumerate(slots.itertuples()):
                for when in snapshot_dates(slot):
                    belt = load_belt(download_geo_window(st, when.date()))
                    if belt.empty:
                        continue
                    me = nearest_set(own.assign(norad_id=insp.norad_id), when).iloc[0]
                    near = neighbours_at(
                        belt, me["lon_deg"], when, args.max_offset, exclude=(insp.norad_id,)
                    )
                    rows.append(
                        near.assign(
                            inspector=insp.norad_id,
                            slot=k,
                            slot_start=slot.start,
                            slot_end=slot.end,
                            slot_lon_deg=slot.lon_deg,
                            snapshot=when,
                        )
                    )
                closest = rows[-1].head(3) if rows else pd.DataFrame()
                names = ", ".join(
                    f"{r.object_name} ({r.arc_km:.0f} km)" for r in closest.itertuples()
                )
                print(
                    f"  slot {k:>2}  {slot.start:%Y-%m-%d} -> {slot.end:%Y-%m-%d}  "
                    f"{slot.lon_deg:8.2f} deg  nearest: {names or '-'}"
                )
            if rows:
                table = pd.concat(rows, ignore_index=True)
                table.to_csv(PROCESSED_DIR / f"neighbours_{insp.norad_id}.csv", index=False)
                targets |= set(table.loc[table["arc_km"] <= args.target_km, "norad_id"])

        if args.download_targets and targets:
            todo = sorted(t for t in targets if not history_path(t).exists())
            print(f"\nDownloading the TLE history of {len(todo)} neighbours "
                  f"({len(targets) - len(todo)} already stored)")  # fmt: skip
            for norad_id in todo:
                update_history(st, norad_id)


if __name__ == "__main__":
    main()
