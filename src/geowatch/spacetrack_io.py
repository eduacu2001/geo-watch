"""Download orbital data from Space-Track and cache it locally.

Space-Track asks users to keep queries under 30/minute and 300/hour and not to download the
same history repeatedly. The ``spacetrack`` client enforces the rate limit; this module adds
the second part by storing everything in Parquet files and only asking for element sets newer
than the last one already on disk.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import spacetrack.operators as op
from spacetrack import SpaceTrackClient

from geowatch.config import GP_HISTORY_DIR, SNAPSHOT_DIR, spacetrack_credentials

log = logging.getLogger(__name__)


def make_client() -> SpaceTrackClient:
    """Space-Track client logged in with the credentials from .env."""
    identity, password = spacetrack_credentials()
    return SpaceTrackClient(identity=identity, password=password)


def history_path(norad_id: int, directory: Path = GP_HISTORY_DIR) -> Path:
    return directory / f"{int(norad_id):05d}.parquet"


def read_history(norad_id: int, directory: Path = GP_HISTORY_DIR) -> pd.DataFrame:
    """Raw gp_history table stored for one object."""
    path = history_path(norad_id, directory)
    if not path.exists():
        raise FileNotFoundError(
            f"No stored history for NORAD {norad_id}. Run scripts/download_history.py first."
        )
    return pd.read_parquet(path)


def _records_to_frame(records: list[dict]) -> pd.DataFrame:
    # Keep every field exactly as Space-Track delivered it; parsing happens in elements.py
    return pd.DataFrame.from_records(records).astype("string")


def update_history(client, norad_id: int, directory: Path = GP_HISTORY_DIR) -> int:
    """Fetch element sets newer than those already stored for one object.

    Returns the number of new element sets added to the local file.
    """
    path = history_path(norad_id, directory)
    existing = pd.read_parquet(path) if path.exists() else None

    query: dict = {"norad_cat_id": int(norad_id), "orderby": "epoch asc"}
    if existing is not None and len(existing):
        # Floor to the second: the last stored set may come back once and is de-duplicated
        last_epoch = pd.Timestamp(existing["EPOCH"].max()).floor("s").to_pydatetime()
        query["epoch"] = op.greater_than(last_epoch)

    records = client.gp_history(**query)
    if not records:
        return 0

    new = _records_to_frame(records)
    combined = new if existing is None else pd.concat([existing, new], ignore_index=True)
    key = ["GP_ID"] if "GP_ID" in combined.columns else ["EPOCH", "TLE_LINE1"]
    combined = combined.drop_duplicates(subset=key).sort_values("EPOCH", kind="stable")

    directory.mkdir(parents=True, exist_ok=True)
    combined.reset_index(drop=True).to_parquet(path, index=False)
    return len(combined) - (0 if existing is None else len(existing))


def download_geo_snapshot(client, directory: Path = SNAPSHOT_DIR, max_age_days: int = 30) -> Path:
    """Latest element set of every object currently in the GEO regime.

    GEO regime here: 0.99-1.01 rev/day, eccentricity < 0.01, not decayed, element set
    younger than ``max_age_days``.
    """
    records = client.gp(
        epoch=op.greater_than(f"now-{max_age_days}"),
        mean_motion=op.inclusive_range(0.99, 1.01),
        eccentricity=op.less_than(0.01),
        decay_date=None,  # sent as 'null-val': objects still in orbit
        orderby="norad_cat_id",
    )
    df = _records_to_frame(records)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"geo_{dt.datetime.now(dt.timezone.utc):%Y%m%d}.parquet"
    df.to_parquet(path, index=False)
    return path


def fetch_satcat(client, norad_ids: Iterable[int]) -> pd.DataFrame:
    """Official catalogue entries (name, launch, country, ...) for the given objects."""
    records = client.satcat(norad_cat_id=[int(n) for n in norad_ids], orderby="norad_cat_id")
    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df["NORAD_CAT_ID"] = df["NORAD_CAT_ID"].astype(int)
    return df


def search_satcat(client, name_fragment: str) -> pd.DataFrame:
    """Catalogue entries whose name contains ``name_fragment`` (case-insensitive)."""
    records = client.satcat(satname=op.like(name_fragment.upper()), orderby="norad_cat_id")
    return pd.DataFrame.from_records(records)
