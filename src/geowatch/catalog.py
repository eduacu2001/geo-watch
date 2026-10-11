"""The list of objects the project follows (config/objects.csv)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from geowatch.config import CATALOG_FILE


def load_catalog(path: Path = CATALOG_FILE) -> pd.DataFrame:
    """Load the tracked-object catalogue, ignoring comment lines."""
    df = pd.read_csv(path, comment="#", dtype={"norad_id": int}, skipinitialspace=True)
    df["notes"] = df["notes"].fillna("")
    return df


def select(
    catalog: pd.DataFrame,
    norad_ids: list[int] | None = None,
    group: str | None = None,
    role: str | None = None,
) -> pd.DataFrame:
    """Filter the catalogue by NORAD ID, group or role (all filters optional)."""
    out = catalog
    if norad_ids:
        out = out[out["norad_id"].isin(norad_ids)]
    if group:
        out = out[out["group"] == group]
    if role:
        out = out[out["role"] == role]
    return out
