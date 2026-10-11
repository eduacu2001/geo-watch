"""Project paths and credentials."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# src/geowatch/config.py -> project root is two levels above the package folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
GP_HISTORY_DIR = RAW_DIR / "gp_history"  # one Parquet file per object
SNAPSHOT_DIR = RAW_DIR / "snapshots"  # current state of the GEO belt
BENCHMARK_DIR = RAW_DIR / "benchmark"  # Shorten et al. ground-truth dataset
FIGURES_DIR = PROJECT_ROOT / "figures"
CATALOG_FILE = PROJECT_ROOT / "config" / "objects.csv"


def spacetrack_credentials() -> tuple[str, str]:
    """Read the Space-Track login from the .env file (or environment variables)."""
    load_dotenv(PROJECT_ROOT / ".env")
    identity = os.getenv("SPACETRACK_IDENTITY")
    password = os.getenv("SPACETRACK_PASSWORD")
    if not identity or not password or identity == "your_email@example.com":
        raise RuntimeError(
            "Space-Track credentials not found. Copy '.env.example' to '.env' in the "
            "project root and fill in SPACETRACK_IDENTITY and SPACETRACK_PASSWORD."
        )
    return identity, password
