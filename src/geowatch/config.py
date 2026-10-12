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
GEO_WINDOW_DIR = RAW_DIR / "geo_windows"  # whole GEO belt around given dates
PROCESSED_DIR = DATA_DIR / "processed"
BENCHMARK_DIR = RAW_DIR / "benchmark"  # Shorten et al. ground-truth dataset
FIGURES_DIR = PROJECT_ROOT / "figures"
CATALOG_FILE = PROJECT_ROOT / "config" / "objects.csv"


EXAMPLE_IDENTITY = "your_email@example.com"
EXAMPLE_PASSWORD = "your_password"


def spacetrack_credentials(env_file: Path | None = None) -> tuple[str, str]:
    """Read the Space-Track login from the .env file (or environment variables).

    Raises a RuntimeError that says exactly what is wrong: no .env file, a missing
    variable, or a .env that still holds the example values.
    """
    env_file = env_file or PROJECT_ROOT / ".env"
    load_dotenv(env_file)
    identity = os.getenv("SPACETRACK_IDENTITY")
    password = os.getenv("SPACETRACK_PASSWORD")

    if not identity and not password and not env_file.exists():
        hint = ""
        if env_file.with_name(".env.txt").exists():
            hint = " Found '.env.txt' instead: rename it to '.env' (Windows hides the .txt)."
        raise RuntimeError(
            f"No .env file in {env_file.parent}. Copy '.env.example' to '.env' and fill in "
            f"your Space-Track login.{hint}"
        )
    if not identity or not password:
        raise RuntimeError(
            f"{env_file} must define both SPACETRACK_IDENTITY and SPACETRACK_PASSWORD."
        )
    if identity == EXAMPLE_IDENTITY or password == EXAMPLE_PASSWORD:
        raise RuntimeError(
            f"{env_file} still contains the example values. Open it, replace them with your "
            "Space-Track email and password, and save the file."
        )
    return identity, password
