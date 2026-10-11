"""Ground-truth manoeuvre dataset of Shorten, Maclean, Roughan & Yang (2023).

D. Shorten et al., "Wide-scale Monitoring of Satellite Lifetimes: Pitfalls and a Benchmark
Dataset", arXiv:2212.08662. Data: github.com/dpshorten/TLE_observation_benchmark_dataset

The dataset is downloaded on demand (not copied into this repository). Manoeuvre times come
from the International DORIS Service and, for the Fengyun satellites, from China's National
Satellite Meteorological Center.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import pandas as pd
import yaml

from geowatch.config import BENCHMARK_DIR
from geowatch.elements import elements_from_tle, read_tle_text

log = logging.getLogger(__name__)

BASE_URL = (
    "https://raw.githubusercontent.com/dpshorten/TLE_observation_benchmark_dataset/"
    "main/processed_files"
)

# name in the dataset -> (NORAD ID, regime)
SATELLITES: dict[str, tuple[int, str]] = {
    "Fengyun-2D": (29640, "GEO"),
    "Fengyun-2E": (33463, "GEO"),
    "Fengyun-2F": (38049, "GEO"),
    "Fengyun-2H": (43491, "GEO"),
    "Fengyun-4A": (41882, "GEO"),
    "Sentinel-3A": (41335, "LEO"),
    "Sentinel-3B": (43437, "LEO"),
    "Sentinel-6A": (46984, "LEO"),
    "Jason-1": (26997, "LEO"),
    "Jason-2": (33105, "LEO"),
    "Jason-3": (41240, "LEO"),
    "SARAL": (39086, "LEO"),
    "CryoSat-2": (36508, "LEO"),
    "Haiyang-2A": (37781, "LEO"),
    "TOPEX": (22076, "LEO"),
}


def name_for(norad_id: int) -> str | None:
    """Dataset name of a NORAD ID, or None if the object is not in the benchmark."""
    for name, (nid, _) in SATELLITES.items():
        if nid == int(norad_id):
            return name
    return None


def download(
    names: list[str] | None = None, directory: Path = BENCHMARK_DIR, overwrite: bool = False
) -> list[Path]:
    """Download TLE and manoeuvre files for the given satellites (default: all GEO ones)."""
    if names is None:
        names = [n for n, (_, regime) in SATELLITES.items() if regime == "GEO"]
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        if name not in SATELLITES:
            raise ValueError(f"'{name}' is not in the benchmark. Options: {list(SATELLITES)}")
        for filename in (f"{name}.tle", f"manoeuvres_{name}.yaml"):
            path = directory / filename
            if overwrite or not path.exists():
                log.info("Downloading %s", filename)
                urllib.request.urlretrieve(f"{BASE_URL}/{filename}", path)
            paths.append(path)
    return paths


def load_manoeuvres(name: str, directory: Path = BENCHMARK_DIR) -> pd.DatetimeIndex:
    """Ground-truth manoeuvre times (UTC) of one benchmark satellite."""
    path = directory / f"manoeuvres_{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/download_benchmark.py first.")
    data = yaml.safe_load(path.read_text())
    return pd.DatetimeIndex(pd.to_datetime(data["manoeuvre_timestamps"])).sort_values()


def load_elements(name: str, directory: Path = BENCHMARK_DIR) -> pd.DataFrame:
    """Mean elements computed from the benchmark's own TLE file."""
    path = directory / f"{name}.tle"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/download_benchmark.py first.")
    pairs = read_tle_text(path.read_text())
    df = elements_from_tle([p[0] for p in pairs], [p[1] for p in pairs])
    return df.drop_duplicates(subset="epoch", keep="last").reset_index(drop=True)
