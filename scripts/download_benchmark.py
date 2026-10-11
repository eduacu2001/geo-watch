"""Download the ground-truth manoeuvre dataset of Shorten et al. (2023) from GitHub.

Usage:
    python scripts/download_benchmark.py          # the 5 GEO satellites (Fengyun)
    python scripts/download_benchmark.py --all    # all 15 satellites (GEO + LEO)
"""

import argparse
import logging

from geowatch import benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="include the LEO satellites")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    names = list(benchmark.SATELLITES) if args.all else None
    benchmark.download(names)
    names = names or [n for n, (_, r) in benchmark.SATELLITES.items() if r == "GEO"]
    for name in names:
        n_man = len(benchmark.load_manoeuvres(name))
        n_tle = len(benchmark.load_elements(name))
        print(f"{name:<12} {n_tle:>6} element sets   {n_man:>5} ground-truth manoeuvres")


if __name__ == "__main__":
    main()
