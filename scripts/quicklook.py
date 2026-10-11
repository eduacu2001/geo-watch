"""Plot longitude, drift rate and inclination of one object over time.

Usage:
    python scripts/quicklook.py --norad 40258                        # Space-Track data
    python scripts/quicklook.py --norad 40258 --start 2025-06-01
    python scripts/quicklook.py --benchmark Fengyun-2F --start 2020-01-01 --end 2021-01-01
    python scripts/quicklook.py --norad 38049 --show                 # also open a window

If the object is in the ground-truth benchmark (and scripts/download_benchmark.py has been
run), its real manoeuvres are drawn as vertical lines.
"""

import argparse

import matplotlib.pyplot as plt

from geowatch import benchmark
from geowatch.catalog import load_catalog
from geowatch.config import FIGURES_DIR
from geowatch.elements import elements_from_history
from geowatch.plots import quicklook
from geowatch.spacetrack_io import read_history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--norad", type=int, help="NORAD ID (uses downloaded Space-Track data)")
    source.add_argument(
        "--benchmark", choices=list(benchmark.SATELLITES), help="use the benchmark's own TLE file"
    )
    parser.add_argument("--start", help="first date to plot, e.g. 2022-01-01")
    parser.add_argument("--end", help="last date to plot")
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    args = parser.parse_args()

    if args.benchmark:
        name = args.benchmark
        norad_id = benchmark.SATELLITES[name][0]
        elements = benchmark.load_elements(name)
        title = f"{name} (NORAD {norad_id}) - benchmark TLEs"
    else:
        norad_id = args.norad
        elements = elements_from_history(read_history(norad_id))
        catalog = load_catalog().set_index("norad_id")
        name = catalog.loc[norad_id, "name"] if norad_id in catalog.index else str(norad_id)
        title = f"{name} (NORAD {norad_id}) - Space-Track TLEs"

    manoeuvres = None
    bench_name = benchmark.name_for(norad_id)
    if bench_name:
        try:
            manoeuvres = benchmark.load_manoeuvres(bench_name)
        except FileNotFoundError:
            print("Tip: run scripts/download_benchmark.py to overlay real manoeuvres.")

    fig = quicklook(elements, title, manoeuvres=manoeuvres, start=args.start, end=args.end)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"quicklook_{norad_id}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
