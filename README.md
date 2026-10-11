# GEO Watch

**Detecting manoeuvres and proximity operations of geostationary satellites from public orbital data.**

Some satellites in geostationary orbit do more than relay TV: Russia's Luch/Olymp spacecraft
have spent a decade parking next to Western communications satellites, and China's SJ-21 towed
a dead Beidou satellite out of the GEO belt before docking with SJ-25 in 2025. This project
reconstructs that activity from the public TLE catalogue: it detects when a satellite
manoeuvres, estimates the Δv, and flags close approaches between objects - and it measures how
reliable those detections are against satellites whose real manoeuvre history is known.

> Status: **phase 1 - data pipeline** (work in progress)

## Roadmap

| Phase | Content | Status |
|---|---|---|
| 1 | Space-Track download with local cache, mean elements, GEO longitude and drift rate, quick-look plots | in progress |
| 2 | Manoeuvre detection (robust statistics → prediction residuals → EKF innovation test), Δv estimation and classification, validation against ground truth | planned |
| 3 | Proximity-operation detection (RIC relative motion, TLE uncertainty bands) | planned |
| 4 | Case studies: Luch/Olymp-K, Luch/Olymp 2, SJ-21 & Beidou-2 G2, SJ-21 & SJ-25 | planned |
| 5 | Interactive website with daily automatic update (GitHub Actions) | planned |

## Setup (Windows, PowerShell)

Requires Python 3.10 or newer and a free [Space-Track](https://www.space-track.org) account.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env      # then edit .env with your Space-Track login
pytest                      # all tests should pass
```

If PowerShell refuses to run `Activate.ps1`, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Usage

```powershell
python scripts/check_catalog.py            # confirm the NORAD IDs in config/objects.csv
python scripts/download_history.py         # full TLE history of every tracked object
python scripts/download_benchmark.py       # ground-truth manoeuvres for validation
python scripts/quicklook.py --norad 40258  # plot Luch (Olymp-K)
python scripts/quicklook.py --benchmark Fengyun-2F --start 2019-01-01 --end 2020-07-01
python scripts/find_object.py "INTELSAT 9" # look up IDs to add to the catalogue
python scripts/download_geo_snapshot.py    # current state of the whole GEO belt
```

`download_history.py` is incremental: later runs only request element sets newer than those
already stored in `data/raw/gp_history/`, which keeps well inside Space-Track's usage limits.

## Method notes

- **Mean elements, not osculating states.** Following Shorten et al. (2023), orbit changes are
  tracked with the mean elements published in each TLE. Propagating TLEs with SGP4 and
  comparing osculating states sampled once a day adds short-period terms that look like noise
  or like fake manoeuvres.
- **Longitude drift rate** is computed by propagating each element set half a sidereal day
  either side of its epoch with SGP4/SDP4. Spanning one full orbit cancels the daily
  oscillation, and SDP4 keeps the deep-space resonance terms. On Fengyun-2F it matches the drift
  observed between consecutive TLEs to about 0.002 deg/day; a J2-only estimate is biased by
  about 0.007 deg/day.

## Project structure

```
config/objects.csv        tracked objects (inspectors, targets, validation satellites)
src/geowatch/
    config.py             paths and credentials
    catalog.py            loading the object list
    spacetrack_io.py      Space-Track queries and local Parquet cache
    elements.py           mean elements, GEO longitude, drift rate, e/i vectors
    benchmark.py          ground-truth manoeuvre dataset
    plots.py              quick-look figures
scripts/                  command-line entry points
tests/                    pytest suite (no network needed)
data/, figures/           generated locally, not committed
```

## Data and credits

- Orbital data: [Space-Track.org](https://www.space-track.org) (18th/19th Space Defense
  Squadron). Raw data is not redistributed in this repository, in line with the Space-Track
  user agreement; only derived results are published.
- Ground-truth manoeuvres: D. Shorten, Y. Yang, J. Maclean, M. Roughan, *Wide-scale Monitoring
  of Satellite Lifetimes: Pitfalls and a Benchmark Dataset*,
  [arXiv:2212.08662](https://arxiv.org/abs/2212.08662) -
  [dataset](https://github.com/dpshorten/TLE_observation_benchmark_dataset). Manoeuvre times
  originate from the International DORIS Service and China's National Satellite
  Meteorological Center.
- Propagation: [sgp4](https://pypi.org/project/sgp4/) by Brandon Rhodes; Space-Track client:
  [spacetrack](https://pypi.org/project/spacetrack/).

Author: Eduardo Acuña
