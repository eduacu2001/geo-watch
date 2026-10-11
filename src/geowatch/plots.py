"""Quick-look figures for exploring one object's orbital history."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reference palette (validated categorical slots 1-2) and recessive ink
SERIES = "#2a78d6"
MANOEUVRE = "#eb6834"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

PANELS = [
    ("lon_deg", "Longitude [deg E]"),
    ("drift_deg_day", "Drift rate [deg/day]"),
    ("inc_deg", "Inclination [deg]"),
]


def _break_wraps(values: pd.Series, jump: float = 180.0) -> np.ndarray:
    """Insert NaN where longitude wraps around +/-180 so the line is not drawn across."""
    v = values.to_numpy(dtype=float).copy()
    wraps = np.abs(np.diff(v)) > jump
    v[1:][wraps] = np.nan
    return v


def quicklook(
    elements: pd.DataFrame,
    title: str,
    manoeuvres: pd.DatetimeIndex | None = None,
    start: str | None = None,
    end: str | None = None,
) -> plt.Figure:
    """Longitude, drift rate and inclination vs time, with optional ground-truth manoeuvres."""
    df = elements
    if start:
        df = df[df["epoch"] >= pd.Timestamp(start)]
    if end:
        df = df[df["epoch"] <= pd.Timestamp(end)]
    if df.empty:
        raise ValueError("No element sets in the selected time window.")

    fig, axes = plt.subplots(
        len(PANELS), 1, figsize=(11, 7.5), sharex=True, facecolor=SURFACE, layout="constrained"
    )
    for ax, (column, label) in zip(axes, PANELS, strict=True):
        y = _break_wraps(df[column]) if column == "lon_deg" else df[column].to_numpy()
        ax.plot(df["epoch"], y, color=SERIES, lw=1.4, marker="o", ms=1.8, label="TLE mean elements")
        if manoeuvres is not None:
            in_window = (manoeuvres >= df["epoch"].min()) & (manoeuvres <= df["epoch"].max())
            shown = manoeuvres[in_window]
            for i, t in enumerate(shown):
                ax.axvline(
                    t,
                    color=MANOEUVRE,
                    lw=0.9,
                    alpha=0.7,
                    label="Ground-truth manoeuvre" if i == 0 else None,
                )
        ax.set_ylabel(label, color=TEXT_SECONDARY)
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)

    axes[0].set_title(title, loc="left", color=TEXT, fontsize=12)
    if manoeuvres is not None:
        axes[0].legend(
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            ncol=2,
            frameon=False,
            fontsize=9,
            labelcolor=TEXT_SECONDARY,
        )
    axes[-1].set_xlabel("Epoch (UTC)", color=TEXT_SECONDARY)
    return fig
