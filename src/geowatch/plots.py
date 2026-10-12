"""Quick-look figures for exploring one object's orbital history."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

# Reference palette (validated categorical slots 1-3) and recessive ink
SERIES = "#2a78d6"
MANOEUVRE = "#eb6834"
DETECTION = "#1baf7a"
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
    detections: pd.DataFrame | None = None,
    start: str | None = None,
    end: str | None = None,
    geo_only: bool = True,
) -> plt.Figure:
    """Longitude, drift rate and inclination vs time.

    Optional overlays: ground-truth manoeuvres (vertical lines) and detected manoeuvres
    (triangles along the top of each panel). With ``geo_only`` the launch and transfer-orbit
    element sets (mean motion outside 0.9-1.1 rev/day) are left out.
    """
    df = elements
    if geo_only:
        df = df[(df["n_rev_day"] > 0.9) & (df["n_rev_day"] < 1.1)]
    if start:
        df = df[df["epoch"] >= pd.Timestamp(start)]
    if end:
        df = df[df["epoch"] <= pd.Timestamp(end)]
    if df.empty:
        raise ValueError("No element sets in the selected time window.")
    t0, t1 = df["epoch"].min(), df["epoch"].max()

    fig, axes = plt.subplots(
        len(PANELS), 1, figsize=(11, 7.5), sharex=True, facecolor=SURFACE, layout="constrained"
    )
    for ax, (column, label) in zip(axes, PANELS, strict=True):
        y = _break_wraps(df[column]) if column == "lon_deg" else df[column].to_numpy()
        ax.plot(df["epoch"], y, color=SERIES, lw=1.4, marker="o", ms=1.8, label="TLE mean elements")
        if manoeuvres is not None:
            shown = manoeuvres[(manoeuvres >= t0) & (manoeuvres <= t1)]
            for i, t in enumerate(shown):
                ax.axvline(
                    t,
                    color=MANOEUVRE,
                    lw=0.9,
                    alpha=0.7,
                    label="Ground-truth manoeuvre" if i == 0 else None,
                )
        if detections is not None and len(detections):
            shown = detections[(detections["t_est"] >= t0) & (detections["t_est"] <= t1)]
            top = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.plot(
                shown["t_est"],
                np.full(len(shown), 0.97),
                linestyle="none",
                marker="v",
                ms=7,
                color=DETECTION,
                markeredgecolor=SURFACE,
                markeredgewidth=0.8,
                transform=top,
                clip_on=False,
                label="Detected manoeuvre",
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
    if manoeuvres is not None or detections is not None:
        axes[0].legend(
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            ncol=3,
            frameon=False,
            fontsize=9,
            labelcolor=TEXT_SECONDARY,
        )
    axes[-1].set_xlabel("Epoch (UTC)", color=TEXT_SECONDARY)
    return fig


def relative_motion_figure(rel: pd.DataFrame, title: str, sigma_km: float = np.nan) -> plt.Figure:
    """Range over time, and the chaser's path around the target in the target's RIC frame."""
    fig = plt.figure(figsize=(11, 7.5), facecolor=SURFACE, layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    ax_range = fig.add_subplot(grid[0, :])
    ax_ir = fig.add_subplot(grid[1, 0])
    ax_ic = fig.add_subplot(grid[1, 1])

    ax_range.plot(rel["time"], rel["range_km"], color=SERIES, lw=1.2)
    if np.isfinite(sigma_km):
        ax_range.fill_between(
            rel["time"],
            (rel["range_km"] - sigma_km).clip(lower=0),
            rel["range_km"] + sigma_km,
            color=SERIES,
            alpha=0.15,
            lw=0,
        )
    ax_range.set_ylabel("Distance [km]", color=TEXT_SECONDARY)
    ax_range.set_ylim(bottom=0)

    for ax, column, label in ((ax_ir, "R_km", "Radial [km]"), (ax_ic, "C_km", "Cross-track [km]")):
        ax.plot(rel["I_km"], rel[column], color=SERIES, lw=0.6, alpha=0.6)
        ax.plot([0], [0], marker="*", ms=12, color=TEXT, linestyle="none")
        ax.annotate("target", (0, 0), textcoords="offset points", xytext=(6, 6), color=TEXT,
                    fontsize=9)  # fmt: skip
        ax.set_xlabel("In-track [km]  (positive = ahead of the target)", color=TEXT_SECONDARY)
        ax.set_ylabel(label, color=TEXT_SECONDARY)

    for ax in (ax_range, ax_ir, ax_ic):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    subtitle = f"  (band: TLE uncertainty ±{sigma_km:.1f} km)" if np.isfinite(sigma_km) else ""
    ax_range.set_title(title + subtitle, loc="left", color=TEXT, fontsize=12)
    return fig
