"""Scoring detections against ground-truth manoeuvre times.

Matching rules (stated up front, because they decide the numbers):

* Ground-truth manoeuvres less than ``merge_days`` apart are one event: TLEs cannot separate
  two burns a day apart (e.g. the paired north-south burns of Fengyun-4A).
* A detection matches a true manoeuvre if the true time lies between ``late_days`` before the
  detection's bracketing interval and ``early_days`` after it. The window is asymmetric on
  purpose: TLEs respond to a manoeuvre with a delay (median ~0.7 days, 90th percentile ~2.2
  days on the Fengyun data), so detections land after the burn, rarely before it.
* Each detection can match at most one true manoeuvre, and vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MatchResult:
    true_positives: int
    false_positives: int
    false_negatives: int
    lags_days: np.ndarray  # detection t_est minus true time, for matched pairs
    matched: pd.DataFrame  # detections that matched a true manoeuvre
    unmatched: pd.DataFrame  # detections with no true manoeuvre (false positives)
    missed: pd.DatetimeIndex  # true manoeuvres with no detection (false negatives)

    @property
    def precision(self) -> float:
        n = self.true_positives + self.false_positives
        return self.true_positives / n if n else float("nan")

    @property
    def recall(self) -> float:
        n = self.true_positives + self.false_negatives
        return self.true_positives / n if n else float("nan")

    @property
    def f1(self) -> float:
        n = 2 * self.true_positives + self.false_positives + self.false_negatives
        return 2 * self.true_positives / n if n else float("nan")


def merge_close_times(times: pd.DatetimeIndex, merge_days: float = 2.0) -> pd.DatetimeIndex:
    """Collapse times closer than merge_days into the first time of each group."""
    times = pd.DatetimeIndex(times).sort_values()
    if len(times) == 0:
        return times
    gaps_days = np.asarray((times[1:] - times[:-1]).total_seconds()) / 86400
    starts_group = np.concatenate([[True], gaps_days > merge_days])
    return times[starts_group]


def match(
    detections: pd.DataFrame,
    truth: pd.DatetimeIndex,
    early_days: float = 1.0,
    late_days: float = 4.0,
    merge_days: float = 2.0,
) -> MatchResult:
    """Match detections to true manoeuvre times (see module docstring for the rules)."""
    truth = merge_close_times(truth, merge_days)
    det = detections.reset_index(drop=True)
    used: set[int] = set()
    lags, missed = [], []
    for true_time in truth:
        lo = det["t_before"] - pd.Timedelta(days=late_days)
        hi = det["t_after"] + pd.Timedelta(days=early_days)
        ok = (lo <= true_time) & (hi >= true_time) & ~det.index.isin(list(used))
        if ok.any():
            j = (det.loc[ok, "t_est"] - true_time).abs().idxmin()
            used.add(j)
            lags.append((det.loc[j, "t_est"] - true_time).total_seconds() / 86400)
        else:
            missed.append(true_time)
    is_used = det.index.isin(list(used))
    return MatchResult(
        true_positives=len(used),
        false_positives=int((~is_used).sum()),
        false_negatives=len(missed),
        lags_days=np.array(lags),
        matched=det[is_used],
        unmatched=det[~is_used],
        missed=pd.DatetimeIndex(missed),
    )
