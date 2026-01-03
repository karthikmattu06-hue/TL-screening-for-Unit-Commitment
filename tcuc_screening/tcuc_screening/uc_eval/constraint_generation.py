# uc_eval/constraint_generation.py
from __future__ import annotations

import numpy as np


def active_lines_from_predicted_loadings(
    yhat_pred_24L: np.ndarray,     # (24, L) predicted loading ratios in [0,1]
    threshold: float,
    policy: str = "any_hour",
) -> np.ndarray:
    """
    Convert predicted loadings into an active line set.

    policy:
      - "any_hour": keep line l if any hour in day has loading >= threshold
      - "max_over_day": equivalent to any_hour, implemented explicitly
    """
    y = np.asarray(yhat_pred_24L, dtype=float)
    assert y.ndim == 2, "Expected shape (24, L)"
    if policy in ("any_hour", "max_over_day"):
        keep = (y.max(axis=0) >= threshold)
    else:
        raise ValueError(f"Unknown policy: {policy}")

    return np.where(keep)[0].astype(int)
