from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .solve_full import SolveResult


@dataclass
class UCEvalMetrics:
    ok: bool
    status_full: str
    status_screened: str

    active_lines_count: int
    total_lines: int
    constraint_reduction: float  # fraction removed

    solve_time_full_sec: float
    solve_time_screened_sec: float
    speedup: float

    objective_full: float | None
    objective_screened: float | None
    objective_gap_rel: float | None  # (screened - full) / full

    # Removed-constraint safety diagnostics evaluated on SCREENED solution
    max_removed_loading: float | None
    max_violation_ratio_removed: float | None
    violation_rate_removed: float | None
    violation_count_removed: int | None


def _get_flows_TL(res: SolveResult):
    """
    Backward/forward compatible accessor.
    Preferred field: flows_TL
    Older field (if any): flows
    """
    if hasattr(res, "flows_TL"):
        return getattr(res, "flows_TL")
    if hasattr(res, "flows"):
        return getattr(res, "flows")
    return None


def _as_2d_float_array(x):
    if x is None:
        return None
    try:
        a = np.asarray(x, dtype=float)
    except Exception:
        return None
    if a.ndim != 2:
        return None
    return a


def compute_uc_eval_metrics(
    full: SolveResult,
    screened: SolveResult,
    f_max: np.ndarray,
    active_lines: np.ndarray,
    *,
    tol: float = 1e-9,
) -> UCEvalMetrics:
    """
    Compares screened vs full solutions and evaluates whether removed constraints
    would be violated when using the screened solution.
    """
    f_max = np.asarray(f_max, dtype=float)
    L = int(f_max.shape[0]) if f_max.ndim == 1 else int(f_max.size)

    # active mask (clip bad indices defensively)
    active_lines = np.asarray(active_lines, dtype=int).ravel()
    active_lines = active_lines[(0 <= active_lines) & (active_lines < L)]
    active_mask = np.zeros(L, dtype=bool)
    if active_lines.size > 0:
        active_mask[active_lines] = True
    removed_mask = ~active_mask
    reduction = float(removed_mask.sum() / L) if L > 0 else 0.0

    # times + speedup
    t_full = float(getattr(full, "solve_time_sec", np.nan))
    t_scr = float(getattr(screened, "solve_time_sec", np.nan))
    speedup = float(t_full / t_scr) if np.isfinite(t_full) and np.isfinite(t_scr) and t_scr > 0 else float("nan")

    # objectives + gap
    obj_full = getattr(full, "objective", None)
    obj_scr = getattr(screened, "objective", None)
    obj_gap = None
    if obj_full is not None and obj_scr is not None:
        try:
            of = float(obj_full)
            os = float(obj_scr)
            if np.isfinite(of) and np.isfinite(os) and of != 0:
                obj_gap = float((os - of) / of)
        except Exception:
            obj_gap = None

    flows_full = _as_2d_float_array(_get_flows_TL(full))
    flows_scr = _as_2d_float_array(_get_flows_TL(screened))

    # basic validity checks
    ok = (
        (obj_full is not None)
        and (obj_scr is not None)
        and (flows_full is not None)
        and (flows_scr is not None)
        and (L > 0)
        and np.isfinite(f_max).all()
        and np.all(f_max > 0)
        and (flows_scr.shape[1] == L)
    )

    max_removed_loading = None
    max_violation_ratio = None
    viol_rate = None
    viol_count = None

    if ok and removed_mask.any():
        flows_removed = np.abs(flows_scr[:, removed_mask])          # (T, L_removed)
        denom = f_max[removed_mask][None, :]                        # (1, L_removed)
        ratios = flows_removed / denom

        finite = np.isfinite(ratios)
        if not finite.all():
            ratios = np.where(finite, ratios, np.inf)

        max_removed_loading = float(np.max(ratios)) if ratios.size else 0.0
        viol = ratios > (1.0 + tol)
        viol_count = int(viol.sum())
        viol_rate = float(viol.mean()) if viol.size else 0.0
        max_violation_ratio = max_removed_loading

    return UCEvalMetrics(
        ok=bool(ok),
        status_full=str(getattr(full, "status", "unknown")),
        status_screened=str(getattr(screened, "status", "unknown")),
        active_lines_count=int(active_mask.sum()),
        total_lines=int(L),
        constraint_reduction=float(reduction),
        solve_time_full_sec=float(t_full),
        solve_time_screened_sec=float(t_scr),
        speedup=float(speedup),
        objective_full=obj_full,
        objective_screened=obj_scr,
        objective_gap_rel=obj_gap,
        max_removed_loading=max_removed_loading,
        max_violation_ratio_removed=max_violation_ratio,
        violation_rate_removed=viol_rate,
        violation_count_removed=viol_count,
    )
