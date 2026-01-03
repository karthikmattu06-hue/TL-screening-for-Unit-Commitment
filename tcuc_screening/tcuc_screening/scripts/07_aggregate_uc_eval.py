#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
from pathlib import Path
import re
import numpy as np
import pandas as pd


def infer_mode_from_filename(p: Path) -> str | None:
    # examples: uc_eval_X_repair_....csv, uc_eval_XY_baseline_....csv
    m = re.search(r"uc_eval_(XY|X|Y)\b", p.name)
    return m.group(1) if m else None


def infer_method_from_filename(p: Path) -> str | None:
    n = p.name.lower()
    if "repair" in n:
        return "RCM+CG"
    if "baseline" in n:
        return "RCM"
    return None


def fmt_int(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    return f"{int(round(x)):,}"


def fmt_float(x, nd=2) -> str:
    if x is None or not np.isfinite(x):
        return "NA"
    return f"{x:.{nd}f}"


def fmt_money(x) -> str:
    if x is None or not np.isfinite(x):
        return "NA"
    # Paper prints large integers with commas; keep as integer dollars
    return f"{int(round(x)):,}"


def build_latex_table(df_mode: pd.DataFrame, mode: str) -> str:
    """
    Table columns in the spirit of Table III:
      Method, Th, NCR(%), T(sec), ΔT(%), C($), ΔC(%), V, V(%)
    """
    # order methods like paper: FN would be baseline, but we show RCM and RCM+CG
    method_order = ["RCM", "RCM+CG"]
    df_mode = df_mode.copy()
    df_mode["method"] = pd.Categorical(df_mode["method"], method_order, ordered=True)
    df_mode = df_mode.sort_values(["method", "threshold"])

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{Transmission line screening results for mode {mode}.}}")
    lines.append(r"\begin{tabular}{l c r r r r r r r}")
    lines.append(r"\hline")
    lines.append(r"Method & Th & NCR(\%) & T (sec) & $\Delta T$ (\%) & C (\$) & $\Delta C$ (\%) & V & V(\%) \\")
    lines.append(r"\hline")

    for _, r in df_mode.iterrows():
        lines.append(
            f"{r['method']} & "
            f"{fmt_float(r['threshold'], 1)} & "
            f"{fmt_float(r['NCR_percent'], 2)} & "
            f"{fmt_int(r['T_screened_sec'])} & "
            f"{fmt_float(r['DeltaT_percent'], 2)} & "
            f"{fmt_money(r['C_screened'])} & "
            f"{fmt_float(r['DeltaC_percent'], 6)} & "
            f"{fmt_int(r['V'])} & "
            f"{fmt_float(r['V_percent'], 4)} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        type=str,
        default="tcuc_screening/results/screened_tcuc/uc_eval_*.csv",
        help="Glob for per-run CSVs produced by 06_run_uc_eval.py",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="tcuc_screening/results/screened_tcuc/aggregated",
        help="Output directory for summary CSV + LaTeX tables",
    )
    ap.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="UC horizon in hours (default 24). Used for NC and V(%).",
    )
    args = ap.parse_args()

    in_paths = [Path(p) for p in glob.glob(args.inputs)]
    if not in_paths:
        raise FileNotFoundError(f"No input CSVs matched: {args.inputs}")

    frames = []
    for p in in_paths:
        df = pd.read_csv(p)
        if "mode" not in df.columns:
            mode = infer_mode_from_filename(p)
            if mode is not None:
                df["mode"] = mode
        if "method" not in df.columns:
            meth = infer_method_from_filename(p)
            # Fallback: use repair_enabled if present
            if meth is None and "repair_enabled" in df.columns:
                meth = "RCM+CG" if bool(df["repair_enabled"].iloc[0]) else "RCM"
            if meth is None:
                # last resort: label unknown
                meth = "RCM"
            df["method"] = meth

        df["source_file"] = p.name
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    # Required columns check
    required = [
        "threshold",
        "constraint_reduction",
        "solve_time_full_sec",
        "solve_time_screened_sec",
        "objective_full",
        "objective_screened",
        "violation_count_removed",
        "total_lines",
        "mode",
        "method",
    ]
    missing = [c for c in required if c not in all_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in inputs: {missing}")

    # Clean numeric types
    for c in [
        "threshold",
        "constraint_reduction",
        "solve_time_full_sec",
        "solve_time_screened_sec",
        "objective_full",
        "objective_screened",
        "violation_count_removed",
        "total_lines",
    ]:
        all_df[c] = pd.to_numeric(all_df[c], errors="coerce")

    # Group and compute Table-III-style aggregates
    # Each row corresponds to one (day, threshold). So Ndays per group is count of rows.
    gcols = ["mode", "method", "threshold"]
    out_rows = []
    for (mode, method, thr), g in all_df.groupby(gcols, dropna=False):
        g = g.copy()
        n_days = int(len(g))
        L = int(pd.Series(g["total_lines"].dropna()).iloc[0])
        H = int(args.horizon)
        NC = L * H * n_days  # total line-hour constraints in this evaluation block

        T_full = float(g["solve_time_full_sec"].sum())
        T_scr = float(g["solve_time_screened_sec"].sum())

        C_full = float(g["objective_full"].sum())
        C_scr = float(g["objective_screened"].sum())

        V = int(pd.Series(g["violation_count_removed"].fillna(0)).sum())

        NCR_percent = float(g["constraint_reduction"].mean() * 100.0)

        DeltaT_percent = float((T_full - T_scr) / T_full * 100.0) if T_full > 0 else np.nan
        DeltaC_percent = float((C_full - C_scr) / C_full * 100.0) if C_full != 0 else np.nan

        V_percent = float(V / NC * 100.0) if NC > 0 else np.nan

        out_rows.append(
            dict(
                mode=str(mode),
                method=str(method),
                threshold=float(thr),
                n_days=n_days,
                horizon_hours=H,
                total_lines=L,
                NC=int(NC),
                NCR_percent=NCR_percent,
                T_full_sec=T_full,
                T_screened_sec=T_scr,
                DeltaT_percent=DeltaT_percent,
                speedup=(T_full / T_scr) if T_scr > 0 else np.nan,
                C_full=C_full,
                C_screened=C_scr,
                DeltaC_percent=DeltaC_percent,
                V=V,
                V_percent=V_percent,
            )
        )

    summary = pd.DataFrame(out_rows).sort_values(["mode", "method", "threshold"]).reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "table3_metrics_summary.csv"
    summary.to_csv(out_csv, index=False)

    # Emit LaTeX tables per mode
    latex_paths = []
    for mode in sorted(summary["mode"].dropna().unique().tolist(), key=lambda x: (len(x), x)):
        df_mode = summary[summary["mode"] == mode]
        tex = build_latex_table(df_mode, mode=mode)
        out_tex = out_dir / f"table3_mode_{mode}.tex"
        out_tex.write_text(tex)
        latex_paths.append(out_tex)

    print("WROTE:")
    print(f" - {out_csv}")
    for p in latex_paths:
        print(f" - {p}")


if __name__ == "__main__":
    main()
