"""S12-01: V3 blend + roster-coverage terms (PA and BF share covered by projected rostered players), constants LOSO.
See RESEARCH_LOG (pre-registered, user OK 2026-11-26).
usage: python model/team3.py | tee work/results/s12_01_team3.txt   (needs work/results/s11_02_preds.parquet)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from park_test import clustered_t  # noqa: E402
from team import RES, SCORED  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def design(g):
    c = lambda col: g[col] - g.groupby("season")[col].transform("mean")  # noqa: E731
    return np.c_[g.U2 - 81, g.p1 - 81, c("pa_cov"), c("bf_cov")]


def run():
    R = pd.read_parquet(RES / "s11_02_preds.parquet")
    R["C1"] = np.nan
    X, y = design(R), (R.act - 81).values
    coef = {}
    for N in R.season.unique():
        m = (R.season.isin(SCORED) & (R.season != N)).values
        b = np.linalg.lstsq(X[m], y[m], rcond=None)[0]
        coef[N] = b
        i = (R.season == N).values
        R.loc[i, "C1"] = 81 + X[i] @ b
    print("LOSO (b_u, b_p, c_pa, c_bf):", {int(n): tuple(np.round(b, 2)) for n, b in sorted(coef.items())})
    v = pd.read_parquet(ROOT / "data" / "vegas" / "win_totals.parquet")[["season", "team_id", "line"]]
    R = R.merge(v, on=["season", "team_id"], how="left")
    R["Vegas"] = R.line - R.groupby("season").line.transform("mean") + 81
    R.to_parquet(RES / "s12_01_preds.parquet", index=False)

    x = R[R.season.isin(SCORED)]
    e1, e2 = x.season <= 2019, x.season >= 2021
    rm = lambda g, k: float(np.sqrt(np.mean((g.act - g[k]) ** 2)))  # noqa: E731
    t = lambda g, b, k: clustered_t(None, None, g.act - g[b], g.act - g[k], np.ones(len(g)), g.season)  # noqa: E731
    print("\nwins per 162, 240 team-seasons. t season-clustered, positive = arm better")
    print("| arm | RMSE | MAE | t vs B1 | t vs B2 | t vs V3 | 2016-19 (t vs V3) | 2021-24 (t vs V3) | slope | SD |")
    for k in ["B1", "B2", "V3", "C1", "Vegas"]:
        f = lambda g, b: "" if k == b else f"{t(g, b, k):.2f}"  # noqa: E731
        print(f"| {k} | {rm(x, k):.2f} | {np.mean(np.abs(x.act - x[k])):.2f} | {f(x, 'B1')} | {f(x, 'B2')} | {f(x, 'V3')} | "
              f"{rm(x[e1], k):.2f} ({f(x[e1], 'V3')}) | {rm(x[e2], k):.2f} ({f(x[e2], 'V3')}) | {np.polyfit(x[k], x.act, 1)[0]:.2f} | {x[k].std():.2f} |")
    print(f"C1 vs Vegas t {t(x, 'Vegas', 'C1'):.2f} (positive = C1 better); corr C1-Vegas {np.corrcoef(x.C1, x.Vegas)[0, 1]:.3f}")
    print("\nby season RMSE:")
    print(R.groupby("season").apply(lambda g: pd.Series({k: rm(g, k) for k in ["B2", "V3", "C1", "Vegas"]}), include_groups=False).round(2).to_string())
    q = pd.qcut(x.Vegas, 5, labels=False)
    print("\nmean error (act - arm) by Vegas quintile:")
    print(x.groupby(q).apply(lambda g: pd.Series({k: (g.act - g[k]).mean() for k in ["V3", "C1", "Vegas"]}), include_groups=False).round(2).to_string())
    b = np.linalg.lstsq(X[R.season.isin(SCORED).values], y[R.season.isin(SCORED).values], rcond=None)[0]
    print("\nall-8-season constants (b_u, b_p, c_pa, c_bf):", np.round(b, 3))
    return R


if __name__ == "__main__":
    run()
