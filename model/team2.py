"""S11-02: U1 (S11-01 N2) + runs-allowed spread stretch + blend with last-year Pythag, constants LOSO. See RESEARCH_LOG.
usage: python model/team2.py | tee work/results/s11_02_team2.txt   (needs work/results/s11_01_preds.parquet)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from park_test import clustered_t  # noqa: E402
from team import RES, SCORED, pythag, team_results  # noqa: E402


def stretch(R, s):
    """re-center RA around the season mean, scale by s, Pythagenpat, re-center wins to 81."""
    out = pd.Series(np.nan, index=R.index)
    for N, g in R.groupby("season"):
        ra = g.ra_p.mean() + s[N] * (g.ra_p - g.ra_p.mean())
        w = 162 * pythag(g.rs_p, ra, 162)
        out[g.index] = w - w.mean() + 81
    return out


def loso_slope(x, y, season, N):
    m = season.isin(SCORED) & (season != N)
    return float(np.polyfit(x[m], y[m], 1)[0])


def run():
    R = pd.read_parquet(RES / "s11_01_preds.parquet")
    res = pd.concat([team_results(y) for y in range(2015, 2025)])
    res["p1"] = 162 * pythag(res.rs, res.ra, res.g)
    R = R.merge(res[["team_id", "season", "p1"]].assign(season=res.season + 1), on=["team_id", "season"], how="left")
    c = lambda col: R[col] - R.groupby("season")[col].transform("mean")  # noqa: E731
    rac, rap = c("ra_a"), c("ra_p")
    s = {N: loso_slope(rap, rac, R.season, N) for N in R.season.unique()}
    R["U2"] = stretch(R, s)
    coef = {}
    for arm, u in (("V1", "N2"), ("V3", "U2")):
        R[arm] = np.nan
        for N in R.season.unique():
            m = R.season.isin(SCORED) & (R.season != N)
            X = np.c_[R.loc[m, u] - 81, R.loc[m, "p1"] - 81]
            b = np.linalg.lstsq(X, R.loc[m, "act"] - 81, rcond=None)[0]
            coef[(arm, N)] = b
            i = R.season == N
            R.loc[i, arm] = 81 + b[0] * (R.loc[i, u] - 81) + b[1] * (R.loc[i, "p1"] - 81)
    R["V2"] = R.U2
    print("RA stretch s (LOSO):", {int(k): round(v, 3) for k, v in sorted(s.items())})
    print("blend (b_u, b_p) LOSO:", {f"{a} {n}": tuple(np.round(b, 3)) for (a, n), b in sorted(coef.items()) if a == "V3"})
    print("V1 blend:", {n: tuple(np.round(b, 3)) for (a, n), b in sorted(coef.items()) if a == "V1"})
    R.to_parquet(RES / "s11_02_preds.parquet", index=False)

    x = R[R.season.isin(SCORED)]
    e1, e2 = x.season <= 2019, x.season >= 2021
    rm = lambda g, k: float(np.sqrt(np.mean((g.act - g[k]) ** 2)))  # noqa: E731
    t = lambda g, b, k: clustered_t(None, None, g.act - g[b], g.act - g[k], np.ones(len(g)), g.season)  # noqa: E731
    print("\nwins per 162, 240 team-seasons. t season-clustered, positive = arm better")
    print("| arm | RMSE | MAE | t vs B1 | t vs B2 | t vs U1 | 2016-19 (t vs B2) | 2021-24 (t vs B2) | slope | SD |")
    for k in ["B1", "B2", "N2", "V1", "V2", "V3"]:
        f = lambda g, b: "" if k == b else f"{t(g, b, k):.2f}"  # noqa: E731
        print(f"| {k} | {rm(x, k):.2f} | {np.mean(np.abs(x.act - x[k])):.2f} | {f(x, 'B1')} | {f(x, 'B2')} | {f(x, 'N2')} | "
              f"{rm(x[e1], k):.2f} ({f(x[e1], 'B2')}) | {rm(x[e2], k):.2f} ({f(x[e2], 'B2')}) | {np.polyfit(x[k], x.act, 1)[0]:.2f} | {x[k].std():.2f} |")
    print("\nby season RMSE:")
    print(R.groupby("season").apply(lambda g: pd.Series({k: rm(g, k) for k in ["B2", "N2", "V1", "V2", "V3"]}), include_groups=False).round(2).to_string())
    return R


if __name__ == "__main__":
    run()
