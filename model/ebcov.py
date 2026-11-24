"""S7-03: S6-01 regression constants from year-to-year covariance. See RESEARCH_LOG (pre-registered).
usage: python model/ebcov.py | tee work/results/s7_03_ebcov.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcomp  # noqa: E402
from marcel import wrmse  # noqa: E402
from park_test import clustered_t, multipliers, run as park_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEASON = ROOT / "data" / "season"
_orig = pcomp.eb_k
STATE = {"cov": set(), "ks": []}


def cov_k(d, col):
    q = d[d.p_tbf >= 100][["player_id", "season", "p_tbf", col]].copy()
    q["r"] = q[col].astype(float) / q.p_tbf
    q["r"] -= q.groupby("season").r.transform(lambda x: np.average(x, weights=q.loc[x.index, "p_tbf"]))
    p = q.merge(q.assign(season=q.season - 1), on=["player_id", "season"], suffixes=("", "_n"))
    p = p[~p.season.isin([2019, 2020])]
    w = 2 / (1 / p.p_tbf + 1 / p.p_tbf_n)
    tau2 = max(np.average(p.r * p.r_n, weights=w), 1e-9)
    obs = np.average(q.r ** 2, weights=q.p_tbf)
    vpb = max(obs - tau2, 1e-9) / np.average(1 / q.p_tbf, weights=q.p_tbf)
    return vpb / tau2


def eb_patch(d, col, var):
    k = cov_k(d, col) if col in STATE["cov"] else _orig(d, col, var)
    STATE["ks"].append((STATE["N"], STATE["arm"], col, round(k)))
    return k


pcomp.eb_k = eb_patch
ARMS = {"E0": set(), "E1": {"k", "ubb", "hbp", "con"}, "E2": {"k", "ubb"}}


def run():
    pit = pd.read_parquet(SEASON / "pit.parquet")
    rv = pd.read_parquet(SEASON / "run_values.parquet").set_index("season")
    mix = pd.read_parquet(SEASON / "park_mix.parquet")
    pf = pd.read_parquet(SEASON / "park_factors.parquet")
    pcon, pmom = pcomp.load_pcon(rv)
    _, P = park_run()
    a, rp = pcomp.ARM_SHIP
    d = pcomp.seasons_frame(pit, rv, pcon, a)
    rows = []
    for N in pcomp.TARGETS:
        mp = multipliers(mix, pf, N, "pit", range(N - 3, N + 1))
        tg = P[P.season == N][["player_id", "p_tbf", "ra9_rel", "season"]].copy()
        for arm, cs in ARMS.items():
            STATE.update(N=N, arm=arm, cov=cs)
            pr, _ = pcomp.project(d, rv, pmom, N, a, mp, role_prior=rp)
            tg[arm] = tg.player_id.map(pr.proj).fillna(0.0)
        rows.append(tg)
    R = pd.concat(rows)
    R.to_parquet(ROOT / "work" / "results" / "s7_03_preds.parquet", index=False)
    print("regression constants (raw BF):")
    print(pd.DataFrame(STATE["ks"], columns=["N", "arm", "comp", "k"]).pivot_table(index="N", columns=["arm", "comp"], values="k").to_string())
    x = R[R.season != 2020]
    e1, e2 = x.season <= 2019, x.season >= 2021

    def t(m, c, b="E0"):
        return clustered_t(None, None, x[m].ra9_rel - x[m][b], x[m].ra9_rel - x[m][c], x[m].p_tbf, x[m].season)
    al = x.season > 0
    print("\n| arm | pooled | t vs shipped | 2016-19 (t) | 2021-24 (t) | slope |")
    for c in ARMS:
        sl = np.polyfit(x[c], x.ra9_rel, 1, w=np.sqrt(x.p_tbf))[0]
        tt = (lambda m: f"{t(m, c):.2f}") if c != "E0" else (lambda m: "")
        print(f"| {c} | {wrmse(x.ra9_rel - x[c], x.p_tbf):.4f} | {tt(al)} | {wrmse(x[e1].ra9_rel - x[e1][c], x[e1].p_tbf):.4f} ({tt(e1)}) | "
              f"{wrmse(x[e2].ra9_rel - x[e2][c], x[e2].p_tbf):.4f} ({tt(e2)}) | {sl:.2f} |")
    print(f"E1 vs E2 t: {t(al, 'E1', 'E2'):.2f}")
    print("\nby season:")
    print(R.groupby("season").apply(lambda g: pd.Series({c: wrmse(g.ra9_rel - g[c], g.p_tbf) for c in ARMS}), include_groups=False).round(4).to_string())
    for nm, m in [("SP", None), ("RP", None)]:
        pass
    return R


if __name__ == "__main__":
    run()
