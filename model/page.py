"""S7-01: pitcher age curve vs the Marcel age factor, on top of shipped S6-01 C2. See RESEARCH_LOG (pre-registered).
usage: python model/page.py | tee work/results/s7_01_page.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcomp  # noqa: E402
pcomp.EB_SHIP = "binom"  # this test was run against the S6-01 constants
from age import AGES, smooth  # noqa: E402
from marcel import age_factor, wrmse  # noqa: E402
from park_test import clustered_t, multipliers, run as park_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEASON = ROOT / "data" / "season"
TUNE = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]


def build():
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
        tg = P[P.season == N][["player_id", "p_tbf", "ra9_rel", "season", "park_804"]].copy()
        ship, _ = pcomp.project(d, rv, pmom, N, a, mp, role_prior=rp)
        pcomp.age_factor = lambda x: np.ones_like(np.asarray(x, dtype=float))
        none, _ = pcomp.project(d, rv, pmom, N, a, mp, role_prior=rp)
        pcomp.age_factor = age_factor
        tg["has_hist"] = tg.player_id.isin(ship.index)
        tg["p_ship"] = tg.player_id.map(ship.proj).fillna(0.0)
        tg["p_none"] = tg.player_id.map(none.proj).fillna(0.0)
        tg["page"] = tg.player_id.map(ship.page)
        rows.append(tg)
    return pd.concat(rows, ignore_index=True)


def run():
    h = build()
    hh = h.has_hist
    ex = h.season != 2020
    print(f"shipped reproduces: {wrmse(h[ex].ra9_rel - h[ex].p_ship, h[ex].p_tbf):.4f} (1.2424 expected); "
          f"no age: {wrmse(h[ex].ra9_rel - h[ex].p_none, h[ex].p_tbf):.4f}")
    ia = pd.Series(np.clip(np.floor(h.page.fillna(28)), AGES[0], AGES[-1]).astype(int), index=h.index)
    for k in ["C0", "A3"]:
        h[f"p_{k}"] = h.p_none
    curves = {}
    for N in TUNE + [2020]:
        tr = ex & (h.season != N) & hh
        i = (h.season == N) & hh
        r = h.ra9_rel - h.p_none
        h.loc[i, "p_C0"] = h.p_none[i] + np.average(r[tr], weights=h.p_tbf[tr])
        cur = smooth((h.p_tbf * r)[tr].groupby(ia[tr]).sum(), h.p_tbf[tr].groupby(ia[tr]).sum())
        curves[N] = cur
        h.loc[i, "p_A3"] = h.p_none[i] + ia[i].map(cur).values
    h["p_A3h"] = 0.5 * h.p_ship + 0.5 * h.p_A3
    # reference curves (in-sample, all 8 targets): residual of no-age, and what Marcel's factor implies
    tr = ex & hh
    r = h.ra9_rel - h.p_none
    full = smooth((h.p_tbf * r)[tr].groupby(ia[tr]).sum(), h.p_tbf[tr].groupby(ia[tr]).sum())
    mimp = ((h.p_ship - h.p_none) * h.p_tbf)[tr].groupby(ia[tr]).sum() / h.p_tbf[tr].groupby(ia[tr]).sum()
    n = h.p_tbf[tr].groupby(ia[tr]).sum()
    print("\nage: measured resid of no-age proj (in-sample, smoothed) | Marcel factor's implied shift | BF")
    for ag in range(21, 41):
        if n.get(ag, 0) > 0:
            print(f"  {ag}: {full[ag]:+.3f} | {mimp.get(ag, np.nan):+.3f} | {n[ag]:.0f}")

    arms = ["ship", "none", "C0", "A3", "A3h"]
    e = h[ex]
    subs = [("2016-24 ex-2020", e.season > 0), ("2016-19", e.season <= 2019), ("2021-24", e.season >= 2021)]
    print("\n| subset | " + " | ".join(arms) + " |")
    for nm, m in subs:
        x = e[m]
        print(f"| {nm} | " + " | ".join(f"{wrmse(x.ra9_rel - x[f'p_{k}'], x.p_tbf):.4f}" for k in arms) + " |")
    print("\nseason-clustered t vs shipped:")
    for nm, m in subs:
        x = e[m]
        print(f"  {nm}: " + "  ".join(
            f"{k} {clustered_t(None, None, x.ra9_rel - x.p_ship, x.ra9_rel - x[f'p_{k}'], x.p_tbf, x.season):.2f}" for k in arms[1:]))
    print("  vs C0: " + "  ".join(
        f"{k} {clustered_t(None, None, e.ra9_rel - e.p_C0, e.ra9_rel - e[f'p_{k}'], e.p_tbf, e.season):.2f}" for k in ["A3", "A3h"]))
    print("\nby season:")
    for N in TUNE:
        x = h[h.season == N]
        print(N, " ".join(f"{k} {wrmse(x.ra9_rel - x[f'p_{k}'], x.p_tbf):.4f}" for k in arms))
    y = h[h.season == 2020]
    print("2020", " ".join(f"{k} {wrmse(y.ra9_rel - y[f'p_{k}'], y.p_tbf):.4f}" for k in arms))
    eh = e[e.has_hist].assign(band=pd.cut(e.page, [0, 25, 28, 31, 34, 60], labels=["<25", "25-27", "28-30", "31-33", "34+"]))
    print("\nmean error (actual - proj, RA9) / RMSE by projected age band, pitchers with history:")
    for b, x in eh.groupby("band", observed=True):
        print(f"  {b:6s} n={len(x):4d}: " + "  ".join(
            f"{k} {np.average(x.ra9_rel - x[f'p_{k}'], weights=x.p_tbf):+.3f}/{wrmse(x.ra9_rel - x[f'p_{k}'], x.p_tbf):.3f}" for k in arms))
    print("  overall: " + "  ".join(f"{k} {np.average(eh.ra9_rel - eh[f'p_{k}'], weights=eh.p_tbf):+.3f}" for k in arms))
    h.to_parquet(ROOT / "work" / "results" / "s7_01_preds.parquet", index=False)
    return h


if __name__ == "__main__":
    run()
