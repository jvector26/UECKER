"""S5-01: hitter age curve vs the Marcel age factor, on top of shipped S4-03 (M-lin). See RESEARCH_LOG.
usage: python model/age.py   -> prints the test, writes work/results/s5_01_age.txt via tee
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import marcel  # noqa: E402
from marcel import COMP, TUNE, WKEY, marcel_hitters, wrmse  # noqa: E402
from park_test import clustered_t  # noqa: E402
from xcon2 import corr  # noqa: E402
from xcon3 import A_AIR_SHIP, A_GB_SHIP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEASON = ROOT / "data" / "season"
AGES = np.arange(20, 41)
# SHIPPED with user OK 2026-11-23 (S5-01, option 1): hitter projection = (1 - W) x shipped Marcel-age projection
#   + W x (no-age projection + A3 residual curve). Curve = data/season/age_curve_hit.parquet (fit on 2016-24 ex-2020
#   targets; refit at the pre-season final check). Pitchers keep the Marcel age factor.
ARM_SHIP = "A3"
W_SHIP = 0.5


def ship_hitters(pred0, lg, page, has_hist, curve=None):
    """Shipped hitter age handling. pred0 = no-age rel-wOBA projection (Marcel + S4-03 contact/speed), lg = league wOBA
    in the player's mix, page = projected (target-season) age."""
    if curve is None:
        curve = pd.read_parquet(SEASON / "age_curve_hit.parquet").set_index("age").resid
    pred0, lg, page = np.asarray(pred0, float), np.asarray(lg, float), np.asarray(page, float)
    af = np.where(has_hist, marcel.age_factor(page), 1.0)
    marcel_p = af * (pred0 + lg) - lg
    ia = np.clip(np.floor(page), AGES[0], AGES[-1]).astype(int)
    a3 = pred0 + np.where(has_hist, curve.reindex(ia).values, 0.0)
    return (1 - W_SHIP) * marcel_p + W_SHIP * a3


def smooth(s, c):
    """[1,2,1] smoothing of binned sums s / counts c (Series indexed by AGES)."""
    s, c = s.reindex(AGES, fill_value=0.0), c.reindex(AGES, fill_value=0.0)
    num = s.shift(1, fill_value=0) + 2 * s + s.shift(-1, fill_value=0)
    den = c.shift(1, fill_value=0) + 2 * c + c.shift(-1, fill_value=0)
    return (num / den.clip(lower=1e-9)).where(den > 0, 0.0)


def delta_curve(bat, N):
    """Measured same-hitter y->y+1 change in rel-wOBA by integer age, seasons < N only."""
    b = bat[~(bat.p_g > 0.5 * bat.b_g) & (bat.season < N)][["player_id", "season", "age", "woba_rel", "woba_den"]]
    p = b.merge(b.assign(season=b.season - 1), on=["player_id", "season"], suffixes=("", "_n"))
    p = p[p.woba_den > 0]
    w = 2 / (1 / p.woba_den + 1 / p.woba_den_n.clip(lower=1e-9))
    a = np.clip(np.floor(p.age), AGES[0], AGES[-1]).astype(int)
    d = smooth((w * (p.woba_rel_n - p.woba_rel)).groupby(a).sum(), w.groupby(a).sum())
    return d  # d[a] = change from age a to a+1


def chain(d, a0, a1):
    """Cumulative change from fractional age a0 to a1 using per-year deltas d (piecewise linear)."""
    cum = pd.Series(np.r_[0, np.cumsum(d.values)], index=np.r_[AGES, AGES[-1] + 1])
    x = cum.index.values.astype(float)
    return np.interp(np.clip(a1, x[0], x[-1]), x, cum.values) - np.interp(np.clip(a0, x[0], x[-1]), x, cum.values)


def marcel_af(age, peak, ys, os_):
    return np.where(age < peak, 1 + ys * (peak - age), 1 - os_ * (age - peak))


def build():
    bat = pd.read_parquet(SEASON / "bat.parquet")
    rv = pd.read_parquet(SEASON / "run_values.parquet").set_index("season")
    xc = pd.read_parquet(SEASON / "xcon3.parquet")
    real_af = marcel.age_factor
    rows = []
    for N in TUNE:
        tgt = bat[(bat.season == N) & (bat.b_pa >= 100) & ~(bat.p_g > 0.5 * bat.b_g)][["player_id", "mlbam", "b_pa", "woba_rel", "age"]]
        marcel.age_factor = lambda a: np.ones_like(np.asarray(a, dtype=float))
        p = marcel_hitters(bat, rv, N, full=True).set_index("player_id")
        marcel.age_factor = real_af
        wN1 = rv.loc[N - 1]
        p["lg"] = sum(wN1[WKEY[c]] * p[f"lg_{c}"] for c in COMP)
        c = corr(bat, xc, N, {"air": ("x_air", "act_air"), "Mlin_gb": ("xm_gb", "act_gb")}).reindex(p.index).fillna(0.0)
        p["pred0"] = p.proj + (1 - A_GB_SHIP) * c.Mlin_gb + (1 - A_AIR_SHIP) * c.air
        hist = bat[bat.season.between(N - 3, N - 1) & bat.player_id.isin(p.index)]
        last = hist.sort_values("season").groupby("player_id").tail(1).set_index("player_id")
        p["age_last"] = last.age
        p["gap"] = N - last.season
        d = delta_curve(bat, N)
        p["delta"] = chain(d, p.age_last.values, (p.age_last + p.gap).values)
        m = tgt.merge(p[["pred0", "lg", "age_last", "gap", "delta"]].reset_index(), on="player_id", how="left")
        m["has_hist"] = m.pred0.notna()
        m[["pred0", "lg", "delta"]] = m[["pred0", "lg", "delta"]].fillna(0.0)
        m["page"] = (m.age_last + m.gap).fillna(m.age)  # projected (target) age
        m["season"] = N
        rows.append(m)
        print(N, "delta curve (21..37 by 2):", " ".join(f"{a}:{d[a]:+.4f}" for a in range(21, 38, 2)))
    return pd.concat(rows, ignore_index=True), bat


def mult(h, f):
    """Apply a multiplicative factor to absolute projected wOBA, keep it league-relative."""
    f = np.where(h.has_hist, f, 1.0)
    return f * (h.pred0 + h.lg) - h.lg


def run():
    h, bat = build()
    hh = h.has_hist
    h["p_ship"] = mult(h, marcel.age_factor(h.page.values))
    h["p_none"] = h.pred0
    h["p_A2"] = h.pred0 + np.where(hh, h.delta, 0.0)
    ex = h.season != 2020
    print(f"\nshipped reproduces: {wrmse(h[ex].woba_rel - h[ex].p_ship, h[ex].b_pa):.5f}; no age term: "
          f"{wrmse(h[ex].woba_rel - h[ex].p_none, h[ex].b_pa):.5f}")

    grid = [(pk, ys, os_) for pk in [26, 27, 28, 29, 30] for ys in [0, .002, .004, .006, .008]
            for os_ in [.003, .005, .007, .009, .011, .013, .015]]
    ia = np.clip(np.floor(h.page), AGES[0], AGES[-1]).astype(int)
    for k in ["C0", "A1", "A3"]:
        h[f"p_{k}"] = np.nan
    picks = {}
    for N in TUNE:
        tr = ex & (h.season != N) & hh
        i = h.season == N
        r = h.woba_rel - h.pred0
        off = np.average(r[tr], weights=h.b_pa[tr])
        h.loc[i, "p_C0"] = h.pred0[i] + np.where(hh[i], off, 0.0)
        sc = {g: wrmse(h.woba_rel[tr] - mult(h[tr], marcel_af(h.page[tr].values, *g)), h.b_pa[tr]) for g in grid}
        g = min(sc, key=sc.get)
        picks[N] = g
        h.loc[i, "p_A1"] = mult(h[i], marcel_af(h.page[i].values, *g))
        cur = smooth((h.b_pa * r)[tr].groupby(ia[tr]).sum(), h.b_pa[tr].groupby(ia[tr]).sum())
        h.loc[i, "p_A3"] = h.pred0[i] + np.where(hh[i], ia[i].map(cur).values, 0.0)
    print("A1 LOSO picks (peak, young slope, old slope):", picks)

    arms = ["ship", "none", "C0", "A1", "A2", "A3"]
    e = h[ex]
    subsets = [("2016-24 ex-2020", e.season > 0), ("2016-19", e.season <= 2019), ("2021-24", e.season >= 2021)]
    print("\n| subset | " + " | ".join(arms) + " |")
    for name, msk in subsets:
        x = e[msk]
        print(f"| {name} | " + " | ".join(f"{wrmse(x.woba_rel - x[f'p_{k}'], x.b_pa):.5f}" for k in arms) + " |")
    print("\nseason-clustered t vs shipped:")
    for name, msk in subsets:
        x = e[msk]
        print(f"  {name}: " + "  ".join(f"{k} {clustered_t(None, None, x.woba_rel - x.p_ship, x.woba_rel - x[f'p_{k}'], x.b_pa, x.season):.2f}"
                                       for k in arms[1:]))
    print("  vs C0: " + "  ".join(f"{k} {clustered_t(None, None, e.woba_rel - e.p_C0, e.woba_rel - e[f'p_{k}'], e.b_pa, e.season):.2f}"
                                  for k in ["A1", "A2", "A3"]))
    print("\nby season:")
    for N in TUNE:
        x = h[h.season == N]
        print(N, " ".join(f"{k} {wrmse(x.woba_rel - x[f'p_{k}'], x.b_pa):.4f}" for k in arms))

    e = e.assign(band=pd.cut(e.page, [0, 25, 28, 31, 34, 60], labels=["<25", "25-27", "28-30", "31-33", "34+"]))
    print("\nmean error (actual - proj) / RMSE by projected age band:")
    for b, x in e.groupby("band", observed=True):
        print(f"  {b:6s} n={len(x):4d}: " + "  ".join(
            f"{k} {np.average(x.woba_rel - x[f'p_{k}'], weights=x.b_pa):+.4f}/{wrmse(x.woba_rel - x[f'p_{k}'], x.b_pa):.4f}" for k in arms))
    print("  overall mean error: " + "  ".join(f"{k} {np.average(e.woba_rel - e[f'p_{k}'], weights=e.b_pa):+.4f}" for k in arms))
    print("  history only:       " + "  ".join(
        f"{k} {np.average(e[e.has_hist].woba_rel - e[e.has_hist][f'p_{k}'], weights=e[e.has_hist].b_pa):+.4f}" for k in arms))
    x32 = e[e.page >= 32]
    print("  32+ mean error:     " + "  ".join(f"{k} {np.average(x32.woba_rel - x32[f'p_{k}'], weights=x32.b_pa):+.4f}" for k in arms))

    ms = pd.read_parquet(ROOT / "data" / "savant" / "sprint_speed.parquet").rename(columns={"player_id": "mlbam"})
    rows = []
    for N in TUNE:
        if N == 2020:
            continue
        hs = ms[ms.season.between(N - 3, N - 1)].sort_values("season").groupby("mlbam").sprint_speed.last()
        rows.append(h[h.season == N].assign(spd=lambda d: d.mlbam.map(hs)))
    s = pd.concat(rows)
    s["grp"] = pd.cut(s.spd, [0, 26.0, 28.5, 40], labels=["slow", "mid", "fast"])
    print("\nmean error by measured speed:")
    for g, x in s.groupby("grp", observed=True):
        print(f"  {g:5s} n={len(x):4d}: " + "  ".join(f"{k} {np.average(x.woba_rel - x[f'p_{k}'], weights=x.b_pa):+.4f}" for k in arms))
    return h


if __name__ == "__main__":
    run()
