"""S4-03: measured sprint speed (Savant) on grounders, vs shipped S3 and the S4-01B SB/3B proxy. See RESEARCH_LOG.
usage: python model/xcon3.py   -> data/season/xcon3.parquet, prints the test
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marcel import TUNE, marcel_hitters, wrmse  # noqa: E402
from park_test import clustered_t  # noqa: E402
from xcon import A_SHIP  # noqa: E402
from xcon2 import contact, corr, speed_scores  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEASON = ROOT / "data" / "season"
K_RUNS = 10.0
# SHIPPED with user OK 2026-11-23 (S4-03): arm M-lin, weight on actual contact results by batted-ball type
ARM_SHIP = "Mlin"
A_GB_SHIP, A_AIR_SHIP = 0.3, 0.1
BAND = 0.5


def measured_speed():
    s = pd.read_parquet(ROOT / "data" / "savant" / "sprint_speed.parquet")
    s = s[["season", "player_id", "sprint_speed", "competitive_runs", "hp_to_1b"]].rename(columns={"player_id": "mlbam"})
    mu = s.groupby("season").apply(lambda d: np.average(d.sprint_speed, weights=d.competitive_runs.clip(lower=1)),
                                   include_groups=False)
    cr = s.competitive_runs.fillna(0)
    s["mspd"] = (cr * s.sprint_speed + K_RUNS * s.season.map(mu)) / (cr + K_RUNS)
    return s, mu


def build(years, rv, bat):
    ids = bat[["season", "mlbam", "player_id", "woba_den"]].dropna(subset=["mlbam"]).astype({"mlbam": int})
    prox = speed_scores(bat)
    ms, mu = measured_speed()
    out, info = [], {}
    for y in years:
        s = contact(y, rv)
        s = s.merge(ids[ids.season == y][["mlbam", "player_id"]], left_on="batter", right_on="mlbam", how="left")
        s = s.merge(prox[prox.season == y][["player_id", "speed"]], on="player_id", how="left")
        s = s.merge(ms[ms.season == y][["mlbam", "mspd"]], on="mlbam", how="left")
        s["speed"] = s.speed.fillna(0.0)
        cov = s[s.gb].mspd.notna().mean()
        s["mspd"] = s.mspd.fillna(mu[y]) - mu[y]
        gb = s[s.gb]
        res = gb.val - gb.xval
        # proxy (S4-01B reproduction)
        sp = gb.speed - gb.speed.mean()
        bp = float(np.sum(sp * res) / np.sum(sp ** 2))
        s["xb"] = s.xval + np.where(s.gb, bp * (s.speed - gb.speed.mean()), 0.0)
        # measured linear
        mc = gb.mspd - gb.mspd.mean()
        bm = float(np.sum(mc * res) / np.sum(mc ** 2))
        s["xm"] = s.xval + np.where(s.gb, bm * (s.mspd - gb.mspd.mean()), 0.0)
        # measured banded, [1,2,1]-smoothed across bands, centered so league total is unchanged
        band = np.floor(s.mspd / BAND).astype(int)
        gband = band[s.gb]
        t = pd.DataFrame({"b": gband, "r": res.values}).groupby("b").r.agg(["sum", "count"])
        t = t.reindex(range(t.index.min() - 1, t.index.max() + 2), fill_value=0)
        sm = (t["sum"].shift(1, fill_value=0) + 2 * t["sum"] + t["sum"].shift(-1, fill_value=0)) / \
             (t["count"].shift(1, fill_value=0) + 2 * t["count"] + t["count"].shift(-1, fill_value=0)).clip(lower=1)
        adj = band.map(sm).fillna(0.0).values
        adj = adj - np.average(adj[s.gb.values])
        s["xmb"] = s.xval + np.where(s.gb, adj, 0.0)
        info[y] = (round(cov, 3), round(bp, 4), round(bm, 4))
        cols = {"act_gb": s.val * s.gb, "act_air": s.val * ~s.gb, "x_gb": s.xval * s.gb, "x_air": s.xval * ~s.gb,
                "xb_gb": s.xb * s.gb, "xm_gb": s.xm * s.gb, "xmb_gb": s.xmb * s.gb}
        d = pd.DataFrame(cols).assign(batter=s.batter).groupby("batter").sum().reset_index().rename(columns={"batter": "mlbam"})
        d["season"] = y
        print(y, f"grounders w/ measured speed {cov:.3f}, beta proxy {bp:.4f}, beta measured (per ft/s) {bm:.4f}; "
                 f"band adj (slowest..fastest) {np.round(sm[t['count'] > 200].values[[0, -1]], 3)}")
        out.append(d)
    return pd.concat(out, ignore_index=True).merge(ids, on=["season", "mlbam"], how="inner"), info


ARMS = {"A": "x_gb", "B": "xb_gb", "Mlin": "xm_gb", "Mbin": "xmb_gb"}


def run():
    bat = pd.read_parquet(SEASON / "bat.parquet")
    rv = pd.read_parquet(SEASON / "run_values.parquet").set_index("season")
    xc, info = build(range(2015, 2027), rv, bat)
    xc.to_parquet(SEASON / "xcon3.parquet", index=False)
    ms, _ = measured_speed()
    q = ms.merge(ms.assign(season=ms.season - 1), on=["mlbam", "season"], suffixes=("", "_n"))
    print("measured speed y/y corr:", round(q.sprint_speed.corr(q.sprint_speed_n), 3))

    cols = {"air": ("x_air", "act_air")} | {f"{k}_gb": (v, "act_gb") for k, v in ARMS.items()}
    rows = []
    for N in TUNE:
        tgt = bat[(bat.season == N) & (bat.b_pa >= 100) & ~(bat.p_g > 0.5 * bat.b_g)][["player_id", "mlbam", "b_pa", "woba_rel"]]
        p = marcel_hitters(bat, rv, N, full=True).set_index("player_id")
        c = corr(bat, xc, N, cols).reindex(p.index).fillna(0.0).mul(p.af, axis=0)
        m = tgt.merge(pd.concat([p[["proj"]], c], axis=1).reset_index(), on="player_id", how="left").fillna(0.0)
        m["season"] = N
        rows.append(m)
    h = pd.concat(rows, ignore_index=True)

    def pred(d, arm, ag, aa):
        return d.proj + (1 - ag) * d[f"{arm}_gb"] + (1 - aa) * d["air"]

    G = np.round(np.arange(0, 1.01, 0.1), 1)
    ex = h[h.season != 2020]
    h["p_marcel"] = h.proj
    h["p_S3"] = pred(h, "A", A_SHIP, A_SHIP)
    print(f"\nshipped S3 check: {wrmse(ex.woba_rel - pred(ex, 'A', A_SHIP, A_SHIP), ex.b_pa):.5f}")
    for arm in ARMS:
        grid = {(ag, aa): wrmse(ex.woba_rel - pred(ex, arm, ag, aa), ex.b_pa) for ag in G for aa in G}
        best = min(grid, key=grid.get)
        print(f"{arm}: in-sample best (a_gb, a_air) {tuple(map(float, best))} {grid[best]:.5f}; "
              "a_gb curve: " + " ".join(f"{ag}:{grid[(ag, best[1])]:.5f}" for ag in G))
        h[f"p_{arm}"] = np.nan
        picks = {}
        for N in TUNE:
            tr = ex[ex.season != N]
            g2 = {(ag, aa): wrmse(tr.woba_rel - pred(tr, arm, ag, aa), tr.b_pa) for ag in G for aa in G}
            b = min(g2, key=g2.get)
            picks[N] = tuple(map(float, b))
            i = h.season == N
            h.loc[i, f"p_{arm}"] = pred(h[i], arm, *b)
        print("   LOSO picks:", picks)

    ex = h[h.season != 2020]
    subsets = [("2016-24 ex-2020", ex.season > 0), ("2016-19", ex.season <= 2019), ("2021-24", ex.season >= 2021)]
    print("\n| subset | Marcel | S3 | A | B proxy | M-lin | M-bin |")
    for name, msk in subsets:
        e = ex[msk]
        print(f"| {name} | " + " | ".join(f"{wrmse(e.woba_rel - e[f'p_{k}'], e.b_pa):.5f}" for k in ["marcel", "S3"] + list(ARMS)) + " |")
    print("\nseason-clustered t vs S3:")
    for name, msk in subsets:
        e = ex[msk]
        print(f"  {name}: " + "  ".join(
            f"{k} {clustered_t(None, None, e.woba_rel - e.p_S3, e.woba_rel - e[f'p_{k}'], e.b_pa, e.season):.2f}" for k in ARMS))
    e = ex
    print("  t M-lin vs B proxy:", round(clustered_t(None, None, e.woba_rel - e.p_B, e.woba_rel - e.p_Mlin, e.b_pa, e.season), 2),
          "| M-bin vs B proxy:", round(clustered_t(None, None, e.woba_rel - e.p_B, e.woba_rel - e.p_Mbin, e.b_pa, e.season), 2))
    print("\nby season (S3 -> A, B, M-lin, M-bin):")
    for N in TUNE:
        e = h[h.season == N]
        print(N, f"{wrmse(e.woba_rel - e.p_S3, e.b_pa):.4f} ->", " ".join(f"{wrmse(e.woba_rel - e[f'p_{k}'], e.b_pa):.4f}" for k in ARMS))

    # mechanism check: history measured speed (last measured season before N)
    rows = []
    for N in TUNE:
        if N == 2020:
            continue
        hs = ms[ms.season.between(N - 3, N - 1)].sort_values("season").groupby("mlbam").sprint_speed.last()
        rows.append(h[h.season == N].assign(spd=lambda d: d.mlbam.map(hs)))
    e = pd.concat(rows)
    e["grp"] = pd.cut(e.spd, [0, 26.0, 28.5, 40], labels=["slow <26", "mid", "fast >28.5"])
    print("\nmean error (actual - proj) / RMSE by measured speed (history):")
    for g, d in e.groupby("grp", observed=True):
        print(f"  {g:10s} n={len(d):4d}: " + "  ".join(
            f"{k} {np.average(d.woba_rel - d[f'p_{k}'], weights=d.b_pa):+.4f}/{wrmse(d.woba_rel - d[f'p_{k}'], d.b_pa):.4f}"
            for k in ["S3", "B", "Mlin", "Mbin"]))
    print(f"  no measured speed n={e.spd.isna().sum()}")
    return h


if __name__ == "__main__":
    run()
