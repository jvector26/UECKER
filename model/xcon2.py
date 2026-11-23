"""S4-01 (speed on grounders) and S4-02 (spray angle): refinements of S3-01 contact quality. See RESEARCH_LOG session 4.

Baseline = shipped model: Marcel + S3-01 (one weight a=.3 on actual contact results).
Arms:
  S3   shipped (a = .3, EV x LA)                      -> reproduces .0333
  A    split weight: a_gb for grounders (LA<10), a_air for the rest
  B    speed-aware grounder expected value (+ beta x speed score), split weights
  S    spray: EV x LA x pull/center/oppo bins, split weights
  SB   spray + speed
Outputs data/season/xcon2.parquet (per hitter-season components). usage: python model/xcon2.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marcel import TUNE, marcel_hitters, wrmse  # noqa: E402
from park_test import clustered_t  # noqa: E402
from xcon import EV_EDGES, HIT, LA_EDGES, OUTS, A_SHIP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "data" / "statcast"
SEASON = ROOT / "data" / "season"
GB_LA = 10
PULL = 15.0
K3 = np.array([[1, 1, 1], [1, 4, 1], [1, 1, 1]], float)


def conv(M):
    P = np.pad(M, 1)
    return sum(K3[i, j] * P[i:i + M.shape[0], j:j + M.shape[1]] for i in range(3) for j in range(3))


def bin_value(e, a, val, shape):
    S, C = np.zeros(shape), np.zeros(shape)
    np.add.at(S, (e, a), val)
    np.add.at(C, (e, a), 1.0)
    return conv(S) / np.maximum(conv(C), 1e-9)


def contact(year, rv):
    cols = ["game_pk", "at_bat_number", "pitch_number", "batter", "stand", "events", "launch_speed", "launch_angle", "hc_x", "hc_y"]
    s = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(SC.glob(f"{year}_*.parquet"))], ignore_index=True)
    s = s[s.events.isin(OUTS | set(HIT))]
    s = s.sort_values("pitch_number").groupby(["game_pk", "at_bat_number"]).tail(1).copy()
    w = rv.loc[year]
    s["val"] = s.events.map({k: w[v] for k, v in HIT.items()}).fillna(0.0)
    s["has_ev"] = s.launch_speed.notna() & s.launch_angle.notna()
    s["e"] = np.clip(np.digitize(s.launch_speed.fillna(0), EV_EDGES), 0, len(EV_EDGES))
    s["a"] = np.clip(np.digitize(s.launch_angle.fillna(0), LA_EDGES), 0, len(LA_EDGES))
    shape = (len(EV_EDGES) + 1, len(LA_EDGES) + 1)
    g = s[s.has_ev]
    V = bin_value(g.e.values, g.a.values, g.val.values, shape)
    s["xval"] = np.where(s.has_ev, V[s.e.values, s.a.values], s.val)
    # spray: + = toward 1B side; flip for LHB so negative = pull
    ang = np.degrees(np.arctan2(s.hc_x - 125.42, 198.27 - s.hc_y))
    ang = np.where(s.stand == "L", -ang, ang)
    s["zone"] = np.select([ang < -PULL, ang > PULL], [0, 2], 1)
    s["has_sp"] = s.has_ev & s.hc_x.notna() & s.hc_y.notna()
    s["xval_s"] = s.xval
    for z in range(3):
        g = s[s.has_sp & (s.zone == z)]
        Vz = bin_value(g.e.values, g.a.values, g.val.values, shape)
        i = s.has_sp & (s.zone == z)
        s.loc[i, "xval_s"] = Vz[s.loc[i, "e"].values, s.loc[i, "a"].values]
    s["gb"] = s.has_ev & (s.launch_angle < GB_LA)
    return s


def speed_scores(bat):
    """Speed score per hitter-season from baserunning: within-season z of shrunk SB-attempt rate and triple share."""
    b = bat[["season", "player_id", "b_sb", "b_cs", "b_1b", "b_ubb", "b_hp", "b_2b", "b_3b"]].copy()
    opp = b.b_1b + b.b_ubb + b.b_hp
    att = b.b_sb + b.b_cs
    xb = b.b_2b + b.b_3b
    out = []
    for y, d in b.assign(opp=opp, att=att, xb=xb).groupby("season"):
        lg_att = d.att.sum() / d.opp.sum()
        lg_3 = d.b_3b.sum() / d.xb.sum()
        r1 = (d.att + 50 * lg_att) / (d.opp + 50)
        r2 = (d.b_3b + 20 * lg_3) / (d.xb + 20)
        big = d.opp >= 50
        z1 = (r1 - r1[big].mean()) / r1[big].std()
        z2 = (r2 - r2[big].mean()) / r2[big].std()
        out.append(pd.DataFrame({"season": y, "player_id": d.player_id, "speed": (z1 + z2) / 2}))
    return pd.concat(out, ignore_index=True)


def build(years, rv, bat):
    ids = bat[["season", "mlbam", "player_id", "woba_den"]].dropna(subset=["mlbam"]).astype({"mlbam": int})
    spd = speed_scores(bat)
    out, betas = [], {}
    for y in years:
        s = contact(y, rv)
        s = s.merge(ids[ids.season == y][["mlbam", "player_id"]], left_on="batter", right_on="mlbam", how="left")
        s = s.merge(spd[spd.season == y][["player_id", "speed"]], on="player_id", how="left")
        s["speed"] = s.speed.fillna(0.0)
        # beta: grounder residual on speed (BIP-level OLS through the origin after centering), for EV/LA and spray xvals
        gb = s[s.gb]
        sp = gb.speed - gb.speed.mean()
        b1 = float(np.sum(sp * (gb.val - gb.xval)) / np.sum(sp ** 2))
        b2 = float(np.sum(sp * (gb.val - gb.xval_s)) / np.sum(sp ** 2))
        betas[y] = (round(b1, 4), round(b2, 4))
        adj = np.where(s.gb, s.speed - gb.speed.mean(), 0.0)
        s["xval_b"] = s.xval + b1 * adj
        s["xval_sb"] = s.xval_s + b2 * adj
        cov = s.has_sp.mean()
        s["air"] = ~s.gb
        agg = {}
        for part in ["gb", "air"]:
            m = s[part]
            for col, nm in [("val", "act"), ("xval", "x"), ("xval_b", "xb"), ("xval_s", "xs"), ("xval_sb", "xsb")]:
                agg[f"{nm}_{part}"] = (s[col] * m)
        d = pd.DataFrame(agg).assign(batter=s.batter, n_gb=s.gb.astype(float)).groupby("batter").sum().reset_index()
        d = d.rename(columns={"batter": "mlbam"})
        d["season"] = y
        print(y, f"contact {len(s)}, spray cov {cov:.3f}, beta gb (EVLA, spray) {betas[y]}, "
                 f"league act/x {s.val.sum() / s.xval.sum():.4f} act/xs {s.val.sum() / s.xval_s.sum():.4f}")
        out.append(d)
    x = pd.concat(out, ignore_index=True).merge(ids, on=["season", "mlbam"], how="inner")
    return x, betas


def corr(bat, xc, N, cols, reg_pa=1200.0, weights=(5, 4, 3)):
    """For each (x_col, act_col) pair: sum w*(x-act) / (sum w*den + reg)."""
    hist = bat[bat.season.between(N - 3, N - 1)][["player_id", "season", "woba_den"]].copy()
    hist["w"] = hist.season.map({N - 1: weights[0], N - 2: weights[1], N - 3: weights[2]})
    need = sorted({c for pair in cols.values() for c in pair})
    hist = hist.merge(xc[["player_id", "season"] + need], on=["player_id", "season"], how="left").fillna({c: 0 for c in need})
    den = (hist.w * hist.woba_den).groupby(hist.player_id).sum() + reg_pa
    return pd.DataFrame({k: (hist.w * (hist[x] - hist[a])).groupby(hist.player_id).sum() / den for k, (x, a) in cols.items()})


ARMS = {  # arm -> (gb expected col, air expected col)
    "A": ("x_gb", "x_air"),
    "B": ("xb_gb", "x_air"),
    "S": ("xs_gb", "xs_air"),
    "SB": ("xsb_gb", "xs_air"),
}


def run():
    bat = pd.read_parquet(SEASON / "bat.parquet")
    rv = pd.read_parquet(SEASON / "run_values.parquet").set_index("season")
    xc, betas = build(range(2015, 2027), rv, bat)
    xc.to_parquet(SEASON / "xcon2.parquet", index=False)

    # speed-score sanity: y/y correlation
    spd = speed_scores(bat)
    q = spd.merge(spd.assign(season=spd.season - 1), on=["player_id", "season"], suffixes=("", "_n"))
    print("speed score y/y corr:", round(q.speed.corr(q.speed_n), 3))

    cols = {}
    for arm, (xg, xa) in ARMS.items():
        cols[f"{arm}_gb"] = (xg, "act_gb")
        cols[f"{arm}_air"] = (xa, "act_air")
    rows = []
    for N in TUNE:
        tgt = bat[(bat.season == N) & (bat.b_pa >= 100) & ~(bat.p_g > 0.5 * bat.b_g)][["player_id", "b_pa", "woba_rel"]]
        p = marcel_hitters(bat, rv, N, full=True).set_index("player_id")
        c = corr(bat, xc, N, cols).reindex(p.index).fillna(0.0).mul(p.af, axis=0)
        m = tgt.merge(pd.concat([p[["proj"]], c], axis=1).reset_index(), on="player_id", how="left").fillna(0.0)
        m["season"] = N
        rows.append(m)
    h = pd.concat(rows, ignore_index=True)

    def pred(d, arm, ag, aa):
        return d.proj + (1 - ag) * d[f"{arm}_gb"] + (1 - aa) * d[f"{arm}_air"]

    G = np.round(np.arange(0, 1.01, 0.1), 1)
    ex = h[h.season != 2020]
    h["p_marcel"] = h.proj
    h["p_S3"] = pred(h, "A", A_SHIP, A_SHIP)
    print(f"\nshipped S3 check: {wrmse(ex.woba_rel - pred(ex, 'A', A_SHIP, A_SHIP), ex.b_pa):.5f}")
    picks = {}
    for arm in ARMS:
        grid = {(ag, aa): wrmse(ex.woba_rel - pred(ex, arm, ag, aa), ex.b_pa) for ag in G for aa in G}
        best = min(grid, key=grid.get)
        same = min(G, key=lambda a: grid[(a, a)])
        print(f"\n{arm}: best (a_gb, a_air) {best} {grid[best]:.5f} | one shared a {same} {grid[(same, same)]:.5f}")
        print("  a_gb curve at best a_air:", "  ".join(f"{ag}:{grid[(ag, best[1])]:.5f}" for ag in G))
        print("  a_air curve at best a_gb:", "  ".join(f"{aa}:{grid[(best[0], aa)]:.5f}" for aa in G))
        h[f"p_{arm}"] = np.nan
        picks[arm] = {}
        for N in TUNE:
            tr = ex[ex.season != N]
            g2 = {(ag, aa): wrmse(tr.woba_rel - pred(tr, arm, ag, aa), tr.b_pa) for ag in G for aa in G}
            b = min(g2, key=g2.get)
            picks[arm][N] = b
            i = h.season == N
            h.loc[i, f"p_{arm}"] = pred(h[i], arm, *b)
        print("  LOSO picks:", picks[arm])

    ex = h[h.season != 2020]
    print("\n| subset | Marcel | S3 shipped | " + " | ".join(ARMS) + " |")
    subsets = [("2016-24 ex-2020", ex.season > 0), ("2016-19", ex.season <= 2019), ("2021-24", ex.season >= 2021),
               ("2018-24 ex-2020", ex.season >= 2018)]
    for name, msk in subsets:
        e = ex[msk]
        cells = [f"{wrmse(e.woba_rel - e[f'p_{k}'], e.b_pa):.4f}" for k in ["marcel", "S3"] + list(ARMS)]
        print(f"| {name} | " + " | ".join(cells) + " |")
    print("\nseason-clustered t vs S3 shipped:")
    for name, msk in subsets:
        e = ex[msk]
        ts = [clustered_t(None, None, e.woba_rel - e.p_S3, e.woba_rel - e[f"p_{k}"], e.b_pa, e.season) for k in ARMS]
        print(f"  {name}: " + "  ".join(f"{k} {t:.2f}" for k, t in zip(ARMS, ts)))
    print("\nby season (S3 -> A, B, S, SB):")
    for N in TUNE:
        e = h[h.season == N]
        print(N, f"{wrmse(e.woba_rel - e.p_S3, e.b_pa):.4f} ->",
              " ".join(f"{wrmse(e.woba_rel - e[f'p_{k}'], e.b_pa):.4f}" for k in ARMS))

    # who moves: speed subgroups (history speed, weighted)
    sp = spd.copy()
    rows = []
    for N in TUNE:
        if N == 2020:
            continue
        hs = sp[sp.season.between(N - 3, N - 1)].groupby("player_id").speed.mean()
        e = h[h.season == N].assign(spd=lambda d: d.player_id.map(hs))
        rows.append(e)
    e = pd.concat(rows)
    e["grp"] = pd.cut(e.spd, [-9, -0.75, 0.75, 9], labels=["slow", "mid", "fast"])
    print("\nmean error (actual - proj) and RMSE by history speed group:")
    for g, d in e.groupby("grp", observed=True):
        print(f"  {g} n={len(d)}: " + "  ".join(
            f"{k} {np.average(d.woba_rel - d[f'p_{k}'], weights=d.b_pa):+.4f}/{wrmse(d.woba_rel - d[f'p_{k}'], d.b_pa):.4f}"
            for k in ["S3", "A", "B", "S", "SB"]))
    return h


if __name__ == "__main__":
    run()
