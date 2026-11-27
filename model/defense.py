"""S14: fielding (Savant FRV) and baserunning (Savant BRV) per-player projections, positional adjustment.

  python model/defense.py test        S14-01 player-level test  -> work/results/s14_01_defense.txt, s14_01_preds.parquet
  python model/defense.py pos         S14-03 positional adjustment league-sum check -> work/results/s14_03_pos.txt

Rates per opportunity (fielding: runs per defensive out; baserunning: runs per runner opportunity), Marcel 5/4/3 over
N-3..N-1, shrunk to 0 (league average; FRV is position-relative, BRV league-relative) with constant k:
rate = sum w x runs / (sum w x opps + k). k chosen leave-target-season-out.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAV = ROOT / "data" / "savant"
SEASON = ROOT / "data" / "season"
RES = ROOT / "work" / "results"
WTS = {1: 5, 2: 4, 3: 3}
TARGETS = [2017, 2018, 2019, 2021, 2022, 2023, 2024]   # FRV/BRV start 2016; 2020 not scored
KGRID = {"fld": [500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 13000, 16000, 20000],
         "br": [5, 10, 20, 30, 40, 60, 80, 100, 130, 160, 200, 300, 400]}
MIN_OPP = {"fld": 300, "br": 20}
POS = {2: "C", 3: "1B", 4: "2B", 5: "3B", 6: "SS", 7: "LF", 8: "CF", 9: "RF"}
POS_ADJ_RAW = {"C": 12.5, "SS": 7.5, "2B": 2.5, "3B": 2.5, "CF": 2.5, "LF": -7.5, "RF": -7.5, "1B": -12.5, "DH": -17.5}
POS_SHIFT = 17.5 / 9  # fixed in advance (S14-03): 9 universal-DH slots sum to 0
POS_ADJ = {k: v + POS_SHIFT for k, v in POS_ADJ_RAW.items()}


def seasons():
    """Player-season table: mlbam, season, fld runs/outs, br runs/opps, outs by position."""
    f = pd.read_parquet(SAV / "frv.parquet").rename(columns={"id": "mlbam"})
    fr = pd.read_parquet(SAV / "framing.parquet")
    fr = fr[(fr.season <= 2017) & fr.player_id.notna()]
    fr = fr.assign(mlbam=fr.player_id.astype(int)).groupby(["mlbam", "season"]).runs_extra_strikes.sum()
    f["fr_add"] = pd.MultiIndex.from_frame(f[["mlbam", "season"]]).map(fr).fillna(0.0).values
    f.loc[f.season > 2017, "fr_add"] = 0.0
    f["fld"] = f.total_runs.fillna(0.0) + f.fr_add
    f = f.groupby(["mlbam", "season"]).agg(fld=("fld", "sum"), fld_n=("outs_total", "sum"),
                                           **{f"o{p}": (f"outs_{p}", "sum") for p in POS}).reset_index()
    b = pd.read_parquet(SAV / "brv.parquet")
    b = b.assign(mlbam=pd.to_numeric(b.player_id, errors="coerce"), br=pd.to_numeric(b.runner_runs_tot, errors="coerce"),
                 br_n=pd.to_numeric(b.N_runner_moved, errors="coerce")).dropna(subset=["mlbam"])
    b = b.assign(mlbam=b.mlbam.astype(int)).groupby(["mlbam", "season"])[["br", "br_n"]].sum().reset_index()
    d = f.merge(b, on=["mlbam", "season"], how="outer")
    bat = pd.read_parquet(SEASON / "bat.parquet")
    bat = bat[bat.mlbam.notna()].assign(mlbam=lambda x: x.mlbam.astype(int))
    pa = bat.groupby(["mlbam", "season"]).agg(pa=("b_pa", "sum"), g_dh=("b_g_dh", "sum"), p_g=("p_g", "sum"),
                                              b_g=("b_g", "sum")).reset_index()
    d = d.merge(pa, on=["mlbam", "season"], how="outer")
    return d.fillna({c: 0.0 for c in d.columns if c not in ("mlbam", "season")})


def project(d, comp, N, k):
    """Shrunk rate per opportunity for season N from N-3..N-1 (all players with any history opps)."""
    h = d[d.season.between(N - 3, N - 1)]
    w = (N - h.season).map(WTS)
    num = (w * h[comp]).groupby(h.mlbam).sum()
    den = (w * h[f"{comp}_n"]).groupby(h.mlbam).sum()
    den = den[den > 0]
    return (num.reindex(den.index) / (den + k)), den


def frame(d, comp, k):
    rows = []
    for N in TARGETS:
        r, den = project(d, comp, N, k)
        t = d[(d.season == N) & (d[f"{comp}_n"] >= MIN_OPP[comp])].set_index("mlbam")
        t = t[t.index.isin(r.index)]
        rows.append(pd.DataFrame({"mlbam": t.index, "season": N, "act": t[comp].values, "n": t[f"{comp}_n"].values,
                                  "rate": r.reindex(t.index).values, "hist_n": den.reindex(t.index).values}))
    x = pd.concat(rows, ignore_index=True)
    x["pred"] = x.rate * x.n
    return x


def sse_by_season(d, comp, k):
    x = frame(d, comp, k)
    return ((x.act - x.pred) ** 2).groupby(x.season).sum()


def clustered_t(e0, e1, season):
    """season-clustered t on per-season mean squared error difference (positive = e1 better)."""
    s = pd.DataFrame({"s": season, "d": e0 ** 2 - e1 ** 2}).groupby("s").d.mean()
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s))), s


def yoy(d, comp, lo):
    a = d[[c for c in ("mlbam", "season", comp, f"{comp}_n")]]
    m = a.merge(a.assign(season=a.season - 1), on=["mlbam", "season"], suffixes=("", "_nx"))
    m = m[(m[f"{comp}_n"] >= lo) & (m[f"{comp}_n_nx"] >= lo) & (m.season != 2019) & (m.season != 2020)]
    return np.corrcoef(m[comp] / m[f"{comp}_n"], m[f"{comp}_nx"] / m[f"{comp}_n_nx"])[0, 1], len(m)


def test():
    d = seasons()
    out, preds = [], []
    for comp, lab in (("fld", "FIELDING (FRV, runs/out)"), ("br", "BASERUNNING (BRV, runs/opp)")):
        S = {k: sse_by_season(d, comp, k) for k in KGRID[comp]}
        kN = {N: min(KGRID[comp], key=lambda k: sum(S[k].drop(N))) for N in TARGETS}
        x = pd.concat([frame(d, comp, kN[N]).query("season == @N") for N in TARGETS], ignore_index=True)
        x["comp"] = comp
        preds.append(x)
        e0, e1 = x.act, x.act - x.pred
        t, s = clustered_t(e0, e1, x.season)
        early, late = x.season <= 2019, x.season >= 2021
        te, _ = clustered_t(e0[early], e1[early], x.season[early])
        tl, _ = clustered_t(e0[late], e1[late], x.season[late])
        rm = lambda e: np.sqrt((e ** 2).mean())  # noqa: E731
        full = 4374 if comp == "fld" else float(d[(d.season.isin(TARGETS)) & (d.pa >= 550)].br_n.median())
        kall = min(KGRID[comp], key=lambda k: sum(S[k]))
        slope = np.polyfit(x.pred, x.act, 1)[0]
        r1, n1 = yoy(d, comp, 2500 if comp == "fld" else 60)
        out += [f"{lab}", f"  LOSO k by season: {kN}  (all-seasons k {kall}; full season = {full:.0f} opps -> shrink n/(n+k) "
                f"{full / (full + kall):.2f} for one full season, {12 * full / 5 / (12 * full / 5 + kall):.2f} for three)",
                f"  y/y r of rate (regulars, >= {2500 if comp == 'fld' else 60} opps both years, ex 2019->20/20->21): {r1:.3f} (n {n1})",
                f"  scored n {len(x)}; RMSE runs: zero {rm(e0):.3f}  projected {rm(e1):.3f}  ({100 * (1 - rm(e1) / rm(e0)):.1f}% lower)",
                f"  season-clustered t {t:.2f}; 2017-19 t {te:.2f} ({rm(e0[early]):.3f} -> {rm(e1[early]):.3f}); "
                f"2021-24 t {tl:.2f} ({rm(e0[late]):.3f} -> {rm(e1[late]):.3f})",
                f"  per-season MSE gain: {s.round(2).to_dict()}",
                f"  calibration slope act~pred {slope:.2f}; SD pred {x.pred.std():.2f} act {x.act.std():.2f}"]
        if comp == "fld":
            c = x.mlbam.isin(d[d.o2 > 0].mlbam)
            for nm, m in (("catchers", c), ("non-catchers", ~c)):
                out.append(f"    {nm}: n {m.sum()} RMSE {rm(e0[m]):.3f} -> {rm(e1[m]):.3f}")
    txt = "\n".join(out)
    print(txt)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "s14_01_defense.txt").write_text(txt + "\n")
    pd.concat(preds).to_parquet(RES / "s14_01_preds.parquet", index=False)


# ---------------------------------------------------------------- positional adjustment
def pos_games(d, papg):
    """Defensive games by position and DH games per player-season (DH = PA-equivalent games not covered in the field)."""
    g = pd.DataFrame({POS[p]: d[f"o{p}"] / 27 for p in POS})
    gf = g.sum(axis=1)
    g["DH"] = np.maximum(0.0, d.pa / papg - gf)
    g.loc[d.p_g > 0.5 * d.b_g.clip(lower=1), "DH"] = 0.0   # pitchers batting
    return g


def pos_runs(g):
    return sum(g[c] * POS_ADJ[c] for c in POS_ADJ) / 162


def pos_check():
    d = seasons()
    bat = pd.read_parquet(SEASON / "bat.parquet")
    out = []
    for y in (2022, 2023, 2024, 2025, 2026):
        x = d[d.season == y].reset_index(drop=True)
        b = bat[(bat.season == y) & ~(bat.p_g > 0.5 * bat.b_g)]
        team_g = 30 * 162
        papg = b.b_pa.sum() / (team_g * 9)    # PA per lineup slot per team-game
        g = pos_games(x, papg)
        tot = g.sum()
        r = pos_runs(g)
        out.append(f"{y}: PA/slot-game {papg:.3f}; games by pos " + " ".join(f"{c} {tot[c]:.0f}" for c in POS_ADJ)
                   + f" (team-games {team_g}); league sum {r.sum():+.1f} runs; raw-scale sum {sum(g[c] * POS_ADJ_RAW[c] for c in POS_ADJ_RAW).sum() / 162:+.1f}")
    txt = "\n".join(out)
    print(txt)
    (RES / "s14_03_pos.txt").write_text(txt + "\n")


if __name__ == "__main__":
    {"test": test, "pos": pos_check, "posb": lambda: None, "dhval": lambda: None, "team": lambda: None}[sys.argv[1]]()


# ---------------------------------------------------------------- S14-03B (post-hoc, after the registered construct failed)
def dh_games(y):
    """Measured DH games: PA as DH (Retrosheet bat_fld_cd 10) / league DH PA per team-game. None if no Retrosheet season."""
    p = ROOT / "data" / "retro" / f"pa_{y}.parquet"
    if not p.exists():
        return None
    e = pd.read_parquet(p, columns=["game_id", "bat_id", "bat_fld_cd", "bat_event_fl"])
    e = e[(e.bat_event_fl == "T") | (e.bat_event_fl == True)]  # noqa: E712
    dh = e[e.bat_fld_cd == 10]
    per_tg = len(dh) / (2 * e.game_id.nunique())
    return dh.groupby("bat_id").size() / per_tg


def pos_check_b():
    d = seasons()
    ppl = pd.read_parquet(ROOT / "data" / "ref" / "people.parquet")
    r2m = ppl.dropna(subset=["key_retro", "key_mlbam"]).set_index("key_retro").key_mlbam.astype(int)
    out = []
    for y in (2022, 2023, 2024, 2025):
        x = d[d.season == y].reset_index(drop=True)
        g = pd.DataFrame({POS[p]: x[f"o{p}"] / 27 for p in POS})
        dh = dh_games(y)
        dhm = dh.groupby(dh.index.map(r2m)).sum()
        g["DH"] = x.mlbam.map(dhm).fillna(0.0).values
        tot = g.sum()
        out.append(f"{y} (B, measured DH): games C {tot['C']:.0f} ... 1B {tot['1B']:.0f} DH {tot['DH']:.0f}; "
                   f"unmatched DH games {dh.sum() - tot['DH']:.0f}; league sum {pos_runs(g).sum():+.1f} runs")
    txt = "\n".join(out)
    print(txt)
    with open(RES / "s14_03_pos.txt", "a") as f:
        f.write(txt + "\n")


if __name__ == "__main__" and sys.argv[1] == "posb":
    pos_check_b()


def dh_games_statcast(y):
    """2026 stand-in (no Retrosheet yet): PA by batters who never appear among their team's fielders (fielder_2..9) in that
    game = DH (+ PH-only) PA; scaled so the league total is one DH per team-game. Validated against Retrosheet on 2025."""
    fs = sorted((ROOT / "data" / "statcast").glob(f"{y}_*.parquet"))
    cols = ["game_pk", "inning_topbot", "batter", "events"] + [f"fielder_{i}" for i in range(2, 10)]
    x = pd.concat([pd.read_parquet(f, columns=cols) for f in fs])
    x["bat_side"] = np.where(x.inning_topbot == "Top", "away", "home")   # batting team
    x["fld_side"] = np.where(x.inning_topbot == "Top", "home", "away")
    fl = x.melt(id_vars=["game_pk", "fld_side"], value_vars=[f"fielder_{i}" for i in range(2, 10)], value_name="pid")
    fl = set(zip(fl.game_pk, fl.fld_side, fl.pid.astype("Int64")))
    pa = x[x.events.notna()]
    infield = [(g, s, b) in fl for g, s, b in zip(pa.game_pk, pa.bat_side, pa.batter)]
    dh = pa[~np.array(infield)]
    tg = 2 * x.game_pk.nunique()
    g = dh.groupby("batter").size()
    return g / g.sum() * tg


if __name__ == "__main__" and sys.argv[1] == "dhval":
    ppl = pd.read_parquet(ROOT / "data" / "ref" / "people.parquet")
    r2m = ppl.dropna(subset=["key_retro", "key_mlbam"]).set_index("key_retro").key_mlbam.astype(int)
    a = dh_games(2025)
    a = a.groupby(a.index.map(r2m)).sum()
    b = dh_games_statcast(2025)
    m = pd.concat([a.rename("retro"), b.rename("statcast")], axis=1).fillna(0)
    print("2025 DH games retro vs statcast stand-in: totals", m.sum().round(0).to_dict(), "r", round(m.corr().iloc[0, 1], 3),
          "MAE", round((m.retro - m.statcast).abs().mean(), 2), "n", len(m))
    big = m[m.retro >= 40]
    print("  DH regulars (>= 40 g): mean retro", round(big.retro.mean(), 1), "statcast", round(big.statcast.mean(), 1))


# ---------------------------------------------------------------- S14-02 team level
def player_proj(d, N, k_fld, k_br):
    """Per-PA projected fielding and baserunning runs for season N (history N-3..N-1), plus position-game mix per PA."""
    out = pd.DataFrame()
    if N - 1 < 2016:
        return out
    rf, _ = project(d, "fld", N, k_fld)
    rb, _ = project(d, "br", N, k_br)
    h = d[d.season.between(N - 3, N - 1)]
    w = (N - h.season).map(WTS)
    wpa = (w * h.pa).groupby(h.mlbam).sum()
    opp_f = (w * h.fld_n).groupby(h.mlbam).sum() / wpa
    opp_b = (w * h.br_n).groupby(h.mlbam).sum() / wpa
    out = pd.DataFrame({"fld_pa": rf.reindex(wpa.index).fillna(0) * opp_f, "br_pa": rb.reindex(wpa.index).fillna(0) * opp_b})
    return out[wpa > 0].replace([np.inf, -np.inf], 0).fillna(0.0)


def loso_k(d):
    k = {}
    for comp in ("fld", "br"):
        S = {kk: sse_by_season(d, comp, kk) for kk in KGRID[comp]}
        k[comp] = {N: min(KGRID[comp], key=lambda kk: sum(S[kk].drop(N, errors="ignore"))) for N in range(2016, 2028)}
    return k


def team_terms():
    import team as T
    d = seasons()
    K = loso_k(d)
    P = pd.read_parquet(RES / "s9_02_preds.parquet")
    bat = pd.read_parquet(SEASON / "bat.parquet")
    rows = []
    for N in T.TARGETS:
        pp = player_proj(d, N, K["fld"][N], K["br"][N])
        hh = P[(P.season == N) & (P.kind == "hit")].set_index("player_id")
        rt = T.roster_team(N, False)
        b1 = bat[bat.season == T.prev(N)]
        pa_T = b1[~(b1.p_g > 0.5 * b1.b_g)].b_pa.sum() / 30
        tm = hh.mlbam.map(rt).values
        for comp in ("fld", "br"):
            rate = hh.mlbam.map(pp[f"{comp}_pa"]).values if len(pp) else np.zeros(len(hh))
            tot, _ = T.rollup(hh.c2, rate, tm, pa_T, 0.0, "N2")
            rows.append(pd.DataFrame({"team_id": tot.index.astype(int), "season": N, "comp": comp, "runs": tot.values}))
    x = pd.concat(rows).pivot_table(index=["team_id", "season"], columns="comp", values="runs").reset_index()
    return x.rename(columns={"fld": "FT", "br": "BT"})


def team_test():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from park_test import clustered_t as ct
    import team as T
    from team2 import stretch, loso_slope
    import team3
    R = pd.read_parquet(RES / "s12_01_preds.parquet")
    tt = team_terms()
    R = R.merge(tt, on=["team_id", "season"], how="left").fillna({"FT": 0.0, "BT": 0.0})
    c = lambda df, col: df[col] - df.groupby("season")[col].transform("mean")  # noqa: E731
    R["FT"], R["BT"] = c(R, "FT"), c(R, "BT")
    # D1: flow through the roll-up (RS += BT, RA -= FT), same LOSO RA stretch, then the C1 blend refit LOSO
    R2 = R.assign(rs_p=R.rs_p + R.BT, ra_p=R.ra_p - R.FT)
    rac, rap = c(R2, "ra_a"), c(R2, "ra_p")
    s = {N: loso_slope(rap, rac, R2.season, N) for N in R2.season.unique()}
    R2["U2"] = stretch(R2, s)
    X1, y = team3.design(R2), (R.act - 81).values
    X0 = team3.design(R)
    X2 = np.c_[X0, (R.FT + R.BT).values]
    for arm, X in (("D1", X1), ("D2", X2), ("C1r", X0)):
        R[arm] = np.nan
        co = {}
        for N in R.season.unique():
            m = (R.season.isin(T.SCORED) & (R.season != N)).values
            b = np.linalg.lstsq(X[m], y[m], rcond=None)[0]
            co[int(N)] = np.round(b, 3)
            i = (R.season == N).values
            R.loc[i, arm] = 81 + X[i] @ b
        print(arm, "LOSO coefs:", co)
    R.to_parquet(RES / "s14_02_preds.parquet", index=False)
    out = []
    x = R[R.season.isin(T.SCORED)]
    sc = x[x.season >= 2017]
    out.append(f"team FT SD {x.FT.std():.1f} runs, BT SD {x.BT.std():.1f}; corr FT with last-year Pythag {np.corrcoef(x.FT, x.p1)[0, 1]:.2f}, "
               f"with C1 {np.corrcoef(x.FT, x.C1)[0, 1]:.2f}; corr (FT+BT) with C1 error {np.corrcoef(x.FT + x.BT, x.act - x.C1)[0, 1]:.3f}")
    out.append(f"check: C1 refit here (C1r) {np.sqrt(np.mean((x.act - x.C1r) ** 2)):.3f} vs stored C1 {np.sqrt(np.mean((x.act - x.C1) ** 2)):.3f}")
    rm = lambda g, k: float(np.sqrt(np.mean((g.act - g[k]) ** 2)))  # noqa: E731
    t = lambda g, b, k: ct(None, None, g.act - g[b], g.act - g[k], np.ones(len(g)), g.season)  # noqa: E731
    out.append("| arm | RMSE 16-24 | t vs C1 | t vs V3 | 2016-19 (t vs C1) | 2021-24 (t vs C1) | RMSE 17-24 (t vs C1) |")
    for k in ["V3", "C1", "D1", "D2", "Vegas"]:
        f = lambda g, b: "" if k == b else f"{t(g, b, k):.2f}"  # noqa: E731
        e1, e2 = x.season <= 2019, x.season >= 2021
        out.append(f"| {k} | {rm(x, k):.3f} | {f(x, 'C1')} | {f(x, 'V3')} | {rm(x[e1], k):.3f} ({f(x[e1], 'C1')}) | "
                   f"{rm(x[e2], k):.3f} ({f(x[e2], 'C1')}) | {rm(sc, k):.3f} ({f(sc, 'C1')}) |")
    out.append("by season RMSE:\n" + R[R.season.isin(T.SCORED)].groupby("season").apply(
        lambda g: pd.Series({k: rm(g, k) for k in ["C1", "D1", "D2"]}), include_groups=False).round(3).to_string())
    txt = "\n".join(out)
    print(txt)
    (RES / "s14_02_team.txt").write_text(txt + "\n")


if __name__ == "__main__" and sys.argv[1] == "team":
    team_test()


# ---------------------------------------------------------------- app export (display; S14 components pending user OK)
def pos_season_runs(d):
    """Positional runs per player-season (S14-03 scale; measured DH games: Retrosheet, Statcast stand-in for 2026+)."""
    ppl = pd.read_parquet(ROOT / "data" / "ref" / "people.parquet")
    r2m = ppl.dropna(subset=["key_retro", "key_mlbam"]).set_index("key_retro").key_mlbam.astype(int)
    out = []
    for y in sorted(d.season.unique()):
        if y < 2016:
            continue
        x = d[d.season == y].reset_index(drop=True)
        g = pd.DataFrame({POS[p]: x[f"o{p}"] / 27 for p in POS})
        dh = dh_games(y)
        if dh is not None:
            dh = dh.groupby(dh.index.map(r2m)).sum()
        elif any((ROOT / "data" / "statcast").glob(f"{y}_*.parquet")):
            dh = dh_games_statcast(y)
        else:
            dh = pd.Series(dtype=float)
        g["DH"] = x.mlbam.map(dh).fillna(0.0).values
        out.append(pd.DataFrame({"mlbam": x.mlbam, "season": y, "pos": pos_runs(g).values}))
    return pd.concat(out)


def app_parts(N):
    """Per-PA projected fielding, baserunning and positional runs for season N, by mlbam."""
    d = seasons()
    k = loso_k(d)
    pp = player_proj(d, N, k["fld"][N], k["br"][N])
    ps = pos_season_runs(d[d.season.between(N - 3, N - 1)])
    h = d[d.season.between(N - 3, N - 1)].merge(ps, on=["mlbam", "season"], how="left").fillna({"pos": 0.0})
    w = (N - h.season).map(WTS)
    wpa = (w * h.pa).groupby(h.mlbam).sum()
    pp["pos_pa"] = ((w * h.pos).groupby(h.mlbam).sum() / wpa).reindex(pp.index).fillna(0.0)
    return pp, k
