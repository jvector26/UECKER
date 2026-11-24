"""S9-01 playing-time test: project PA (hitters) / BF (pitchers) for season N from seasons < N.

Population: players with >= 1 PA (BF) in N-1 or N-2; target = PA_N (BF_N), 0 if absent. Pre-registered in RESEARCH_LOG.
usage: python model/ptime.py   -> prints tables, writes work/results/s9_01_ptime.txt and s9_01_preds.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marcel import marcel_hitters, marcel_pitchers, wrmse  # noqa: E402
from park_test import clustered_t  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEASON = ROOT / "data" / "season"
RETRO = ROOT / "data" / "retro"
RES = ROOT / "work" / "results"

TARGETS = list(range(2016, 2025))
TRAIN0 = 2013
PRO = {2020: 162 / 60}
POS = ["c", "1b", "2b", "3b", "ss", "lf", "cf", "rf"]


def daily_feats(y):
    """per player-season: hitter starts, catcher games, late-season PA/BF, teams (from Retrosheet daily lines)."""
    d = pd.read_parquet(RETRO / f"daily_{y}.parquet")
    dt = pd.to_datetime(d.game_dt.astype(str), format="%Y%m%d")
    late = dt >= dt.max() - pd.Timedelta(days=60)
    d["start"] = (d.seq_ct == 1) & (d.slot_ct >= 1) & (d.b_pa.fillna(0) + d.b_g_dh.fillna(0) > 0) | (d[[f"f_{p}_gs" for p in POS]].fillna(0).sum(axis=1) > 0)
    d["fg"] = d[[f"f_{p}_g" for p in POS]].fillna(0).sum(axis=1).clip(upper=1)
    d["late_pa"] = np.where(late, d.b_pa.fillna(0), 0)
    d["late_bf"] = np.where(late, d.p_tbf.fillna(0), 0)
    g = d.groupby("player_id")
    out = pd.DataFrame({
        "starts": g.start.sum(), "fgames": g.fg.sum(), "cgames": g.f_c_g.sum(),
        "late_pa": g.late_pa.sum(), "late_bf": g.late_bf.sum(), "nteams": g.team_id.nunique(),
    })
    out["season"] = y
    return out.reset_index()


def feats_any(y):
    """Retrosheet daily lines when posted, else Statcast (validated on 2025, model/ptime_statcast.py)."""
    if (RETRO / f"daily_{y}.parquet").exists():
        return daily_feats(y)
    from ptime_statcast import daily_feats_statcast
    return daily_feats_statcast(y)


def build_frames(Ns=range(TRAIN0, 2025)):
    Ns = list(Ns)
    last = max(Ns) - 1
    bat = pd.read_parquet(SEASON / "bat.parquet")
    pit = pd.read_parquet(SEASON / "pit.parquet")
    rv = pd.read_parquet(SEASON / "run_values.parquet").set_index("season")
    bat, pit = bat[bat.season <= max(last, 2025)].copy(), pit[pit.season <= max(last, 2025)].copy()
    df = pd.concat([feats_any(y) for y in range(2010, last + 1)])
    # seasons from the Stats API (no Retrosheet yet) lack DH/PH games and pitching games in bat.parquet
    sc = df[df.columns.intersection(["b_g_dh", "b_g_ph"]).tolist() + ["player_id", "season"]] if "b_g_dh" in df else None
    miss = bat.b_g_dh.isna()
    if miss.any() and sc is not None:
        k = bat.loc[miss, ["player_id", "season"]].merge(sc, on=["player_id", "season"], how="left")
        bat.loc[miss, "b_g_dh"] = k.b_g_dh.fillna(0).values
        bat.loc[miss, "b_g_ph"] = k.b_g_ph.fillna(0).values
        k = bat.loc[miss, ["player_id", "season"]].merge(pit[["player_id", "season", "p_g"]], on=["player_id", "season"], how="left")
        bat.loc[miss, "p_g"] = k.p_g.fillna(0).values
    df = df[["player_id", "season", "starts", "fgames", "cgames", "late_pa", "late_bf", "nteams"]]
    # role: a player-season is a "pitcher season" if mostly pitching
    hb = bat.merge(df, on=["player_id", "season"], how="left")
    hb = hb[~(hb.p_g.fillna(0) > 0.5 * hb.b_g)]
    pp = pit.merge(bat[["player_id", "season", "b_g", "p_g"]].rename(columns={"p_g": "pg_b"}), on=["player_id", "season"], how="left")
    pp = pp[~(pp.p_g < 0.5 * pp.b_g.fillna(0))].merge(df, on=["player_id", "season"], how="left")
    pp["p_ip"] = pp.p_out / 3
    for f in (hb, pp):
        f["pro"] = f.season.map(PRO).fillna(1.0)
    lg_bfo = pit.groupby("season").apply(lambda g: g.p_tbf.sum() / g.p_out.sum(), include_groups=False)

    H, P = [], []
    for N in Ns:
        # ---------- hitters
        h = {k: hb[hb.season == N - k].set_index("player_id") for k in (1, 2, 3)}
        ids = h[1].index.union(h[2].index)
        x = pd.DataFrame(index=ids)
        for k in (1, 2, 3):
            x[f"pa{k}"] = (h[k].b_pa * h[k].pro).reindex(ids).fillna(0)
            x[f"pa{k}_raw"] = h[k].b_pa.reindex(ids).fillna(0)
        h1 = h[1].reindex(ids)
        x["start_sh"] = (h1.starts / h1.b_g).fillna(0)
        x["late_sh"] = (h1.late_pa / h1.b_pa).fillna(0)
        x["c_sh"] = (h1.cgames / h1.fgames.where(h1.fgames > 0)).fillna(0)
        x["dh_sh"] = (h1.b_g_dh / h1.b_g).fillna(0)
        x["ph_sh"] = (h1.b_g_ph / h1.b_g).fillna(0)
        x["nteams"] = h1.nteams.fillna(0)
        last = hb[hb.season.between(N - 3, N - 1) & hb.player_id.isin(ids)].sort_values("season").groupby("player_id").tail(1).set_index("player_id")
        x["age"] = (last.age + (N - last.season)).reindex(ids)
        mh = marcel_hitters(bat, rv, N).set_index("player_id")
        x["q"] = mh.proj.reindex(ids)
        x["wpa"] = 5 * x.pa1 + 4 * x.pa2 + 3 * x.pa3
        tgt = hb[hb.season == N].set_index("player_id").b_pa
        x["y"] = tgt.reindex(ids).fillna(0)
        x["b0"] = 0.5 * x.pa1_raw + 0.1 * x.pa2_raw + 200
        x["b1"] = 0.5 * x.pa1 + 0.1 * x.pa2 + 200
        x["season"] = N
        H.append(x.reset_index())
        # ---------- pitchers
        p = {k: pp[pp.season == N - k].set_index("player_id") for k in (1, 2, 3)}
        ids = p[1].index.union(p[2].index)
        x = pd.DataFrame(index=ids)
        for k in (1, 2, 3):
            x[f"bf{k}"] = (p[k].p_tbf * p[k].pro).reindex(ids).fillna(0)
            x[f"ip{k}"] = (p[k].p_ip * p[k].pro).reindex(ids).fillna(0)
            x[f"ip{k}_raw"] = p[k].p_ip.reindex(ids).fillna(0)
        p1 = p[1].reindex(ids)
        p2 = p[2].reindex(ids)
        x["gs_sh"] = (p1.p_gs / p1.p_g).fillna(0)
        x["bf_app"] = (p1.p_tbf / p1.p_g).fillna(0)
        x["late_sh"] = (p1.late_bf / p1.p_tbf).fillna(0)
        x["nteams"] = p1.nteams.fillna(0)
        last = pp[pp.season.between(N - 3, N - 1) & pp.player_id.isin(ids)].sort_values("season").groupby("player_id").tail(1).set_index("player_id")
        x["age"] = (last.age + (N - last.season)).reindex(ids)
        mp = marcel_pitchers(pit, rv, N).set_index("player_id")
        x["q"] = mp.proj.reindex(ids)
        tgt = pp[pp.season == N].set_index("player_id").p_tbf
        x["y"] = tgt.reindex(ids).fillna(0)
        # Marcel role from most recent season with games
        gs_role = np.where(p1.p_g.notna(), p1.p_gs / p1.p_g, p2.p_gs / p2.p_g)
        base = np.where(gs_role >= 0.5, 60, 25)
        bfo = lg_bfo.loc[N - 1]
        x["b0"] = 3 * bfo * (0.5 * x.ip1_raw + 0.1 * x.ip2_raw + base)
        x["b1"] = 3 * bfo * (0.5 * x.ip1 + 0.1 * x.ip2 + base)
        x["season"] = N
        P.append(x.reset_index())
    return pd.concat(H, ignore_index=True), pd.concat(P, ignore_index=True)


HFEAT = ["pa1", "pa2", "pa3", "start_sh", "late_sh", "c_sh", "dh_sh", "ph_sh", "nteams", "age", "q", "wpa"]
PFEAT = ["bf1", "bf2", "bf3", "gs_sh", "bf_app", "late_sh", "nteams", "age", "q"]


def fit_predict(F, feats):
    F = F.copy()
    F["c1"], F["cols"] = np.nan, np.nan
    for N in TARGETS:
        tr = F[(F.season >= TRAIN0) & (F.season < N) & (F.season != 2020)]
        te = F.season == N
        gb = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=40, random_state=0)
        gb.fit(tr[feats], tr.y)
        F.loc[te, "c1"] = np.clip(gb.predict(F.loc[te, feats]), 0, None)
        Xtr = tr[feats].fillna(tr[feats].median())
        lr = LinearRegression().fit(Xtr, tr.y)
        F.loc[te, "cols"] = np.clip(lr.predict(F.loc[te, feats].fillna(tr[feats].median())), 0, None)
    return F[F.season.isin(TARGETS)]


def ship_hitters_pt(N, train_last=2024):
    """S9-01 C1 production: hitter PA projection for season N for everyone with PA in N-1 or N-2.
    Trained on targets TRAIN0..train_last ex-2020 (holdout 2025-26 added at the pre-season final)."""
    H, _ = build_frames(list(range(TRAIN0, train_last + 1)) + [N])
    tr = H[(H.season <= train_last) & (H.season != 2020)]
    gb = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=40, random_state=0)
    gb.fit(tr[HFEAT], tr.y)
    te = H[H.season == N].copy()
    te["pa_proj"] = np.clip(gb.predict(te[HFEAT]), 0, None)
    return te[["player_id", "season", "pa_proj", "b0"] + HFEAT].rename(columns={"b0": "pa_marcel"})


def report(F, name, unit, bands, lines):
    def rm(g, c):
        return float(np.sqrt(np.mean((g.y - g[c]) ** 2)))
    ex = F[F.season != 2020]
    arms = ["b0", "b1", "c1", "cols"]
    base = "b0" if rm(ex, "b0") <= rm(ex, "b1") else "b1"
    lines.append(f"\n{name}: n={len(ex)} (ex-2020), target {unit}; better baseline = {base}")
    lines.append(f"{'arm':6s} {'RMSE':>7s} {'t vs ' + base:>9s} {'16-19':>7s} {'t':>6s} {'21-24':>7s} {'t':>6s} {'mean err':>9s} {'played RMSE':>12s}")
    e1, e2 = ex[ex.season <= 2019], ex[ex.season >= 2021]
    one = np.ones(len(ex))
    for a in arms:
        t = clustered_t(None, None, ex.y - ex[base], ex.y - ex[a], one, ex.season) if a != base else np.nan
        t1 = clustered_t(None, None, e1.y - e1[base], e1.y - e1[a], np.ones(len(e1)), e1.season) if a != base else np.nan
        t2 = clustered_t(None, None, e2.y - e2[base], e2.y - e2[a], np.ones(len(e2)), e2.season) if a != base else np.nan
        pl = ex[ex.y > 0]
        lines.append(f"{a:6s} {rm(ex, a):7.1f} {t:9.2f} {rm(e1, a):7.1f} {t1:6.2f} {rm(e2, a):7.1f} {t2:6.2f} {(ex[a] - ex.y).mean():9.1f} {rm(pl, a):12.1f}")
    by = ex.groupby("season").apply(lambda g: pd.Series({a: rm(g, a) for a in arms}), include_groups=False)
    lines.append("by season RMSE:\n" + by.round(1).T.to_string())
    t20 = F[F.season == 2020]
    lines.append("2020 as target (prorated x162/60 for comparability): " + ", ".join(f"{a} {rm(t20.assign(y=t20.y * 162 / 60), a):.1f}" for a in arms))
    ex = ex.assign(band=pd.cut(ex[bands[0]], bands[1], labels=bands[2], right=False))
    g = ex.groupby("band", observed=True)
    tab = pd.DataFrame({"n": g.size(), "mean y": g.y.mean(), "zero share": g.y.apply(lambda s: (s == 0).mean())})
    for a in ["b0" if base == "b0" else "b1", "c1"]:
        tab[f"{a} err"] = g.apply(lambda s: (s[a] - s.y).mean(), include_groups=False)
        tab[f"{a} RMSE"] = g.apply(lambda s: rm(s, a), include_groups=False)
    lines.append(f"by {bands[0]} band:\n" + tab.round(2).to_string())


if __name__ == "__main__" and "--ship" in sys.argv:
    out = ship_hitters_pt(2027)
    out.to_parquet(SEASON / "hit_pt_2027.parquet")
    print(out.sort_values("pa_proj", ascending=False).head(15).round(1).to_string(index=False))
    print(f"n={len(out)}  total PA {out.pa_proj.sum():.0f} (Marcel {out.pa_marcel.sum():.0f}); >=400 PA: {(out.pa_proj >= 400).sum()}")
elif __name__ == "__main__":
    RES.mkdir(parents=True, exist_ok=True)
    H, P = build_frames()
    Hp, Pp = fit_predict(H, HFEAT), fit_predict(P, PFEAT)
    L = ["S9-01 playing time (targets 2016-2024; headline ex-2020)"]
    report(Hp, "HITTERS", "PA", ("pa1", [0, 1, 100, 300, 500, 2000], ["none N-1", "1-99", "100-299", "300-499", "500+"]), L)
    report(Pp, "PITCHERS", "BF", ("bf1", [0, 1, 100, 300, 600, 2000], ["none N-1", "1-99", "100-299", "300-599", "600+"]), L)
    txt = "\n".join(L)
    print(txt)
    (RES / "s9_01_ptime.txt").write_text(txt + "\n")
    pd.concat([Hp.assign(kind="hit"), Pp.assign(kind="pit")]).to_parquet(RES / "s9_01_preds.parquet")
