"""S6-01: pitcher component model. Project K, uBB, HBP and contact value (actual/expected blend) separately, each regressed
by its own empirical-Bayes reliability, then rebuild RA9. See RESEARCH_LOG S6-01 (pre-registered).

usage: python model/pcomp.py   -> builds data/season/pcon.parquet (Statcast contact allowed) if missing, prints S6-01 test
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marcel import age_factor, wrmse  # noqa: E402
from park_test import clustered_t, multipliers, run as park_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "data" / "statcast"
SEASON = ROOT / "data" / "season"
TARGETS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
HIT = {"single": "lw_1b", "double": "lw_2b", "triple": "lw_3b", "home_run": "lw_hr"}
OUTS = {"field_out", "force_out", "grounded_into_double_play", "double_play", "triple_play", "fielders_choice",
        "fielders_choice_out", "sac_fly", "sac_fly_double_play", "field_error"}
EV_EDGES = np.arange(40, 124, 3)
LA_EDGES = np.arange(-80, 84, 4)
ARM_SHIP = (0.3, False)  # S6-01 C2 shipped with user OK 2026-11-23: expected-contact blend a=.3, no role prior
# S7-03 shipped with user OK 2026-11-24: regression constants from same-pitcher year-to-year covariance ("cov");
# "binom" = the S6-01 sampling-noise constants (kept for reproducing old tests).
EB_SHIP = "cov"
WTS = {1: 1.0, 2: 2 / 3, 3: 1 / 3}  # Marcel 3/2/1, normalized so N-1 = 1 (k_c is in raw BF)


def statcast_contact(year, rv):
    cols = ["game_pk", "at_bat_number", "pitch_number", "pitcher", "events", "launch_speed", "launch_angle"]
    s = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(SC.glob(f"{year}_*.parquet"))], ignore_index=True)
    s = s[s.events.isin(OUTS | set(HIT))]
    s = s.sort_values("pitch_number").groupby(["game_pk", "at_bat_number"]).tail(1)
    w = rv.loc[year]
    s["val"] = s.events.map({k: w[v] - w["lw_out"] for k, v in HIT.items()}).fillna(0.0)
    s["has_ev"] = s.launch_speed.notna() & s.launch_angle.notna()
    e = np.clip(np.digitize(s.launch_speed.fillna(0), EV_EDGES), 0, len(EV_EDGES))
    a = np.clip(np.digitize(s.launch_angle.fillna(0), LA_EDGES), 0, len(LA_EDGES))
    g = s[s.has_ev]
    S = np.zeros((len(EV_EDGES) + 1, len(LA_EDGES) + 1))
    C = np.zeros_like(S)
    np.add.at(S, (e[s.has_ev.values], a[s.has_ev.values]), g.val.values)
    np.add.at(C, (e[s.has_ev.values], a[s.has_ev.values]), 1.0)
    k = np.array([[1, 1, 1], [1, 4, 1], [1, 1, 1]], float)

    def conv(M):
        P = np.pad(M, 1)
        return sum(k[i, j] * P[i:i + M.shape[0], j:j + M.shape[1]] for i in range(3) for j in range(3))

    V = conv(S) / np.maximum(conv(C), 1e-9)
    s["xval"] = np.where(s.has_ev, V[e, a], s.val)
    out = s.groupby("pitcher").agg(sc_n=("val", "size"), sc_act=("val", "sum"), sc_x=("xval", "sum")).reset_index()
    out["season"] = year
    # event-level second moments for the sampling variance of the blend (per BIP event)
    mom = {f"m2_{al}": float(((al * s.val + (1 - al) * s.xval) ** 2).sum() / len(s)) for al in (0.0, 0.3, 0.5, 1.0)}
    mom.update({f"m1_{al}": float((al * s.val + (1 - al) * s.xval).sum() / len(s)) for al in (0.0, 0.3, 0.5, 1.0)})
    return out.rename(columns={"pitcher": "mlbam"}), dict(season=year, **mom)


def load_pcon(rv):
    f, fm = SEASON / "pcon.parquet", SEASON / "pcon_moments.parquet"
    if f.exists() and fm.exists():
        return pd.read_parquet(f), pd.read_parquet(fm).set_index("season")
    rows, moms = [], []
    for y in range(2015, 2025):
        o, m = statcast_contact(y, rv)
        rows.append(o)
        moms.append(m)
        print("statcast contact", y, len(o), flush=True)
    pc, pm = pd.concat(rows, ignore_index=True), pd.DataFrame(moms)
    pc.to_parquet(f, index=False)
    pm.to_parquet(fm, index=False)
    return pc, pm.set_index("season")


def seasons_frame(pit, rv, pcon, a):
    """Per pitcher-season component totals; con = blended contact value (runs rel. to an out)."""
    d = pit[pit.season <= 2024].copy()
    d["k"], d["ubb"], d["hbp"] = d.p_so, d.p_bb - d.p_ibb, d.p_hp
    w = rv.loc[d.season]
    d["b1"] = d.p_h - d.p_2b - d.p_3b - d.p_hr
    d["con_act"] = (d.b1.values * (w.lw_1b - w.lw_out).values + d.p_2b.values * (w.lw_2b - w.lw_out).values
                    + d.p_3b.values * (w.lw_3b - w.lw_out).values + d.p_hr.values * (w.lw_hr - w.lw_out).values)
    d = d.merge(pcon, on=["mlbam", "season"], how="left")
    has = d.sc_n.notna()
    # replace the Statcast-covered part of actual with the blend; uncovered remainder stays actual
    d["con"] = d.con_act
    d.loc[has, "con"] = d.con_act[has] - (1 - a) * (d.sc_act[has] - d.sc_x[has])
    return d


def eb_k(d, col, var_per_bf):
    """k (raw BF) = sampling variance per BF / true between-pitcher variance; pooled over the given seasons."""
    q = d[d.p_tbf >= 100]
    taus = []
    for s, g in q.groupby("season"):
        r = g[col] / g.p_tbf
        m = np.average(r, weights=g.p_tbf)
        obs = np.average((r - m) ** 2, weights=g.p_tbf)
        samp = var_per_bf[s] * np.average(1.0 / g.p_tbf, weights=g.p_tbf)
        taus.append((obs - samp, len(g)))
    tau2 = max(np.average([t for t, _ in taus], weights=[n for _, n in taus]), 1e-9)
    vpb = np.mean([var_per_bf[s] for s in q.season.unique()])
    return vpb / tau2


def cov_k(d, col):
    """S7-03: k (raw BF) = sampling variance per BF / true variance, true variance = BF-weighted lag-1 covariance of
    league-relative rates for the same pitcher (both seasons BF >= 100; 2019->20, 2020->21 pairs dropped)."""
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


COMPS = ["k", "ubb", "hbp", "con"]


def project(d, rv, pmom, N, a, mp, role_prior=False):
    hist = d[d.season.between(N - 3, N - 1)].copy()
    lg = d[d.season < N].groupby("season")[COMPS + ["p_tbf", "p_out", "p_r"]].sum()
    lgrate = lg[COMPS].div(lg.p_tbf, axis=0)
    lg_rbf = lg.p_r / lg.p_tbf
    # park: allocate run factor to contact, per season
    pm = mp.set_index(["player_id", "season"]).runs
    hist["m"] = pd.MultiIndex.from_frame(hist[["player_id", "season"]]).map(pm).fillna(1.0).values
    hist["con"] = hist.con - (hist.m - 1) * hist.season.map(lg_rbf) * hist.p_tbf
    # EB constants on seasons < N (all available history 2010+)
    eb = d[d.season < N]
    eb = eb.assign(con=eb.con)  # contact not park-neutralized for EB (small effect on spread)
    ks = {}
    for c in ["k", "ubb", "hbp"]:
        p = lgrate[c]
        ks[c] = eb_k(eb, c, (p * (1 - p)).to_dict())
    al = {0.0: 0.0, 0.3: 0.3, 0.5: 0.5, 1.0: 1.0}[a]
    vcon = {}
    w = rv
    for s in lg.index:
        p_bip = 1 - lgrate.loc[s, ["k", "ubb", "hbp"]].sum() - 0.01
        if s in pmom.index:
            m2, m1 = pmom.loc[s, f"m2_{al}"], pmom.loc[s, f"m1_{al}"]
        else:  # pre-Statcast seasons hold actual contact only: use 2015's actual-only moments
            m2, m1 = pmom.loc[2015, "m2_1.0"], pmom.loc[2015, "m1_1.0"]
        vcon[s] = p_bip * m2 - (p_bip * m1) ** 2
    ks["con"] = eb_k(eb, "con", vcon)
    if EB_SHIP == "cov":
        ks = {c: cov_k(eb, c) for c in COMPS}
    hist["w"] = (N - hist.season).map(WTS)
    hist["wbf"] = hist.w * hist.p_tbf
    wbf = hist.groupby("player_id").wbf.sum()
    out = pd.DataFrame(index=wbf.index)
    if role_prior:
        hist["gs"] = hist.w * hist.p_gs
        hist["g"] = hist.w * hist.p_g
        sp = (hist.groupby("player_id").gs.sum() / hist.groupby("player_id").g.sum()) > 0.5
        e2 = d[d.season.between(N - 3, N - 1)].copy()
        e2["sp"] = (e2.p_gs / e2.p_g) > 0.5
    wN1 = rv.loc[N - 1]
    rel = 0.0
    for c in COMPS:
        hist[f"lg_{c}"] = hist.season.map(lgrate[c])
        pl = (hist.w * hist[c]).groupby(hist.player_id).sum()
        lgw = (hist.wbf * hist[f"lg_{c}"]).groupby(hist.player_id).sum() / wbf
        prior = lgw
        if role_prior:
            # role offset: BF-weighted mean (rate - league) by role over N-3..N-1, pitchers BF >= 100
            q = e2[e2.p_tbf >= 100]
            off = ((q[c] / q.p_tbf - q.season.map(lgrate[c])) * q.p_tbf).groupby(q.sp).sum() / q.p_tbf.groupby(q.sp).sum()
            prior = lgw + sp.reindex(lgw.index).map(off).fillna(0.0)
        r = (pl + ks[c] * prior) / (wbf + ks[c])
        dv = 1.0 if c == "con" else (wN1[{"k": "lw_k", "ubb": "lw_bb", "hbp": "lw_hbp"}[c]] - wN1["lw_out"])
        rel = rel + (r - lgw) * dv
    lg_out_bf = lg.loc[N - 1, "p_out"] / lg.loc[N - 1, "p_tbf"]
    lgr = lg.loc[N - 1, "p_r"] / lg.loc[N - 1, "p_out"]
    ra9_rel = 27 * rel / lg_out_bf
    # target park mix: add (m_N - 1) x lg R/BF, then age (as shipped: runs per out / af)
    mN = pm.xs(N, level="season") if N in pm.index.get_level_values("season") else pd.Series(dtype=float)
    mN = mN.reindex(out.index).fillna(1.0)
    ra9_rel = ra9_rel + 27 * (mN - 1) * lg_rbf.loc[N - 1] / lg_out_bf
    last = hist.sort_values("season").groupby("player_id").tail(1).set_index("player_id")
    age = (last.age + (N - last.season)).reindex(out.index).values
    af = age_factor(age)
    absr = lgr + ra9_rel / 27
    out["proj"] = 27 * (absr / af - lgr)
    out["page"] = age  # projected (target-season) age, used by S7-01
    return out, ks


def run(arms):
    pit = pd.read_parquet(SEASON / "pit.parquet")
    rv = pd.read_parquet(SEASON / "run_values.parquet").set_index("season")
    mix = pd.read_parquet(SEASON / "park_mix.parquet")
    pf = pd.read_parquet(SEASON / "park_factors.parquet")
    pcon, pmom = load_pcon(rv)
    _, P = park_run()
    frames = {a: seasons_frame(pit, rv, pcon, a) for a in {x[0] for x in arms.values()}}
    rows, kl = [], []
    for N in TARGETS:
        mp = multipliers(mix, pf, N, "pit", range(N - 3, N + 1))
        tg = P[P.season == N][["player_id", "p_tbf", "ra9_rel", "season", "base_804", "park_804"]].copy()
        for name, (a, rp) in arms.items():
            pr, ks = project(frames[a], rv, pmom, N, a, mp, role_prior=rp)
            tg[name] = tg.player_id.map(pr.proj).fillna(0.0)
            kl.append(dict(season=N, arm=name, **{k: round(v) for k, v in ks.items()}))
        rows.append(tg)
    return pd.concat(rows), pd.DataFrame(kl)


if __name__ == "__main__":
    arms = {"C1": (1.0, False), "C2": (0.3, False), "C3": (0.3, True), "C2_a0": (0.0, False), "C2_a5": (0.5, False)}
    R, K = run(arms)
    R.to_parquet(ROOT / "work" / "results" / "s6_01_preds.parquet", index=False)
    print("EB regression constants (raw BF), by season:")
    print(K[K.arm.isin(["C1", "C2"])].to_string(index=False))
    cols = ["zero", "base_804", "park_804"] + list(arms)
    R["zero"] = 0.0
    x = R[R.season != 2020]
    e1, e2 = x.season.between(2016, 2019), x.season.between(2021, 2024)
    print("\nrel-RA9 BF-weighted RMSE, targets 2016-24 ex-2020")
    for c in cols:
        r = wrmse(x.ra9_rel - x[c], x.p_tbf)
        t = clustered_t(None, None, x.ra9_rel - x.park_804, x.ra9_rel - x[c], x.p_tbf, x.season) if c not in ("zero", "park_804") else np.nan
        t1 = clustered_t(None, None, x[e1].ra9_rel - x[e1].park_804, x[e1].ra9_rel - x[e1][c], x[e1].p_tbf, x[e1].season) if c not in ("zero", "park_804") else np.nan
        t2 = clustered_t(None, None, x[e2].ra9_rel - x[e2].park_804, x[e2].ra9_rel - x[e2][c], x[e2].p_tbf, x[e2].season) if c not in ("zero", "park_804") else np.nan
        sl = np.polyfit(x[c], x.ra9_rel, 1, w=np.sqrt(x.p_tbf))[0] if c != "zero" else np.nan
        print(f"  {c:9s} {r:.4f}  t vs shipped {t:5.2f}  (16-19 {t1:5.2f}, 21-24 {t2:5.2f})  "
              f"16-19 {wrmse(x[e1].ra9_rel - x[e1][c], x[e1].p_tbf):.4f}  21-24 {wrmse(x[e2].ra9_rel - x[e2][c], x[e2].p_tbf):.4f}  slope {sl:.2f}")
    print("\nby season:")
    print(x.groupby("season").apply(lambda g: pd.Series({c: wrmse(g.ra9_rel - g[c], g.p_tbf) for c in cols}), include_groups=False).round(3).to_string())
    y = R[R.season == 2020]
    print("2020:", {c: round(wrmse(y.ra9_rel - y[c], y.p_tbf), 3) for c in cols})
