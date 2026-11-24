"""S9-02 playing time with opening-day rosters + IL history (pre-registered in RESEARCH_LOG).

usage: python model/ptime2.py  -> work/results/s9_02_ptime2.txt, s9_02_preds.parquet
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from park_mix_statcast import id_map  # noqa: E402
from park_test import clustered_t  # noqa: E402
from ptime import HFEAT, PFEAT, RES, TARGETS, TRAIN0, build_frames  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "data" / "statsapi"
RFEAT = ["od_active", "od_40", "od_minors", "od_il", "od_il60", "il_sh1", "il_sh2", "il_stints3"]

PLACE = re.compile(r"(placed|transferred) .* (to|on) the \d+-day (injured|disabled) list", re.I)
ACT = re.compile(r"(activated|reinstated) .* from the \d+-day (injured|disabled) list", re.I)
RETRO = re.compile(r"retroactive to (\w+ \d+, \d{4})", re.I)


def season_span(y):
    g = pd.read_parquet(API / f"games_{y}.parquet")
    d = pd.to_datetime(g[g.status == "Final"].date)
    return d.min(), d.max()


def il_days(y):
    """per mlbam: days on the IL during the regular season of y, and number of stints started."""
    t = pd.read_parquet(API / f"transactions_{y}.parquet").dropna(subset=["mlbam", "desc"])
    t = t[t.desc.str.contains("injured list|disabled list", case=False)].copy()
    t["kind"] = np.where(t.desc.str.contains(ACT), "act", np.where(t.desc.str.contains(PLACE), "place", None))
    t = t[t.kind.notna()]
    rd = t.desc.str.extract(RETRO)[0]
    t["d"] = pd.to_datetime(rd, format="%B %d, %Y", errors="coerce").fillna(pd.to_datetime(t.effective.fillna(t.date), errors="coerce"))
    t = t.dropna(subset=["d"]).sort_values(["mlbam", "d", "kind"], ascending=[True, True, False])  # place before act same day
    s0, s1 = season_span(y)
    out = {}
    for pid, g in t.groupby("mlbam"):
        days, stints, open_ = 0.0, 0, None
        for k, d, desc in zip(g.kind, g.d, g.desc):
            if k == "place":
                if open_ is None:
                    open_ = d
                    if not desc.lower().startswith("transferred") and "transferred" not in desc.lower():
                        stints += 1
            elif open_ is not None:
                a, b = max(open_, s0), min(d, s1)
                days += max((b - a).days, 0)
                open_ = None
        if open_ is not None:
            days += max((s1 - max(open_, s0)).days, 0)
        out[pid] = (days, stints)
    df = pd.DataFrame.from_dict(out, orient="index", columns=["il_days", "il_stints"])
    df["il_sh"] = df.il_days / ((s1 - s0).days + 1)
    return df


def roster_feats(y):
    r = pd.read_parquet(API / f"rosters_{y}.parquet")
    a = r[(r.roster == "active")].groupby("mlbam").size().gt(0)
    f = r[r.roster == "40Man"]
    sd = f.status_desc.fillna("")
    out = pd.DataFrame({
        "od_active": a,
        "od_40": f.groupby("mlbam").size().gt(0),
        "od_minors": sd.str.contains("Minors").groupby(f.mlbam).any(),
        "od_il": sd.str.contains("Injured").groupby(f.mlbam).any(),
        "od_il60": sd.str.contains("60").groupby(f.mlbam).any(),
    }).fillna(False).astype(float)
    return out


def add_feats(F):
    im = id_map()
    rev = pd.Series(im.index, index=im.values)
    rev = rev[~rev.index.duplicated()]
    F = F.copy()
    F["mlbam"] = F.player_id.map(rev)
    il = {y: il_days(y) for y in range(2010, int(F.season.max()))}
    rf = {y: roster_feats(y) for y in sorted(F.season.unique()) if (API / f"rosters_{y}.parquet").exists()}
    parts = []
    for N, g in F.groupby("season"):
        g = g.copy()
        r = rf.get(N)
        for c in ["od_active", "od_40", "od_minors", "od_il", "od_il60"]:
            g[c] = g.mlbam.map(r[c]).fillna(0.0) if r is not None else np.nan
        for k in (1, 2):
            d = il.get(N - k)
            g[f"il_sh{k}"] = g.mlbam.map(d.il_sh).fillna(0.0) if d is not None else np.nan
        g["il_stints3"] = sum(g.mlbam.map(il[N - k].il_stints).fillna(0) if (N - k) in il else 0 for k in (1, 2, 3))
        parts.append(g)
    return pd.concat(parts)


def fit(F, feats, kind, train0):
    pred = pd.Series(np.nan, index=F.index)
    for N in TARGETS:
        tr = F[(F.season >= train0) & (F.season < N) & (F.season != 2020)]
        te = F.season == N
        if kind == "gb":
            m = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=40, random_state=0)
            m.fit(tr[feats], tr.y)
            pred[te] = np.clip(m.predict(F.loc[te, feats]), 0, None)
        else:
            med = tr[feats].median()
            m = LinearRegression().fit(tr[feats].fillna(med), tr.y)
            pred[te] = np.clip(m.predict(F.loc[te, feats].fillna(med)), 0, None)
    return pred


def ship_pt(N, train_last=2024):
    """S9-02 production (user OK 2026-11-24): hitter PA (H2) and pitcher BF (C2) for season N.
    Needs data/statsapi/rosters_<N>.parquet (opening-day rosters). Trained on targets TRAIN0..train_last ex-2020;
    add 2025-26 at the pre-season final."""
    if not (API / f"rosters_{N}.parquet").exists():
        raise FileNotFoundError(f"rosters_{N}.parquet missing: opening-day rosters not pulled yet (winter stand-in is untested)")
    H, P = build_frames(list(range(TRAIN0, train_last + 1)) + [N])
    out = []
    for F, feats, kind, unit in [(add_feats(H), HFEAT, "hit", "pa"), (add_feats(P), PFEAT, "pit", "bf")]:
        f = feats + RFEAT
        tr = F[(F.season <= train_last) & (F.season != 2020)]
        m = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=40, random_state=0)
        m.fit(tr[f], tr.y)
        te = F[F.season == N].copy()
        te["proj"] = np.clip(m.predict(te[f]), 0, None)
        te["kind"] = kind
        out.append(te[["player_id", "mlbam", "season", "kind", "proj", "b0"] + f].rename(columns={"b0": "marcel"}))
    return pd.concat(out, ignore_index=True)


def rm(g, c):
    return float(np.sqrt(np.mean((g.y - g[c]) ** 2)))


def table(F, arms, refs, L, name):
    ex = F[(F.season != 2020) & F.season.isin(TARGETS)]
    e1, e2 = ex[ex.season <= 2019], ex[ex.season >= 2021]
    L.append(f"\n{name} n={len(ex)}")
    for ref in refs:
        L.append(f"  vs {ref}:")
        for a in arms:
            if a == ref:
                continue
            ts = [clustered_t(None, None, e.y - e[ref], e.y - e[a], np.ones(len(e)), e.season) for e in (ex, e1, e2)]
            L.append(f"    {a:8s} t {ts[0]:6.2f}  (16-19 {ts[1]:6.2f}, 21-24 {ts[2]:6.2f})")
    pl = ex[ex.y > 0]
    L.append(f"  {'arm':8s} {'RMSE':>7s} {'16-19':>7s} {'21-24':>7s} {'mean err':>9s} {'played':>8s}")
    for a in arms:
        L.append(f"  {a:8s} {rm(ex, a):7.1f} {rm(e1, a):7.1f} {rm(e2, a):7.1f} {(ex[a] - ex.y).mean():9.1f} {rm(pl, a):8.1f}")
    by = ex.groupby("season").apply(lambda g: pd.Series({a: rm(g, a) for a in arms}), include_groups=False)
    L.append("  by season:\n" + by.round(1).T.to_string())
    t20 = F[F.season == 2020].assign(y=lambda d: d.y * 162 / 60)
    L.append("  2020 (x162/60): " + ", ".join(f"{a} {rm(t20, a):.1f}" for a in arms))


def bands(F, col, cuts, labels, arms, L):
    ex = F[(F.season != 2020) & F.season.isin(TARGETS)]
    ex = ex.assign(band=pd.cut(ex[col], cuts, labels=labels, right=False))
    g = ex.groupby("band", observed=True)
    tab = pd.DataFrame({"n": g.size(), "mean y": g.y.mean()})
    for a in arms:
        tab[f"{a} err"] = g.apply(lambda s: (s[a] - s.y).mean(), include_groups=False)
        tab[f"{a} rmse"] = g.apply(lambda s: rm(s, a), include_groups=False)
    L.append(f"  by {col}:\n" + tab.round(1).to_string())
    for nm, m in [("on OD IL", ex.od_il == 1), ("not on any OD 40-man/active", (ex.od_40 == 0) & (ex.od_active == 0))]:
        s = ex[m]
        L.append(f"  {nm}: n={len(s)} mean y {s.y.mean():.0f}; " + ", ".join(f"{a} err {(s[a] - s.y).mean():+.0f} rmse {rm(s, a):.0f}" for a in arms))


if __name__ == "__main__" and "--ship" in sys.argv:
    N = int(sys.argv[sys.argv.index("--ship") + 1])
    out = ship_pt(N)
    out.to_parquet(ROOT / "data" / "season" / f"pt_{N}.parquet")
    for k, g in out.groupby("kind"):
        print(f"{k}: n={len(g)} total {g.proj.sum():.0f} (Marcel {g.marcel.sum():.0f}); on OD IL {int(g.od_il.sum())}, "
              f"proj of OD-IL players {g[g.od_il == 1].proj.mean():.0f}; top: " + ", ".join(f"{p} {v:.0f}" for p, v in g.nlargest(5, "proj")[["player_id", "proj"]].values))
elif __name__ == "__main__":
    H, P = build_frames()
    H, P = add_feats(H), add_feats(P)
    L = ["S9-02 playing time + opening-day rosters + IL history (targets 2016-24 ex-2020; all arms trained on targets 2013..N-1)."]
    for F, feats, nm in [(H, HFEAT, "HITTERS (PA)"), (P, PFEAT, "PITCHERS (BF)")]:
        F["r0"] = fit(F, feats, "gb", TRAIN0)
        F["c2"] = fit(F, feats + RFEAT, "gb", TRAIN0)
        F["c2ols"] = fit(F, feats + RFEAT, "ols", TRAIN0)
        table(F, ["b0", "r0", "c2", "c2ols"], ["b0", "r0", "c2"], L, nm)
        bands(F, "bf1" if "bf1" in F else "pa1", [0, 1, 100, 300, 600, 3000], ["none N-1", "1-99", "100-299", "300-599", "600+"],
              ["b0", "r0", "c2", "c2ols"], L)
    txt = "\n".join(L)
    print(txt)
    (RES / "s9_02_ptime2.txt").write_text(txt + "\n")
    pd.concat([H.assign(kind="hit"), P.assign(kind="pit")]).to_parquet(RES / "s9_02_preds.parquet")
