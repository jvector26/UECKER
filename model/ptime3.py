"""S9-03 winter stand-in for playing time (pre-registered in RESEARCH_LOG).

Features known by Dec 1 of N-1: winter 40-man (data/statsapi/rosters_winter_<N>.parquet), FA / outright moves after the
N-1 season, IL status at the end of N-1, plus S9-02's IL history.
usage: python model/ptime3.py              -> work/results/s9_03_ptime3.txt, s9_03_preds.parquet
       python model/ptime3.py --ship N     -> data/season/pt_winter_<N>.parquet (only after user OK)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ptime import HFEAT, PFEAT, RES, TARGETS, TRAIN0, build_frames  # noqa: E402
from ptime2 import ACT, API, PLACE, RETRO, RFEAT, add_feats, bands, fit, season_span, table  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WFEAT = ["w40", "w_fa", "w_out", "il_end", "il_end60", "il_sh1", "il_sh2", "il_stints3"]
OUTMOVE = re.compile(r"outright|released|designated .* for assignment", re.I)


def il_end(y):
    """per mlbam: on the IL on the last regular-season day of y (open stint), and whether that stint is 60-day."""
    t = pd.read_parquet(API / f"transactions_{y}.parquet").dropna(subset=["mlbam", "desc"])
    t = t[t.desc.str.contains("injured list|disabled list", case=False)].copy()
    t["kind"] = np.where(t.desc.str.contains(ACT), "act", np.where(t.desc.str.contains(PLACE), "place", None))
    t = t[t.kind.notna()]
    rd = t.desc.str.extract(RETRO)[0]
    t["d"] = pd.to_datetime(rd, format="%B %d, %Y", errors="coerce").fillna(pd.to_datetime(t.effective.fillna(t.date), errors="coerce"))
    _, s1 = season_span(y)
    t = t[t.d <= s1].sort_values(["mlbam", "d", "kind"], ascending=[True, True, False])
    last = t.groupby("mlbam").tail(1)
    last = last[last.kind == "place"]
    return pd.DataFrame({"il_end": 1.0, "il_end60": last.desc.str.contains("60-day").astype(float).values}, index=last.mlbam.values)


def winter_feats(N):
    """features as of the winter snapshot before season N."""
    f = API / f"rosters_winter_{N}.parquet"
    r = pd.read_parquet(f) if f.exists() else None
    snap = pd.Timestamp(r.date.iloc[0]) if r is not None and len(r) else pd.Timestamp(f"{N - 1}-12-01")
    _, s1 = season_span(N - 1)
    t = pd.read_parquet(API / f"transactions_{N - 1}.parquet").dropna(subset=["mlbam", "desc"])
    t["d"] = pd.to_datetime(t.date, errors="coerce")
    t = t[(t.d > s1) & (t.d < snap)]
    on40 = set(r.mlbam) if r is not None else set()
    fa = set(t[t.type.eq("Declared Free Agency") | t.desc.str.contains("elected free agency", case=False)].mlbam) - on40
    out = set(t[t.desc.str.contains(OUTMOVE)].mlbam) - on40
    ids = on40 | fa | out
    df = pd.DataFrame(index=sorted(ids))
    df["w40"] = df.index.isin(on40).astype(float)
    df["w_fa"] = df.index.isin(fa).astype(float)
    df["w_out"] = df.index.isin(out).astype(float)
    return df, r is not None


def add_winter(F):
    F = add_feats(F)  # S9-02 features: od_* (ceiling arm) + IL history
    ie = {y: il_end(y) for y in range(2010, int(F.season.max()))}
    parts = []
    for N, g in F.groupby("season"):
        g = g.copy()
        w, have = winter_feats(N)
        for c in ["w40", "w_fa", "w_out"]:
            g[c] = g.mlbam.map(w[c]).fillna(0.0)
        if not have:
            g["w40"] = np.nan
        e = ie.get(N - 1)
        for c in ["il_end", "il_end60"]:
            g[c] = g.mlbam.map(e[c]).fillna(0.0) if e is not None else np.nan
        parts.append(g)
    return pd.concat(parts)


def w0(F, feats):
    """sensitivity: shipped C2 trees (trained on opening-day features) scored with winter values plugged in."""
    pred = pd.Series(np.nan, index=F.index)
    for N in TARGETS:
        tr = F[(F.season >= TRAIN0) & (F.season < N) & (F.season != 2020)]
        m = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=40, random_state=0)
        m.fit(tr[feats + RFEAT], tr.y)
        te = F[F.season == N].copy()
        te["od_40"] = te.w40
        te["od_active"] = te.w40
        te["od_minors"] = 0.0
        te["od_il"] = te.il_end60
        te["od_il60"] = te.il_end60
        pred[te.index] = np.clip(m.predict(te[feats + RFEAT]), 0, None)
    return pred


def ship_winter(N, train_last=2024):
    H, P = build_frames(list(range(TRAIN0, train_last + 1)) + [N])
    out = []
    for F, feats, kind in [(add_winter(H), HFEAT, "hit"), (add_winter(P), PFEAT, "pit")]:
        f = feats + WFEAT
        tr = F[(F.season <= train_last) & (F.season != 2020)]
        m = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=40, random_state=0)
        m.fit(tr[f], tr.y)
        te = F[F.season == N].copy()
        te["proj"] = np.clip(m.predict(te[f]), 0, None)
        te["kind"] = kind
        out.append(te[["player_id", "mlbam", "season", "kind", "proj", "b0"] + f].rename(columns={"b0": "marcel"}))
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__" and "--ship" in sys.argv:
    N = int(sys.argv[sys.argv.index("--ship") + 1])
    out = ship_winter(N)
    out.to_parquet(ROOT / "data" / "season" / f"pt_winter_{N}.parquet")
    for k, g in out.groupby("kind"):
        print(f"{k}: n={len(g)} total {g.proj.sum():.0f} (Marcel {g.marcel.sum():.0f}); top: "
              + ", ".join(f"{p} {v:.0f}" for p, v in g.nlargest(5, "proj")[["player_id", "proj"]].values))
elif __name__ == "__main__":
    H, P = build_frames()
    H, P = add_winter(H), add_winter(P)
    L = ["S9-03 winter stand-in (Dec 1 of N-1). Targets 2016-24 ex-2020; all arms trained on targets 2013..N-1."]
    for F, feats, nm in [(H, HFEAT, "HITTERS (PA)"), (P, PFEAT, "PITCHERS (BF)")]:
        ex = F[F.season.isin(TARGETS)]
        L.append(f"\n{nm} winter coverage: on Dec 40-man {ex.w40.mean():.3f}, FA {ex.w_fa.mean():.3f}, out {ex.w_out.mean():.3f}, "
                 f"IL at season end {ex.il_end.mean():.3f}; OD-40 among Dec-40 {ex[ex.w40 == 1].od_40.mean():.3f}, "
                 f"among FA {ex[ex.w_fa == 1].od_40.mean():.3f}")
        F["r0"] = fit(F, feats, "gb", TRAIN0)
        F["c2"] = fit(F, feats + RFEAT, "gb", TRAIN0)
        F["w1"] = fit(F, feats + WFEAT, "gb", TRAIN0)
        F["w1ols"] = fit(F, feats + WFEAT, "ols", TRAIN0)
        F["w0"] = w0(F, feats)
        arms = ["b0", "r0", "w1", "w1ols", "w0", "c2"]
        table(F, arms, ["b0", "r0", "w1"], L, nm)
        bands(F, "bf1" if "bf1" in F else "pa1", [0, 1, 100, 300, 600, 3000], ["none N-1", "1-99", "100-299", "300-599", "600+"], arms, L)
        ex = F[(F.season != 2020) & F.season.isin(TARGETS)]
        for lab, m in [("FA in winter", ex.w_fa == 1), ("on Dec 40-man", ex.w40 == 1), ("IL at N-1 end", ex.il_end == 1)]:
            s = ex[m]
            L.append(f"  {lab}: n={len(s)} mean y {s.y.mean():.0f}; " + ", ".join(f"{a} {(s[a] - s.y).mean():+.0f}/{np.sqrt(((s[a] - s.y) ** 2).mean()):.0f}" for a in arms))
    txt = "\n".join(L)
    print(txt)
    (RES / "s9_03_ptime3.txt").write_text(txt + "\n")
    pd.concat([H.assign(kind="hit"), P.assign(kind="pit")]).to_parquet(RES / "s9_03_preds.parquet")
