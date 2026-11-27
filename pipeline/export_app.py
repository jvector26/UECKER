"""Export the shipped 2027 projections to the app's JSON (site/data/). No model math here: rates, playing time, rosters and
the team board all come from the shipped code / files (model/team.py, model/team2.py). App v1, session 13.

usage: python pipeline/export_app.py [N]      (default 2027; run after `python model/team2.py --ship N`)

Validation (required, raises on failure): rolling the exported players up by team with team.rollup(mode="N2") plus the same
filler and centering as team.project_season must reproduce team_wins_N rs and (un-stretched) ra.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
import team as T  # noqa: E402
import defense as D  # noqa: E402
import pcomp  # noqa: E402
from team import pythag  # noqa: E402

OUT = ROOT / "site" / "data"
TOL = 0.05  # runs
# Run breakdown (S14). Hitters: only components that cleared the research bar AND have the user's OK are shown.
# bat = shipped. fld / br (S14-01 passed player-level) and pos (S14-03) wait for the user's OK: add the key here to show.
SHOW_HIT = ("bat", "br", "fld", "pos")  # S14: fld/br (S14-01) and pos (S14-03B) approved by user 2026-11-27
PARTS_P = ("k", "ubb", "hbp", "con", "age")   # + "park" in the sum check (always 0 for players: park is applied at team level)
LABEL = {"bat": "Batting", "br": "Baserunning", "fld": "Fielding", "pos": "Position",
         "k": "Strikeouts", "ubb": "Walks", "hbp": "Hit batters", "con": "Contact", "park": "Park", "age": "Age"}
PLUS = {"bat": "the bat", "br": "baserunning", "fld": "the glove", "pos": "a premium position",
        "k": "missing bats", "ubb": "limiting walks", "hbp": "rarely hitting batters", "con": "weak contact allowed",
        "park": "a pitcher-friendly park", "age": "age (still improving)"}
MINUS = {"bat": "the bat", "br": "baserunning", "fld": "the glove", "pos": "an easy position",
         "k": "few strikeouts", "ubb": "walks", "hbp": "hit batters", "con": "hard contact allowed",
         "park": "a hitter-friendly park", "age": "age decline"}
ABBR = {108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC", 113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
        118: "KC", 119: "LAD", 120: "WSH", 121: "NYM", 133: "ATH", 134: "PIT", 135: "SD", 136: "SEA", 137: "SF", 138: "STL",
        139: "TB", 140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI", 144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL"}
DIV = {t: d for d, ts in {"AL East": (110, 111, 139, 141, 147), "AL Central": (114, 116, 118, 142, 145),
                          "AL West": (108, 117, 133, 136, 140), "NL East": (120, 121, 143, 144, 146),
                          "NL Central": (112, 113, 134, 138, 158), "NL West": (109, 115, 119, 135, 137)}.items() for t in ts}


def context(N):
    """Same inputs as team2.ship(N)."""
    winter = not (T.API / f"rosters_{N}.parquet").exists()
    pt = pd.read_parquet(T.SEASON / (f"pt_winter_{N}.parquet" if winter else f"pt_{N}.parquet"))
    bat = pd.read_parquet(T.SEASON / "bat.parquet")
    rv = pd.read_parquet(T.SEASON / "run_values.parquet").set_index("season")
    xc = pd.read_parquet(T.SEASON / "xcon3.parquet")
    PF = T.pitcher_frames(upto=N - 1)
    return winter, pt, bat, rv, xc, PF


def players(N):
    winter, pt, bat, rv, xc, PF = context(N)
    pN = T.prev(N)
    rate_h = T.hitter_rates(bat, rv, xc, N, None)         # shipped curve (curve=None), as team2.ship (curve="ship")
    rate_p = T.pitcher_rates(PF, N)
    q1 = PF[0][PF[0].season == pN]
    obf = q1.p_out.sum() / q1.p_tbf.sum()
    ws = rv.loc[pN].woba_scale
    rt = T.roster_team(N, winter)
    P = pt[["player_id", "mlbam", "kind", "proj"]].copy()
    P["rate"] = np.where(P.kind == "hit", P.player_id.map(rate_h), P.player_id.map(rate_p))
    P["team_id"] = P.mlbam.map(rt)
    # value: hitters runs above average = rel_wOBA / wOBA scale x PA; pitchers runs saved = -rel_RA9 x BF x outs/BF / 27
    h = P.kind == "hit"
    P["runs"] = np.where(h, P.rate / ws * P.proj, -P.rate * P.proj * obf / 27)
    P["rate_std"] = np.where(h, P.rate / ws * 600, -P.rate * 200 / 9)  # per 600 PA / per 200 IP
    # breakdown: pitchers' shipped components (same projection call as team.pitcher_rates, parts=True; display only)
    pit, rvp, mix, pf, pmom, d, a = PF
    mp = T.multipliers(mix, pf, N, "pit", range(N - 3, N))
    pp, _ = pcomp.project(d, rvp, pmom, N, a, mp, parts=True)
    assert np.allclose(pp.proj.reindex(rate_p.index), rate_p, equal_nan=True), "parts projection != shipped pitcher rate"
    for c in PARTS_P + ("park",):
        P[f"x_{c}"] = np.where(h, np.nan, -P.player_id.map(pp[f"part_{c}"]) * P.proj * obf / 27)
    # hitters: bat (shipped) + S14 components (computed for checks; shown only if in SHOW_HIT)
    hp, _ = D.app_parts(N)
    P["x_bat"] = np.where(h, P.runs, np.nan)
    for c in ("fld", "br", "pos"):
        P[f"x_{c}"] = np.where(h, P.mlbam.map(hp[f"{c}_pa"]).fillna(0.0) * P.proj, np.nan)
    return P, dict(winter=winter, ws=ws, obf=obf, bat=bat, pit=PF[0], rv=rv, PF=PF)


def validate(N, P, k):
    """Roll the exported players up exactly as team.project_season(mode N2, park on) and compare to the shipped board."""
    bat, pit, rv, PF = k["bat"], k["pit"], k["rv"], k["PF"]
    pN = T.prev(N)
    winter = k["winter"]
    hist = pd.read_parquet(T.RES / "s9_02_preds.parquet")[["player_id", "mlbam", "season", "kind"]]
    Pc = pd.concat([hist, P[["player_id", "mlbam", "kind"]].assign(season=N)])
    fh, fp, _, _ = T.filler(bat, pit, Pc, N, winter)
    b1 = bat[bat.season == pN]
    pa_T = b1[~(b1.p_g > 0.5 * b1.b_g)].b_pa.sum() / 30
    q1 = pit[pit.season == pN]
    bf_T = q1.p_tbf.sum() / 30
    lgR = rv.loc[pN].runs / 30
    board = pd.read_parquet(T.SEASON / f"team_wins_{N}.parquet").set_index("team_id")
    teams = board.index.values
    PFAC = pd.read_parquet(T.SEASON / "park_factors.parquet")
    sched = N if (T.API / f"games_{N}.parquet").exists() else N - 1
    pfa = min(N, int(PFAC["asof"].max()))
    hp = T.home_park(sched)
    pfr = PFAC[(PFAC["asof"] == pfa) & (PFAC["comp"] == "runs")].set_index("park_id").pf
    park = pd.Series(teams, index=teams).map(hp).map(pfr).fillna(1.0)
    H, Q = P[P.kind == "hit"].set_index("player_id"), P[P.kind == "pit"].set_index("player_id")
    rs_rel, _ = T.rollup(H.proj, H.rate.values, H.team_id.values, pa_T, fh, "N2")
    ra_rel, _ = T.rollup(Q.proj, Q.rate.values, Q.team_id.values, bf_T, fp, "N2")
    rs_rel = rs_rel.reindex(teams).fillna(pa_T * fh) / k["ws"]
    ra_rel = ra_rel.reindex(teams).fillna(bf_T * fp) * k["obf"] / 27 + lgR * (park - 1) / 2
    rs, ra = lgR + rs_rel - rs_rel.mean(), lgR + ra_rel - ra_rel.mean()
    s = json.loads((T.SEASON / "team_v3.json").read_text())["s_ra"]
    ra_board = board.ra.mean() + (board.ra - board.ra.mean()) / s   # undo the V3 stretch
    d_rs, d_ra = (rs - board.rs).abs().max(), (ra - ra_board).abs().max()
    print(f"validation: max |dRS| {d_rs:.4f}, max |dRA| {d_ra:.4f} runs (tol {TOL}); filler {fh:.4f} / {fp:.4f}")
    if d_rs > TOL or d_ra > TOL:
        raise SystemExit("export does not reproduce the shipped board")
    return dict(fill_h=fh, fill_p=fp, pa_T=pa_T, bf_T=bf_T, lgR=lgR)


def r(x, n=1):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def history(bat, pit, N):
    b = bat[bat.season.between(N - 3, N - 1) & (bat.b_pa > 0)]
    hb = {}
    for m, g in b.groupby("mlbam"):
        hb[int(m)] = [[int(x.season), int(x.b_pa), r(x.woba, 3), r(x.woba_rel, 3), int(x.b_hr), int(x.b_bb), int(x.b_so)]
                      for x in g.sort_values("season").itertuples()]
    p = pit[pit.season.between(N - 3, N - 1) & (pit.p_tbf > 0)]
    hp = {}
    for m, g in p.groupby("mlbam"):
        hp[int(m)] = [[int(x.season), int(x.p_tbf), r(x.p_out / 3, 1), int(x.p_so), int(x.p_bb), int(x.p_hr), r(x.ra9, 2),
                       int(x.p_g), int(x.p_gs)] for x in g.sort_values("season").itertuples()]
    return hb, hp


def comment(parts):
    """One line from the biggest plus and the biggest minus (|runs| < 0.5 ignored)."""
    if not parts:
        return None
    hi, lo = max(parts, key=lambda x: x[1]), min(parts, key=lambda x: x[1])
    p = f"Best: {PLUS[hi[0]]} ({hi[1]:+.1f})" if hi[1] >= 0.5 else None
    m = f"weak spot: {MINUS[lo[0]]} ({lo[1]:+.1f})" if lo[1] <= -0.5 else None
    if p and m:
        return f"{p}; {m}."
    if p:
        return f"{p}; no weak spot."
    if m:
        return f"Weak spot: {MINUS[lo[0]]} ({lo[1]:+.1f}); nothing above average."
    return "Average across the board."


def trivia(kind, hist, pt):
    """One true line from the data."""
    if not hist:
        return f"No MLB {'plate appearances' if kind == 'hit' else 'batters faced'} in the last three seasons."
    if kind == "hit":
        best = max(hist, key=lambda x: x[4])
        if best[4] >= 20:
            return f"Hit {best[4]} home runs in {best[0]}."
        tot = sum(x[1] for x in hist)
        return f"{tot:,} plate appearances over the last {len(hist)} season{'s' if len(hist) > 1 else ''}."
    big = max(hist, key=lambda x: x[1])
    if big[8] >= 20:
        return f"Made {big[8]} starts in {big[0]}."
    if big[3] >= 100:
        return f"Struck out {big[3]} batters in {big[0]}."
    return f"Faced {big[1]:,} batters in {big[0]}."


def main(N=2027):
    P, k = players(N)
    v = validate(N, P, k)
    hh, pq = P[P.kind == "hit"], P[P.kind == "pit"]
    dp = (pq[[f"x_{c}" for c in PARTS_P + ("park",)]].sum(axis=1) - pq.runs).abs().max()
    print(f"pitcher parts sum to total: max diff {dp:.6f} runs")
    if dp > 1e-6:
        raise SystemExit("pitcher breakdown does not add up")
    ros = hh[hh.team_id.notna()]
    print(f"hitter S14 components (rostered, not shown unless in SHOW_HIT): league sum fld {ros.x_fld.sum():+.1f}, "
          f"br {ros.x_br.sum():+.1f}, pos {ros.x_pos.sum():+.1f} runs; SD fld {ros.x_fld.std():.2f} br {ros.x_br.std():.2f} pos {ros.x_pos.std():.2f}")
    people = pd.read_parquet(T.API / "people.parquet").drop_duplicates("mlbam").set_index("mlbam")
    ros = pd.read_parquet(T.API / ("rosters_winter_%d.parquet" % N if k["winter"] else "rosters_%d.parquet" % N))
    rpos = ros.drop_duplicates("mlbam").set_index("mlbam").pos
    board = pd.read_parquet(T.SEASON / f"team_wins_{N}.parquet")
    hb, hp = history(k["bat"], k["pit"], N)
    ref = pd.Timestamp(f"{N}-07-01")
    out = []
    for x in P.itertuples():
        m = int(x.mlbam)
        pe = people.loc[m] if m in people.index else None
        name = pe["name"] if pe is not None else x.player_id
        birth = pd.Timestamp(pe["birth"]) if pe is not None and pd.notna(pe["birth"]) else None
        pos = rpos.get(m) or (pe["pos"] if pe is not None else None)
        hist = (hb if x.kind == "hit" else hp).get(m, [])
        keys = SHOW_HIT if x.kind == "hit" else PARTS_P
        parts = [[c, r(getattr(x, f"x_{c}"), 1)] for c in keys]
        # displayed value = sum of shown parts (hitters: batting + approved S14 parts; the board/win totals stay batting-only)
        tot = sum(getattr(x, f"x_{c}") for c in keys) if x.kind == "hit" else x.runs
        tot_std = tot / x.proj * 600 if x.kind == "hit" and x.proj > 0 else (x.rate_std if x.kind != "hit" else 0.0)
        out.append({
            "id": f"{m}{'h' if x.kind == 'hit' else 'p'}", "m": m, "n": name, "k": x.kind,
            "t": None if pd.isna(x.team_id) else int(x.team_id),
            "pos": pos if x.kind == "hit" or pos not in (None, "") else "P",
            "b": pe["bats"] if pe is not None else None, "th": pe["throws"] if pe is not None else None,
            "age": r((ref - birth).days / 365.25, 1) if birth is not None else None,
            "pt": int(round(x.proj)), "rate": r(x.rate, 4 if x.kind == "hit" else 3), "runs": r(tot, 1),
            "std": r(tot_std, 1), "bat": r(x.runs, 1) if x.kind == "hit" else None, "h": hist, "bd": parts, "cm": comment(parts), "tv": trivia(x.kind, hist, x.proj)})
    teams = []
    for b in board.itertuples():
        teams.append({"id": int(b.team_id), "name": b.team, "ab": ABBR.get(int(b.team_id)), "div": DIV.get(int(b.team_id)),
                      "w": r(b.wins), "rs": r(b.rs, 0), "ra": r(b.ra, 0), "rd": r(b.rs - b.ra, 0),
                      "py": r(b.last_pythag), "uw": r(b.uecker_wins), "pc": r(b.pa_cov, 3), "bc": r(b.bf_cov, 3)})
    meta = {"season": N, "built": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            "model": board.model.iloc[0], "inputs": board.inputs.iloc[0],
            "parts": {c: LABEL[c] for c in SHOW_HIT + PARTS_P}, "labels": {"wins": f"Wins: {board.model.iloc[0]} on {board.inputs.iloc[0]}",
                       "hit": "Hitters: shipped rel-wOBA model (Marcel + contact quality + sprint speed + age curve + history prior)",
                       "pit": "Pitchers: shipped component model (K, BB, HBP, contact; covariance regression; park)",
                       "pt": "Playing time: " + ("S9-03 winter stand-in" if k["winter"] else "S9-02 roster + IL")},
            "woba_scale": r(k["ws"], 3), "lg_runs_team": r(v["lgR"], 1), "fill": [r(v["fill_h"], 4), r(v["fill_p"], 3)],
            "n_players": len(out),
            # wins RMSE 2016-24 ex-2020, from work/RESEARCH_LOG.md (S11-02, S12-01); yardstick display only
            "backtest": [["UECKER (2027 board)", 10.28, "V3, the model behind this board"],
                         ["Vegas win totals", 9.66, "the yardstick"],
                         ["Last year's Pythag", 10.86, "regressed to the mean"],
                         ["Last year's wins", 11.08, "regressed to the mean"]]}
    OUT.mkdir(parents=True, exist_ok=True)
    dump = lambda o: json.dumps(o, separators=(",", ":"), ensure_ascii=False)  # noqa: E731
    (OUT / "teams.json").write_text(dump(teams))
    (OUT / "players.json").write_text(dump(out))
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {len(teams)} teams, {len(out)} players ({sum(1 for o in out if o['t'] is None)} free agents)",
          f"players.json {(OUT / 'players.json').stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2027)
