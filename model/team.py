"""S11-01: team win totals from UECKER player projections. See RESEARCH_LOG (pre-registered).
usage: python model/team.py | tee work/results/s11_01_team.txt
Holdout 2025-26 is never scored here.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import marcel  # noqa: E402
import pcomp  # noqa: E402
from age import AGES, ship_hitters, smooth  # noqa: E402
from hprior import mu_table  # noqa: E402
from hprior2 import ship_prior  # noqa: E402
from marcel import COMP, WKEY, marcel_hitters  # noqa: E402
from park_test import clustered_t, multipliers  # noqa: E402
from xcon2 import corr  # noqa: E402
from xcon3 import A_AIR_SHIP, A_GB_SHIP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEASON, API, RETRO, RES = ROOT / "data" / "season", ROOT / "data" / "statsapi", ROOT / "data" / "retro", ROOT / "work" / "results"
TARGETS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
SCORED = [y for y in TARGETS if y != 2020]


def prev(N):
    """last full season before N (2020 skipped for league totals)."""
    return 2019 if N == 2021 else N - 1


# ---------------------------------------------------------------- team results
def team_results(y):
    g = pd.read_parquet(API / f"games_{y}.parquet")
    g = g[g.status.isin(["Final", "Completed Early"])].drop_duplicates("game_pk")
    h = pd.DataFrame({"team_id": g.home_id, "rs": g.home_score, "ra": g.away_score})
    a = pd.DataFrame({"team_id": g.away_id, "rs": g.away_score, "ra": g.home_score})
    x = pd.concat([h, a])
    x["w"] = (x.rs > x.ra).astype(int)
    r = x.groupby("team_id").agg(g=("w", "size"), w=("w", "sum"), rs=("rs", "sum"), ra=("ra", "sum"))
    r["season"] = y
    return r.reset_index()


def pythag(rs, ra, g):
    x = ((rs + ra) / g) ** 0.287
    return rs ** x / (rs ** x + ra ** x)


def _games(y):
    g = pd.read_parquet(API / f"games_{y}.parquet")
    return g[g.status.isin(["Final", "Completed Early"])].drop_duplicates("game_pk")


def _matched(y):
    g = _games(y)
    gl = pd.read_parquet(RETRO / f"gamelog_{y}.parquet").rename(columns={"home": "hcode"})
    gl["date"] = pd.to_datetime(gl.date.astype(str)).dt.strftime("%Y-%m-%d")
    return g.merge(gl, left_on=["date", "away_score", "home_score"], right_on=["date", "vis_r", "home_r"])


def home_park(y):
    """team_id -> main home park (Retrosheet code) on season y's schedule (pregame info). Without a Retrosheet gamelog
    for y, Stats API venue ids are mapped to park codes learned from earlier seasons."""
    if (RETRO / f"gamelog_{y}.parquet").exists():
        m = _matched(y)
        return m.groupby("home_id").park.agg(lambda s: s.mode().iloc[0])
    vm = pd.concat([_matched(z)[["venue_id", "park"]] for z in range(2016, y) if (RETRO / f"gamelog_{z}.parquet").exists()])
    vmap = vm.groupby("venue_id").park.agg(lambda s: s.mode().iloc[0])
    g = _games(y)
    return g.groupby("home_id").venue_id.agg(lambda s: s.mode().iloc[0]).map(vmap)


# ---------------------------------------------------------------- rosters
def roster_team(N, winter=False):
    """mlbam -> team_id. Opening day: active roster wins ties; winter: Dec-1 40-man."""
    r = pd.read_parquet(API / (f"rosters_winter_{N}.parquet" if winter else f"rosters_{N}.parquet"))
    if not winter:
        r = r.assign(o=(r.roster != "active").astype(int)).sort_values("o")
    return r.drop_duplicates("mlbam").set_index("mlbam").team_id


# ---------------------------------------------------------------- player rates for everyone
def hitter_rates(bat, rv, xc, N, curve):
    real = marcel.age_factor
    marcel.age_factor = lambda a: np.ones_like(np.asarray(a, dtype=float))
    p = marcel_hitters(bat, rv, N, full=True).set_index("player_id")
    marcel.age_factor = real
    wN1 = rv.loc[N - 1]
    p["lg"] = sum(wN1[WKEY[c]] * p[f"lg_{c}"] for c in COMP)
    c = corr(bat, xc, N, {"air": ("x_air", "act_air"), "Mlin_gb": ("xm_gb", "act_gb")}).reindex(p.index).fillna(0.0)
    p["pred0"] = p.proj + (1 - A_GB_SHIP) * c.Mlin_gb + (1 - A_AIR_SHIP) * c.air
    hist = bat[bat.season.between(N - 3, N - 1) & bat.player_id.isin(p.index)]
    last = hist.sort_values("season").groupby("player_id").tail(1).set_index("player_id")
    p["page"] = last.age + (N - last.season)
    mu, _, _ = mu_table(bat, N)
    p["pred0s"] = p.pred0 + ship_prior(bat, N, p.index.values, np.ones(len(p), bool), mu)
    p["rate"] = ship_hitters(p.pred0s, p.lg, p.page, np.ones(len(p), bool), curve=curve)
    return p.rate


def loso_curves():
    """S5-01 A3 curve refit leaving the target season out, on the S8-03 (Q1) no-age residuals (as in the player tests)."""
    h = pd.read_parquet(RES / "s8_03_preds.parquet")
    ia = np.clip(np.floor(h.page), AGES[0], AGES[-1]).astype(int)
    r = h.woba_rel - h.pred0_Q1
    out = {}
    for N in TARGETS:
        tr = (h.season != 2020) & (h.season != N) & h.has_hist
        out[N] = smooth((h.b_pa * r)[tr].groupby(ia[tr]).sum(), h.b_pa[tr].groupby(ia[tr]).sum())
    return out, h


def pitcher_frames(upto=2024):
    pit = pd.read_parquet(SEASON / "pit.parquet")
    rv = pd.read_parquet(SEASON / "run_values.parquet").set_index("season")
    mix = pd.read_parquet(SEASON / "park_mix.parquet")
    pf = pd.read_parquet(SEASON / "park_factors.parquet")
    pcon, pmom = pcomp.load_pcon(rv)
    a, _ = pcomp.ARM_SHIP
    return pit, rv, mix, pf, pmom, pcomp.seasons_frame(pit, rv, pcon, a, upto), a


def pitcher_rates(PF, N, with_target_mix=False):
    pit, rv, mix, pf, pmom, d, a = PF
    mp = multipliers(mix, pf, N, "pit", range(N - 3, N + 1) if with_target_mix else range(N - 3, N))
    pr, _ = pcomp.project(d, rv, pmom, N, a, mp)
    return pr.proj


# ---------------------------------------------------------------- filler rates
def filler(bat, pit, P, N, winter=False):
    """PT-weighted mean actual rel rate, seasons 2013..N-1 ex-2020, of players without a projection or not on a roster."""
    hs, ps = [], []
    for M in range(2013, N):
        if M == 2020:
            continue
        rt = roster_team(M, winter)
        pm = P[P.season == M]
        cov_h = set(pm[(pm.kind == "hit") & pm.mlbam.isin(rt.index)].player_id)
        cov_p = set(pm[(pm.kind == "pit") & pm.mlbam.isin(rt.index)].player_id)
        b = bat[(bat.season == M) & ~(bat.p_g > 0.5 * bat.b_g) & (bat.b_pa > 0)]
        hs.append(b[~b.player_id.isin(cov_h)][["b_pa", "woba_rel"]])
        q = pit[(pit.season == M) & (pit.p_tbf > 0)]
        ps.append(q[~q.player_id.isin(cov_p)][["p_tbf", "ra9_rel"]])
    h, p = pd.concat(hs).dropna(), pd.concat(ps).dropna()
    return (np.average(h.woba_rel, weights=h.b_pa), np.average(p.ra9_rel, weights=p.p_tbf),
            h.b_pa.sum() / (len(hs) * 30), p.p_tbf.sum() / (len(ps) * 30))


# ---------------------------------------------------------------- team roll-up
def rollup(pt, rate, team, total, fill_rate, mode):
    """pt, rate: per player (index player); team: team_id per player. Returns per-team rel sum (PT x rate) and PT used."""
    x = pd.DataFrame({"pt": pt, "rate": rate, "team": team}).dropna(subset=["team"])
    x["rate"] = x.rate.fillna(fill_rate)
    tot = x.groupby("team").pt.sum()
    if mode == "N0":
        s = pd.Series(1.0, index=tot.index)
    elif mode == "N1":
        s = total / tot
    else:  # N2: scale down only, fill shortfall
        s = np.minimum(1.0, total / tot)
    x["pt_s"] = x.pt * x.team.map(s)
    out = (x.pt_s * x.rate).groupby(x.team).sum()
    used = x.groupby("team").pt_s.sum()
    if mode == "N2":
        out = out + (total - used).clip(lower=0) * fill_rate
    return out, used


def project_season(N, ctx, winter=False, modes=("N0", "N1", "N2"), teams=None, sched=None, pf_asof=None, curve="loso"):
    """teams / sched (season whose schedule gives home parks) / pf_asof / curve override for a live season (no results)."""
    bat, rv, xc, curves, PF, P, res, PFAC = ctx
    pN = prev(N)
    rate_h = hitter_rates(bat, rv, xc, N, curves[N] if curve == "loso" else None)
    rate_p = pitcher_rates(PF, N)
    col = "w1" if winter else "c2"
    pn = P[P.season == N]
    rt = roster_team(N, winter)
    fh, fp, _, _ = filler(bat, PF[0], P, N, winter)
    lg = rv.loc[pN]
    b1 = bat[(bat.season == pN)]
    pa_T = b1[~(b1.p_g > 0.5 * b1.b_g)].b_pa.sum() / 30
    q1 = PF[0][PF[0].season == pN]
    bf_T = q1.p_tbf.sum() / 30
    obf = q1.p_out.sum() / q1.p_tbf.sum()
    lgR = lg.runs / 30
    teams = res[res.season == N].team_id.values if teams is None else teams
    hp = home_park(N if sched is None else sched)
    pfr = PFAC[(PFAC["asof"] == (N if pf_asof is None else pf_asof)) & (PFAC["comp"] == "runs")].set_index("park_id").pf
    park = pd.Series(teams, index=teams).map(hp).map(pfr).fillna(1.0)
    hh, pp = pn[pn.kind == "hit"].set_index("player_id"), pn[pn.kind == "pit"].set_index("player_id")
    out = {}
    for mode in modes:
        rs_rel, pa_used = rollup(hh[col], hh.index.map(rate_h).values, hh.mlbam.map(rt).values, pa_T, fh, mode)
        ra_rel, bf_used = rollup(pp[col], pp.index.map(rate_p).values, pp.mlbam.map(rt).values, bf_T, fp, mode)
        rs_rel = rs_rel.reindex(teams).fillna(pa_T * fh) / lg.woba_scale
        ra_rel = ra_rel.reindex(teams).fillna(bf_T * fp) * obf / 27
        for pk in (True, False):
            ra_p = ra_rel + (lgR * (park - 1) / 2 if pk else 0.0)
            rs, ra = lgR + rs_rel - rs_rel.mean(), lgR + ra_p - ra_p.mean()
            w = 162 * pythag(rs, ra, 162)
            out[(mode, pk)] = pd.DataFrame({"w": w - w.mean() + 81, "rs": rs, "ra": ra,
                                            "pa_cov": pa_used.reindex(teams).values / pa_T,
                                            "bf_cov": bf_used.reindex(teams).values / bf_T})
    return out, (fh, fp)


# ---------------------------------------------------------------- baselines
def baselines(res, N):
    r = res.copy()
    r["wp"] = r.w / r.g * 162
    r["py"] = 162 * pythag(r.rs, r.ra, r.g)
    lag = r[["team_id", "season", "wp", "py"]].assign(season=r.season + 1)
    m = r.merge(lag, on=["team_id", "season"], suffixes=("", "_1"))
    tr = m[(m.season < N) & (m.season != 2020) & (m.season != 2021)]
    out = {}
    for c in ("wp", "py"):
        x, y = tr[f"{c}_1"] - 81, tr.wp - 81
        k = float((x * y).sum() / (x * x).sum())
        cur = m[m.season == N].set_index("team_id")
        out[c] = (81 + k * (cur[f"{c}_1"] - 81), k)
    return out


def run():
    bat = pd.read_parquet(SEASON / "bat.parquet")
    rv = pd.read_parquet(SEASON / "run_values.parquet").set_index("season")
    xc = pd.read_parquet(SEASON / "xcon3.parquet")
    curves, h8 = loso_curves()
    PF = pitcher_frames()
    P = pd.read_parquet(RES / "s9_02_preds.parquet")
    P = P.merge(pd.read_parquet(RES / "s9_03_preds.parquet")[["player_id", "season", "kind", "w1"]], on=["player_id", "season", "kind"], how="left")
    res = pd.concat([team_results(y) for y in range(2010, 2025)])
    PFAC = pd.read_parquet(SEASON / "park_factors.parquet")
    ctx = (bat, rv, xc, curves, PF, P, res, PFAC)

    # --- validation: rates re-run for everyone must match the shipped player-test preds on the scored players
    for N in (2018, 2023):
        rh = hitter_rates(bat, rv, xc, N, curves[N])
        x = h8[(h8.season == N) & h8.has_hist]
        print(f"check hitters {N}: max |diff| vs s8_03 p_Q1 = {np.abs(x.player_id.map(rh) - x.p_Q1).max():.2e}")
        s7 = pd.read_parquet(RES / "s7_03_preds.parquet")
        rp = pitcher_rates(PF, N, with_target_mix=True)
        y = s7[s7.season == N]
        print(f"check pitchers {N}: max |diff| vs s7_03 E1 = {np.abs(y.player_id.map(rp).fillna(0) - y.E1).max():.2e}")

    rows = []
    for N in TARGETS:
        cur = res[res.season == N].set_index("team_id")
        act = cur.w / cur.g * 162
        bl = baselines(res, N)
        base = pd.DataFrame({"season": N, "act": act, "B0": 81.0, "B1": bl["wp"][0], "B2": bl["py"][0]})
        o, f = project_season(N, ctx)
        ow, _ = project_season(N, ctx, winter=True, modes=("N2",))
        for (mode, pk), d in o.items():
            base[f"{mode}{'' if pk else '_nopark'}"] = d.w
        base["W1"] = ow[("N2", True)].w
        base["rs_p"], base["ra_p"] = o[("N2", True)].rs, o[("N2", True)].ra
        base["rs_a"], base["ra_a"] = cur.rs / cur.g * 162, cur.ra / cur.g * 162
        base["pa_cov"], base["bf_cov"] = o[("N2", True)].pa_cov, o[("N2", True)].bf_cov
        base["wpa_cov"], base["wbf_cov"] = ow[("N2", True)].pa_cov, ow[("N2", True)].bf_cov
        rows.append(base.reset_index())
        print(f"{N}: k1 {bl['wp'][1]:.3f} k2 {bl['py'][1]:.3f}  filler rel-wOBA {f[0]:+.4f} rel-RA9 {f[1]:+.3f}  "
              f"roster coverage PA {base.pa_cov.mean():.2f} BF {base.bf_cov.mean():.2f} (winter {base.wpa_cov.mean():.2f}/{base.wbf_cov.mean():.2f})")
    R = pd.concat(rows, ignore_index=True)
    R.to_parquet(RES / "s11_01_preds.parquet", index=False)
    report(R)
    diag(R, res)
    return R


def report(R):
    arms = ["B0", "B1", "B2", "N2", "N1", "N0", "N2_nopark", "W1"]
    x = R[R.season.isin(SCORED)]
    e1, e2 = x.season <= 2019, x.season >= 2021

    def rm(g, c):
        return float(np.sqrt(np.mean((g.act - g[c]) ** 2)))

    def t(g, b, c):
        return clustered_t(None, None, g.act - g[b], g.act - g[c], np.ones(len(g)), g.season)
    print("\nwins per 162, 240 team-seasons (2016-19, 2021-24). t = season-clustered, positive = arm better than B")
    print("| arm | RMSE | MAE | t vs B1 | t vs B2 | 2016-19 RMSE (t vs B2) | 2021-24 RMSE (t vs B2) | slope | SD proj |")
    for c in arms:
        sl = np.polyfit(x[c], x.act, 1)[0] if c != "B0" else np.nan
        tt = lambda g, b: "" if c in ("B0", b) else f"{t(g, b, c):.2f}"
        print(f"| {c} | {rm(x, c):.2f} | {np.mean(np.abs(x.act - x[c])):.2f} | {tt(x, 'B1')} | {tt(x, 'B2')} | "
              f"{rm(x[e1], c):.2f} ({tt(x[e1], 'B2')}) | {rm(x[e2], c):.2f} ({tt(x[e2], 'B2')}) | {sl:.2f} | {x[c].std():.2f} |")
    print(f"SD actual {x.act.std():.2f}.  N2 vs N1 t {t(x, 'N1', 'N2'):.2f}, N2 vs N0 t {t(x, 'N0', 'N2'):.2f}, "
          f"N2 vs nopark t {t(x, 'N2_nopark', 'N2'):.2f}")
    print("\nby season RMSE:")
    print(R.groupby("season").apply(lambda g: pd.Series({c: rm(g, c) for c in arms}), include_groups=False).round(2).to_string())
    x = x.assign(err=x.act - x.N2)
    print("\nbiggest N2 misses:")
    print(x.reindex(x.err.abs().sort_values(ascending=False).index).head(10)[["season", "team_id", "act", "N2", "B2", "err"]].round(1).to_string(index=False))


def diag(R, res):
    """Diagnostic only: runs scored / allowed vs last year's (regressed), each centered within season."""
    r = res.assign(rs=res.rs / res.g * 162, ra=res.ra / res.g * 162)
    lag = r[["team_id", "season", "rs", "ra"]].assign(season=r.season + 1)
    x = R[R.season.isin(SCORED)].merge(lag, on=["team_id", "season"], how="left")
    print("\ndiagnostic (not a test): runs per 162, centered within season")
    for s_ in ("rs", "ra"):
        a = x[f"{s_}_a"] - x.groupby("season")[f"{s_}_a"].transform("mean")
        pz = x[f"{s_}_p"] - x.groupby("season")[f"{s_}_p"].transform("mean")
        l1 = x[s_] - x.groupby("season")[s_].transform("mean")
        k = float((l1 * a).sum() / (l1 * l1).sum())
        print(f"  {s_.upper()}: SD actual {a.std():.1f}  UECKER r {np.corrcoef(pz, a)[0, 1]:.3f} RMSE {np.sqrt(np.mean((a - pz) ** 2)):.1f} "
              f"slope {np.polyfit(pz, a, 1)[0]:.2f} | last year regressed (k {k:.2f}, in-sample) r {np.corrcoef(l1, a)[0, 1]:.3f} "
              f"RMSE {np.sqrt(np.mean((a - k * l1) ** 2)):.1f}")


if __name__ == "__main__":
    run()
