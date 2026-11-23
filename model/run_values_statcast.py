"""RE24 linear weights + wOBA weights from Statcast pitch data, for seasons Retrosheet hasn't posted yet (2026).
Same method as model/run_values.py, adapted: one row per PA (last pitch), base-out state from outs_when_up/on_1b..3,
runs on a PA = batting team's score at the next PA (or end of half) minus score now. Non-PA runs (SB, WP) are folded
into the surrounding PA, which Retrosheet's event-level version does not do; validated on 2025 before use.

usage: python model/run_values_statcast.py 2025 2026  -> prints comparison, appends/updates 2026 row in run_values.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "data" / "statcast"
OUTF = ROOT / "data" / "season" / "run_values.parquet"

EV = {"field_out": "out", "force_out": "out", "grounded_into_double_play": "out", "double_play": "out",
      "triple_play": "out", "fielders_choice": "out", "fielders_choice_out": "out", "sac_fly": "out",
      "sac_bunt": "out", "sac_fly_double_play": "out", "sac_bunt_double_play": "out", "strikeout": "k",
      "strikeout_double_play": "k", "walk": "bb", "intent_walk": "ibb", "hit_by_pitch": "hbp", "catcher_interf": "xi",
      "field_error": "roe", "single": "1b", "double": "2b", "triple": "3b", "home_run": "hr"}
SAC = {"sac_fly", "sac_bunt", "sac_fly_double_play", "sac_bunt_double_play"}


def load_pa(year):
    cols = ["game_pk", "game_date", "inning", "inning_topbot", "at_bat_number", "pitch_number", "batter", "pitcher",
            "stand", "p_throws", "outs_when_up", "on_1b", "on_2b", "on_3b", "events", "home_score", "away_score",
            "home_team", "away_team"]
    s = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(SC.glob(f"{year}_*.parquet"))], ignore_index=True)
    s = s.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    first = s.groupby(["game_pk", "at_bat_number"]).head(1).set_index(["game_pk", "at_bat_number"])
    last = s.groupby(["game_pk", "at_bat_number"]).tail(1).set_index(["game_pk", "at_bat_number"])
    pa = last.copy()
    # state and score at the START of the PA
    for c in ["outs_when_up", "on_1b", "on_2b", "on_3b", "home_score", "away_score"]:
        pa[c] = first[c]
    pa = pa.reset_index()
    pa["top"] = pa.inning_topbot == "Top"
    pa["bat_score"] = np.where(pa.top, pa.away_score, pa.home_score)
    pa["half"] = pa.game_pk.astype(str) + "_" + pa.inning.astype(str) + "_" + pa.top.astype(int).astype(str)
    pa["bases"] = pa.on_1b.notna().astype(int) + 2 * pa.on_2b.notna().astype(int) + 4 * pa.on_3b.notna().astype(int)
    pa["state"] = pa.outs_when_up * 8 + pa.bases
    # batting team's score after this PA: score at start of next PA in the game (any half; same team's column)
    pa = pa.sort_values(["game_pk", "at_bat_number"])
    nxt_home = pa.groupby("game_pk").home_score.shift(-1)
    nxt_away = pa.groupby("game_pk").away_score.shift(-1)
    pa["bat_score_next"] = np.where(pa.top, nxt_away, nxt_home)
    pa["last_in_game"] = pa.groupby("game_pk").at_bat_number.transform("max") == pa.at_bat_number
    pa["runs_on_play"] = pa.bat_score_next - pa.bat_score
    pa["ev"] = pa.events.map(EV)
    return pa


def season_run_values(year):
    pa = load_pa(year)
    # use only halves that are not the game's final half (those reach 3 outs) and fully observed
    final_half = pa[pa.last_in_game].set_index("game_pk").half
    f = pa[~pa.half.isin(set(final_half))].copy()
    f = f[f.runs_on_play.notna() & (f.runs_on_play >= 0)]
    f["cum"] = f.groupby("half").runs_on_play.cumsum()
    tot = f.groupby("half").runs_on_play.transform("sum")
    f["runs_roi"] = tot - f.cum + f.runs_on_play
    re = f.groupby("state").runs_roi.mean()
    f["re_start"] = f.state.map(re)
    f["re_next"] = f.groupby("half").re_start.shift(-1).fillna(0.0)
    f["rv"] = f.re_next - f.re_start + f.runs_on_play
    real = f[f.ev.notna()]
    lw = real.groupby("ev").rv.mean()
    allpa = pa[pa.ev.notna()]
    n = allpa.ev.value_counts()
    out_val = (lw["out"] * n["out"] + lw["k"] * n["k"] + lw["roe"] * n["roe"]) / (n["out"] + n["k"] + n["roe"])
    rel = {k: lw[k] - out_val for k in ["bb", "hbp", "1b", "2b", "3b", "hr"]}
    pa_n = len(allpa)
    obp_num = n["bb"] + n["hbp"] + n["1b"] + n["2b"] + n["3b"] + n["hr"]
    obp_den = pa_n - n.get("ibb", 0) - n.get("xi", 0) - allpa.events.isin(SAC).sum()
    lg_obp = obp_num / obp_den
    scale = lg_obp / (sum(rel[k] * n[k] for k in rel) / obp_den)
    runs = pa.groupby("game_pk")[["home_score", "away_score"]].max().sum().sum()  # lower bound; replaced below
    row = {"season": year, "pa": pa_n, "lg_obp": lg_obp, "woba_scale": scale, "lw_out": out_val, "lw_k": lw["k"]}
    for k in rel:
        row[f"lw_{k}"] = lw[k]
        row[f"w_{k}"] = rel[k] * scale
    row["lg_woba"] = lg_obp
    row["source"] = "statcast"
    return row


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]] or [2025, 2026]
    old = pd.read_parquet(OUTF)
    rows = pd.DataFrame([season_run_values(y) for y in years]).set_index("season")
    keys = ["pa", "lg_woba", "woba_scale", "w_bb", "w_hbp", "w_1b", "w_2b", "w_3b", "w_hr", "lw_out"]
    cmp = rows[keys].copy()
    o = old.set_index("season")
    for y in years:
        if y in o.index:
            cmp.loc[f"{y} retro"] = o.loc[y, keys]
    print(cmp.round(4).to_string())
    if 2026 in rows.index:
        g = pd.read_parquet(ROOT / "data" / "statsapi" / "games_2026.parquet")
        g = g[g.status == "Final"]
        r26 = rows.loc[[2026]].reset_index()
        r26["runs"] = g.away_score.sum() + g.home_score.sum()
        r26["r_per_pa"] = r26.runs / r26.pa
        old = old[old.season != 2026]
        if "source" not in old:
            old["source"] = "retrosheet"
        pd.concat([old, r26[old.columns]], ignore_index=True).to_parquet(OUTF, index=False)
        print("wrote 2026 row")
