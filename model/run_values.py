"""Per-season run values (RE24 linear weights) and wOBA weights, built from Retrosheet events.
First principles: no external weights. Output data/season/run_values.parquet (one row per season).

Event codes (Retrosheet): 2 out, 3 K, 14 BB, 15 IBB, 16 HBP, 17 interference, 18 error, 19 FC,
20 1B, 21 2B, 22 3B, 23 HR.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RETRO = ROOT / "data" / "retro"
OUTF = ROOT / "data" / "season" / "run_values.parquet"

EV = {2: "out", 3: "k", 14: "bb", 15: "ibb", 16: "hbp", 17: "xi", 18: "roe", 19: "out", 20: "1b", 21: "2b", 22: "3b", 23: "hr"}


def load_events(year):
    e = pd.read_parquet(RETRO / f"pa_{year}.parquet")
    e["half"] = e.game_id + "_" + e.inn_ct.astype(str) + "_" + e.bat_home_id.astype(str)
    dests = e[["bat_dest_id", "run1_dest_id", "run2_dest_id", "run3_dest_id"]].to_numpy()
    e["runs_on_play"] = (dests >= 4).sum(axis=1)
    e["bases"] = (e.base1_run_id.notna().astype(int) + 2 * e.base2_run_id.notna().astype(int)
                  + 4 * e.base3_run_id.notna().astype(int))
    e["state"] = e.outs_ct * 8 + e.bases
    return e


def season_run_values(year):
    e = load_events(year)
    # runs from this event to end of half-inning; keep only half-innings that reach 3 outs
    e["cum"] = e.groupby("half").runs_on_play.cumsum()
    tot = e.groupby("half").runs_on_play.transform("sum")
    e["runs_roi"] = tot - e.cum + e.runs_on_play  # runs scored from start of this event to end of half
    outs_end = e.groupby("half").apply(lambda g: (g.outs_ct + g.event_outs_ct).iloc[-1], include_groups=False)
    full = outs_end[outs_end == 3].index
    f = e[e.half.isin(full)].copy()
    re = f.groupby("state").runs_roi.mean()
    f["re_start"] = f.state.map(re)
    f["re_next"] = f.groupby("half").re_start.shift(-1).fillna(0.0)  # 0 after 3rd out
    f["rv"] = f.re_next - f.re_start + f.runs_on_play
    pa = f[f.bat_event_fl == "T"].copy()
    pa["ev"] = pa.event_cd.map(EV)
    lw = pa.groupby("ev").rv.mean()
    # league totals over all PA (not just full innings) for wOBA scale
    allpa = e[e.bat_event_fl == "T"].copy()
    allpa["ev"] = allpa.event_cd.map(EV)
    n = allpa.ev.value_counts()
    out_val = (lw["out"] * n["out"] + lw["k"] * n["k"] + lw["roe"] * n["roe"]) / (n["out"] + n["k"] + n["roe"])
    rel = {k: lw[k] - out_val for k in ["bb", "hbp", "1b", "2b", "3b", "hr"]}
    pa_n = len(allpa)
    obp_num = n["bb"] + n["hbp"] + n["1b"] + n["2b"] + n["3b"] + n["hr"]
    obp_den = pa_n - n.get("ibb", 0) - n.get("xi", 0) - (allpa.event_cd.isin([2, 19]) & (allpa.ab_fl == "F")).sum()  # drop sac
    lg_obp = obp_num / obp_den
    raw = sum(rel[k] * n[k] for k in rel) / obp_den
    scale = lg_obp / raw
    runs = e.runs_on_play.sum()
    row = {"season": year, "pa": pa_n, "runs": runs, "r_per_pa": runs / pa_n, "lg_obp": lg_obp, "woba_scale": scale,
           "lw_out": out_val, "lw_k": lw["k"]}
    for k in rel:
        row[f"lw_{k}"] = lw[k]
        row[f"w_{k}"] = rel[k] * scale
    row["lg_woba"] = lg_obp  # by construction
    return row


if __name__ == "__main__":
    rows = [season_run_values(y) for y in range(2010, 2026)]
    df = pd.DataFrame(rows)
    OUTF.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTF, index=False)
    pd.set_option("display.width", 200)
    print(df[["season", "r_per_pa", "lg_woba", "woba_scale", "w_bb", "w_hbp", "w_1b", "w_2b", "w_3b", "w_hr", "lw_out"]].round(3))
