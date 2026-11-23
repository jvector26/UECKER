"""Player-season tables from Retrosheet daily lines.
Outputs data/season/bat.parquet and data/season/pit.parquet (player_id = Retrosheet id, one row per player-season,
teams combined), with wOBA / RA9 and league-relative versions using UECKER's own run values.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RETRO = ROOT / "data" / "retro"
SEASON = ROOT / "data" / "season"

BAT = ["b_g", "b_pa", "b_ab", "b_h", "b_2b", "b_3b", "b_hr", "b_bb", "b_ibb", "b_so", "b_hp", "b_sh", "b_sf", "b_sb",
       "b_cs", "b_g_dh", "b_g_ph", "p_g"]
PIT = ["p_g", "p_gs", "p_out", "p_tbf", "p_r", "p_er", "p_h", "p_2b", "p_3b", "p_hr", "p_bb", "p_ibb", "p_so", "p_hp",
       "p_sf", "p_sh", "p_go", "p_ao", "p_pitch"]


def birthdates():
    b = pd.read_csv(ROOT / "data" / "ref" / "retro_bio.csv", dtype=str, usecols=["id", "birthdate", "bats", "throws"])
    b["birth"] = pd.to_datetime(b.birthdate, format="%Y%m%d", errors="coerce")
    return b.rename(columns={"id": "player_id"})[["player_id", "birth", "bats", "throws"]]


def build(y0=2010, y1=2025):
    rv = pd.read_parquet(SEASON / "run_values.parquet").set_index("season")
    bats, pits = [], []
    for y in range(y0, y1 + 1):
        d = pd.read_parquet(RETRO / f"daily_{y}.parquet")
        g = d.groupby("player_id")
        b = g[BAT].sum()
        b["teams"] = g.team_id.nunique()
        b["season"] = y
        bats.append(b.reset_index())
        p = g[PIT].sum()
        p = p[p.p_g > 0]
        p["season"] = y
        pits.append(p.reset_index())
    bat = pd.concat(bats, ignore_index=True)
    pit = pd.concat(pits, ignore_index=True)
    bio = birthdates()

    # hitters
    bat["b_1b"] = bat.b_h - bat.b_2b - bat.b_3b - bat.b_hr
    bat["b_ubb"] = bat.b_bb - bat.b_ibb
    bat["woba_den"] = bat.b_ab + bat.b_ubb + bat.b_sf + bat.b_hp
    w = rv.loc[bat.season]
    num = (w.w_bb.values * bat.b_ubb + w.w_hbp.values * bat.b_hp + w.w_1b.values * bat.b_1b + w.w_2b.values * bat.b_2b
           + w.w_3b.values * bat.b_3b + w.w_hr.values * bat.b_hr)
    bat["woba"] = num / bat.woba_den.where(bat.woba_den > 0)
    bat["lg_woba"] = w.lg_woba.values
    bat["woba_rel"] = bat.woba - bat.lg_woba
    bat = bat.merge(bio, on="player_id", how="left")
    bat["age"] = (pd.to_datetime(bat.season.astype(str) + "-07-01") - bat.birth).dt.days / 365.25

    # pitchers
    pit["ra9"] = 27 * pit.p_r / pit.p_out.where(pit.p_out > 0)
    lg = pit.groupby("season")[["p_r", "p_out", "p_tbf"]].sum()
    lg["lg_ra9"] = 27 * lg.p_r / lg.p_out
    pit["lg_ra9"] = pit.season.map(lg.lg_ra9)
    pit["ra9_rel"] = pit.ra9 - pit.lg_ra9
    pit = pit.merge(bio, on="player_id", how="left")
    pit["age"] = (pd.to_datetime(pit.season.astype(str) + "-07-01") - pit.birth).dt.days / 365.25

    bat.to_parquet(SEASON / "bat.parquet", index=False)
    pit.to_parquet(SEASON / "pit.parquet", index=False)
    return bat, pit


if __name__ == "__main__":
    bat, pit = build()
    q = bat[(bat.season == 2025) & (bat.b_pa >= 500)].sort_values("woba", ascending=False)
    print(q[["player_id", "b_pa", "woba", "woba_rel", "age"]].head(5).round(3))
    print(pit.groupby("season").apply(lambda g: (27 * g.p_r.sum() / g.p_out.sum()), include_groups=False).round(2).to_dict())
    print("missing ages:", bat.age.isna().sum(), pit.age.isna().sum())
