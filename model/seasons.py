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


def idmap():
    p = pd.read_parquet(ROOT / "data" / "ref" / "people.parquet").dropna(subset=["key_mlbam", "key_retro"])
    return pd.Series(p.key_retro.values, index=p.key_mlbam.astype(int).values)


def statsapi_season(y):
    """Season totals from the MLB Stats API, renamed to the Retrosheet column set (for seasons Retrosheet hasn't
    posted). Verified on 2025: every hitter's PA and pitcher's BF match Retrosheet exactly."""
    sa = ROOT / "data" / "statsapi"
    h = pd.read_parquet(sa / f"hitting_{y}.parquet")
    b = pd.DataFrame({"mlbam": h.mlbam, "b_g": h.gamesPlayed, "b_pa": h.plateAppearances, "b_ab": h.atBats,
                      "b_h": h.hits, "b_2b": h.doubles, "b_3b": h.triples, "b_hr": h.homeRuns, "b_bb": h.baseOnBalls,
                      "b_ibb": h.intentionalWalks, "b_so": h.strikeOuts, "b_hp": h.hitByPitch, "b_sh": h.sacBunts,
                      "b_sf": h.sacFlies, "b_sb": h.stolenBases, "b_cs": h.caughtStealing})
    p = pd.read_parquet(sa / f"pitching_{y}.parquet")
    q = pd.DataFrame({"mlbam": p.mlbam, "p_g": p.gamesPitched, "p_gs": p.gamesStarted, "p_out": p.outs,
                      "p_tbf": p.battersFaced, "p_r": p.runs, "p_er": p.earnedRuns, "p_h": p.hits, "p_2b": p.doubles,
                      "p_3b": p.triples, "p_hr": p.homeRuns, "p_bb": p.baseOnBalls, "p_ibb": p.intentionalWalks,
                      "p_so": p.strikeOuts, "p_hp": p.hitBatsmen, "p_sf": p.sacFlies, "p_sh": p.sacBunts,
                      "p_go": p.groundOuts, "p_ao": p.airOuts, "p_pitch": p.numberOfPitches})
    m = idmap()
    for d in (b, q):
        d["player_id"] = d.mlbam.map(m).fillna("m" + d.mlbam.astype(str))  # 'm<mlbam>' if no Retrosheet id yet
        d["season"] = y
    return b, q


def build(y0=2010, y1=2025, extra=(2026,)):
    rv = pd.read_parquet(SEASON / "run_values.parquet").set_index("season")
    bats, pits = [], []
    for y in extra:
        b, q = statsapi_season(y)
        bats.append(b)
        pits.append(q)
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
    bat = pd.concat(bats, ignore_index=True).sort_values(["season", "player_id"], ignore_index=True)
    pit = pd.concat(pits, ignore_index=True).sort_values(["season", "player_id"], ignore_index=True)
    rmap = pd.Series(idmap().index, index=idmap().values)
    for d in (bat, pit):
        d["mlbam"] = d.mlbam.fillna(d.player_id.map(rmap)).astype("Int64")
    bio = birthdates()
    # fill bio gaps (2026 debutants) from Stats API people
    sp = pd.read_parquet(ROOT / "data" / "statsapi" / "people.parquet")[["mlbam", "birth", "bats", "throws"]]
    sp["birth"] = pd.to_datetime(sp.birth, errors="coerce")
    sp = sp.drop_duplicates("mlbam").set_index("mlbam")

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
    for c in ["birth", "bats", "throws"]:
        bat[c] = bat[c].fillna(bat.mlbam.map(sp[c]))
    bat["age"] = (pd.to_datetime(bat.season.astype(str) + "-07-01") - bat.birth).dt.days / 365.25

    # pitchers
    pit["ra9"] = 27 * pit.p_r / pit.p_out.where(pit.p_out > 0)
    lg = pit.groupby("season")[["p_r", "p_out", "p_tbf"]].sum()
    lg["lg_ra9"] = 27 * lg.p_r / lg.p_out
    pit["lg_ra9"] = pit.season.map(lg.lg_ra9)
    pit["ra9_rel"] = pit.ra9 - pit.lg_ra9
    pit = pit.merge(bio, on="player_id", how="left")
    for c in ["birth", "bats", "throws"]:
        pit[c] = pit[c].fillna(pit.mlbam.map(sp[c]))
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
