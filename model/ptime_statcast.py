"""Playing-time features from Statcast pitch data, for seasons with no Retrosheet daily lines yet (2026).

Same columns as ptime.daily_feats plus b_g / b_pa / b_g_dh / b_g_ph. Validated against Retrosheet on 2025:
python model/ptime_statcast.py --validate
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from park_mix_statcast import id_map  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "data" / "statcast"
FLD = [f"fielder_{i}" for i in range(2, 10)]


def load(y):
    fs = sorted(SC.glob(f"{y}_*.parquet"))
    d = pd.concat([pd.read_parquet(f, columns=["game_pk", "game_date", "game_type", "home_team", "away_team", "inning_topbot",
                                                 "at_bat_number", "pitch_number", "batter", "pitcher", "events"] + FLD) for f in fs])
    return d[d.game_type == "R"]


def daily_feats_statcast(y):
    d = load(y)
    top = d.inning_topbot == "Top"
    d["bteam"] = np.where(top, d.away_team, d.home_team)
    d["fteam"] = np.where(top, d.home_team, d.away_team)
    dt = pd.to_datetime(d.game_date)
    d["late"] = dt >= dt.max() - pd.Timedelta(days=60)
    pa = d[d.events.notna()].drop_duplicates(["game_pk", "at_bat_number"])
    # batting: per game-batter PA, start = among first 9 distinct batters for his team in the game
    pa = pa.sort_values(["game_pk", "at_bat_number"])
    first = pa.drop_duplicates(["game_pk", "bteam", "batter"])
    first = first.assign(order=first.groupby(["game_pk", "bteam"]).cumcount())
    bg = pa.groupby(["game_pk", "batter"]).agg(pa=("events", "size"), team=("bteam", "first"),
                                               late_pa=("late", "sum")).reset_index()
    bg = bg.merge(first[["game_pk", "batter", "order"]], on=["game_pk", "batter"])
    bg["start"] = bg.order < 9
    # fielding: any of fielder_2..9 during the game; catcher = fielder_2
    f = d[["game_pk", "fteam"] + FLD].drop_duplicates()
    fl = f.melt(id_vars=["game_pk", "fteam"], value_vars=FLD, var_name="pos", value_name="pid").dropna()
    fl["pid"] = fl.pid.astype(int)
    fg = fl.groupby(["game_pk", "pid"]).agg(c=("pos", lambda s: (s == "fielder_2").any()), team=("fteam", "first")).reset_index()
    fg["fg"] = True
    g = bg.rename(columns={"batter": "pid"}).merge(fg, on=["game_pk", "pid"], how="outer", suffixes=("", "_f"))
    g["team"] = g.team.fillna(g.team_f)
    g["pa"] = g.pa.fillna(0)
    g["late_pa"] = g.late_pa.fillna(0)
    g["start"] = g.start.fillna(False).astype(bool)
    g["fg"] = g.fg.fillna(False).astype(bool)
    g["c"] = g.c.fillna(False).astype(bool)
    # fielders who start: in the field in the top/bottom of the 1st is not tracked; count a fielder start if he batted
    # in the first nine OR fielded in the first fielding half-inning
    first_f = d[d.at_bat_number == d.groupby(["game_pk", "fteam"]).at_bat_number.transform("min")][["game_pk"] + FLD]
    ff = first_f.melt(id_vars="game_pk", value_vars=FLD, value_name="pid").dropna()
    ff = set(zip(ff.game_pk, ff.pid.astype(int)))
    g["start"] = g.start | np.array([(a, b) in ff for a, b in zip(g.game_pk, g.pid)])
    g["dh"] = g.start & ~g.fg  # validated 2025: corr .995 with Retrosheet; PH corr .93 (defensive subs who bat count as PH)
    g["ph"] = (g.pa > 0) & ~g.start
    # pitcher BF late (for completeness)
    pb = pa.groupby("pitcher").agg(bf=("events", "size"), late_bf=("late", "sum"), pteams=("fteam", "nunique"))
    a = g.groupby("pid").agg(b_g=("game_pk", "nunique"), b_pa=("pa", "sum"), starts=("start", "sum"), fgames=("fg", "sum"),
                             cgames=("c", "sum"), late_pa=("late_pa", "sum"), nteams=("team", "nunique"),
                             b_g_dh=("dh", "sum"), b_g_ph=("ph", "sum"))
    a = a.join(pb[["late_bf", "pteams"]], how="outer").fillna(0)
    a["nteams"] = np.maximum(a.nteams, a.pteams)
    a = a.drop(columns="pteams")
    a.index = a.index.map(id_map())
    a = a[a.index.notna()].rename_axis("player_id").reset_index()
    a["season"] = y
    return a


if __name__ == "__main__":
    if "--validate-pit" in sys.argv:
        from ptime import daily_feats
        r = daily_feats(2025).set_index("player_id")
        pt = pd.read_parquet(ROOT / "data" / "season" / "pit.parquet").query("season == 2025").set_index("player_id")
        r = r.join(pt[["p_tbf"]], how="inner")
        r = r[r.p_tbf >= 20]
        j = r.join(daily_feats_statcast(2025).set_index("player_id"), rsuffix="_sc", how="inner")
        print(f"2025 pitchers BF>=20: {len(r)}, matched {len(j)}")
        for c in ["late_bf", "nteams"]:
            print(f"  {c:8s} mean retro {j[c].mean():.2f} statcast {j[c + '_sc'].mean():.2f} MAE {np.abs(j[c] - j[c + '_sc']).mean():.3f}")
    if "--validate" in sys.argv:
        from ptime import daily_feats
        r = daily_feats(2025).set_index("player_id")
        b = pd.read_parquet(ROOT / "data" / "season" / "bat.parquet").query("season == 2025").set_index("player_id")
        r = r.join(b[["b_g", "b_pa", "b_g_dh", "b_g_ph", "p_g"]])
        r = r[(r.b_pa >= 50) & ~(r.p_g > 0.5 * r.b_g)]
        s = daily_feats_statcast(2025).set_index("player_id")
        j = r.join(s, rsuffix="_sc", how="inner")
        print(f"2025 hitters PA>=50: retro {len(r)}, matched {len(j)}")
        for c in ["b_g", "b_pa", "starts", "fgames", "cgames", "late_pa", "nteams", "b_g_dh", "b_g_ph", "late_bf"]:
            x, z = j[c].astype(float), j[f"{c}_sc"].astype(float)
            print(f"  {c:8s} mean retro {x.mean():7.2f} statcast {z.mean():7.2f}  MAE {np.abs(x - z).mean():6.2f}  corr {np.corrcoef(x, z)[0, 1]:.4f}")
        for nm, num, den in [("start_sh", "starts", "b_g"), ("c_sh", "cgames", "fgames"), ("late_sh", "late_pa", "b_pa"),
                             ("dh_sh", "b_g_dh", "b_g"), ("ph_sh", "b_g_ph", "b_g")]:
            x = (j[num] / j[den]).fillna(0)
            z = (j[f"{num}_sc"] / j[f"{den}_sc"]).fillna(0)
            print(f"  {nm:8s} MAE {np.abs(x - z).mean():.4f}  corr {np.corrcoef(x, z)[0, 1]:.4f}")
