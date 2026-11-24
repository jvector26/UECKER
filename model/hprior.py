"""S8-02: playing-time-aware prior for hitters. Regress toward the mean of hitters with similar history PA instead of league
average. See RESEARCH_LOG (pre-registered).  usage: python model/hprior.py | tee work/results/s8_02_hprior.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from age import build as age_build  # noqa: E402
from hcomp import age_ship_loso  # noqa: E402
from marcel import TUNE, wrmse  # noqa: E402
from park_test import clustered_t  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEASON = ROOT / "data" / "season"
REG = 1200.0
EDGES = [-1, 100, 250, 500, 800, np.inf]
LABELS = ["<100", "100-249", "250-499", "500-799", "800+"]


def hist_pa(bat, M):
    """raw 5/4/3-weighted PA over M-3..M-1 per player (rows of any kind count as history, as in Marcel)."""
    h = bat[bat.season.between(M - 3, M - 1)]
    w = h.season.map({M - 1: 5, M - 2: 4, M - 3: 3})
    return (w * h.woba_den).groupby(h.player_id).sum()


def band(raw, has):
    b = pd.cut(raw / 5, EDGES, labels=LABELS, right=False).astype(object)
    return np.where(has, b, "rookie")


def mu_table(bat, N):
    rows = []
    for M in range(2013, N):
        if M == 2020:
            continue
        t = bat[(bat.season == M) & (bat.b_pa >= 100) & ~(bat.p_g > 0.5 * bat.b_g)][["player_id", "b_pa", "woba_rel"]]
        raw = hist_pa(bat, M)
        has = t.player_id.isin(raw.index).values
        rows.append(t.assign(band=band(t.player_id.map(raw).fillna(0).values, has)))
    x = pd.concat(rows)
    mu = x.groupby("band").apply(lambda g: np.average(g.woba_rel, weights=g.b_pa), include_groups=False)
    return mu, float(np.average(x.woba_rel, weights=x.b_pa)), x.groupby("band").size()


def run():
    bat = pd.read_parquet(SEASON / "bat.parquet")
    h, _ = age_build()
    h["pred0_PC"], h["pred0_P1"], h["band"] = np.nan, np.nan, ""
    print("mu by history-PA band (PA-weighted mean rel-wOBA, seasons 2013..N-1 ex-2020), and the all-hitter mean:")
    for N in TUNE:
        mu, mall, n = mu_table(bat, N)
        i = h.season == N
        raw = h[i].player_id.map(hist_pa(bat, N)).fillna(0).values
        hh = h[i].has_hist.values
        b = band(raw, hh)
        shrink = np.where(hh, REG / (raw + REG), 1.0)
        h.loc[i, "band"] = b
        h.loc[i, "pred0_PC"] = h[i].pred0.values + shrink * mall
        h.loc[i, "pred0_P1"] = h[i].pred0.values + shrink * pd.Series(b).map(mu).values
        print(N, " ".join(f"{k}:{mu.get(k, np.nan):+.4f}" for k in ["rookie"] + LABELS), f"| all {mall:+.4f}",
              "| n", n.reindex(["rookie"] + LABELS).fillna(0).astype(int).tolist() if N == 2024 else "")
    h["p_P0"] = age_ship_loso(h, "pred0")
    h["p_PC"] = age_ship_loso(h, "pred0_PC")
    h["p_P1"] = age_ship_loso(h, "pred0_P1")

    arms = ["P0", "PC", "P1"]
    e = h[h.season != 2020]
    e1, e2 = e.season <= 2019, e.season >= 2021

    def t(base, k, m=None):
        x = e if m is None else e[m]
        return clustered_t(None, None, x.woba_rel - x[f"p_{base}"], x.woba_rel - x[f"p_{k}"], x.b_pa, x.season)

    print("\nrel-wOBA PA-weighted RMSE, targets 2016-24 ex-2020")
    print("| arm | pooled | t vs P0 | 2016-19 (t) | 2021-24 (t) | slope |")
    for k in arms:
        sl = np.polyfit(e[f"p_{k}"], e.woba_rel, 1, w=np.sqrt(e.b_pa))[0]
        row = [f"{wrmse(e.woba_rel - e[f'p_{k}'], e.b_pa):.5f}", "" if k == "P0" else f"{t('P0', k):.2f}"]
        for m in (e1, e2):
            x = e[m]
            row.append(f"{wrmse(x.woba_rel - x[f'p_{k}'], x.b_pa):.5f}" + ("" if k == "P0" else f" ({t('P0', k, m):.2f})"))
        print(f"| {k} | " + " | ".join(row) + f" | {sl:.3f} |")
    print(f"P1 vs PC: t {t('PC', 'P1'):.2f} (16-19 {t('PC', 'P1', e1):.2f}, 21-24 {t('PC', 'P1', e2):.2f})")
    print("\nby season:")
    for N in TUNE:
        x = h[h.season == N]
        print(N, " ".join(f"{k} {wrmse(x.woba_rel - x[f'p_{k}'], x.b_pa):.4f}" for k in arms))
    print("\nmean error / RMSE by history-PA band:")
    for b in ["rookie"] + LABELS:
        x = e[e.band == b]
        print(f"  {b:8s} n={len(x):4d}: " + "  ".join(
            f"{k} {np.average(x.woba_rel - x[f'p_{k}'], weights=x.b_pa):+.4f}/{wrmse(x.woba_rel - x[f'p_{k}'], x.b_pa):.4f}" for k in arms))
    e = e.assign(pa_band=pd.cut(e.b_pa, [0, 250, 450, 800], labels=["100-249", "250-449", "450+"]))
    print("mean error by TARGET PA (not usable pregame; for reference):")
    for b, x in e.groupby("pa_band", observed=True):
        print(f"  {b:8s}: " + "  ".join(f"{k} {np.average(x.woba_rel - x[f'p_{k}'], weights=x.b_pa):+.4f}" for k in arms))
    print("with-history only, pooled:", {k: round(wrmse(e[e.has_hist].woba_rel - e[e.has_hist][f"p_{k}"], e[e.has_hist].b_pa), 5) for k in arms})
    y = h[h.season == 2020]
    print("2020 as target:", {k: round(wrmse(y.woba_rel - y[f"p_{k}"], y.b_pa), 4) for k in arms})
    h.to_parquet(ROOT / "work" / "results" / "s8_02_preds.parquet", index=False)


if __name__ == "__main__":
    run()
