"""S8-03: S8-02's history-PA prior applied only below a cutoff of weighted history PA. See RESEARCH_LOG (pre-registered).
usage: python model/hprior2.py | tee work/results/s8_03_hprior2.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from age import build as age_build  # noqa: E402
from hcomp import age_ship_loso  # noqa: E402
from hprior import LABELS, REG, band, hist_pa, mu_table  # noqa: E402
from marcel import TUNE, wrmse  # noqa: E402
from park_test import clustered_t  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEASON = ROOT / "data" / "season"
CUTS = {"Q1": 500, "Q250": 250, "Q800": 800, "P1": np.inf}


def run():
    bat = pd.read_parquet(SEASON / "bat.parquet")
    h, _ = age_build()
    h["band"], h["hpa"] = "", np.nan
    for k in CUTS:
        h[f"pred0_{k}"] = np.nan
    for N in TUNE:
        mu, _, _ = mu_table(bat, N)
        i = h.season == N
        raw = h[i].player_id.map(hist_pa(bat, N)).fillna(0).values
        hh = h[i].has_hist.values
        b = band(raw, hh)
        shrink = np.where(hh, REG / (raw + REG), 1.0)
        add = shrink * pd.Series(b).map(mu).values
        h.loc[i, "band"], h.loc[i, "hpa"] = b, raw / 5
        for k, c in CUTS.items():
            h.loc[i, f"pred0_{k}"] = h[i].pred0.values + np.where(raw / 5 < c, add, 0.0)
    h["p_Q0"] = age_ship_loso(h, "pred0")
    for k in CUTS:
        h[f"p_{k}"] = age_ship_loso(h, f"pred0_{k}")
    arms = ["Q0"] + list(CUTS)
    e = h[h.season != 2020]
    e1, e2 = e.season <= 2019, e.season >= 2021

    def t(base, k, m=None):
        x = e if m is None else e[m]
        return clustered_t(None, None, x.woba_rel - x[f"p_{base}"], x.woba_rel - x[f"p_{k}"], x.b_pa, x.season)

    print("rel-wOBA PA-weighted RMSE, targets 2016-24 ex-2020 (Q1 = candidate, cutoff 500; Q250/Q800 sensitivity; P1 = S8-02)")
    print("| arm | pooled | t vs Q0 | 2016-19 (t) | 2021-24 (t) | slope |")
    for k in arms:
        sl = np.polyfit(e[f"p_{k}"], e.woba_rel, 1, w=np.sqrt(e.b_pa))[0]
        row = [f"{wrmse(e.woba_rel - e[f'p_{k}'], e.b_pa):.5f}", "" if k == "Q0" else f"{t('Q0', k):.2f}"]
        for m in (e1, e2):
            x = e[m]
            row.append(f"{wrmse(x.woba_rel - x[f'p_{k}'], x.b_pa):.5f}" + ("" if k == "Q0" else f" ({t('Q0', k, m):.2f})"))
        print(f"| {k} | " + " | ".join(row) + f" | {sl:.3f} |")
    print(f"Q1 vs P1: t {t('P1', 'Q1'):.2f}")
    print("\nby season:")
    for N in TUNE:
        x = h[h.season == N]
        print(N, " ".join(f"{k} {wrmse(x.woba_rel - x[f'p_{k}'], x.b_pa):.4f}" for k in arms))
    print("\nmean error / RMSE by history-PA band:")
    for b in ["rookie"] + LABELS:
        x = e[e.band == b]
        print(f"  {b:8s} n={len(x):4d}: " + "  ".join(
            f"{k} {np.average(x.woba_rel - x[f'p_{k}'], weights=x.b_pa):+.4f}/{wrmse(x.woba_rel - x[f'p_{k}'], x.b_pa):.4f}" for k in arms))
    y = h[h.season == 2020]
    print("2020 as target:", {k: round(wrmse(y.woba_rel - y[f"p_{k}"], y.b_pa), 4) for k in arms})
    h.to_parquet(ROOT / "work" / "results" / "s8_03_preds.parquet", index=False)


if __name__ == "__main__":
    run()
