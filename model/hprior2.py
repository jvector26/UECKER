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
# SHIPPED with user OK 2026-11-24 (S8-03, arm Q1): hitters with weighted history PA < CUT_SHIP regress toward the mean of
# hitters with similar history PA (mu by band) instead of league average. Applied to the no-age projection, before the
# S5-01 age handling. Production mu table data/season/hit_pa_prior.parquet; age curve refit on the new residuals
# (data/season/age_curve_hit.parquet; the S5-01 curve is kept as age_curve_hit_s5.parquet). Refit both at the pre-season final.
ARM_SHIP, CUT_SHIP = "Q1", 500


def ship_prior(bat, N, player_ids, has_hist, mu=None):
    """Shipped S8-03 additive shift (rel-wOBA) to the no-age hitter projection for season N."""
    if mu is None:
        mu = pd.read_parquet(SEASON / "hit_pa_prior.parquet").set_index("band").mu
    raw = pd.Series(player_ids).map(hist_pa(bat, N)).fillna(0).values
    hh = np.asarray(has_hist, bool)
    b = band(raw, hh)
    shrink = np.where(hh, REG / (raw + REG), 1.0)
    return np.where(raw / 5 < CUT_SHIP, shrink * pd.Series(b).map(mu).fillna(0.0).values, 0.0)


def build_production():
    """mu table on all seasons through 2024 (ex-2020), age curve refit in-sample on 2016-24 ex-2020 targets."""
    from age import AGES, smooth
    from age import ship_hitters
    bat = pd.read_parquet(SEASON / "bat.parquet")
    mu, _, n = mu_table(bat, 2025)
    pd.DataFrame({"band": mu.index, "mu": mu.values, "n": n.reindex(mu.index).values}).to_parquet(SEASON / "hit_pa_prior.parquet", index=False)
    h, _ = age_build()
    h["pred0s"] = np.nan
    for N in TUNE:
        i = h.season == N
        h.loc[i, "pred0s"] = h[i].pred0.values + ship_prior(bat, N, h[i].player_id.values, h[i].has_hist.values, mu)
    ex = (h.season != 2020) & h.has_hist
    ia = np.clip(np.floor(h.page), AGES[0], AGES[-1]).astype(int)
    r = h.woba_rel - h.pred0s
    cur = smooth((h.b_pa * r)[ex].groupby(ia[ex]).sum(), h.b_pa[ex].groupby(ia[ex]).sum())
    pd.DataFrame({"age": AGES, "resid": cur.reindex(AGES).values,
                  "n_pa": h.b_pa[ex].groupby(ia[ex]).sum().reindex(AGES, fill_value=0).values}).to_parquet(SEASON / "age_curve_hit.parquet", index=False)
    h["p"] = ship_hitters(h.pred0s, h.lg, h.page, h.has_hist)
    e = h[h.season != 2020]
    print("production mu:", mu.round(4).to_dict())
    print("production (in-sample curve) rel-wOBA RMSE 2016-24 ex-2020:", round(wrmse(e.woba_rel - e.p, e.b_pa), 5))
    print("production curve:", " ".join(f"{a}:{v:+.4f}" for a, v in cur.reindex(AGES).items()))


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
    build_production() if "--ship" in sys.argv else run()
