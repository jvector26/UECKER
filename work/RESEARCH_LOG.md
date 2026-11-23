# UECKER research log
Every test, with numbers. Newest at the bottom. Pre-registration (expected sign/size) is written BEFORE the run.

Conventions (session 1):
- Scoring targets are relative to that season's league, so league-wide environment shifts (2019 ball, 2022 dead ball)
  don't count against player skill: hitters = wOBA - lg wOBA; pitchers = RA9 - lg RA9.
- Projections use only data from seasons < N (including the weights/league rates of those seasons).
- wOBA weights and run values are UECKER's own (RE24 linear weights from Retrosheet, `model/run_values.py`).
- Tuning window: targets 2016-2024. Holdout: 2025-2026, not scored until the pre-season final check.
- 2020 (60 games) as a target: reported, but headline = 2016-2024 excluding 2020.

---

## S1-01 Run values 2010-2025 (sanity check, not a model test)
RE24 linear weights per season from Retrosheet events. League wOBA scaled to league OBP.
Result: weights in the familiar range, e.g. 2019 HR 1.94 / 2B 1.21 / 1B ~0.88, wOBA scale 1.18, lg wOBA .322;
2014 (low offense) HR 2.15, scale 1.32. Stored in data/season/run_values.parquet.

## S1-02 Marcel baseline (PRE-REGISTERED before running)
Implementation (Tango's Marcel, as public method, not a target):
- Hitters: component rates per PA (uBB, HBP, 1B, 2B, 3B, HR), weights 5/4/3 for N-1/N-2/N-3, regress with 1200 PA
  of league average (weighted units), age factor +0.006/yr under 29, -0.003/yr over 29 on positive events.
  Projected wOBA uses season N-1's weights; minus the projected league wOBA built the same way.
- Pitchers: runs per out, weights 3/2/1, regress with 134 IP (402 outs) of league average, same age factor (inverted).
  Plus a component version (K, uBB, HBP, HR per BF regressed, converted to runs with own linear weights).
- Players with no MLB history get league average (Marcel default).
Test: targets 2016-2024 ex-2020, hitters PA>=100 weighted by PA; pitchers BF>=100 weighted by BF.
Comparison baselines: league average for everyone (0), and last season only (unregressed).
Expected: Marcel beats both. Hitter rel-wOBA weighted RMSE around .030-.035; pitcher rel-RA9 RMSE around 1.2-1.6.
Constants are Marcel's published ones; I also report 0.5x/2x regression as a sensitivity check only (no tuning).

### S1-02 result (targets 2016-2024 ex-2020; PA- or BF-weighted RMSE)
Hitters, wOBA - lg wOBA (n ~450/season):

| Season | League avg | Last season only | Marcel |
|---|---|---|---|
| 2016 | .0380 | .0564 | .0327 |
| 2017 | .0393 | .0571 | .0346 |
| 2018 | .0397 | .0563 | .0329 |
| 2019 | .0412 | .0479 | .0354 |
| 2021 | .0401 | .0650 | .0347 |
| 2022 | .0393 | .0492 | .0342 |
| 2023 | .0378 | .0499 | .0342 |
| 2024 | .0387 | .0472 | .0342 |
| **Pooled** | **.0393** | **.0539** | **.0341** |
(2020 as target: .0461 / .0515 / .0442.) Marcel beats league average in every season. AS EXPECTED.
Sensitivity (not tuning): 600 PA regression .0350, 2400 PA .0337.

Pitchers, RA9 - lg RA9 (n ~520/season, BF>=100):
pooled: league avg 1.309, last season only 2.534, Marcel RA 1.325, Marcel component (K/BB/HBP/HR) 1.323.
**NOT AS EXPECTED: Marcel does not beat "everyone is league average" on next-season RA9.** Checked for a bug: data
reproduces known stability (pitcher K% year-to-year r .74, BB% .53, HR% .31, RA9 .26 at 200+ BF). Marcel projections
correlate only .18 with next RA9 and the fitted slope is .44, i.e. it under-regresses for this pool (reliever-heavy).
With 2x regression (804 weighted outs) Marcel RA is 1.288, which does beat league average (1.309).
On a component-runs target (K/BB/HBP/HR only, no BIP/sequencing luck): league avg .926, Marcel comp .882 (402) / .882 (804).
Takeaways: (1) single-season RA9 is mostly noise; differences between models will be small and need the season-clustered
t-test. (2) The regression constant for pitchers matters; my reading of Marcel's 134 IP in weighted units may be
too light. Open decision for the user: which pitcher baseline is the bar (see HANDOFF decisions).
