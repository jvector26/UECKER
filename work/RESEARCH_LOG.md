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

## Session 2 (2026-11-23)

### S2-00 data: 2026 season added
- `model/run_values_statcast.py`: RE24/wOBA weights from Statcast PA rows. Check on 2025 vs Retrosheet: lg wOBA identical
  (.3153), weights within .01 (uBB .725/.717, 1B .884/.884, 2B 1.230/1.225, HR 1.976/1.998), 3B .05 (small n). 2026 row written (source=statcast).
- `model/seasons.py`: 2026 player-seasons from Stats API (Retrosheet 2026 not posted). Check on 2025: every hitter PA, pitcher BF,
  R, outs, HBP match Retrosheet exactly. Every 2010-2026 player maps Retrosheet <-> MLBAM (Chadwick); `mlbam` column added.

### S2-01 park factors (PRE-REGISTERED before running)
Method (`model/parks.py`): component park factors per PA (uBB, HBP, 1B, 2B, 3B, HR, K) by batter hand, plus runs per PA.
"As of season N" factors use seasons N-3..N-1 only: rate in park (both teams) / rate in the home team's road games (both teams).
Shrunk toward 1 by empirical Bayes on the log ratio (true spread = observed variance minus sampling variance); no tuned constants.
Players' seasons get a PA-weighted multiplier over the parks they actually played in.
Test: Marcel on park-neutral history, then re-apply each player's target-season park mix (schedule info; uses actual games, so
mid-season trades leak slightly, noted). Same targets/metrics as S1-02. Pitchers: runs per out neutralized by the run factor.
Expected: small gain. Hitters .0341 -> about .0333-.0338, concentrated in Rockies and team-changers. Pitchers Marcel RA9
improves by ~.005-.015 (1.325 and 1.288 versions). If hitters don't improve at all, park factors still go into game odds
(they're needed there), but don't get used in player projections.

### S2-01 result (targets 2016-2024 ex-2020)
Park factors sane: as of 2026, runs PF Coors 1.16 (top), Seattle 0.90 (bottom); HR high Cincinnati 1.19, low Chase/Kauffman/Oakland.
EB shrink spread (tau of log PF): HR .11, 3B .27-.33, 2B .08-.09, 1B .04-.05, K .05-.07, uBB .01-.03, runs .05.

| | base Marcel | park-adjusted | season-clustered t |
|---|---|---|---|
| Hitters rel-wOBA | .0341 | .0340 | 1.05 |
| Pitchers RA9, Marcel 402 | 1.3251 | 1.3152 | 4.61 |
| Pitchers RA9, Marcel 804 | 1.2884 | 1.2786 | 3.67 |
Same sign both eras (pitchers 804: 2016-19 1.292->1.286, 2021-24 1.284->1.271). 2020: hitters .0442->.0444, pitchers 1.677->1.671.
- **Pitchers: AS EXPECTED, passes** (about -.010, at the top of the expected range).
- **Hitters: NOT AS EXPECTED, no real gain.** Big-park and multi-team subgroups flat. Rockies-heavy hitters (n=111) get WORSE:
  .0332 -> .0343, mean error -.0007 -> -.0067 (park version over-projects them). Consistent with the home/road method
  overstating Coors for hitters (Rockies' own road numbers are depressed, the "Coors hangover"), and with raw history already
  carrying the same park when the player stays put. Per the pre-registration: park factors go into game odds; hitter projections
  stay unadjusted for now. Shipping the pitcher adjustment needs the user's OK.
- User OK 2026-11-23: research bar approved; S2-01 pitcher park adjustment shipped.

## Session 3 (2026-11-23, evening)

### S3-00 data: 2026 park mix
- `model/park_mix_statcast.py`: Stats API venue -> Retrosheet park map learned on 2025 (2425/2434 games matched on date,
  score, home SP). Check: 2025 mix rebuilt from Statcast vs Retrosheet: 365,852 PA rows both, abs diff 4 by player/park/hand.
  2026 rows appended. Tropicana (Rays back home) -> STP01 by hand. Neutral sites (Mexico City 157 PA, Field of Dreams 79,
  Las Vegas Ballpark 514) get factor 1.0.

### S3-01 hitter contact quality (PRE-REGISTERED before running)
Idea: on balls in play, exit velocity x launch angle says more about future production than whether the ball found a hole.
Method (`model/xcon.py`): UECKER's own expected contact value. For each season, bin batted balls (AB-ending contact + SF,
no sac bunts) by EV (3 mph) x LA (4 deg); bin value = mean of UECKER wOBA weights for that season's outcomes (out/ROE = 0),
lightly smoothed toward neighbors; no MLB xwOBA column used. Per hitter-season: delta = (expected - actual contact value)
/ woba_den. Balls with no EV/LA keep their actual value. Pre-2015 seasons: delta = 0.
Projection: Marcel on adjusted history, i.e. contact value = a*actual + (1-a)*expected. a chosen leave-one-target-season-out
from {0, .1, ..., 1}; full curve reported. Known gaps: no sprint speed (fast hitters' grounders), no spray angle.
Expected: a around .3-.6. Hitters rel-wOBA .0341 -> about .0331-.0337, t > 2.5, same sign 2016-19 and 2021-24. Bigger gain
on 2018-24 targets (full 3-year Statcast history). If it doesn't beat Marcel, contact quality waits for a fuller model
(speed + spray) rather than shipping.

### S3-01 result (targets 2016-2024 ex-2020)
EV/LA present on 97-99.7% of contact; league actual/expected contact value 1.001-1.003 every year (calibrated).
Hitter-season delta sd .020 wOBA (PA>=300). Y/y per-BIP: expected value predicts next year's actual slightly better than actual does (.622 vs .610).
Curve (a = weight on actual): 0 .03349, .2 .03330, .3 .03327, .4 .03327, .6 .03340, 1 (Marcel) .03413. LOSO picks a = .3 (7 seasons), .4 (2).

| subset | Marcel | Marcel+xcon | season-clustered t |
|---|---|---|---|
| 2016-24 ex-2020 | .0341 | .0333 | 4.70 |
| 2016-19 | .0339 | .0335 | 3.24 |
| 2021-24 | .0343 | .0331 | 6.64 |
| 2018-24 ex-2020 (full Statcast history) | .0343 | .0334 | 3.79 |
Every target season improves (2018 smallest, .0329 -> .0328). 2020: .0442 -> .0425.
- **AS EXPECTED, passes.** Size at the good end of the pre-registered range; a at the low end (.3). Needs user OK to ship.
- Next refinements (separate pre-registrations): sprint speed on grounders, spray angle, K%/BB% stabilization.
- User OK 2026-11-23: S3-01 shipped with a = .3 (`A_SHIP` in model/xcon.py).

## Session 4 (2026-11-23, late)
Baseline for both tests = the shipped model: Marcel + S3-01 contact quality (a = .3), hitters rel-wOBA **.0333**
(targets 2016-24 ex-2020). Pass = better pooled, season-clustered t > 2.5 vs .0333, same sign 2016-19 and 2021-24.
Tuning parameters chosen leave-one-target-season-out, full grids reported.

### S4-01 hitter speed on grounders (PRE-REGISTERED before running)
Idea: fast hitters beat out grounders that EV x LA calls outs, so S3-01 under-rates them and over-trusts the expected side.
Real sprint speed lives on Savant (Actions-only); first test with in-house proxies, fetch sprint speed only if these show signal.
- A, split trust: separate weight on actual for grounders (LA < 10) vs air balls (a_gb, a_air on 0..1 by .1).
  Expected: a_gb > a_air (about .5-.7 vs .2-.3). Gain .0333 -> .0331-.0333; may not reach t 2.5 on its own.
- B, speed-aware expected value: speed score per hitter-season = mean of within-season z-scores of
  (SB+CS)/(1B+uBB+HBP) and 3B/(2B+3B), shrunk by opportunities. Grounder expected value += beta x speed, beta fit per season
  on that season's grounders (history seasons only). Expected: beta > 0 every season; gain .0001-.0003.
- If A or B passes, its size sets whether pulling real sprint speed through Actions is worth it.

### S4-02 spray angle (PRE-REGISTERED before running)
Idea: at the same EV x LA, pulled air balls are worth more (shorter fences, HR) and pulled grounders less (shifts pre-2023);
a hitter who pulls his fly balls is better than EV x LA says.
Method: spray angle from hc_x/hc_y, sign flipped for LHB so negative = pull; 3 zones (pull < -15 deg, center, oppo > 15).
Expected value bins = EV x LA x zone (same 3x3 smoothing within zone). Balls missing spray keep EV x LA value.
Weight on actual re-tuned (expected to stay near .3). Expected: gain .0333 -> .0329-.0332, both eras.
Combined A/B + spray reported too.

### S4 results (targets 2016-2024 ex-2020; `model/xcon2.py`, output `work/results/s4_xcon2.txt`)
Shipped S3 reproduces .03327. Speed score y/y corr .64. Grounder speed beta > 0 every season (.014-.022).
| subset | Marcel | S3 shipped | A split a | B speed | S spray | SB |
|---|---|---|---|---|---|---|
| 2016-24 ex-2020 | .0341 | .0333 | .0332 | .0332 | .0333 | .0333 |
| t vs S3 | | | 1.27 | **3.11** | -0.53 | -0.24 |
| 2016-19 (t) | .0339 | .0335 | .0335 (0.22) | .0334 (0.92) | .0334 (0.32) | .0334 (0.43) |
| 2021-24 (t) | .0343 | .0331 | .0330 (1.81) | .0329 (7.35) | .0332 (-0.72) | .0331 (-0.49) |
Raw RMSE: S3 .03327, A .03317 (a_gb .6, a_air .2), B .03312 (a_gb .4-.5, a_air .1-.2). B better in 7 of 8 seasons (2017 worse by .0001).
- **A: direction as expected (a_gb .6 > a_air .2), size at the top of the range, fails t (1.27).** Not shipped on its own.
- **B: passes the bar as written** (t 3.11, positive both eras, size in the pre-registered range). But the mechanism only half
  matches: the gain is all slow hitters (mean error -.0122 -> -.0089); fast hitters go from -.0009 to -.0043 (RMSE .0311 -> .0315).
  Exploratory age x speed split (not pre-registered): slow hitters are over-projected in every age band, so it is not just aging;
  the linear speed term over-corrects the fast end. 2016-19 support is weak (t 0.92).
- **S spray: NOT AS EXPECTED, fails** (worse pooled; 2024 .0324 -> .0329). Likely sparse 3-zone bins and the 2023 shift ban
  changing what a pulled grounder is worth. Rejected as built.
- Side finding (exploratory): projections run high overall, mean error S3 -.0045, Marcel -.0059, worst at 32+ (-.0085 mid-speed).
  Suggests the Marcel age curve is too kind to older hitters; candidate for its own pre-registered test.
- Recommendation: hold B pending the user's call. Options: ship B now at partial weight, or first pull real sprint speed (Savant,
  via Actions) to test the mechanism with a non-linear speed term.

### User rule 2026-11-23: measured values over surrogates or intermediate models whenever available.
Data: `data/savant/sprint_speed.parquet` (Savant leaderboard 2015-2026, min=0: ~600 players/season; sprint_speed ft/s,
hp_to_1b, competitive_runs; player_id = MLBAM). Fetched via Actions backfill job `running`.

### S4-03 measured sprint speed on grounders (PRE-REGISTERED before running)
Replaces the SB/3B proxy in S4-01B. Baseline = shipped S3 (.0333); also compared with proxy B (.0332).
Speed per hitter-season = sprint speed shrunk to the season mean by competitive runs (k = 10 runs); hitters with no
measured speed get the season mean. Split weights a_gb/a_air tuned LOSO as in S4-01.
- M-lin: grounder expected value += beta x (speed - season mean), beta fit per season on that season's grounders.
- M-bin: grounder expected value += smoothed mean grounder residual in the hitter's speed band (0.5 ft/s bands, per season),
  allowing a non-linear shape (the proxy over-credited the fast end).
Expected: beta > 0 every season. Pooled .0333 -> .0330-.0332, t > 2.5, both eras positive; at least as good as proxy B.
Mechanism check: slow-hitter mean error improves AND fast-hitter mean error does not worsen by more than .001
(the proxy failed this). If measured speed doesn't beat the proxy, the proxy result was mostly not about speed.

### S4-03 result (targets 2016-2024 ex-2020; `model/xcon3.py`, `work/results/s4_03_xcon3.txt`)
Measured speed on 97-100% of grounders; y/y corr .92 (proxy .64). Beta > 0 every season (.014-.019 wOBA-weight per ft/s).
| subset | Marcel | S3 | A | B proxy | M-lin | M-bin |
|---|---|---|---|---|---|---|
| 2016-24 ex-2020 | .03413 | .03327 | .03321 | .03316 | **.03306** | .03308 |
| t vs S3 | | | 1.27 | 3.11 | **3.46** | 3.98 |
| 2016-19 (t) | .03393 | .03347 | .03345 (0.22) | .03342 (0.92) | .03333 (1.26) | .03332 (1.66) |
| 2021-24 (t) | .03433 | .03307 | .03297 (1.81) | .03289 (7.35) | .03279 (7.27) | .03284 (6.11) |
M-lin weights (LOSO): a_gb .3, a_air .1. M-lin vs proxy B: t 2.21. M-lin better or equal to S3 in all 8 seasons.
- **Pooled bar: PASSES** (M-lin and M-bin), size in range, both eras positive, beats the proxy. Shape: M-bin no better than linear.
- **Mechanism criterion as written: FAILS.** Fast-hitter mean error -.0005 -> -.0040 (limit was .001 worse). But net of the
  shared over-projection (overall mean error about -.004, see S4 side finding), the speed groups line up:
  slow/mid/fast relative error S3 -.0064/+.0012/+.0039 -> M-lin -.0025/+.0010/+.0001 (spread .0103 -> .0026).
  S3 was under-rating fast hitters relative to everyone else; their absolute error looked good only because it offset the
  general over-projection. The criterion should have been written relative to the pool; noted, not re-scored.
- Recommendation: ship M-lin (needs user OK). The over-projection (age curve) is the next pre-registered test; it should
  bring fast-hitter absolute error back near zero.
- User OK 2026-11-23: S4-03 M-lin shipped (a_gb .3, a_air .1; constants in model/xcon3.py). Hitter baseline now .03306.

## Session 5 (2026-11-23)

### S5-01 hitter age curve (PRE-REGISTERED before running)
Baseline = shipped S4-03 M-lin (.03306). Problem: projections run high (mean error about -.004 overall, about -.0085 at 32+).
Marcel's age factor (+.006/yr under 29, -.003/yr over, multiplying absolute wOBA) is a published rule of thumb, not measured.
Arms (all replace the Marcel age factor; contact/speed corrections unchanged):
- C0 control: no age term, flat offset fit LOSO. If C0 matches the age arms, the problem is level, not age.
- A1 re-tuned Marcel shape: peak age and both slopes, grid, LOSO.
- A2 measured delta-method curve: same-hitter consecutive-season change in rel-wOBA by age, harmonic-mean-PA weighted,
  fit only on seasons before N (2010+), smoothed, chained from last-season age to target age. Additive. Not tuned on targets.
- A3 residual curve: (actual - no-age projection) by projected age, smoothed bins, LOSO. Additive.
Expected: older hitters lose more than Marcel says; peak about 26-28. Pooled .03306 -> .0328-.0330, t > 2.5, both eras.
Mechanism check: 32+ mean error moves toward 0 by >= .003, age bands' mean errors tighter, and the best age arm beats C0
(t > 1.5); otherwise it is a level fix, not aging. Side check: fast-hitter absolute mean error back near 0.

### S5-01 result (targets 2016-2024 ex-2020; `model/age.py`, `work/results/s5_01_age.txt`)
Shipped constants (a_gb .3 / a_air .1, fixed) reproduce .03303 (the .03306 headline used LOSO-picked weights).
| subset | shipped | no age | C0 flat | A1 re-tuned | A2 delta | A3 residual |
|---|---|---|---|---|---|---|
| 2016-24 ex-2020 | .03303 | .03332 | .03323 | **.03280** | .03290 | .03280 |
| t vs shipped | | -2.99 | -1.36 | **2.08** | 0.63 | 2.01 |
| 2016-19 (t) | .03328 | | | .03316 (4.11) | .03332 (-0.34) | .03316 (2.88) |
| 2021-24 (t) | .03278 | | | .03244 (1.56) | .03248 (0.75) | .03244 (1.49) |
A1 LOSO picks: peak 27-28, +.004/yr below, -.005/yr above (Marcel: 29, +.006, -.003). 6 of 8 seasons better; 2021 worse.
Mean error by projected age, shipped -> A1: <25 -.0052 -> -.0011; 25-27 -.0033 -> +.0001; 28-30 -.0019 -> +.0006;
31-33 -.0066 -> -.0021; 34+ -.0077 -> -.0003. 32+: -.0070 -> -.0009. Overall -.0042 -> -.0004. Fast hitters -.0040 -> -.0001.
- **Pooled bar: FAILS as written** (best t 2.08 < 2.5), though size is in range and both eras are positive.
- **Mechanism: PASSES clearly.** Age arms beat flat C0 (t 4.3-4.7); every age band moves toward 0; 32+ fixed. It is aging, not level.
  Marcel's curve is too kind at BOTH ends: too much growth for young hitters and too little decline for old ones.
- A2 measured delta-method: over-corrects 25-30 (mean error +.0028 to +.0042), t 0.63. Not shipped. Likely the known
  delta-method selection bias (y+1 pairs exist only for players who kept jobs); a measured curve needs a survivor fix first.
- Exploratory, NOT pre-registered: partial weight on A1 vs shipped: w .25 .03292 (t 4.86); w .5 .03284 (t 3.83; 16-19 7.23,
  21-24 2.33); w .75 .03280 (t 2.91); w 1 .03280 (t 2.08). Half weight keeps most of the gain with lower variance.
- Recommendation (needs user OK): ship A1 at half weight (rule 6: partial weight in proportion to evidence), flagged as
  post-hoc. Alternative: hold, and pre-register a survivor-corrected measured curve.

### User call 2026-11-23: HOLD S5-01 A1. Pre-register a measured curve with a job-retention fix.

### S5-02 measured age curve, survivor-corrected (PRE-REGISTERED before running)
Baseline = shipped (.03303 with fixed constants). Also shown: S5-01 A1 (.03280) and A2 raw delta (.03290).
Facts before running: of hitters with 100+ PA in season y, the share with 100+ PA in y+1 is .84 (<=25), .79 (25-31),
.75 (31-34), .59 (34+). Those who drop out are missing from the delta method, and they are mostly the decliners.
A second bias, probably the bigger one for A2's 25-30 overshoot: y is a noisy season, and pairs lean toward good-luck y
seasons (more PA), so raw y->y+1 change mixes regression to the mean with aging. Marcel already regresses, so A2 double-counts.
All arms: pairs from seasons before N only (2010+), y 100+ PA, [1,2,1]-smoothed by integer age, chained last-season age ->
target age, additive, nothing tuned on targets.
- D0: raw delta on this pair set (y+1 100+ PA), harmonic-mean-PA weights. Reference.
- D1 job retention: D0 weighted by 1/P(return), P from a logistic fit (seasons < N) on age bins, y rel-wOBA, log y PA;
  weights capped at 5. Assumes dropouts look like returners with the same age/level/PA (observable selection only).
- D2 regression: y side shrunk toward league as Marcel does for one season (PA / (PA + 240)), y+1 raw.
- D3: both fixes.
Expected: D1 alone makes old-age declines steeper (more negative 32+ deltas) and on its own may over-correct.
D2 fixes the 25-30 overshoot. D3 best: pooled .03303 -> .0328-.0329, t > 2.5, both eras; age-band mean errors within
+/- .002. Caveat stated up front: the bar scores only hitters who reach 100 PA, i.e. survivors, so a survivor-corrected
curve can be more right about talent and still score no better here; if D1 helps talent but not the score, note it
for the playing-time model.

### S5-02 result (`model/age2.py`, `work/results/s5_02_age2.txt`)
| subset | shipped | A1 | A2 | D0 | D1 retention | D2 regression | D3 both |
|---|---|---|---|---|---|---|---|
| 2016-24 ex-2020 | .03303 | .03280 | .03290 | .03289 | .03289 | .03285 | .03286 |
| t vs shipped | | 2.08 | 0.63 | 0.74 | 1.55 | 1.51 | 1.48 |
| 2016-19 (t) | | 4.11 | -0.34 | -0.24 | 0.81 | 0.45 | 0.47 |
| 2021-24 (t) | | 1.56 | 0.75 | 0.85 | 1.32 | 1.52 | 1.47 |
vs A1: every D arm worse (t -1.1 to -2.4). 32+ mean error: shipped -.0070, A1 -.0009, D0 -.0022, D1 -.0064, D2 -.0048, D3 -.0051.
- **FAILS.** No measured arm reaches t 2.5 or beats A1; all D arms still under-project 31+ (-.004 to -.007).
- **D1 sign NOT AS EXPECTED:** the retention weights made old-age declines SHALLOWER (cum 27->33 -.045 -> -.026/-.028), not
  steeper. The weights up-weight returners who looked like likely dropouts (low y wOBA, low y PA); those are mostly bad-luck
  seasons that bounce back, so observable-based weighting re-imports regression to the mean. The real survivor bias
  (decline that shows up only in the missing y+1) can't be seen in observables, so this fix can't remove it.
- D2 (regression fix) did what was expected for 25-30 (+.0037 -> +.0002) but over-shallows 31+.
- Takeaway: the year-to-year delta method can't measure old-age decline here without an assumed value for dropouts
  (the Lichtman "phantom season"), which is a judgment call, not a measurement. A3 (age curve measured from UECKER's
  own projection residuals, LOSO) is the measured option that works: .03280, t 2.01, same as A1.

### S5-01 SHIPPED with user OK 2026-11-23 (option 1): A3 residual age curve at half weight
Definition: hitter projection = .5 x shipped (Marcel age factor) + .5 x (no-age projection + A3 curve).
(A first pass halved A3 alone with no Marcel term, i.e. half the total age effect: .03291, t 1.26. Wrong reading of
"half weight"; the A1 half-weight figure was a blend with shipped, so A3 is scored the same way.)
LOSO: .03303 -> **.03283**, t 3.75; 2016-19 .03313 (t 6.96); 2021-24 .03254 (t 2.28). 7 of 8 seasons better (2021 .0334 -> .0335).
Mean error by age: <25 -.0028, 25-27 -.0019, 28-30 -.0008, 31-33 -.0038, 34+ -.0040; 32+ -.0070 -> -.0036; overall -.0042 -> -.0023.
Post-hoc weight (not pre-registered); flagged. Production curve fit on all 2016-24 ex-2020 targets (in-sample .03278):
`data/season/age_curve_hit.parquet`; `model/age.py` `ship_hitters`, `ARM_SHIP`, `W_SHIP`. Refit at the pre-season final check.
New hitter baseline to beat: .03283.

## Session 6 (2026-11-23, night)

### S6-01 pitcher component model (PRE-REGISTERED before running)
Idea: next-season RA9 is mostly noise, so project the parts that carry signal separately, each regressed by its own
measured reliability, then rebuild runs. Baseline = shipped S2-01 (park-adjusted Marcel, 804 outs): **1.2786**.
Must beat all three bars (1.325 / 1.288 / 1.309) AND shipped 1.2786, t > 2.5, same sign 2016-19 and 2021-24.
Method (`model/pcomp.py`):
- Per pitcher-season, per BF: K, uBB, HBP rates, and contact value CON = runs of batted balls (1B/2B/3B/HR, errors = outs)
  relative to an out, with UECKER's RE24 weights of that season. Run value rel. league per BF = sum (rate - lg) x (lw - lw_out).
- Expected contact (Statcast 2015+): CON_x from the pitcher's allowed EV x LA, same bins/smoothing as S3-01 hitters.
  History contact = a x actual + (1-a) x expected; pre-2015 actual only.
- Weights 3/2/1 (Marcel). Regression per component by empirical Bayes, fit only on seasons < N:
  k_c = sampling variance per BF / true between-pitcher variance (BF >= 100 pitcher-seasons). No constants tuned on the target.
- Park: shipped run factor allocated to contact (CON history minus (m-1) x lg R/BF; target adds (m_N - 1) x lg R/BF).
- Age: Marcel factor as shipped. RA9 rel = 27 x rel runs per BF / lg outs per BF (linear; calibration slope reported, not used).
Arms: C1 components, actual contact only. C2 = C1 + expected contact, a = .3 (as hitters). C3 = C2 + role prior
(regress toward history-role mean, SP if weighted GS/G > .5; relievers run about -.14 RA9 vs league).
Sensitivity only (not tuning): a = 0 / .5.
Expected: all C arms beat the three bars. C1 ~ 1.28 (about even with shipped: FIP-style failed in S1 because it dropped
contact and used one regression constant). C2 1.265-1.275 (expected contact helps less than for hitters; pitchers control
contact weakly). C3 another -.003 to -.008. Best arm 1.255-1.272. (Earlier chat guess 1.24-1.26 revised down after rereading
S1-02: the predictable spread is small.) If no arm reaches t 2.5 vs shipped, report the failure; nothing ships without user OK.
