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

### S6-01 result (targets 2016-24 ex-2020; BF-weighted rel-RA9 RMSE; t = season-clustered vs shipped park_804)
| Arm | Pooled | t | 2016-19 (t) | 2021-24 (t) | calib. slope |
|---|---|---|---|---|---|
| League avg | 1.3093 | | 1.3303 | 1.2872 | |
| Marcel 804 | 1.2884 | | 1.2924 | 1.2843 | .61 |
| Shipped S2-01 | 1.2786 | | 1.2862 | 1.2707 | .65 |
| C1 components, actual contact | 1.2627 | 4.09 | 1.2679 (2.60) | 1.2572 (3.32) | .71 |
| **C2 + expected contact a=.3** | **1.2424** | **7.42** | 1.2487 (7.89) | 1.2357 (3.73) | .82 |
| C3 = C2 + role prior | 1.2442 | 5.76 | 1.2517 (4.61) | 1.2366 (3.25) | .80 |
| sens. a=0 / a=.5 | 1.2394 / 1.2466 | 6.45 / 7.51 | | | |
C2 beats shipped in all 8 seasons (2018 1.252 -> 1.215, 2021 1.364 -> 1.308). 2020 as target: 1.671 -> 1.676 (flat).
EB constants (raw BF, N-1 weight = 1): K ~68, uBB ~192, HBP 630-850, contact 380-430 actual / 260-390 blend (falls as
Statcast years accumulate). Starters 1.150 -> 1.105, relievers 1.450 -> 1.424; mean error SP +.142 -> +.082, RP -.039 -> +.008.
- **C2 PASSES, BETTER THAN EXPECTED** (pre-reg 1.265-1.275; got 1.242). Clears all three pitcher bars and shipped, t 7.4, both eras.
  Why bigger than expected: separate regression per component matters most (K is ~6x more reliable than contact), and
  expected contact strips most of the BABIP/HR-per-fly noise.
- **C3 NOT AS EXPECTED:** role prior slightly worse (relievers' mean error goes to +.14; the history-role offset double-counts
  what their K/BB already show).
- a=0 (all expected contact) is .003 better, but a=.3 was pre-registered; not switching post-hoc.
- Awaiting user OK to ship C2 (`model/pcomp.py`, arm C2). Preds: `work/results/s6_01_preds.parquet`.

- **S6-01 SHIPPED with user OK 2026-11-23** (arm C2, a=.3). New pitcher baseline to beat: **1.2424**. `ARM_SHIP` in `model/pcomp.py`.

## Session 7 (2026-11-24)

### S7-01 pitcher age curve (PRE-REGISTERED before running)
Idea: pitchers still use Marcel's hitter-style age factor on runs per out. Measure the age effect from UECKER's own
S6-01 residuals instead, exactly as S5-01 did for hitters. Baseline = shipped S6-01 C2: **1.2424**.
Method (`model/page.py`): rebuild shipped C2 with and without the age factor (P0 = no age). Projected age = last
history age + gap. LOSO over targets 2016-24 ex-2020, pitchers with history only:
- C0: P0 + constant offset (BF-weighted mean residual of the other seasons). Separates overall bias from age.
- A3: P0 + residual curve by integer projected age (BF-weighted mean residual, [1,2,1]-smoothed, ages 20-40).
- A3h: .5 x shipped + .5 x A3 (the shipped hitter form).
Pass: beat 1.2424, season-clustered t > 2.5, same sign 2016-19 and 2021-24. Also must beat C0 (else it is bias, not age).
Expected: small. Pitcher RA9 is noisy and Marcel's factor is mild. A3 / A3h 1.238-1.242, t 0.5-2.5; likely below the bar.
Shape guess: flatter than Marcel before 27, steeper decline 33+. C0 alone may capture part of it (C2 starters ran +.08 high).
If no arm passes, report the failure; nothing ships without user OK.

### S7-01 result (`model/page.py`, `work/results/s7_01_page.txt`; targets 2016-24 ex-2020, rel-RA9 BF-wtd RMSE)
| arm | pooled | t vs shipped | 2016-19 (t) | 2021-24 (t) |
|---|---|---|---|---|
| shipped C2 (Marcel age) | 1.2424 | | 1.2487 | 1.2357 |
| no age | 1.2445 | -2.10 | 1.2498 (-0.63) | 1.2389 (-3.10) |
| C0 constant | 1.2444 | -2.10 | 1.2498 | 1.2389 |
| A3 residual curve | 1.2437 | -1.94 | 1.2499 (-1.75) | 1.2374 (-1.17) |
| A3h half weight | 1.2428 | -1.27 | 1.2490 (-1.01) | 1.2363 (-0.83) |
A3 vs C0 t 0.68; A3h vs C0 t 1.76. 2020 target: shipped 1.676 best.
Measured curve (in-sample) vs Marcel's implied shift: young pitchers improve LESS than Marcel says (age 24: -.04 vs -.12 RA9),
34+ decline STEEPER (age 36: +.21 vs +.10). Mean error 34+: shipped +.098, A3 +.009; <25: shipped +.047, A3 -.013.
- **FAILS.** No arm beats shipped; the age factor is worth .002 RMSE and Marcel's version is already as good as a measured curve.
- Shape matched the pre-reg guess (flatter young, steeper 33+) but the bias fix doesn't reduce RMSE: 34+ is only 400 targets
  and RA9 noise swamps a ~.1 run shift. Size expectation (1.238-1.242) NOT met: no gain at all.
- Keep Marcel's age factor for pitchers. Possible later use: the measured 34+ decline for playing time / IL risk, not rate.
- Kept `out["page"]` in `pcomp.project` (output column only; projections unchanged, reproduces 1.2424).

### S7-02 pitcher stuff and plate discipline (PRE-REGISTERED before running)
Idea: K and uBB are the reliable pitcher components; pitch-level outcomes (whiffs, called strikes, balls) and pitch
physics (stuff) should measure them faster than K/BB counts, as expected contact did for batted balls.
Baseline = shipped S6-01 C2 **1.2424**. Pass = beat it, season-clustered t > 2.5, same sign 2016-19 and 2021-24.
Method (`model/stuff.py`), Statcast 2015+, history only (target-season data never used):
- Plate discipline per pitcher-season (non-intentional pitches): whiff/pitch, called strike/pitch, foul/pitch, ball/pitch,
  zone%. xK/BF and xuBB/BF = BF-weighted linear fit on pitcher-seasons < N (BF >= 100).
- Stuff: LightGBM P(whiff | swing) from physics only (velo, spin, spin axis, movement, release point, extension,
  differences from the pitcher-season primary fastball, same-hand flag; no location, no count, no pitch-type label;
  LHP mirrored). Trained on seasons < N, pitcher-grouped 2-fold out-of-fold so a pitcher's own pitches never score him.
  Stuff score = mean OOF P(whiff|swing) over all his pitches.
- History K, uBB = a x actual + (1-a) x expected, a = .3 (pre-set, same as contact). Pre-2015 actual only.
- EB constant for the blended K/uBB: binomial sampling variance no longer applies, so true variance = BF-weighted
  year-to-year covariance of same-pitcher rates (seasons < N), sampling variance = observed - true. Control arm uses this
  same EB on actual K/uBB, so the arms isolate the new information, not the EB method.
Arms: B0 shipped. B1c control (covariance EB, actual). B1 plate-discipline xK/xuBB. B2 = B1 + stuff score in the xK and
xuBB fits. B3 stuff only (xK/xuBB from stuff score + zone%, no observed swing outcomes).
Expected: B1c ~ shipped (+/- .002). B1 1.236-1.241 (t 1.5-3.5; K is already reliable, uBB gains more). B2 a further
-.001 to -.004 vs B1. B3 worse than B1 (observed whiffs carry more than physics). Best arm 1.233-1.240. If no arm passes,
report the failure; nothing ships without user OK.

### S7-02 result (`model/stuff_build.py`, `model/stuff.py`, `work/results/s7_02_stuff.txt`)
| arm | pooled | t vs shipped | 2016-19 (t) | 2021-24 (t) | slope |
|---|---|---|---|---|---|
| B0 shipped C2 | 1.2424 | | 1.2487 | 1.2357 | .82 |
| B1c control: covariance EB, actual K/uBB | 1.2395 | 3.20 | 1.2477 (1.02) | 1.2310 (6.80) | .85 |
| B1 plate-discipline xK/xuBB | 1.2429 | -0.38 | 1.2511 (-1.02) | 1.2344 (1.70) | .83 |
| B2 B1 + stuff | 1.2433 | -0.76 | 1.2511 (-1.08) | 1.2352 (0.60) | .83 |
| B3 stuff + zone% only | 1.2460 | -1.21 | 1.2574 (-2.13) | 1.2342 (0.63) | .85 |
vs B1c: B1 t -4.57, B2 -5.34, B3 -3.07. B2 vs B1 t -2.65. Fit R2 (pitcher-seasons): plate discipline xK .92, xuBB .69-.71;
stuff adds ~.001; stuff + zone% alone xK .38-.40, xuBB .12-.18. Stuff model: LightGBM P(whiff|swing), OOF by pitcher.
- **FAILS.** No new-information arm beats shipped or the control; stuff adds nothing once a season of observed swings exists.
- NOT AS EXPECTED: plate-discipline blend hurts vs control (pre-reg B1 1.236-1.241, got 1.2429). Over a full season K is
  already ~92% explained by whiff/called/foul/ball rates, so the blend mostly swaps signal for fit error. Stuff's value, if any,
  is at small samples (call-ups, early season), which this season-level bar can't see.
- UNEXPECTED LEAD (not a pass): the control alone, i.e. regressing K and uBB with constants from same-pitcher year-to-year
  covariance instead of binomial noise, gains .003 (t 3.2) but fails the era test (2016-19 t 1.02). Covariance constants
  (raw BF): K ~160 (shipped 68), uBB ~330 (192), HBP ~850, contact ~410-480. Binomial EB ignores year-to-year talent change,
  so S6-01 under-regresses K and uBB, consistent with calibration slope .82. Candidate S7-03 (needs its own pre-registration).

### S7-03 covariance regression constants for S6-01 (PRE-REGISTERED before running)
Idea: S6-01 sets each component's regression constant k from binomial / event sampling noise. That treats true talent as
fixed, but talent drifts year to year, so the projection under-regresses (calibration slope .82). Measure k instead from
same-pitcher year-to-year covariance: tau2 = BF-weighted lag-1 covariance of league-relative rates (pairs both BF >= 100,
seasons < N, 2019->20 and 20->21 pairs dropped), sampling variance per BF = (observed variance - tau2) / E[1/BF], k = ratio.
Nothing tuned on targets. Baseline shipped C2 **1.2424**; pass = t > 2.5 pooled, same sign 2016-19 and 2021-24.
Arms: E1 = covariance k for all four components (K, uBB, HBP, contact) = the candidate.
E2 = covariance k for K and uBB only (this is S7-02's control B1c, already seen: 1.2395, t 3.2, 2016-19 t 1.02;
reference only, cannot be "pre-registered" now).
Expected: E1 1.237-1.240, slope .88-.95. 2016-19 is the risk: E2 was weak there, and contact k moves less (410-480 vs
260-430 shipped), so E1 may reach t 2.5 pooled and still miss the 2016-19 era test. Honest odds of a full pass: under half.

### S7-03 result (`model/ebcov.py`, `work/results/s7_03_ebcov.txt`)
| arm | pooled | t vs shipped | 2016-19 (t) | 2021-24 (t) | slope |
|---|---|---|---|---|---|
| E0 shipped C2 | 1.2424 | | 1.2487 | 1.2357 | .82 |
| **E1 covariance k, all four** | **1.2360** | **3.03** | 1.2469 (1.71) | 1.2246 (4.48) | **.95** |
| E2 covariance k, K/uBB only (seen before) | 1.2395 | 3.20 | 1.2477 (1.02) | 1.2310 (6.80) | .85 |
E1 vs E2 t 2.54. E1 better than shipped in all 9 seasons incl. 2020 (1.676 -> 1.668). k (raw BF): K 159-169 (was 67-69),
uBB 324-350 (189-198), HBP 799-968 (634-855), contact 406-488 (262-391).
- **E1 PASSES the written bar** (pooled t 3.03 > 2.5; same sign both eras), slightly better than expected (pre-reg 1.237-1.240).
  Calibration slope .82 -> .95, so the under-regression is mostly fixed. 2016-19 gain is small (t 1.71).
- Correction: S7-02 said B1c/E2 "fails the era test" with 2016-19 t 1.02; the bar asks for same sign, not t > 2.5 per era.
  E2 would also have passed on the letter; E1 beats it (t 2.54) anyway.
- Awaiting user OK to ship E1.

- **S7-03 SHIPPED with user OK 2026-11-24** (E1). `EB_SHIP = "cov"`, `cov_k` in `model/pcomp.py`; production reproduces 1.2360.
  Old tests (page/stuff/ebcov) pin `EB_SHIP = "binom"` so they reproduce. New pitcher baseline to beat: **1.2360**.

## Session 8 (2026-11-24)

### S8-01 hitter component projection with covariance regression constants (PRE-REGISTERED before running)
Idea: the hitter version of S6-01 + S7-03. Shipped hitters (Marcel + S4-03 contact + S5-01 age) regress every component
toward league with one fixed constant (1200 PA on 5/4/3 weights = 240 raw PA at N-1 weight 1). Measure each component's
own constant from same-hitter year-to-year covariance instead. Baseline = shipped **.03283** (rel-wOBA, PA-weighted RMSE,
targets 2016-24 ex-2020). Pass = pooled season-clustered t > 2.5 vs shipped AND same sign in 2016-19 and 2021-24.
Method (`model/hcomp.py`):
- Per hitter-season, per wOBA PA (AB + uBB + SF + HBP): K, uBB, HBP rates and contact value CON = wOBA-weighted 1B/2B/3B/HR
  (that season's UECKER weights). History contact = shipped S4-03 blend: grounders .3 actual + .7 expected (EV x LA + measured
  sprint speed), air .1 actual + .9 expected (`xcon3.parquet` xm_gb / x_air), Statcast 2015+; uncovered remainder and
  pre-2015 stay actual (same replacement as pcomp).
- Weights 5/4/3 (Marcel hitters, as shipped), normalized N-1 = 1. Each component regressed to the league rate in the player's
  mix of seasons by its own k = `cov_k` logic from `model/pcomp.py`: tau2 = PA-weighted lag-1 covariance of season-centered
  rates, same hitter, both seasons PA >= 100, pitcher-hitters excluded, 2019->20 and 20->21 pairs dropped, seasons < N only.
  Nothing tuned on targets.
- uBB, HBP valued with N-1 wOBA weights (as Marcel); contact is already in wOBA units.
- Age: shipped S5-01 method unchanged: .5 x Marcel age factor on absolute wOBA + .5 x (no-age projection + A3 residual curve),
  curve refit LOSO on each arm's own no-age residuals (same procedure that produced .03283).
Arms:
- H0 shipped (must reproduce .03283).
- **HC control** = H1's split and contact blend, every component regressed by Marcel's single constant (240 raw PA).
  HC vs H0 = effect of the split; H1 vs HC = effect of the new regression alone.
- **H1 candidate** = components per PA as specified (uBB, HBP, contact per PA), covariance k each. Note: in wOBA a K and an
  out in play are both worth 0, so with contact per PA the K component carries no weight; K's constant is reported only.
- H2 variant (reported, not the candidate): contact value per ball in play (BIP = PA - K - uBB - HBP) with its own covariance k,
  projected contact per PA = (1 - K - uBB - HBP) x CON/BIP. Makes K live. Needs its own call if it beats H1.
Expected: HC ~ shipped (+/- .0001): Marcel is already a component split (uBB/HBP/1B/2B/3B/HR) with one constant, and the contact
correction already uses the same 1200 denominator, so only bookkeeping differs. Covariance k (raw PA, guesses): uBB 150-250,
HBP 200-400, contact/PA 350-600 (the 70-90% expected blend is less noisy than raw BABIP, but talent drifts). So the main change
is heavier regression on contact. H1 .0326-.0328 (gain .00003-.0002), t 1-3, calibration slope moves toward 1 if shipped is
under-regressed. H2 within +/- .0001 of H1. Honest odds of a full pass: under half; hitters have less room than pitchers
(Marcel's 1200 was tuned on hitters; pitchers' Marcel constant was not suited to components).
Report calibration slope (weighted LS of actual on projection) for every arm. Nothing ships without user OK.

### S8-01 result (`model/hcomp.py`, `work/results/s8_01_hcomp.txt`, preds `work/results/s8_01_preds.parquet`)
| arm | pooled | t vs H0 | 2016-19 (t) | 2021-24 (t) | slope |
|---|---|---|---|---|---|
| H0 shipped (reproduced) | .03283 | | .03313 | .03254 | .970 |
| HC control: split, Marcel constant 240 | .03285 | -1.82 | .03314 (-1.40) | .03255 (-1.14) | .966 |
| H1 candidate: covariance k, contact per PA | .03289 | -0.75 | .03312 (+0.10) | .03267 (-1.15) | 1.036 |
| H2 variant: contact per BIP, K live | .03318 | -3.45 | .03347 (-1.78) | .03288 (-3.87) | .933 |
H1 vs HC (regression alone) t -0.57 (16-19 +0.20, 21-24 -0.99). H2 vs H1 t -1.71. H1 better in 3 of 8 seasons (2016, 2017, 2023).
Covariance k (raw PA, N-1 weight 1; Marcel = 240): K 78-98 (no weight in wOBA), uBB 170-183, HBP 287-330,
contact/PA 377-492 (falls as Statcast years accumulate); contact/BIP 166-240.
- **FAILS.** H1 .03289 vs .03283 (t -0.75); signs split by era. Nothing to ship.
- HC as expected (within .00002; the split itself adds nothing, Marcel is already a component model).
- NOT AS EXPECTED on calibration: shipped hitters are NOT under-regressed (slope .970, unlike pitchers' .82). Covariance k
  regresses contact ~2x harder and overshoots (slope 1.036: projections now too compressed). The extra uBB/HBP precision
  doesn't offset it. Q1 (lowest projections) mean error +.0005 -> -.0022, q5 -.0023 -> -.0003: spread squeezed.
- Why hitters differ from pitchers: Marcel's 1200 PA was built for hitters; S6-01's per-component binomial constants were
  the outlier there. Also the error pattern by target PA (100-249: -.025, 450+: +.005) is a selection effect (regulars are
  better than league average), which a heavier pull to league average makes worse. A playing-time-aware prior (regress
  toward the mean of hitters with similar PA) is the natural follow-up; that's a different test.
- H2 (contact per BIP) clearly worse: compounding K/uBB/HBP into the BIP share adds noise.

### S8-02 playing-time-aware prior for hitters (PRE-REGISTERED before running)
Idea: Marcel regresses every hitter toward league average, but hitters who play are better than league average, and how
much they played is pregame information. S8-01's errors ran -.025 (target PA 100-249) to +.005 (450+). Target-season PA is
NOT usable (it's partly caused by performance in N), so the prior uses history PA only. Regress toward the mean of hitters
with similar history PA instead. Baseline shipped **.03283**; pass = pooled season-clustered t > 2.5 and same sign 2016-19 /
2021-24. Calibration slope reported.
Method (`model/hprior.py`), on the shipped no-age projection (Marcel + S4-03 contact), shipped S5-01 age handling after
(A3 curve refit LOSO on each arm's own no-age residuals, half weight with Marcel age factor):
- h = weighted history PA (woba_den, 5/4/3 normalized so N-1 = 1) over N-3..N-1. Bands: no history (rookie), (0,100),
  [100,250), [250,500), [500,800), 800+.
- mu(band) = PA-weighted mean rel-wOBA in season M of hitters with PA >= 100 in M (pitcher-hitters out, same population as
  the scored targets), banded by their own h over M-3..M-1; M from 2013 (full 3-year window) to N-1, M = 2020 dropped.
  Seasons < N only; nothing fit on targets.
- Since Marcel's pull is linear, regressing toward league + mu equals adding shrink x mu, shrink = 1200 / (raw 5/4/3 PA + 1200).
  Rookies (no history, now projected 0) get mu(rookie) in full.
Arms:
- P0 shipped (reproduces .03283).
- **PC control** = one mu for everyone (all-band mean, same population), same shrink. Separates "regress toward hitters who
  play" from "regress toward hitters with similar playing time".
- **P1 candidate** = mu by history-PA band.
Expected: mu rises with h (rookies ~ -.010 to -.015, 800+ ~ +.010 to +.015). P1 .03265-.03278 (t 1.5-4), much of the gain
from rookies and short-history hitters (now projected league average). PC smaller (~.0328) since it pulls part-timers up
too. P1 vs PC t > 1. Slope near 1 (shipped .970). Risks: the A3 age curve already absorbs some of this (young = short
history), and much of the target-PA pattern is endogenous, so the reachable gain is a fraction of what S8-01's table suggests.

### S8-02 result (`model/hprior.py`, `work/results/s8_02_hprior.txt`, preds `work/results/s8_02_preds.parquet`)
| arm | pooled | t vs P0 | 2016-19 (t) | 2021-24 (t) | slope |
|---|---|---|---|---|---|
| P0 shipped (reproduced) | .03283 | | .03313 | .03254 | .970 |
| PC control: one mu for all | .03304 | -4.94 | .03324 (-4.69) | .03284 (-6.71) | .975 |
| P1 candidate: mu by history-PA band | .03276 | 2.07 | .03303 (2.39) | .03250 (0.73) | .932 |
P1 vs PC t 4.96 (16-19 3.29, 21-24 3.86). P1 better or equal in 7 of 8 seasons (2023 .0320 -> .0321); 2020 .0421 -> .0423.
mu (2024 fit): rookie -.003, <100 -.011, 100-249 -.009, 250-499 -.005, 500-799 -.002, 800+ +.013; all-hitter mean +.0045.
Mean error by history band, P0 -> P1: <100 -.0082 -> +.0004, 100-249 -.0059 -> -.0005, 250-499 -.0034 -> -.0020,
500-799 -.0017 -> -.0011, 800+ -.0006 -> -.0027, rookies -.0054 -> -.0048.
- **FAILS the written bar** (pooled t 2.07 < 2.5), though same sign both eras and inside the pre-registered range
  (.03265-.03278). Gain .00007.
- As expected: the band shape matters, not the level (PC is clearly worse; P1 vs PC t 4.96, both eras). Part-timers' bias is
  removed.
- NOT AS EXPECTED: rookies barely move (mu ~ 0; rookies who reach 100 PA are near league average, so the "projected 0"
  default was already right on average). The gain is all from short-history hitters.
- Cost: regulars (800+) get pulled up too far (mean error -.0006 -> -.0027) and the slope drops .970 -> .932. Marcel's
  pull for regulars was already offset elsewhere (age curve / survivor effects), so a flat band mean double-counts there.
  A follow-up would apply the prior only below ~500 history PA; that is post-hoc and would need its own pre-registration.

### S8-03 history-PA prior below 500 PA only (PRE-REGISTERED before running; POST-HOC origin flagged)
Idea: S8-02's band prior fixed part-timers but over-lifted regulars. Apply it only to hitters with weighted history PA
< 500 (bands rookie, <100, 100-249, 250-499); 500+ keep shipped. **Flag: the 500 cutoff was chosen after seeing S8-02's
by-band errors on the same 2016-24 targets, so this t is optimistic (forking paths).** Same bar: baseline **.03283**,
pooled season-clustered t > 2.5, same sign 2016-19 / 2021-24; slope reported. mu, shrink, age handling exactly as S8-02.
Arms: Q0 shipped; **Q1 candidate** cutoff 500; sensitivity only (not chosen from): cutoffs 250 and 800; P1 (S8-02, no
cutoff) shown for reference.
Expected: 800+ mean error back to ~ -.0006, slope back to ~.96. Q1 .03272-.03276, t 2-3.5 (S8-02 lost a little on
regulars; that comes back). Odds of a pass about even. Because of the flag, even a pass would be weaker evidence than a
clean one; if it passes I'd suggest re-checking on the 2025-26 holdout at the pre-season final before trusting it fully.

### S8-03 result (`model/hprior2.py`, `work/results/s8_03_hprior2.txt`, preds `work/results/s8_03_preds.parquet`)
| arm | pooled | t vs Q0 | 2016-19 (t) | 2021-24 (t) | slope |
|---|---|---|---|---|---|
| Q0 shipped | .03283 | | .03313 | .03254 | .970 |
| **Q1 cutoff 500 (candidate)** | **.03272** | **3.90** | .03303 (3.08) | .03241 (2.34) | .948 |
| Q250 (sensitivity) | .03274 | 3.54 | .03304 (2.13) | .03243 (2.54) | .951 |
| Q800 (sensitivity) | .03272 | 3.61 | .03302 (2.75) | .03242 (2.17) | .947 |
| P1 (S8-02, no cutoff) | .03276 | 2.07 | .03303 (2.39) | .03250 (0.73) | .932 |
Q1 vs P1 t 2.13. Q1 better in 6 of 8 seasons, tied 2021 and 2023; 2020 as target .0421 -> .0422.
Mean error by history band, Q0 -> Q1: <100 -.0082 -> .0000, 100-249 -.0059 -> -.0008, 250-499 -.0034 -> -.0025,
500-799 -.0017 -> -.0024, 800+ -.0006 -> -.0011 (regulars move a little through the refit age curve).
- **Q1 PASSES the written bar** (t 3.90, both eras positive), at the top of the expected range (.03272-.03276).
- Robust to the cutoff: 250 / 500 / 800 all t 3.5-3.9, so it isn't a knife-edge pick. That softens, but doesn't remove,
  the post-hoc flag (cutoff idea came from S8-02 on the same targets).
- Slope .970 -> .948 (slightly more spread than ideal); 2020 flat-to-worse.
- Suggest re-checking on the 2025-26 holdout at the pre-season final. Awaiting user OK to ship Q1.

- **S8-03 SHIPPED with user OK 2026-11-24** (Q1, cutoff 500). `model/hprior2.py` `ship_prior`, `CUT_SHIP`; `python model/hprior2.py --ship`
  builds `data/season/hit_pa_prior.parquet` (mu through 2024 ex-2020: rookie -.0042, <100 -.0106, 100-249 -.0102, 250-499 -.0057)
  and refits `age_curve_hit.parquet` on the new residuals (in-sample .03264; S5 curve kept as `age_curve_hit_s5.parquet`).
  The refit curve is higher at young ages (21: +.0064 -> +.0106), partly offsetting the prior for young part-timers; the
  LOSO test already includes that interaction. New hitter baseline to beat: **.03272**.

---

## Session 9 (2026-11-24): playing time (user picked this over game model / win totals / more rate work)

### S9-01 hitter + pitcher playing-time model (PRE-REGISTERED before running)
Question: how many PA (hitters) / BF (pitchers) will a player get in season N, from data in seasons < N only?
New target, so a new bar, same shape as the rate bar:
- **Population** (pregame info only, no rosters yet): hitters = non-pitchers (p_g <= .5 b_g) with >= 1 PA in N-1 or N-2;
  pitchers = players with >= 1 BF in N-1 or N-2 who were mainly pitchers (p_g > .5 b_g; position players pitching excluded).
  **Target = PA_N / BF_N, 0 if he didn't play.** Rookies (no MLB history) are a depth-chart question, not scored here.
- **Metric**: unweighted RMSE of PA (BF). Targets 2016-2024 ex-2020; 2020 reported. Pass = better pooled, season-clustered
  t > 2.5, same sign 2016-19 and 2021-24. Also reported (not the bar): mean error, RMSE among those who played (PA_N >= 1;
  endogenous, closer to what Marcel's formula was meant for), and by history band.
- **Baselines**: B0 Marcel as published: PA = .5 PA1 + .1 PA2 + 200; pitchers IP = .5 IP1 + .1 IP2 + 60 (GS/G >= .5 in
  N-1, else 25), converted to BF with the N-1 league BF/out. B1 = same with 2020 history prorated x162/60 (obvious fix).
  Candidate must beat the better of B0/B1.
- **Candidate C1**: gradient-boosted trees (sklearn HistGradientBoostingRegressor, squared loss, fixed params: lr .05,
  300 iters, max_leaf_nodes 15, min_samples_leaf 40, no tuning), trained on target seasons 2013..N-1 ex-2020 (so N-3 >= 2010
  exists), predictions clipped at 0. Features, all from N-1..N-3 (2020 counts prorated):
  hitters: PA1/PA2/PA3, games started share N-1, share of N-1 PA in the last 60 days of the season, catcher share of
  fielding games N-1, DH/PH share N-1, teams in N-1 (traded/waived), age in N, Marcel rel-wOBA projection and its weighted PA.
  pitchers: BF1/BF2/BF3, GS/G N-1, BF per appearance N-1, late-season BF share N-1, teams N-1, age in N, Marcel rel-RA9 proj.
- **Sensitivity only (not chosen from)**: C-OLS = linear regression on the same features.
Expected: Marcel's flat +200 / +60 is far too high for players who leave MLB, so B0/B1 should have a big positive mean
error (hitters ~ +60-100 PA) and RMSE around 180-210 PA (hitters) / 150-200 BF (pitchers). C1 removes the bias and uses
role/age/quality: RMSE 10-20% lower, t well above 2.5, both eras. Among those who played, gain smaller (~5-10%).
Risk: much of C1's gain may come from attrition (retirements/releases) that opening-day rosters would tell us anyway; the
"played" subset and the history bands show how much is left once that's accounted for. A roster-conditioned version
(40-man / opening-day rosters from the Stats API via Actions) would be the follow-up.

### S9-01 result (`model/ptime.py`, `work/results/s9_01_ptime.txt`, preds `work/results/s9_01_preds.parquet`)
Unweighted RMSE, targets 2016-24 ex-2020; better baseline is B0 (published Marcel; prorating 2020 history made it worse).
| | arm | RMSE | t vs B0 | 2016-19 (t) | 2021-24 (t) | mean err | RMSE among those who played |
|---|---|---|---|---|---|---|---|
| Hitters (PA, n 6176) | B0 Marcel | 200.0 | | 202.4 | 197.7 | +112.5 | 173.6 |
| | B1 Marcel, 2020 prorated | 201.7 | -1.51 | 202.4 | 201.1 | +123.5 | 174.2 |
| | **C1 boosted trees** | **140.7** | **15.58** | 139.7 (38.9) | 141.7 (7.50) | +3.6 | 155.8 |
| | C-OLS (sensitivity) | 140.3 | 15.29 | 138.8 | 141.7 | +8.1 | 152.1 |
| Pitchers (BF, n 7625) | B0 Marcel | 183.3 | | 190.1 | 176.9 | +87.9 | 170.4 |
| | B1 | 185.7 | -1.44 | | | +96.7 | 172.1 |
| | **C1 boosted trees** | **152.6** | **5.00** | 153.4 (17.9) | 152.0 (2.01) | +8.5 | 172.4 |
| | C-OLS (sensitivity) | 149.3 | 8.68 | 151.5 | 147.3 | +7.1 | 165.5 |
C1 better than B0 in all 8 seasons for hitters; pitchers 7 of 8 (2021: 171.9 vs 161.2, history includes prorated 2020).
2020 as target (x162/60): hitters B0 202.9 / C1 136.8; pitchers 199.3 / 159.4.
By N-1 band, mean error B0 -> C1: hitters none-in-N-1 +192 -> 0, 1-99 +142 -> -1, 100-299 +123 -> +4, 300-499 +79 -> +16,
500+ +18 -> 0. Pitchers none +121 -> -2, 1-99 +79 -> 0, 100-299 +71 -> +10, 300-599 +86 -> +17, 600+ +106 -> +40.
- **Both PASS the written bar** (hitters t 15.6, pitchers t 5.0; positive in both eras). Inside/above the expected range.
- AS EXPECTED: Marcel's flat +200 PA / +60 IP is the main error (mean +112 PA / +88 BF); C1 is unbiased overall.
- AS FEARED: much of the gain is attrition. Among players who did play, hitters improve 173.6 -> 155.8 (-10%), but
  **pitchers don't: C1 172.4 vs Marcel 170.4.** OLS (sensitivity, not chosen) does better for pitchers (149.3; played 165.5),
  which suggests the trees overfit the pitcher role mix (starters 600+ still +40 BF high; RMSE 250+ there = injuries).
- Starter-level BF is where the money is for team wins and it's the weakest spot. Next: opening-day rosters (Stats API,
  Actions) so the test conditions on who's actually on a roster, plus an injury/IL history feature for pitchers.

- **S9-01 HITTERS SHIPPED with user OK 2026-11-24** (C1 trees). Pitcher half held (no edge among pitchers who played);
  goes to S9-02 with opening-day rosters + IL history. `model/ptime.py` `ship_hitters_pt`; `python model/ptime.py --ship`
  writes `data/season/hit_pt_2027.parquet` (794 hitters with PA in 2025/26; 170k PA total vs Marcel 269k).
  2026 features come from Statcast (`model/ptime_statcast.py`, no Retrosheet 2026 yet). Validated on 2025 vs Retrosheet,
  hitters 50+ PA: PA/starts/fielding games/catcher games/late PA/teams corr >= .999; DH share .987; PH share .90
  (defensive subs who bat count as PH). Trained on targets 2013-2024; refit with 2025-26 at the pre-season final.

### S9-02 pitcher (and hitter) playing time with opening-day rosters + IL history (PRE-REGISTERED before running)
Data (Stats API via Actions, `pipeline/statsapi_fetch.py rosters/transactions`): each team's active and 40-man roster on
opening day (40-man carries status: active / 10-15-60 day IL / etc.), and all transactions (IL placements/activations).
All of it is known before opening day, so it's pregame info.
New features: on any opening-day active roster; on any 40-man; on the IL at opening day (and 60-day); IL days in N-1
(placements matched to activations, capped at season end) and IL stints N-1..N-3; roster position (SP/RP listing not
available on the roster call, so "P").
Same population, targets, metric and t-test as S9-01. Arms:
- B0 Marcel PT; R0 = S9-01 C1 trees (no roster info).
- **Pitchers, candidate C2** = S9-01 C1 trees + the new features. Pass = beats B0 AND R0 (t > 2.5 each, same sign both eras)
  AND, because of S9-01's weak spot, RMSE among pitchers who pitched (BF_N >= 1) no worse than B0.
- Pitchers, C2-OLS = OLS on the same features: second candidate (S9-01 hinted OLS suits pitchers; that hint came from the
  same targets, so flagged). Only preferred over C2 if it beats C2 with t > 2.5.
- **Hitters, H2** = shipped C1 + the new features vs shipped C1, t > 2.5 both eras.
Expected: roster status is a strong attrition signal, so the "unconditional" gain is large (pitchers ~152 -> ~125-135 BF,
hitters ~141 -> ~120-130 PA). Opening-day IL status should cut starters' over-projection (600+ band +40 BF -> under +20).
Among pitchers who pitched: C2 ~160-165 vs Marcel 170 (pass of the extra check is maybe 60/40). Hitters H2 t > 5.
Risk: 40-man by date may not be served historically for every season; if so, active roster only (reported).

### S9-02 result (`model/ptime2.py`, `work/results/s9_02_ptime2.txt`, preds `work/results/s9_02_preds.parquet`)
Data: Stats API rosters on each team's own first game date (first pull used the league opener, which gave spring
rosters for Tokyo/Seoul seasons; fixed) and transactions, backfilled 2010-2026 so every arm trains on targets 2013..N-1 as
registered. IL days parsed from placed/transferred/activated (injured or disabled list; retroactive dates honored):
every IL transaction classified; 455-814 stints/season.
| | arm | RMSE | t vs B0 | t vs R0 (16-19, 21-24) | mean err | played RMSE |
|---|---|---|---|---|---|---|
| Hitters PA | B0 Marcel | 200.0 | | | +112.5 | 173.6 |
| | R0 shipped S9-01 | 140.7 | 15.58 | | +3.6 | 155.8 |
| | **H2 = R0 + roster/IL** | **123.5** | 19.27 | **14.52** (8.83, 10.68) | +1.8 | 146.1 |
| | H2-OLS | 123.7 | 22.08 | 11.06 | +6.6 | 143.6 |
| Pitchers BF | B0 Marcel | 183.3 | | | +87.9 | 170.4 |
| | R0 (S9-01 C1) | 152.6 | 5.00 | | +8.5 | 172.4 |
| | **C2 = R0 + roster/IL** | **130.1** | 12.04 | **7.96** (27.9, 4.31) | +6.8 | **156.2** |
| | C2-OLS | 130.9 | 15.07 | 5.79 | +7.2 | 154.1 |
C2-OLS vs C2: t -0.63 (pitchers), -0.14 (hitters): OLS not preferred. Every season better than R0 for both.
On the IL at opening day: pitchers mean error B0 +179 / R0 +104 / C2 +14; hitters +168 / +95 / +1.
Starters (600+ BF in N-1): mean error +40 -> +32, RMSE 253 -> 224 (OLS +6 / 219).
- **Pitcher C2 PASSES all three parts** (vs Marcel t 12.0, vs S9-01 t 7.96, both eras; among pitchers who pitched 156.2
  vs Marcel 170.4). **Hitter H2 PASSES** (vs shipped t 14.5, both eras). Both inside the expected ranges.
- AS EXPECTED: opening-day IL status is the big one; it removes almost all the bias for injured players.
- NOT AS EXPECTED: starters still run +32 BF high (expected under +20). Trees under-fit the top end; OLS gets the bias
  but not better RMSE. Starter workload (innings caps, pitch counts, six-man rotations) is the next playing-time item.
- Production caveat: opening-day rosters for 2027 exist only in late March. Until then the features need a stand-in
  (current 40-man + IL status, od_active = unknown). That stand-in hasn't been tested.

- **S9-02 SHIPPED with user OK 2026-11-24** (hitters H2, pitchers C2; supersedes S9-01 hitters). `model/ptime2.py` `ship_pt`;
  `python model/ptime2.py --ship N` -> `data/season/pt_<N>.parquet` (needs `rosters_<N>.parquet`). Smoke run as of 2026
  (not scored, holdout): 783 hitters 171k PA, 1015 pitchers 163k BF; top SP ~736 BF. 2027 waits for opening-day rosters or
  a tested winter stand-in (next). Statcast features now carry pitchers' teams (2025 check: late BF MAE .14, teams exact).

### S9-03 winter stand-in for playing time (PRE-REGISTERED before pulling the data, 2026-11-24)
Question: can 2027 playing time be projected in December, before opening-day rosters exist, and how much of S9-02's
roster gain survives? Snapshot date: **Dec 1 of N-1** (Stats API 40-man by date, each team; 2027 uses the latest date
available when it's run). Everything used must be known by that date.
Why not just plug the winter 40-man into the shipped C2 model: in December most veteran free agents are on no 40-man,
so C2 would read them as released/minor leaguers. So the main candidate is **retrained on winter features**.
Winter features (`WFEAT`): w40 = on any 40-man Dec 1 N-1; w_fa = declared/elected free agency between the end of the
N-1 regular season and Dec 1 and not on a 40-man Dec 1; w_out = outrighted / released / DFA'd in that window and not on a
40-man; il_end = on the IL on the last day of the N-1 regular season (open stint), il_end60 = that stint is 60-day;
plus the S9-02 IL history (il_sh1, il_sh2, il_stints3, all known by Dec 1).
Same population, targets (2016-24 ex-2020), metric, trees/params and training window as S9-02. Arms:
- B0 Marcel; R0 = S9-01 trees (no rosters); C2 = shipped S9-02 (opening-day rosters; the ceiling, not beatable by design).
- **W1 = S9-01 features + WFEAT, trees, retrained.** Pass = beats R0 with t > 2.5, same sign 2016-19 and 2021-24
  (pitchers also: RMSE among those who pitched no worse than Marcel, as in S9-02). If it passes it ships as the
  production model until opening-day rosters exist, then C2 takes over.
- W0 (sensitivity) = shipped C2 model with winter values plugged in (od_40 = od_active = w40, od_il = od_il60 = il_end60,
  od_minors 0). Expected to be clearly worse than W1 (the FA problem). Not a candidate.
- W1-OLS (sensitivity only).
Expected: Dec 40-man is a much weaker attrition signal than the opening-day active roster (FAs re-sign; non-roster
invitees make teams). W1 recovers ~40-60% of the R0 -> C2 gain: hitters 140.7 -> ~128-133 PA (C2 123.5), pitchers
152.6 -> ~138-145 BF (C2 130.1); t vs R0 > 4 for both. il_end should cut the injured-starter bias (TJ cases) but less
than OD IL status did. W0 worse than W1 by > 5 PA/BF, possibly worse than R0.
Risk: the API may not serve historical 40-man rosters for Dec dates (if empty/odd, report and fall back to
transactions-only features: W1 without w40).

### S9-03 result (`model/ptime3.py`, `work/results/s9_03_ptime3.txt`, preds `work/results/s9_03_preds.parquet`)
Data: `data/statsapi/rosters_winter_<N>.parquet`, 40-man on Dec 1 of N-1 for all 30 teams, 2011-2026 (31-41 per team;
the API serves historical dates fine), plus 2027 as of 2026-11-24 (1150 players). Dec 40-man players are on an OD 40-man
93% of the time; winter FAs 56-57%.
| | arm | RMSE | t vs R0 (16-19, 21-24) | mean err | played RMSE |
|---|---|---|---|---|---|
| Hitters PA | B0 Marcel | 200.0 | | +112.5 | 173.6 |
| | R0 S9-01 (no rosters) | 140.7 | | +3.6 | 155.8 |
| | **W1 winter, retrained** | **137.7** | **4.56** (2.46, 4.08) | +3.3 | 154.3 |
| | W1-OLS (sens.) | 137.0 | 5.14 | +6.3 | 150.5 |
| | W0 plug into C2 (sens.) | 154.1 | -6.30 | +12.5 | 173.1 |
| | C2 opening day (ceiling) | 123.5 | 14.52 | +1.8 | 146.1 |
| Pitchers BF | B0 Marcel | 183.3 | | +87.9 | 170.44 |
| | R0 | 152.6 | | +8.5 | 172.35 |
| | **W1** | **150.1** | **2.52** (3.45, 1.44) | +8.1 | **170.36** |
| | W1-OLS (sens.) | 146.9 | 1.84 | +5.5 | 164.3 |
| | W0 (sens.) | 167.0 | -5.01 | +30.0 | 185.6 |
| | C2 (ceiling) | 130.1 | 7.96 | +6.8 | 156.2 |
W1 better than R0 in 8/8 seasons (hitters) and 7/8 (pitchers; 2023 140.9 vs 140.8). Winter FAs: mean error R0 +34 -> W1 +10
(hitters), +24 -> +7 (pitchers). On the OD IL: W1 +109 PA / +114 BF (R0 +95 / +104, C2 +1 / +14): IL status at the end of
N-1 says almost nothing about opening-day IL. Starters 600+ BF still +40.
- Hitters W1 PASSES (t 4.56, both eras positive). Pitchers W1 passes the letter of all three parts, barely
  (t 2.52; 21-24 only t 1.44; among those who pitched 170.36 vs Marcel 170.44).
- NOT AS EXPECTED: winter info recovers only ~15% of the opening-day gain (hitters 3.0 of 17.2 PA, pitchers 2.5 of 22.5 BF),
  not the 40-60% registered. The S9-02 gain is mostly March information (who made the active roster, who's hurt in March).
- AS EXPECTED: plugging winter rosters into the opening-day model (W0) is clearly worse than no rosters (FAs read as cut).
- Implication: December 2027 projections are about S9-01 quality; spring injury news and roster battles are where the
  remaining ~17-20 PA/BF of error lives. Re-pulling rosters through spring (e.g. weekly 40-man + IL from mid-Feb) is the
  natural follow-up rather than more winter features.

- **S9-03 W1 SHIPPED with user OK 2026-11-24** (hitters and pitchers) as the pre-opening-day playing-time model.
  `python model/ptime3.py --ship 2027` -> `data/season/pt_winter_2027.parquet` (40-man as of 2026-11-24): 794 hitters 172k PA
  (Marcel 269k), 1023 pitchers 165k BF (Marcel 260k). S9-02 replaces it once opening-day rosters exist.

---

## Session 10 (2026-11-24): starter workload

### S10-01 starter workload in playing time (PRE-REGISTERED before running)
Problem: S9-02 C2 over-projects starters (600+ BF in N-1) by +32 BF (W1 +40). Diagnostic on existing S9-02 preds (no new
model run): by target season 2016 +44, 2017 +23, 2018 +15, 2019 +4, **2021 +125** (history = 2020 prorated x162/60),
2022 -1, 2023 +33, 2024 +18; mean actual BF of that group fell 577 -> 495. Two suspects: (a) workload info the trees don't
see (pitch counts, career-high workload, innings jumps, 2020-prorated history), (b) drift: training on 2013..N-1 equally
weights older seasons with heavier starter workloads.
Same population, targets (2016-24 ex-2020), metric, t-test, tree params as S9-02. Applied on top of BOTH shipped models
(C2 opening day = primary; W1 winter = secondary). Pitchers only; hitters get K2 as a sensitivity (not a candidate).
- K1 workload features (all from N-1..N-3, prorated where counts): gs1 (GS N-1), pit_app (pitches per appearance N-1),
  pit_gs (pitches per game for GS share >= .5, else 0), maxbf (max BF N-1..N-3), jump = bf1 - max(bf2, bf3),
  h2020 = 1 if any of N-1..N-3 is 2020 (prorated history), lg_spbf = league mean BF per start in N-1 (season trend).
- K2 recency weights: training rows weighted 0.8^(N-1-season) (fixed, not tuned).
- **K3 = K1 + K2: the one candidate.** K1, K2 alone are reported as a decomposition (not chosen from).
Pass: K3 beats its base (C2; and separately W1) with pooled t > 2.5, same sign in 2016-19 and 2021-24, AND RMSE among
pitchers who pitched no worse than the base. Ship to each base it passes on.
Also reported: 600+ band mean error and RMSE, by season; 300-599 band.
Expected: overall pitchers C2 130.1 -> ~128-129 (t ~2-3; it's a minority of pitchers so the pooled gain is small);
600+ bias +32 -> +10-20, RMSE 224 -> ~215 (most starter error is injuries, which this doesn't fix). 2021 still high but
better (h2020). K2 helps 2021-24 more than 2016-19. Honest odds of passing the pooled bar: ~50/50.
Risk: lg_spbf is a season-level constant; trees can't extrapolate it below the training range (2027 will be the lowest).
Six-man / team rotation usage is not in this test (needs team-level starts; follow-up if this falls short).
