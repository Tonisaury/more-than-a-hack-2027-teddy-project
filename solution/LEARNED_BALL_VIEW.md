# Learned ball view — notes for review (branch `ander/work`)

**Shipped on 2026-11-17, with the dead-ball weight of the section "Adopted on top" (0.36).** This
file was written for the team to decide whether the change ships, and it lists what TECHNICAL.md would have to say if it does.

## What changed

One thing: **who is on the ball** is no longer read with the hand-written formula in
`prep.ball_view` (nearest player within 2.5 m, damped by relative speed) but with a classifier
trained on other matches (`mtah/ballmodel.py`, stored in `models/ball_view.pkl`).

Everything else is the shipped pipeline: dead-ball cues (`out`, `still`, `moving_in`), the
goalkeeper rule, the first pass and its pseudo-labels, the player view, the sticky decoder and its
costs. `model.LEARNED_BALL_VIEW = False` gives back the previous submission.

- **Training data:** the DFL open dataset (Bassek, Rein, Weber & Memmert 2025, Scientific Data;
  7 matches of the 2022/23 Bundesliga and 2. Bundesliga; TRACAB tracking at 25 Hz; CC BY 4.0;
  figshare DOI 10.6084/m9.figshare.28196177). Each ball frame carries `BallStatus` (in play or
  not) and `BallPossession` (home / away). These are labels on other games, made by the data
  provider, publicly available. No label of any kind on the challenge match is used.
- **What it sees:** 47 label-free features around the ball, built from `prep.ball_view`'s own
  outputs: control and distance of each side's nearest player, ball speed, acceleration and
  change of direction, distance to the boundary, and summaries of the half second, second and
  three seconds before and after.
- **What it answers:** how much likelier home is than away to have the ball, on frames where the
  ball is in the data. It is trained on in-play frames only and has no say on dead ball.
- **Made to look like our tracking** (`dfl_eval.degraded_ball`): the challenge tracker's ball-gap
  pattern is laid over the DFL ball, the ball in play gets a slow positional wander (0.8 m / 1 s
  and 1.2 m / 2 s, as in `standin_eval.py`), and the ball is removed from half of the stoppages.
  The gap pattern is the only thing taken from the challenge match: which frames have a ball. No
  label and nothing a person decided.
- **Symmetric:** every training frame is also given with the sides swapped, and predictions are
  averaged with their mirror image, so it cannot prefer either side.
- **Two constants:** `K_LEARNED = 0.5` (0.25–1.0 is a plateau) and the cap of one unit of
  evidence per frame (below). The team-change cost stays at 4.

## Evidence (matches the classifier never saw)

`spellF1`, "controlled" reading; the "loose" reading agrees in every cell.

| test | shipped | learned | cells won |
|---|---|---|---|
| DFL, leave one match out, stoppages not given: J03WMX, J03WN1, J03WOH (`ballmodel_eval.py --targets ...`) | 0.452 | 0.529 | 6 / 6 |
| same, the other four matches: J03WOY, J03WPY, J03WQQ, J03WR9 | 0.481 | 0.543 | 7 / 8 |
| all seven DFL matches (`ballmodel_eval.py`) | 0.469 | 0.537 | 13 / 14 |
| Metrica sample games, never used for training, stoppages given (`ballmodel_eval.py --metrica`) | 0.595 | 0.637 | 8 / 8 |

Under the "loose" reading the learned version wins all 14 DFL cells (+0.076 on average). The one
cell it loses under the "controlled" reading is on one of the four matches measured last.

Possessions found on the first three DFL matches: 230 → 286 of 335; on the other four, 235 → 279 of 314. The gain holds across the grid of
`K_LEARNED` × team-change cost (12 settings), with one exception on Metrica (k 1.0, cost 2).

On the challenge match (no ground truth; label-free counts):

| | shipped | learned |
|---|---|---|
| stoppages | 97 | 99 |
| stoppages under 3 s | 3 | 10 |
| possessions | 271 | 333 |
| share of graded frames called dead | 42% | 40% |
| public five minutes (pipeline check only): possessions (truth 17) / spellF1 / macroF1 | 14 / 0.452 / 0.846 | 17 / 0.506 / 0.790 |

`run.py` is deterministic with the stored model (two runs, same file), and
`ballmodel_eval.py --train-final` reproduces the stored model byte for byte on the same machine.

## What went wrong on the way (worth a line in "tried and dropped")

1. **A three-class version (home / away / dead)** scored higher still on the DFL matches (+0.10)
   and broke the challenge match: 185 stoppages, 73 of them under three seconds, 451 possessions.
   In the DFL data a dead ball is a frozen placeholder on every frame, so the classifier learnt
   "dead = motionless" and called every moving ball alive, including a ball being carried to a
   corner flag. The stand-in matches cannot show this, because they share the flaw. Fixed by
   training on in-play frames only.
2. **Uncapped evidence** (up to 2.3 units per frame) still took 11% of the previous dead time
   away: the classifier is sure about a ball in a player's hands during a stoppage, and outweighed
   the player view. Capped at one unit per frame, the formula's maximum: it says who, and weighs
   no more against "dead" than the formula did.
3. **"Lost on its way over the line" as a dead-ball cue**: when a ball vanishes heading for a
   boundary line within a second, call it out. On gaps laid over a ball that was really there,
   the truth was dead within a second in 2 of 20 cases. Dropped; the code is not in this branch.
4. **Boundary snapping (`refine.py`) part by part**: team changes only gives +0.006 on DFL (6/6)
   and +0.002 on Metrica; restarts and exits give nothing. Still off.
5. **Dead-ball switching costs 6–24, player-view weight 0.15–0.6, team cost 2–12 on DFL**: the
   shipped values are at or next to the best in each sweep (`dfl_eval.py`). An independent
   confirmation of the constants chosen on the Metrica games.

6. **Averaging the player view over three seeds, and a second self-training round**
   (`rounds_eval.py`, 6 DFL cells, on top of the learned ball view). Averaging: -0.002. Second
   round, where the player view is also taught on the frames without a ball that the first decode
   is sure about (49% -> 96% of the match pseudo-labelled): -0.001 with one seed, +0.002 with
   three. Neither beats the single-seed, single-round version in more than half of the cells.
   Not adopted. The seed still decides about 5% of the possessions (`seed_stability.py`), but
   removing that lottery does not move the score on other matches.

7. **A richer ball classifier.** Three additions, screened leave-one-match-out on 6 DFL cells and
   on the 8 Metrica cells (player view kept from the first version): each side's second-nearest
   player and how close the ball's path ahead and behind passes to a player (who it is going to,
   who it came from); eight degradations of each training match instead of two (wander 0.5-1.6 m,
   two placements of the gap pattern); a larger model (600 trees, 63 leaves). The new features
   alone: 0.000. Everything together: +0.009 on DFL and +0.004 on Metrica ("controlled"; +0.003
   and +0.008 "loose"). Real but a quarter of the metric's noise band, so the first version stays
   and the code is not in this branch.

## Adopted on top: give the stoppages back (`model.LEARNED_DEAD_WEIGHT = 0.36`)

Reviewing this branch on the challenge match (label-free counts): the learned version turns 156 s of
what was dead ball into a possession and only 28 s the other way, and inside the public five
minutes its dead-ball share is 10% (truth 20%, previous version 17%). Neither bench can see this:
on Metrica the stoppages are given, and on DFL a dead ball is a frozen placeholder that `still`
catches by itself.

Cause: the cap limits the classifier's maximum, not its typical strength. On the challenge match's
frames with a ball that is not out or still, the formula's |net| averages 0.28 (zero on 39% of
frames); the learned |net| averages 0.78 and sits at the cap on 59%. So it weighs 2.8 times more
against "dead" than the formula did.

Change (`model.LEARNED_DEAD_WEIGHT = 0.36`, nine lines in `model.ball_emissions`): both H and A
give back `(1 - 0.36) * |net|`. Home-against-away is exactly as before; alive-against-dead is
scaled to the formula's level. 0.36 is the ratio above, measured without labels, not searched.

| challenge match | previous (formula) | learned, weight 1.0 | learned, weight 0.36 |
|---|---|---|---|
| possessions | 271 | 333 | 334 |
| share of graded frames called dead | 42.0% | 40.0% | 41.5% |
| dead before -> possession now | | 156 s | 79 s |
| stoppages under 3 s | 3 | 10 | 6 |
| public five minutes (pipeline check): dead share (truth 20%) | 17% | 10% | 17% |
| public five minutes: spellF1 / macroF1 | 0.452 / 0.846 | 0.506 / 0.790 | 0.533 / 0.833 |

File with weight 0.36: SHA-256 `48314f01c8d295187aee1f2f172d534df4150c672303889f08aa470397874348`.

**On DFL, stoppages not given (14 cells, final pass on the cached player view):** weight 1.0 → 0.537,
0.6 → 0.533, 0.36 → 0.531; each beats the formula in 13 of 14. Within noise, and this bench cannot see
what the weight gives back. How to rerun it: `python solution/ballmodel_eval.py`
now prints three extra rows, "dead weight 1.0 / 0.6 / 0.36" (final pass only, on the cached player
view). If 0.36 is not worse than 1.0 there, it should ship; if it is clearly worse, weight 1.0 is
the version in `ander/work`.

## Known limits

- The DFL possession flag is the provider's and is not documented in detail ("the player in
  control of the ball"); it changes hands on touches (median possession 3 s). Training uses the
  same "controlled" folding as `standin_eval.py`.
- Our tracking is automatic and noisier than anything in either test bench; item 1 above is an
  example of a difference the benches could not see. There may be others.
- The stand-in matches have no broadcast feed or tagging, so the player view is shape-only there.
- Blind stoppage starts are late (median 1.4 s on DFL, when the ball is not in the data).
  Correcting that moves `spellF1` by 0.002, so it is left alone.
- macroF1 on the public window is lower than before. Not investigated; 17 possessions.
- A fifth of the possessions is still missed on the DFL matches.

## If it ships: what TECHNICAL.md needs

**Done in this branch** (sections 1, 2.2, 2.5, 3, 4, 5, 6, 7, 8 and 9, and RUN.md). The list is
kept as a checklist for the review; if the team keeps the hand-written version instead, revert
TECHNICAL.md and RUN.md to `main` and set `model.LEARNED_BALL_VIEW = False`.

- Section 1 and 4 currently say nothing is trained on any other match. That stops being true:
  the DFL dataset, its source, licence, who labelled it (the provider) and how much (7 matches,
  452,354 training rows after side-swapping).
- Section 2.2: the ball view's who-is-on-it part; section 2.5: the second test bench.
- Section 3: the items above.
- Section 8: `models/ball_view.pkl` must be in the public repository; retraining it needs the
  DFL data (2.6 GB, `dfl.fetch`) and the challenge bundle (for the gap pattern).
- Section 9: written with an AI coding assistant.

## Other new files

| file | what |
|---|---|
| `mtah/dfl.py` | loads the DFL matches into the pipeline's format |
| `dfl_eval.py` | test bench on DFL matches with stoppages not given; scores stoppage edges too |
| `ballmodel_eval.py` | leave-one-match-out and Metrica tests of the learned ball view; `--train-final` |
| `diagnose.py` | label-free counts of a run's output (segments, edges, what each stoppage rests on) |
| `seed_stability.py` | how much the answer depends on the player view's seed (5% of possessions) |
| `rounds_eval.py` | seed averaging and a second self-training round, measured on DFL matches |
