# Technical document — two views and a sticky state machine

More than a Hack 2027, Phase 1 (ball possession), match BAR-ATH.
Team **Teddy Project**: Toni Sagalés, Ander Barrio, Luis Boix.
Code: `solution/`. One command regenerates the submission: `python solution/run.py`.
The submitted file has SHA-256 `48314f01c8d295187aee1f2f172d534df4150c672303889f08aa470397874348`; it
was regenerated on two machines.

## 1. The approach in one paragraph

A possession is an interval, and for four graded frames in ten the ball is not in the data at all.
So the system does two things. It looks at each frame through **two independent views** — a
*ball view* that only asks who is on the ball, and a *player view* that never sees the ball and
reads the situation from twenty players, two cameras and the Smart Tagging — and it joins them
with a **sticky three-state hidden Markov model** (`H`, `A`, `D`) that only changes state when
the accumulated evidence pays a switching cost. No human label from this match is used anywhere.
Two things are learned. The ball view's reading of *who is on the ball* comes from a classifier
trained on seven other, public matches that carry the answer on every frame (section 2.2). The
player view is trained on pseudo-labels that the ball view and two physical rules produce on this
match. Nine other public matches in all serve as test benches (section 2.5).

Why this rather than something else:

- A per-frame classifier flickers, and the ranking metric (`spellF1`) counts possessions. A state
  model with a switching cost produces intervals by construction. The baseline's four variants
  show the same thing: smoothing moves `spellF1` by a factor of five.
- The ball is missing in long stretches (median gap about ten seconds, some over two minutes), so
  smoothing or interpolating the ball cannot cover them. Something that does not need the ball has
  to.
- With no labels allowed on this match, the only teacher available is the data itself. The ball
  view is reliable where the ball is seen; it teaches a model that also works where it is not.
- "Which side is on the ball, given the players around it" is the one question that looks the same
  in every match, so it is the one question other matches are allowed to teach. Whether play is
  stopped depends on what this match's cameras and tagging show, and stays with this match.

## 2. Pipeline

| step | module | what happens |
|---|---|---|
| 1 | `data.py` | Parse both ATD files, Smart Tagging and both homographies. Frame index = tactical video frame. |
| 2 | `sync.py` | Measure the broadcast → tactical time mapping; verify every broadcast frame against the tactical players. |
| 3 | `prep.py`, `ballmodel.py` | Clean and fuse the ball track; per-frame ball-view evidence; who is on the ball from the classifier in `models/ball_view.pkl`. |
| 4 | `model.py` + `hmm.py` | HMM on the ball view alone → posteriors → confident pseudo-labels. |
| 5 | `playerview.py` | 141 ball-free features per frame; gradient-boosted classifier, cross-fitted. |
| 6 | `model.py` + `hmm.py` | HMM on both views → Viterbi path → `frame,label` CSV. |
| — | `standin.py`, `standin_eval.py`, `dfl.py`, `dfl_eval.py`, `ballmodel_eval.py` | Not part of the run: the test benches on other matches, and the training of the ball classifier (sections 2.2 and 2.5). |

### 2.1 Synchronising the broadcast recording

The broadcast tracking is on its own timeline and no mapping is provided. Both trackings are in
pitch coordinates, so the mapping can be measured.

1. **Coarse.** Masked FFT cross-correlation of the two ball trajectories in 160-second chunks.
   Each chunk votes for `tactical_frame − broadcast_frame`; the votes form two tight clusters
   (+4014 and +26918 frames — one per half; the broadcast recording skips most of half-time).
2. **Fine.** Per 60-second window, the shift within ±15 frames that makes the broadcast players
   land on the tactical players. The result drifts by at most three frames across the match.
3. **Verify.** A broadcast frame is trusted only if the mean distance from each of its players to
   the nearest tactical player of the same team is under 1.5 m (at least five players). Replays
   and cutaways show another moment, fail this test and drop out with no special handling.
   90,882 of 164,488 broadcast frames pass.
4. **Correct.** The median residual of the matched players is the local homography error of the
   broadcast frame; the same shift is applied to its ball position.

Where both recordings see the ball, the mapped broadcast ball is a median 0.72 m from the tactical
one (82% within 3 m). It fills tactical gaps on 3.3% of graded frames (59.3% → 62.6%; 62.8% after
bridging gaps of up to one second). That is a small gain. The larger payoff of the
synchronisation is a different signal entirely: **whether the TV director is on the live wide
shot** at each tactical frame, and where that camera is pointing.

### 2.2 Ball view

Per frame with a ball position:

- `control_home`, `control_away` ∈ [0, 1]: `max(0, 1 − d / 2.5 m)` for the nearest player of each
  side, damped when the ball is moving fast relative to that player (a ball flying past a player
  is not held by him). The evidence is the difference of the two, so a 50/50 duel is neutral.
- `out`: ball more than 0.4 m beyond the boundary. `still`: positional spread under 0.2 m over one
  second (a placed ball). Both are evidence for `D`, and cancel control evidence — a player
  standing over a placed ball, or holding it on the touchline, does not possess it.
- `moving_in`: in the pitch and faster than 4 m/s — mild evidence against `D`.
- A loose ball adds a 0.02 tie-break for the side that last clearly controlled it, so a switch is
  placed at the first frame of the new side's control and never earlier.

Isolated ball detections that imply more than 45 m/s in and out are removed first.

**Who is on the ball: learned from other matches.** The control values above are a formula
written by hand, and it looks at one frame at a time: it never asks what a touch looks like. So
the *direction* of the control evidence (home or away) is taken from a classifier instead
(`mtah/ballmodel.py`), and the formula's outputs become its inputs.

- **Training data.** The DFL open dataset (Bassek, Rein, Weber & Memmert 2025, *Scientific Data*;
  seven matches of the 2022/23 Bundesliga and 2. Bundesliga; TRACAB optical tracking at 25 Hz;
  CC BY 4.0; figshare DOI 10.6084/m9.figshare.28196177). Each ball frame carries the provider's
  `BallStatus` (in play or not) and `BallPossession` (home or away). Short spells sandwiched
  inside the other side's play are folded into it first, as in section 2.5.
- **Made to look like our tracking.** The DFL ball is hand-checked and present on every frame.
  Before training, the challenge tracker's own ball-gap pattern is laid over it (which frames
  have a ball: no label, nothing a person decided), the ball in play is given the slow wander of
  section 2.5 (0.8 m over 1 s and 1.2 m over 2 s), and the ball is removed from a random half of
  the stoppages.
- **Features** (47, none of them a label): control and distance of each side's nearest player,
  ball speed, acceleration and change of direction, distance to the boundary, and summaries of
  the half second, second and three seconds before and after each frame.
- **It answers one question.** It is trained on in-play frames only and returns how much likelier
  home is than away to have the ball. It has no say on whether the ball is dead: that stays with
  `out`, `still`, `moving_in` and the player view (section 3 explains why).
- **Symmetric.** Every training frame is also given with the sides swapped, and each prediction
  is averaged with its mirror image, so it cannot prefer either side.
- **How it enters.** Half a unit of evidence per unit of log-odds, capped at one unit per frame,
  the formula's own maximum. `model.LEARNED_BALL_VIEW = False` gives back the formula.
- **It says who, not whether.** The cap limits the classifier's maximum, not its typical
  strength. On this match's frames with a ball that is neither out nor still, the formula's
  evidence averages 0.28 units (zero on 39% of frames); the classifier's averages 0.78 and sits
  at the cap on 59%. At full weight it therefore argued against "dead ball" 2.8 times harder
  than the formula ever did, and took dead time away (section 3). So both sides give back
  `(1 − 0.36) × |net|` (`model.LEARNED_DEAD_WEIGHT`): home against away is exactly what the
  classifier says, alive against dead weighs what the formula's did. 0.36 is the ratio of the two
  averages above, measured on this match without any label; it was not searched.

A `HistGradientBoostingClassifier` (300 trees), 452,354 training rows. The trained model is in the
repository (`solution/models/ball_view.pkl`, 1.2 MB); `python solution/ballmodel_eval.py
--train-final` rebuilds it.

**Goalkeeper rule.** A keeper who catches or smothers the ball hides it from the tracker, and the
decoder is then left with the player view, which reads team shapes — and at that moment the
shapes still say the other side is attacking. So the ball view carries one piece of evidence into
frames without a ball. The goalkeeper of a side is found by position, not identity: its player
nearest its own goal, if inside its own penalty area. If in the ten frames before the ball
disappears his grip reached 0.3, no opponent's was higher and the ball was not out, his side
gets one unit per frame (the same as firm control of a visible ball) for up to six seconds — the
Laws' limit on holding the ball — or until the ball is seen again. On this match it fires seven
times, 41 seconds in all (section 2.5 for how it was checked).

### 2.3 Player view

A `HistGradientBoostingClassifier` over 141 features, none of which uses the ball:

- **Team shape and movement**, in "home attacks to the right" coordinates so both halves look the
  same: centroid, spread, quantiles of player x and y, mean velocity, speed quantiles, share of
  players walking, compactness; differences between the two sides.
- **Cameras**: the pitch point at the centre of each camera's image and its zoom, from the
  homographies; pan speed; whether the broadcast feed is on the verified live wide shot (also
  averaged over ±2 s and ±6 s); which side has players nearest to where the TV camera points.
- **Smart Tagging**: every phase code as a signed indicator (+1 home, −1 away), time since and
  until the nearest set-piece tag, set-piece type, goal intervals. The classifier learns how far
  to trust each tag from the pseudo-labels; no tag is copied across as a label.
- **Context**: how ten core features differ 2 s and 5 s before and after.

**Pseudo-labels** (127,307 of 158,857 graded frames; H 36%, A 23%, D 41%):

1. the ball-view HMM's posterior is at least 0.95 and the ball is seen;
2. "everybody is walking and the TV has cut away" → `D`: mean player speed over ±6 s in the
   slowest 40% of this match *and* the broadcast on the live shot for under 25% of those seconds.
   A free kick being set up with the ball in view looks like possession to the ball view; frames
   that are walking with the TV mostly away are never taken as live labels;
3. one second before to two seconds after each tagged set-piece start, and the wait before the
   tagged second-half kick-off → `D`.

**Cross-fitting.** The match is cut into one-minute blocks dealt into five folds; every frame is
scored by a model that never trained on its own block. Held-out agreement with the pseudo-labels
is 87.1% (recall H 0.87, A 0.76, D 0.93) — the player view recovers who has the ball, without
the ball, about four times in five.

### 2.4 Fusion

Emission log-scores are summed and decoded per half with Viterbi:

- ball view: up to about one unit per frame;
- player view: `0.25 × log p`, split into "in play or not" (always full weight) and "which side"
  (full weight where the ball is not seen, a quarter where it is — the ball view is the teacher
  the classifier learnt from, and a student should not overrule the teacher on frames the teacher
  can see);
- a change of state costs 12 units, except a change of team while the ball is in view, which
  costs 4. With the ball in view a clean change of control is direct evidence, and a sixth of a
  second of it is enough; without the ball, or in and out of a stoppage, the decoder has only
  indirect evidence and asks for three times as much. The figure 4 is the one constant in the
  system chosen by measurement, and it was measured on other matches (section 2.5).

Frames outside the two halves are written as `X`. There are no abstentions inside them.

### 2.5 Choosing constants without touching the provided match

The public five minutes may not be used as a tuning set, and with 17 possessions they could not
support one. So the decoder is tested on **other matches that have answers**: Metrica Sports'
open sample data (`github.com/metrica-sports/sample-data`, Sample Games 1 and 2 — the same
coordinate convention as the challenge tracking, with hand-made event data).
`solution/standin_eval.py` downloads it, and `solution/mtah/standin.py` builds the test:

- **Truth** is derived mechanically from the events: dead ball from the end of the last event
  before each set piece until that set piece; otherwise the team of the latest pass, shot, set
  piece, recovery, lost ball or ball out. Two readings are scored, because events over-count
  changes of hands compared with "controlled possession": *loose* (every owning event counts) and
  *controlled* (a spell under two seconds sandwiched inside the other team's play is a touch, not
  a change). A change to the decoder is adopted only if it helps under both.
- **Degradation.** The sample tracking is hand-corrected and far cleaner than ours: its ball sits
  a median 0.3 m from the nearest player; ours 1.9 m. A decoder tuned on clean data over-splits on
  noisy data — we saw exactly that (section 3). So the stand-in ball is given a slow positional
  wander sized, from label-free statistics alone, to bracket the challenge tracking (0.8 m over
  1 s and 1.2 m over 2 s → median distance to the nearest player 1.4 m and 1.9 m), and each is run
  with the sample data's own ball gaps (ball in view on 62% of frames; ours 63%) and with the
  challenge tracker's gap pattern laid over it (37%, the pessimistic case). That gives four
  scenarios × two games. `python solution/standin_eval.py --noise` prints the comparison.
- **What can be judged.** These games have no broadcast recording, homography or Smart Tagging,
  which is what our stoppage detection leans on. So stoppages are *given* from the truth in these
  tests, and only the who-has-it part of the decoder is measured.

Result (`python solution/standin_eval.py --sweep`; `spellF1`, mean of the eight cells):

| who-has-it decoder, stoppages given | loose truth | controlled truth | possessions found (true: 369 / 281) |
|---|---|---|---|
| organisers' baseline (nearest player + 2 s vote) | 0.358 | 0.370 | 433 |
| team-change cost 12 (previous version) | 0.473 | 0.569 | 190 |
| cost 8 | 0.489 | 0.584 | 198 |
| cost 6 | 0.495 | 0.588 | 208 |
| **cost 4 (shipped)** | **0.505** | **0.595** | 220 |
| cost 3 | 0.504 | 0.590 | 234 |
| cost 2 | 0.498 | 0.572 | 255 |

Cost 4 beats cost 12 in 8 of 8 game × scenario cells under both readings (mean +0.032 and +0.026).
The gain is modest and it is a gain on other matches under an imitation of our noise; it is not a
measurement on the provided match.

**The goalkeeper rule** (section 2.2) was checked the same way (`python solution/standin_eval.py
--keeper`). Its premise holds: over the eight cells it fires 14 times, and in those frames the
event truth gives the ball to the keeper's side 92% of the time, to the other side 0% and calls
it dead 2% (with a ten-second window instead of six the first figure falls to 83%, so six stays).
It does not move `spellF1` there (0.5045 and 0.5950 with or without it), because on those
matches the player view already gets these frames right. On the provided match it does change
the answer: in the frames where it fires for the away keeper, the label was his side on 31% of
them before and 68% after, the rest dead ball. So it is adopted on the strength of its premise,
not of a measured gain.

#### A second test bench, where stoppages can be measured

The sample games above give the decoder nothing to find a stoppage with, so stoppages have to be
handed over. The DFL matches carry a per-frame "in play" flag, so `solution/dfl_eval.py` runs the
whole decoder on them with nothing given (ball degraded as described in section 2.2; two wander
scenarios per match), and scores the dead-ball class and the edges of stoppages as well. Like the
sample games they have no broadcast feed or tagging, so the player view is shape-only there.

On three of these matches the constants chosen on the sample games were swept again
(`spellF1`, "controlled" reading, six match × scenario cells): team-change cost with the ball in
view 2 / 3 / **4** / 6 / 8 / 12 → 0.437 / 0.447 / **0.452** / 0.444 / 0.440 / 0.435; switching
costs into and out of dead ball from 6 to 24 → within 0.001; player-view weight 0.15 / **0.25** /
0.40 / 0.60 → 0.462 / **0.452** / 0.434 / 0.417 (the lower weight wins there because the
shape-only player view is weaker than ours, which also has cameras and tagging). The shipped
values are at or next to the best in every sweep: a second, independent dataset agrees with them.

**The learned ball view** (section 2.2) was tested on matches its classifier never saw
(`python solution/ballmodel_eval.py`): leave-one-match-out on the seven DFL matches, and the two
Metrica sample games, which were not used for training at all.

| test (`spellF1`, "controlled" reading) | formula | learned | cells won |
|---|---|---|---|
| DFL, each match scored by a model trained on the other six, stoppages not given | 0.469 | 0.537 | 13 / 14 |
| Metrica sample games, classifier trained on the seven DFL matches, stoppages given | 0.595 | 0.637 | 8 / 8 |

Under the "loose" reading it wins all 22 cells (+0.076 on DFL, +0.064 on Metrica). On DFL it
finds 282 of 323 true possessions per match where the formula found 233. The gain holds across
the grid of evidence scale (0.25, 0.5, 1.0) × team-change cost (2, 4, 8, 12), with one exception
on Metrica; the cost stays at 4. These are gains on other matches under an imitation of our
noise. They are not a measurement on the provided match.

## 3. Assets used, and dropped

| asset | used for |
|---|---|
| tactical ATD | everything: ball view, team shapes, velocities |
| broadcast ATD | time synchronisation, ball gap filling, live-shot detection |
| tactical homography | where the tactical camera points, zoom |
| broadcast homography | where the TV camera points, zoom |
| Smart Tagging | features of the player view; set-piece starts and the second-half kick-off as `D` pseudo-labels |
| both videos | **not used.** No pixels are read. |

Tried and dropped:

- **A missing ball as evidence for `D` inside the HMM.** It is the baseline's rule and it is right
  on average, but as a per-frame term over minute-long gaps it swamps every other signal; the
  player view learns the same thing with context.
- **Snapping decoded boundaries onto ball events** (`refine.py`: first touch of the new side, the
  kick that restarts play, the frame the ball leaves the pitch). No measurable change on the
  stand-in matches (0.473 → 0.473, 0.495 → 0.496), so it is in the repository but switched off.
- **Silencing the player view's who-has-it opinion wherever the ball is in view.** On the clean
  stand-in this was the best setting (+0.02). With the stand-in ball made as noisy as ours it
  still wins at the lower noise level but loses by 0.03–0.06 at the higher one (over the four
  scenarios: 0.503 vs 0.505 loose, 0.570 vs 0.595 controlled), and on the provided match it nearly
  doubled the number of possessions (191 → 334). A noisy ball needs the second opinion. The
  quarter weight stays. This is why section 2.5 degrades the stand-in before trusting it.
- **Other constants.** Gap cost 8–24 and player-view weight 0.15 move `spellF1` by under 0.005;
  weight 0.4 costs 0.015. A wider control radius helps slightly (0.478 at 1.5 m, 0.495 at 2.5 m,
  0.505 at 4.5 m) but the radius also shapes the pseudo-labels and the stoppage evidence, which
  the stand-in cannot judge, so it is left at 2.5 m.

- **A three-class learned ball view (home / away / dead).** It scored higher still on the DFL
  matches (+0.10) and broke the provided match: 185 stoppages, 73 of them under three seconds,
  451 possessions. In the DFL data a dead ball is a frozen placeholder on every frame, so the
  classifier learnt "dead = motionless" and called every moving ball alive, including a ball
  being carried to a corner flag. The stand-in matches cannot show this, because they share the
  flaw; the label-free counts of the provided match did. Hence the two-class version, trained on
  in-play frames only.
- **The learned evidence at full weight against dead ball.** Capped at one unit it still turned
  156 s of what the hand-written version called dead into possession (28 s the other way), most
  of it with the ball in view and moving: a ball being carried or rolled back during a stoppage.
  Scaled to the formula's strength (section 2.2) that falls to 79 s, with the possessions
  unchanged (334 against 333).
- **Uncapped learned evidence** (up to 2.3 units per frame). It still took 11% of the previous
  dead time away, because the classifier is sure about a ball in a player's hands during a
  stoppage. Capped at one unit, the formula's maximum.
- **"Lost on its way over the line" as a dead-ball cue.** When the ball vanishes while heading
  for a boundary line it would cross within a second, call it out. On gaps laid over a stand-in
  ball that was really still there, the event truth was dead within a second in 2 of 20 cases: a
  ball heading for the touchline is usually kept in. Dropped.
- **Boundary snapping, part by part.** On DFL, snapping team changes only: +0.006 (6 of 6 cells);
  restarts only and exits only: 0.000. On the sample games +0.002. Still switched off.
- **Moving stoppage starts that were placed without a ball.** On DFL they are late by a median
  1.4 s (starts placed with the ball in view: 0.1 s early). Moving them earlier by up to two
  seconds changes `spellF1` by at most +0.002. Left alone.
- **Averaging the player view over three seeds, and a second self-training round**
  (`rounds_eval.py`). The seed of the player view's classifier decides about 5% of the
  possessions (`seed_stability.py`), but averaging does not move the score on DFL (−0.002). A
  second round, in which the player view is also taught on the frames without a ball that the
  first decode is sure about (49% → 96% of the match pseudo-labelled), gives −0.001 to +0.002.
- **A richer ball classifier**: each side's second-nearest player, how close the ball's path
  ahead and behind passes to a player, eight degradations of each training match instead of two,
  and a model twice the size. Together +0.009 on DFL and +0.004 on the sample games. A quarter of
  the metric's noise band; the simpler one ships.

## 4. Training data

No pre-trained model from anyone else. No hand labels made by us on this or any other game.

**Used for training: the DFL open dataset** (Bassek, Rein, Weber & Memmert 2025, "An integrated
dataset of spatiotemporal and event data in elite soccer", *Scientific Data*; figshare DOI
10.6084/m9.figshare.28196177; CC BY 4.0). Seven matches of the 2022/23 Bundesliga and
2. Bundesliga, about 1.0 million tracking frames. The labels are the provider's own per-frame
possession and ball-status flags, published with the tracking; we did not label anything. From
them, 452,354 training rows (in-play frames with a ball, every third one, each also with the
sides swapped) train the ball view's classifier (section 2.2). The dataset is public and is
downloaded by `mtah/dfl.py`. None of it is about the provided game.

**Used for testing only: Metrica Sports' open sample games 1 and 2** (tracking and event data
about two other, anonymised matches). Nothing is trained on them. They are downloaded by
`standin_eval.py`.

Neither dataset is needed to regenerate the submission: the trained classifier is in the
repository. Both are needed to rebuild it or to rerun the tests of section 2.5.

The player view is trained on pseudo-labels produced by the algorithm from the provided match.

The public five minutes of ground truth were not used for training or for searching parameters;
see section 7 for exactly how they were used.

## 5. The possession rules implemented

- **Controlled possession** is "enough evidence to pay the switching cost". With the ball in
  view the evidence comes from the learned ball view: at most one unit per frame, so a change of
  side needs at least a sixth of a second of it if that side then keeps the ball, a third of a
  second if it is given straight back (the cost is paid twice), and more when the classifier is
  unsure. The classifier sees the second before and after each frame, so a deflection or a ball
  flying past a player, with the same side on the ball before and after, is weak evidence and
  changes nothing.
- **Loose and contested balls** stay with the last side that had control (neutral evidence, plus
  the tie-break). A switch is placed at the first frame of the new side's control.
- **Dead ball** is one of: ball beyond the boundary; ball placed and motionless; the scene looks
  stopped to the player view (walking players, TV cutaway, set-piece context); the wait before
  the second-half kick-off. Dead-ball ownership is not modelled.
- **Restarts** take the side the ball view sees on the ball once it is played.

## 6. Results

Public five minutes (17 possessions — a pipeline check, not a ranking; intervals are the
scorer's 5–95% bootstrap):

| | macroF1 | spellF1 | possessions | mean duration |
|---|---|---|---|---|
| ground truth | | | 17 | 14.2 s |
| **two views, learned ball view (submission)** | **0.833** (0.61–0.89) | **0.533** (0.36–0.70) | 16 | 15.6 s |
| … with the learned evidence at full weight against dead ball | 0.790 (0.59–0.86) | 0.506 (0.34–0.67) | 17 | 15.9 s |
| two views, hand-written ball view (previous version) | 0.846 (0.61–0.90) | 0.452 (0.27–0.62) | 14 | 17.8 s |
| … without the goalkeeper rule | 0.841 | 0.440 | 13 | 19.3 s |
| … with team-change cost 12 | 0.843 | 0.471 | 11 | 22.8 s |
| player view only | 0.815 | 0.430 | 10 | 25.2 s |
| ball view only (learned) | 0.570 | 0.516 | 14 | 20.8 s |
| baseline_smooth125 | 0.581 | 0.424 | 16 | 18.5 s |
| baseline_smooth50 (published floor) | 0.621 | 0.326 | 26 | 10.8 s |
| baseline_raw | 0.597 | 0.038 | 131 | 2.0 s |

Per class on the public window: F1 H 0.85, A 0.87, D 0.77 (hand-written version: 0.86, 0.88, 0.79;
learned at full weight: 0.86, 0.84, 0.67); alive/dead F1 0.95; team accuracy where both agree the
ball is alive 0.91.

**This window did not choose between the hand-written and the learned ball view.** The learned
one scores higher on `spellF1` (0.533 against 0.452) and slightly lower on `macroF1` (0.833
against 0.846); both differences are a possession or a stoppage or two out of 17 and 4, far
inside the intervals. It did play a part in the weight against dead ball, and section 7 says
exactly which. The tests that are allowed to decide — nine other
matches, 22 match × scenario cells — favour the submitted version in 21 under one reading of the
truth and in all 22 under the other (section 2.5). The same held one step earlier: the
hand-written version with cost 4 scored lower here than with cost 12 (0.452 against 0.471) and
was kept, because the sample games favoured it in all eight cells.

Whole match, as predicted (no ground truth available): 334 possessions (163 home, 171 away), mean
11.1 s, median 6.7 s; 100 stoppages; ball in play 62.0 of 105.9 graded minutes; Barcelona 58% of
in-play time. (Learned at full weight against dead ball: 333 possessions, 99 stoppages, 63.5
minutes in play.) (Hand-written ball view: 271 possessions, mean 13.6 s, 97 stoppages, 61.4 minutes
in play. The public window's own rate, about four possessions per minute of live play, would put
the match near 260.)

## 7. Limitations, stated plainly

- **The public window cannot tell us the final score.** Its interval for `spellF1` is 0.27–0.67
  and it has the ball in view about 88% of the time, against 59% for the match. The published baseline
  drops from 0.33 there to 0.24 on the full match; expect ours to move too.
- **How the public ground truth was used.** To check frame alignment and file format, and to
  understand what the labels mean — in particular that long stoppages exist with the ball in view
  and no set-piece tag, which is what motivated pseudo-label rule 2. The pipeline was scored on
  it after each structural revision, thirteen times in all: seven up to the hand-written version,
  once to confirm that that version regenerated on a second machine scored the same, once for
  the learned ball view and once more when a second machine regenerated it, twice for the weight
  against dead ball (0.36 and 0.6) and once for the final file (the three-class and uncapped
  variants of section 3 were rejected on label-free counts and never scored here).
  **One decision was prompted by it, and we say so.** When the learned ball view was reviewed,
  its dead-ball share inside this window (10%, against 20% in the ground truth and 17% for the
  hand-written version) and its dead-ball F1 (0.67 against 0.79) were read, together with the
  label-free count that 156 s of previously dead time had become possession. That is what made
  us look for the cause. The remedy was then fixed without this window: the weight 0.36 is the
  measured ratio of the two readings' strengths (section 2.2), the only value tried on principle;
  0.6 was run once as a midpoint and not adopted. On the DFL matches the three weights are within
  0.006 of each other (1.0: 0.537, 0.6: 0.533, 0.36: 0.531; each beats the formula in 13 of 14
  cells), and that bench cannot see stoppages lost to a ball in view, because its dead ball is a
  motionless placeholder. So this choice rests on the mechanism and on label-free counts of the
  provided match, with the public window as the alarm, not as the judge. The per-class confusion of the hand-written version
  was read once: dead frames called live were its largest error, which is what sent us to
  measure stoppage edges on other matches. No constant was searched against it, and it never
  entered training. With 17 possessions it could not have supported a search anyway.
  One influence should be stated: when a setting chosen on the *clean* stand-in nearly doubled
  the possession count on this match while the public score did not move, that made us doubt the
  clean stand-in, measure its noise against ours with label-free statistics, and redo the test on
  a degraded one (sections 2.5 and 3). The setting was then rejected on the stand-in, not here.
- **What watching the video was used for.** A team member watched the match with the model's
  output overlaid (`review/`) and reported that goalkeeper situations looked wrong. We measured
  it in the data — the tracking does not mislabel the keepers, but when a keeper holds the ball
  and it is lost, the decoder used to follow team shapes — and wrote the general, position-based
  rule of section 2.2. No frame of this match was labelled, no rule or constant refers to a
  frame, and the review page cannot store a label for this match.
- **The learned ball view comes from another league, another tracker and an imitation of our
  noise.** The three-class failure in section 3 is a difference between those matches and ours
  that no test on them could show; we found it in the label-free counts of our own match. There
  may be others we did not find. The provider's possession flag is not documented in detail and
  changes hands on touches (median possession 3 s), where the challenge asks for controlled
  possession; training folds short spells, but the two definitions are not the same.
- **Dead ball may have got slightly worse.** The submitted version calls 40.0% of graded frames
  dead where the hand-written one called 42.0%, and has ten stoppages under three seconds where
  that one had three; on the public window the dead-ball F1 fell from 0.79 to 0.67 (four
  stoppages). The other matches cannot measure this, because their dead ball is a placeholder.
- **The stand-in is an imitation.** Its truth comes from event data, not from the organisers'
  annotation rules; its noise is a smooth wander fitted to two summary statistics of ours, and
  its nearest-team flicker is still higher than ours (about 50 flips a minute against 28). The
  +0.03 it measures is the best evidence we could build within the rules. It is not a promise.
- **Stoppage detection is untested off this match.** The stand-in games have no broadcast
  recording or tagging, so everything said about `D` rests on the public five minutes and on
  self-consistency.
- **An eighth of the possessions is still missed on other matches** (282 found of 323 per DFL
  match), and some are invented: on the public window we output 16 possessions for 17, of which
  10 match a true one at IoU 0.5.
- **Boundaries are soft.** Only 6 of 20 public transitions are within one second. The player view
  is built from features spanning several seconds, so it is slow at restarts (on the DFL matches
  a stoppage start placed without the ball is a median 1.4 s late); `spellF1` at IoU 0.6–0.7
  (0.36) is well below IoU 0.3–0.5 (0.61–0.73).
- **Stoppages without any cue are missed**: a quick throw-in with the ball lost, players still
  moving and the TV still live.
- **A goalkeeper carrying the ball before a goal kick** is read as possession by the ball view.
- **Dead-ball share is self-consistent, not verified.** The model calls 42% of graded frames
  dead, and 83% of the frames with no ball. That matches typical effective playing time, but
  nothing in the public window can confirm it.

With more time: a test bench that also has a broadcast feed and a real ball during stoppages, so
stoppage detection can be measured off this match and learned the way who-is-on-the-ball now is;
training matches tracked automatically rather than degraded by us; pixels around the look-at
point of the broadcast camera to recover the ball where both trackers lose it.

## 8. Reproduction

```bash
pip install -r solution/requirements.txt
export MTAH_DATA_DIR=/path/to/bundle        # contains tactical/ and broadcast/
python solution/run.py --name <team_name>
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
```

Python 3.12, numpy 1.26.4, pandas 2.3.3, scikit-learn 1.5.2. Fixed seed; no GPU; about four
minutes on a laptop. Two consecutive runs produce byte-identical files. The match data is not in
the repository and is never written into it.

`run.py` reads the trained ball classifier from `solution/models/ball_view.pkl` (a scikit-learn
pickle: it needs the scikit-learn version above). To rebuild it from the public data instead:

```bash
python solution/ballmodel_eval.py --train-final solution/models/ball_view.pkl   # downloads 2.6 GB on first use
```

That needs the provided bundle too, because the training matches are given the provided match's
ball-gap pattern. Two rebuilds on one machine give the same file byte for byte; we have not
checked that across operating systems, which is why the trained file is committed.

## 9. How this was built

The code and this document were written with an AI coding assistant (Claude, a cloud-hosted LLM)
directed by the team. The method contains no LLM: nothing in `run.py` calls a model other than
its two gradient-boosted classifiers (the stored ball classifier and the player view it trains).
