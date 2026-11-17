"""Evidence -> labels.

Two "experts" each give every frame a log-score for H, A and D:

  ball view    -- who is on the ball, is it out, is it placed, is it being played (prep.ball_view)
  player view  -- a classifier that never sees the ball: team shapes, movement, cameras, tagging
                  (playerview.py), trained on labels the ball view itself is sure about

and a sticky HMM (hmm.py) turns the summed scores into intervals.

Scales: each expert can contribute about one unit of evidence per frame at most, and changing
state costs SWITCH_COST units -- the "controlled possession" rule of the challenge, expressed as a
number. These were first set by reasoning. One of them, the cost of a change of team while the
ball is in view, was then chosen on other matches that have answers (standin_eval.py), never on
the provided match: 12 units there asks for half a second of undisputed control, 4 units for a
sixth of a second, and the lower figure finds more of the short possessions without inventing
more than it finds.
"""

import numpy as np

from . import data as D
from . import hmm

# Params
K_CONTROL = 1.0        # ball view: evidence per frame for the side firmly on the ball
EPS_LAST_TOUCH = 0.02  # tie-break: a loose ball stays with the side that last controlled it
MIN_TOUCH = 0.3        # control level that counts as "had it" for the tie-break
W_OUT = 1.5            # ball beyond the boundary -> dead
W_STILL = 1.0          # ball placed and motionless -> dead
W_MOVING = 0.3         # ball travelling inside the pitch -> not dead
W_KEEPER_HOLD = 1.0    # ball that vanished in a goalkeeper's hands -> his side (prep.keeper_hold): as much as
                       # firm control of a visible ball. Premise checked on the stand-in (standin_eval.py --keeper)
LEARNED_BALL_VIEW = True   # read who is on the ball with the classifier trained on other matches (ballmodel.py,
                           # solution/models/ball_view.pkl) instead of the hand-written formula below. On matches the
                           # classifier never saw it beats the formula in every game x scenario cell (ballmodel_eval.py)
K_LEARNED = 0.5        # evidence per unit of the learned ball view's log-odds; 0.25-1.0 is a plateau on other matches
LEARNED_DEAD_WEIGHT = 0.36   # how much of the learned evidence also counts against "dead ball". The classifier is
                             # far surer of WHO than the formula was (mean strength 0.78 against 0.28 on this match's
                             # frames, at its cap on 59% of them), so at full weight it took dead time away that
                             # the formula never contested. 0.36 is that ratio, measured without labels, not searched:
                             # home-against-away is untouched, alive-against-dead weighs what the formula's did
SWITCH_COST = 12.0     # evidence units to change state (any pair of states)
TEAM_COST_BALL_SEEN = 4.0    # ... for a change of team while the ball is in view. Chosen on the stand-in
                             # matches (standin_eval.py --sweep): better than 12 in 8 of 8 game x scenario
                             # cells under both readings of the truth; 3-6 is a plateau
SHARPEN = False              # move decoded boundaries onto ball events (refine.py). Off: no measurable
                             # gain on the stand-in matches, so it is not shipped
PLAYER_VIEW_WEIGHT = 0.25   # scales the classifier's log-probabilities to ~1 unit per frame
TEAM_DISCOUNT_WHEN_BALL_SEEN = 0.25   # the player view's say on H-vs-A where the ball is in view: a quarter.
                                      # Stand-in matches: 0 is best on clean tracking and worst once the ball
                                      # is as noisy as ours; 0.25 is best or near-best in every scenario
PROB_FLOOR = 0.01           # classifier probabilities are clipped here before the log
CONFIDENT = 0.95            # posterior needed for a frame to become a pseudo-label


def ball_emissions(bv):
    """(n, 3) log-scores from the ball track alone. All zeros where there is no ball."""
    n = len(bv["has"])
    e = np.zeros((n, 3))
    dead_cue = bv["out"] | bv["still"]
    if bv.get("learned") is not None:             # who is on the ball: the classifier trained on other matches
        net = np.where(dead_cue, 0.0, np.clip(K_LEARNED * bv["learned"], -1.0, 1.0))   # capped at the formula's one
        # unit per frame: the classifier says WHO, and must not weigh more against "dead" than the formula did
    else:                                         # ... or the hand-written formula
        net = np.where(dead_cue, 0.0, bv["control_home"] - bv["control_away"])
    e[:, hmm.H] = K_CONTROL * np.maximum(net, 0.0)
    e[:, hmm.A] = K_CONTROL * np.maximum(-net, 0.0)
    if bv.get("learned") is not None:            # both sides give back the same amount: H - A stays net,
        fade = (1.0 - LEARNED_DEAD_WEIGHT) * K_CONTROL * np.abs(net)   # the lead over D shrinks to the formula's scale
        e[:, hmm.H] -= fade
        e[:, hmm.A] -= fade

    # loose-ball rule: carry the last side that clearly had it
    touch = np.where(net > MIN_TOUCH, 1, np.where(net < -MIN_TOUCH, -1, 0))
    idx = np.where(touch != 0, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    last = touch[idx]
    e[:, hmm.H] += EPS_LAST_TOUCH * (last == 1)
    e[:, hmm.A] += EPS_LAST_TOUCH * (last == -1)

    if "keeper_hold" in bv:                       # the one piece of ball evidence on frames without a ball
        e[:, hmm.H] += W_KEEPER_HOLD * (bv["keeper_hold"] == 1)
        e[:, hmm.A] += W_KEEPER_HOLD * (bv["keeper_hold"] == -1)

    e[:, hmm.D] = W_OUT * bv["out"] + W_STILL * (bv["still"] & ~bv["out"]) - W_MOVING * bv["moving_in"]
    return e


def player_emissions(proba, ball_seen=None, weight=None, discount=None):
    """(n, 3) log-scores from the player-view classifier's probabilities (NaN rows -> zeros).

    The classifier answers two questions and they are weighted separately:

      is the ball in play?   D against H+A -- always at full weight. Walking players and a TV
                             cutaway say "stopped" whether or not the ball is in view.
      which side has it?     H against A -- at full weight only where the ball is NOT seen.
                             Where it is seen the ball view is the teacher these probabilities
                             were learnt from, and a student should not overrule its teacher on
                             the frames the teacher can actually see.
    """
    p = np.clip(proba, PROB_FLOOR, 1.0)
    alive = np.clip(p[:, 0] + p[:, 1], PROB_FLOOR, 1.0)
    base = PLAYER_VIEW_WEIGHT if weight is None else weight
    discount = TEAM_DISCOUNT_WHEN_BALL_SEEN if discount is None else discount
    weight = np.full(len(p), float(base))
    if ball_seen is not None:
        weight = np.where(ball_seen, base * discount, weight)
    e = np.zeros_like(p)
    # log p(H) = log p(alive) + log p(H | alive): the two terms carry the two questions
    e[:, hmm.H] = base * np.log(alive) + weight * np.log(p[:, 0] / alive)
    e[:, hmm.A] = base * np.log(alive) + weight * np.log(p[:, 1] / alive)
    e[:, hmm.D] = base * np.log(p[:, 2])
    return np.where(np.isnan(e), 0.0, e)


def decode(emissions, forced_dead=None, want_posteriors=False, team_cost=None, switch_cost=None):
    """Run the HMM inside each half. Returns codes (n,) with -1 outside the halves, and
    optionally posteriors (n, 3) with NaN outside.

    switch_cost overrides SWITCH_COST for all three pairs; team_cost (per frame) overrides the
    H<->A cost only.
    """
    cost = SWITCH_COST if switch_cost is None else switch_cost
    trans = hmm.transition_matrix(cost, cost, cost)
    codes = np.full(D.N_FRAMES, -1, dtype=np.int8)
    post = np.full((D.N_FRAMES, 3), np.nan) if want_posteriors else None
    for first, stop in D.HALVES:
        e = emissions[first:stop].copy()
        if forced_dead is not None:
            fd = forced_dead[first:stop]
            e[fd, hmm.H] -= 1e3
            e[fd, hmm.A] -= 1e3
        codes[first:stop] = hmm.viterbi(e, trans, None if team_cost is None else team_cost[first:stop])
        if want_posteriors:
            post[first:stop] = hmm.posteriors(e, trans)
    return (codes, post) if want_posteriors else codes


def to_labels(codes):
    labels = np.full(len(codes), "X", dtype="<U1")
    labels[codes == hmm.H] = "H"
    labels[codes == hmm.A] = "A"
    labels[codes == hmm.D] = "D"
    return labels
