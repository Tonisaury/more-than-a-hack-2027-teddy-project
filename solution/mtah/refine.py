"""Sharpen the edges of decoded possessions without adding or removing any.

The decoder is good at saying *that* possession changed and soft about *when*: the player view is
built from features that span seconds. The ranking metric pays for exact edges (a perfect labelling
that is one second late scores 0.65), so after decoding every boundary is moved to the nearest
physical event the ball track shows, inside a small window. Nothing is created or deleted, and a
boundary never moves past its neighbours, so the possession count cannot change.

  team change, ball seen   -> the first frame from which the new side is on the ball and the old side
                              does not touch it again
  dead -> live, ball seen  -> the kick: a placed or slow ball that leaves at playing speed
  live -> dead, ball seen  -> the frame the ball crosses the boundary line

Where the ball is not seen around a boundary, the boundary stays where the decoder put it.
"""

import numpy as np

from . import data as D
from . import hmm

# Params (physical units; chosen on the stand-in matches, see standin_eval.py)
BACK = 2.0            # seconds a boundary may move earlier
FORWARD = 1.0         # seconds a boundary may move later
TOUCH = 0.3           # control level (0-1) that counts as being on the ball
KICK_SPEED = 6.0      # m/s the ball reaches after a restart kick
REST_SPEED = 2.0      # m/s below which the ball was waiting to be played
KICK_WITHIN = 0.4     # seconds from rest to kick speed


def _team_switch(t, lo, hi, net, new_sign):
    """First frame in [lo, hi) from which the new side touches and the old side never does again."""
    signed = new_sign * net[lo:hi]
    new_touch = np.flatnonzero(signed > TOUCH)
    if not len(new_touch):
        return t
    old_touch = np.flatnonzero(signed < -TOUCH)
    last_old = old_touch[-1] if len(old_touch) else -1
    after = new_touch[new_touch > last_old]
    return int(lo + after[0]) if len(after) else t


def _kick(t, lo, hi, speed, has):
    """Frame nearest to t in [lo, hi) where a resting ball reaches kick speed within KICK_WITHIN."""
    step = max(1, int(KICK_WITHIN * D.FPS))
    best, best_gap = t, None
    for f in range(max(lo, step), hi):
        if has[f] and has[f - step] and speed[f] > KICK_SPEED and speed[f - step] < REST_SPEED:
            gap = abs(f - step - t)
            if best_gap is None or gap < best_gap:
                best, best_gap = f - step, gap
    return best


def _goes_out(t, lo, hi, out):
    idx = np.flatnonzero(out[lo:hi])
    if not len(idx):
        return t
    first = lo + idx[0]
    return int(first) if not (first > lo and out[first - 1]) else t


def refine(codes, bv, team=True, restarts=True, exits=True):
    """A copy of codes with each boundary snapped to a ball event, per half."""
    out = codes.copy()
    net = np.nan_to_num(bv["control_home"] - bv["control_away"])
    speed = np.nan_to_num(bv["ball_speed"], nan=0.0)
    back, forward = int(BACK * D.FPS), int(FORWARD * D.FPS)
    for first, stop in D.HALVES:
        window = codes[first:stop]
        cuts = (np.flatnonzero(window[1:] != window[:-1]) + 1 + first).tolist()
        bounds = [first] + cuts + [stop]
        moved = list(cuts)
        for i, t in enumerate(cuts):
            a, b = int(codes[t - 1]), int(codes[t])
            lo = max(bounds[i] + 1, t - back)                 # never reach the previous boundary
            hi = min(bounds[i + 2] - 1, t + forward)          # ... or the next one
            if i > 0:
                lo = max(lo, moved[i - 1] + 1)
            if hi <= lo:
                continue
            if a != hmm.D and b != hmm.D and team:
                moved[i] = _team_switch(t, lo, hi, net, 1.0 if b == hmm.H else -1.0)
            elif a == hmm.D and restarts:
                moved[i] = _kick(t, lo, hi, speed, bv["has"])
            elif b == hmm.D and exits:
                moved[i] = _goes_out(t, lo, hi, bv["out"])
        for i, (t, new) in enumerate(zip(cuts, moved)):
            if new < t:
                out[new:t] = codes[t]
            elif new > t:
                out[t:new] = codes[t - 1]
    return out
