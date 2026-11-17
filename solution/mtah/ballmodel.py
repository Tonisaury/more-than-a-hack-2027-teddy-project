"""A learned ball view: who has the ball, from the ball and the players around it.

prep.ball_view turns the ball track into evidence with a formula written by hand: the nearest
player within 2.5 m is "on the ball", damped when the ball is flying past him. That formula looks
at one frame at a time and never asks what a touch looks like -- the ball changing speed or
direction next to a player, or one side being near it for the last second and the other for the
next.

This module learns that from OTHER matches that have the answer on every frame (the DFL open
dataset, mtah/dfl.py), with their ball degraded to look like an automatic tracker's
(dfl_eval.degraded_ball). The classifier sees only label-free geometry around the ball, built
from prep.ball_view's own outputs, and returns how much likelier home is than away to have it, for every frame
that has a ball and is in play. Nothing about the challenge match is used to train it.

Home and away are interchangeable to the classifier: every training frame is also given with the
two sides swapped, so it cannot learn a preference for either.
"""

import warnings

import numpy as np

from . import data as D
from . import prep

# Params
WINDOWS = (12, 25, 75)        # frames of past and of future summarised next to each frame (0.5 s, 1 s, 3 s)
FAR = 30.0                    # m; nearest-player distances are capped here
TRAIN_STRIDE = 3              # every 3rd frame with a ball; neighbours are near-duplicates
PROB_FLOOR = 0.01             # probabilities are clipped here before the log


def _past_future(values, w, how):
    """Summary of the w frames before and the w frames after each frame (the frame itself excluded)."""
    n = len(values)
    padded = np.concatenate((np.full(w, np.nan), values.astype(np.float64), np.full(w + 1, np.nan)))
    view = np.lib.stride_tricks.sliding_window_view(padded, w)        # view[i] = values[i - w : i]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = getattr(np, how)(view, axis=1)
    return summary[:n], summary[w + 1:w + 1 + n]


def features(bv, bx, by):
    """(n, k) float32 features and their names. Rows without a ball are NaN and are never scored.

    Columns come in home/away pairs so that `swap_sides` can exchange them.
    """
    has = bv["has"]
    gone = ~has
    c = [np.where(gone, np.nan, bv["control_home"]), np.where(gone, np.nan, bv["control_away"])]
    d = [np.minimum(bv["nearest_home"], FAR), np.minimum(bv["nearest_away"], FAR)]
    bvx, bvy = prep.velocities(bx, by, 2, prep.MAX_BALL_SPEED)
    step = 3
    accel = np.full(len(bx), np.nan)
    turn = np.full(len(bx), np.nan)
    ax, ay = bvx[2 * step:] - bvx[:-2 * step], bvy[2 * step:] - bvy[:-2 * step]
    accel[step:-step] = np.hypot(ax, ay) * D.FPS / (2.0 * step)
    with np.errstate(invalid="ignore", divide="ignore"):
        dot = bvx[2 * step:] * bvx[:-2 * step] + bvy[2 * step:] * bvy[:-2 * step]
        norm = np.hypot(bvx[2 * step:], bvy[2 * step:]) * np.hypot(bvx[:-2 * step], bvy[:-2 * step])
        turn[step:-step] = np.where(norm > 1.0, dot / norm, np.nan)
        edge = np.minimum(np.minimum(bx, D.PITCH_LENGTH - bx), np.minimum(by, D.PITCH_WIDTH - by))

    f = {"control_h": c[0], "control_a": c[1], "near_h": d[0], "near_a": d[1],
         "speed": bv["ball_speed"], "accel": accel, "turn": turn, "edge": edge,
         "out": np.where(gone, np.nan, bv["out"]), "still": np.where(gone, np.nan, bv["still"]),
         "moving": np.where(gone, np.nan, bv["moving_in"])}
    seen = has.astype(np.float64)
    for w in WINDOWS:
        for side, tag in ((0, "h"), (1, "a")):
            past, future = _past_future(c[side], w, "nanmean")
            f["control_{}_past{}".format(tag, w)], f["control_{}_next{}".format(tag, w)] = past, future
            past, future = _past_future(d[side], w, "nanmin")
            f["near_{}_past{}".format(tag, w)], f["near_{}_next{}".format(tag, w)] = past, future
        past, future = _past_future(seen, w, "nanmean")
        f["seen_past{}".format(w)], f["seen_next{}".format(w)] = past, future
        past, future = _past_future(accel, w, "nanmax")
        f["accel_past{}".format(w)], f["accel_next{}".format(w)] = past, future
    names = sorted(f)
    X = np.column_stack([f[k] for k in names]).astype(np.float32)
    X[gone] = np.nan
    return X, names


def swap_sides(X, names):
    """The same frames seen with home and away exchanged."""
    order = []
    for name in names:
        if name.startswith(("control_h", "near_h")):
            other = name.replace("_h", "_a", 1)
        elif name.startswith(("control_a", "near_a")):
            other = name.replace("_a", "_h", 1)
        else:
            other = name
        order.append(names.index(other))
    return X[:, order]


def training_rows(X, names, truth, has, halves):
    """Rows and labels for training: frames with a ball while it is in play, each also with sides swapped.

    Dead-ball frames are left out on purpose. In the DFL matches a dead ball is a frozen placeholder, which
    looks nothing like a real ball being carried to a corner flag; a classifier taught "dead" from it calls
    every moving ball alive. So this model answers one question only -- which side has it -- and whether
    the ball is in play stays with the cues that do not depend on it (model.ball_emissions, the player view).
    """
    inside = np.zeros(len(truth), dtype=bool)
    for first, stop in halves:
        inside[first:stop] = True
    rows = np.flatnonzero(has & inside & (truth <= 1))[::TRAIN_STRIDE]
    y = truth[rows].astype(np.int8)
    return np.concatenate((X[rows], swap_sides(X[rows], names))), np.concatenate((y, 1 - y))


def fit(X, y, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=100,
                                         l2_regularization=1.0, early_stopping=False, random_state=seed)
    clf.fit(X, y)
    return clf


def log_odds(clf, X, names, has):
    """(n,) log of P(home has it) / P(away has it); zero where there is no ball. Averaged with the side-swapped
    view, so the answer for home is exactly the mirror of the answer for away."""
    out = np.zeros(len(X))
    rows = np.flatnonzero(has)
    if not len(rows):
        return out
    home = list(clf.classes_).index(0)
    p = clf.predict_proba(X[rows])[:, home]
    q = clf.predict_proba(swap_sides(X[rows], names))[:, home]
    p = np.clip(0.5 * (p + 1.0 - q), PROB_FLOOR, 1.0 - PROB_FLOOR)
    out[rows] = np.log(p / (1.0 - p))
    return out
