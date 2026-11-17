"""A three-state sticky hidden Markov model over H / A / D, in log space.

The emissions are per-frame log-scores built elsewhere; this file only knows how to turn them
into the single best label sequence (Viterbi) or per-frame posteriors (forward-backward).

"Sticky" is the whole point. The ranking metric counts possessions, and a possession is an
interval: a state may only change when the accumulated evidence for the new state exceeds a fixed
switching cost. That is how "a simple touch is not a change of possession" is implemented --
a touch is a few frames of evidence, and a few frames do not pay for two switches.
"""

import numpy as np

H, A, D = 0, 1, 2
N_STATES = 3


def transition_matrix(cost_team_switch, cost_to_dead, cost_from_dead):
    """Log-score added when moving from row state to column state. Staying is free."""
    t = np.zeros((N_STATES, N_STATES))
    t[H, A] = t[A, H] = -cost_team_switch
    t[H, D] = t[A, D] = -cost_to_dead
    t[D, H] = t[D, A] = -cost_from_dead
    return t


def viterbi(emissions, trans, team_cost=None):
    """Most likely state sequence. emissions is (n, 3) log-scores.

    team_cost, if given, is the cost of an H<->A change *into* each frame (length n); it replaces
    the matrix's constant. That lets a change of possession be cheaper where the ball is in view
    than where the decoder is guessing in the dark.
    """
    n = len(emissions)
    back = np.zeros((n, N_STATES), dtype=np.int8)
    score = emissions[0].copy()
    t00, t10, t20 = trans[0, 0], trans[1, 0], trans[2, 0]
    t01, t11, t21 = trans[0, 1], trans[1, 1], trans[2, 1]
    t02, t12, t22 = trans[0, 2], trans[1, 2], trans[2, 2]
    s0, s1, s2 = float(score[0]), float(score[1]), float(score[2])
    em = emissions.tolist()
    tc = None if team_cost is None else (-np.asarray(team_cost, dtype=float)).tolist()
    back_list = []
    for i in range(1, n):
        e = em[i]
        if tc is not None:
            t10 = t01 = tc[i]
        # ties resolve to "stay", so a switch never happens earlier than the evidence demands
        a0, b0 = s0 + t00, 0
        if s1 + t10 > a0: a0, b0 = s1 + t10, 1
        if s2 + t20 > a0: a0, b0 = s2 + t20, 2
        a1, b1 = s1 + t11, 1
        if s0 + t01 > a1: a1, b1 = s0 + t01, 0
        if s2 + t21 > a1: a1, b1 = s2 + t21, 2
        a2, b2 = s2 + t22, 2
        if s0 + t02 > a2: a2, b2 = s0 + t02, 0
        if s1 + t12 > a2: a2, b2 = s1 + t12, 1
        s0, s1, s2 = a0 + e[0], a1 + e[1], a2 + e[2]
        back_list.append((b0, b1, b2))
    path = np.zeros(n, dtype=np.int8)
    state = int(np.argmax([s0, s1, s2]))
    path[n - 1] = state
    for i in range(n - 2, -1, -1):
        state = back_list[i][state]
        path[i] = state
    return path


def _logsumexp_rows(m):
    top = m.max(axis=1, keepdims=True)
    return (top + np.log(np.exp(m - top).sum(axis=1, keepdims=True)))[:, 0]


def posteriors(emissions, trans):
    """Per-frame state probabilities by forward-backward. (n, 3), rows sum to 1."""
    n = len(emissions)
    fwd = np.zeros((n, N_STATES))
    bwd = np.zeros((n, N_STATES))
    fwd[0] = emissions[0]
    for i in range(1, n):
        fwd[i] = _logsumexp_rows((fwd[i - 1][:, None] + trans).T) + emissions[i]
    for i in range(n - 2, -1, -1):
        bwd[i] = _logsumexp_rows(trans + (bwd[i + 1] + emissions[i + 1])[None, :])
    log_post = fwd + bwd
    log_post -= log_post.max(axis=1, keepdims=True)
    post = np.exp(log_post)
    return post / post.sum(axis=1, keepdims=True)
