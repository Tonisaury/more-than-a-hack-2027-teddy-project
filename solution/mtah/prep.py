"""Clean kinematics and the "ball view": what the ball track alone says about each frame.

All constants are physical (metres, seconds, metres per second) and were set from football
reasoning, not fitted to the public ground truth.
"""

import warnings

import numpy as np

from . import data as D

# Params
MAX_PLAYER_SPEED = 12.0      # m/s; a faster frame-to-frame jump is an identity switch, not a run
MAX_BALL_SPEED = 45.0        # m/s; faster is a false detection
INTERP_MAX_GAP = 25          # frames; ball gaps up to 1 s are bridged linearly
CONTROL_RADIUS = 2.5         # m; beyond this nobody is "on the ball"
FAST_BALL = 9.0              # m/s relative speed at which a nearby ball is passing by, not held
OUT_MARGIN = 0.4             # m beyond the line before the ball counts as out
STILL_WINDOW = 25            # frames; a ball that has not moved for this long is "placed"
STILL_SPREAD = 0.20          # m; positional spread below which the ball is not moving
MOVING_BALL = 4.0            # m/s; an in-field ball this fast is being played
KEEPER_BOX = (16.5, 20.16)   # m; the penalty area: depth from the goal line, half-width
KEEPER_GRIP = 0.3            # control level at which a goalkeeper is taken to have gathered the ball
KEEPER_LOOKBACK = 10         # frames before the ball vanishes in which he must have had that grip
KEEPER_HOLD_SECONDS = 6.0    # how long a ball that vanished in a keeper's hands is assumed to stay there
                             # (the Laws' limit on holding the ball; after it we assume nothing)


def attack_signs(tr):
    """+1 where the home team attacks towards +x, -1 where it attacks towards -x. Per frame.

    Read from the data: in each half the home side's mean x sits on the side of its own goal.
    """
    sign = np.ones(D.N_FRAMES)
    for first, stop in D.HALVES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            home = np.nanmean(tr["px"][first:stop][:, tr["team"] == D.HOME])
            away = np.nanmean(tr["px"][first:stop][:, tr["team"] == D.AWAY])
        sign[first:stop] = 1.0 if home < away else -1.0
    return sign


def velocities(x, y, step, max_speed):
    """Central-difference velocity over +/- step frames, NaN across gaps and jumps."""
    dt = 2.0 * step / D.FPS
    vx = np.full_like(x, np.nan)
    vy = np.full_like(y, np.nan)
    vx[step:-step] = (x[2 * step:] - x[:-2 * step]) / dt
    vy[step:-step] = (y[2 * step:] - y[:-2 * step]) / dt
    with np.errstate(invalid="ignore"):
        bad = np.hypot(vx, vy) > max_speed
    vx[bad] = np.nan
    vy[bad] = np.nan
    return vx, vy


def clean_ball(bx, by):
    """Drop physically impossible ball jumps (isolated false detections)."""
    bx, by = bx.copy(), by.copy()
    step = np.hypot(np.diff(bx), np.diff(by)) * D.FPS          # NaN where either side is missing
    with np.errstate(invalid="ignore"):
        jump = step > MAX_BALL_SPEED
    # a detection that jumps in AND out is a spike; remove it
    spike = np.zeros(len(bx), dtype=bool)
    spike[1:-1] = jump[:-1] & jump[1:]
    bx[spike] = np.nan
    by[spike] = np.nan
    return bx, by


def interpolate_short_gaps(bx, by, max_gap):
    """Linear bridge over gaps of at most max_gap frames. Returns (bx, by, observed_mask)."""
    observed = ~np.isnan(bx)
    idx = np.flatnonzero(observed)
    out_x, out_y = bx.copy(), by.copy()
    if len(idx) < 2:
        return out_x, out_y, observed
    gaps = np.flatnonzero(np.diff(idx) > 1)
    for g in gaps:
        a, b = idx[g], idx[g + 1]
        if b - a - 1 <= max_gap:
            w = (np.arange(a + 1, b) - a) / float(b - a)
            out_x[a + 1:b] = bx[a] + w * (bx[b] - bx[a])
            out_y[a + 1:b] = by[a] + w * (by[b] - by[a])
    return out_x, out_y, observed


def rolling_spread(x, y, window):
    """Centred rolling positional std (m) of the ball; NaN unless the whole window is present."""
    half = window // 2
    ok = ~np.isnan(x)
    x0, y0 = np.where(ok, x, 0.0), np.where(ok, y, 0.0)

    def roll(v):
        c = np.concatenate(([0.0], np.cumsum(v)))
        lo = np.maximum(0, np.arange(len(v)) - half)
        hi = np.minimum(len(v), np.arange(len(v)) + half + 1)
        return c[hi] - c[lo], hi - lo

    n, size = roll(ok.astype(float))
    sx, _ = roll(x0)
    sy, _ = roll(y0)
    sxx, _ = roll(x0 * x0)
    syy, _ = roll(y0 * y0)
    full = n == size
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (sxx / n - (sx / n) ** 2) + (syy / n - (sy / n) ** 2)
    return np.where(full, np.sqrt(np.maximum(var, 0.0)), np.nan)


def keeper_hold(has, keeper, control, out):
    """+1 / -1 on the frames where the ball has just vanished in the home / away goalkeeper's hands.

    A keeper who catches or smothers the ball hides it from the tracker. Without this the decoder is left
    with the player view, which reads team shapes -- and at that moment the shapes still say the *other*
    side is attacking. So: if in the last frames before the ball disappears the goalkeeper (the player of
    a side nearest its own goal, inside its penalty area) had a grip on it, no opponent had a firmer one
    and the ball was not out, the ball is taken to be his for up to KEEPER_HOLD_SECONDS or until it is
    seen again, whichever comes first.
    """
    hold = np.zeros(len(has), dtype=np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([1], has.astype(np.int8), [1]))))
    for a, b in zip(edges[::2], edges[1::2]):                # [a, b) is a stretch with no ball
        lo = max(0, a - KEEPER_LOOKBACK)
        if a == 0 or out[lo:a].any():
            continue
        for side, value in ((D.HOME, 1), (D.AWAY, -1)):
            best = keeper[side, lo:a].max()
            if best >= KEEPER_GRIP and control[1 - side, lo:a].max() < best:
                hold[a:min(b, a + int(KEEPER_HOLD_SECONDS * D.FPS))] = value
    return hold


def ball_view(tr, bx, by):
    """Everything the ball track says, per frame.

    control_home / control_away  in [0, 1]: how firmly the nearest player of that side is on the
                                 ball. Close and moving with it -> 1; far, or the ball flying
                                 past -> 0.
    out                          the ball is beyond the pitch boundary
    still                        the ball has been motionless for a second (a placed ball)
    moving_in                    the ball is inside the pitch and travelling at playing speed
    """
    has = ~np.isnan(bx)
    pvx, pvy = velocities(tr["px"], tr["py"], 2, MAX_PLAYER_SPEED)
    bvx, bvy = velocities(bx, by, 2, MAX_BALL_SPEED)
    ball_speed = np.hypot(bvx, bvy)

    control = np.zeros((2, len(bx)))
    nearest = np.full((2, len(bx)), np.nan)
    keeper = np.zeros((2, len(bx)))             # grip of each side's goalkeeper, found by position, not by name
    sign = attack_signs(tr)
    idx = np.flatnonzero(has)
    for lo in range(0, len(idx), 20000):
        sel = idx[lo:lo + 20000]
        dx = tr["px"][sel] - bx[sel, None]
        dy = tr["py"][sel] - by[sel, None]
        dist = np.hypot(dx, dy)
        rel = np.hypot(pvx[sel] - bvx[sel, None], pvy[sel] - bvy[sel, None])
        rel = np.where(np.isnan(rel), FAST_BALL / 2.0, rel)       # unknown -> neutral
        grip = np.clip(1.0 - dist / CONTROL_RADIUS, 0.0, 1.0) * (0.25 + 0.75 * np.exp(-(rel / FAST_BALL) ** 2))
        grip = np.where(np.isnan(grip), 0.0, grip)
        for side in (D.HOME, D.AWAY):
            cols = tr["team"] == side
            control[side, sel] = grip[:, cols].max(axis=1)
            d_side = np.where(np.isnan(dist[:, cols]), np.inf, dist[:, cols]).min(axis=1)
            nearest[side, sel] = d_side
            # the goalkeeper is whoever of this side stands nearest its own goal, if that is inside its penalty area
            own_goal = np.where((sign[sel] > 0) == (side == D.HOME), 0.0, D.PITCH_LENGTH)
            depth = np.abs(tr["px"][sel][:, cols] - own_goal[:, None])
            wide = np.abs(tr["py"][sel][:, cols] - D.PITCH_WIDTH / 2.0)
            to_goal = np.where(np.isnan(depth), np.inf, np.hypot(depth, wide))
            who = to_goal.argmin(axis=1)
            rows = np.arange(len(sel))
            with np.errstate(invalid="ignore"):
                in_box = (depth[rows, who] < KEEPER_BOX[0]) & (wide[rows, who] < KEEPER_BOX[1])
            keeper[side, sel] = np.where(in_box, grip[:, cols][rows, who], 0.0)

    with np.errstate(invalid="ignore"):
        out = has & ((bx < -OUT_MARGIN) | (bx > D.PITCH_LENGTH + OUT_MARGIN)
                     | (by < -OUT_MARGIN) | (by > D.PITCH_WIDTH + OUT_MARGIN))
        spread = rolling_spread(bx, by, STILL_WINDOW)
        still = has & (spread < STILL_SPREAD)
        moving_in = has & ~out & (ball_speed > MOVING_BALL)
    return {
        "keeper_hold": keeper_hold(has, keeper, control, out),
        "has": has, "control_home": control[D.HOME], "control_away": control[D.AWAY],
        "nearest_home": nearest[D.HOME], "nearest_away": nearest[D.AWAY],
        "out": out, "still": still, "moving_in": moving_in, "ball_speed": ball_speed,
        "pvx": pvx, "pvy": pvy,
    }
