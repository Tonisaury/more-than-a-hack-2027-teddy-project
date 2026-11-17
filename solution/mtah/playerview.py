"""The "player view": guess H / A / D for a frame WITHOUT looking at the ball.

The ball is missing on roughly four graded frames in ten, in stretches that last from seconds to
minutes, so something other than the ball has to carry those stretches. This module describes
every frame by things that are always available --

  * the shape and movement of both teams (where the blocks are, how spread, which way they run)
  * what the two cameras are doing (where they point, how far they are zoomed, and whether the TV
    director has cut away from the live wide shot)
  * the automatic Smart Tagging (phases with a team, set pieces)

-- and trains a gradient-boosted classifier to predict the label from them.

No human label is involved. The training targets are pseudo-labels: frames on which the ball view
(prep.py + hmm.py) is already confident, plus frames around tagged set-piece restarts. The model
learns "what does the pitch look like when Barcelona clearly has the ball / when the ball is
clearly dead", and is then asked the same question on frames where the ball cannot be seen.
Predictions are cross-fitted over one-minute blocks, so no frame is scored by a model that was
trained on its own neighbourhood.
"""

import warnings

import numpy as np

from . import data as D

# Params
SLOW = 1.0                  # m/s; below this a player is walking or standing
NEAR_CAMERA = 8.0           # m; "close to where the TV camera is pointing"
SHIFTS = (-125, -50, 50, 125)        # frames; temporal context for a core set of features
BLOCK = 1500                # frames per cross-fitting block (60 s)
N_FOLDS = 5
TRAIN_STRIDE = 3            # use every 3rd labelled frame; neighbours are near-duplicates
SET_PIECE_DEAD = (-25, 50)  # frames around a tagged set-piece start that are taken as dead
CAP_SECONDS = 120.0
WALKING_QUANTILE = 0.40     # "everybody is walking": mean speed in the slowest 40% of the match
TV_MOSTLY_AWAY = 0.50       # share of the last/next 6 s the TV feed spent on the live wide shot
TV_AWAY = 0.25

ATTACK_CODES = ("BUILD UP", "PROGRESSION", "ATTACKING TRANSITION", "CREATING CHANCES",
                "LONG BALL", "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX")
DEFEND_CODES = ("DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD",
                "DEFENDING IN ATTACKING THIRD", "DEFENSIVE TRANSITION")
SET_PIECE_TYPES = ("throw-in", "corner kick", "goal kick", "free kick",
                   "kick off (after goal)", "kick off (start)")


def _nan(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return func(*args, **kwargs)


def _row_quantiles(values, qs):
    """Row-wise quantiles ignoring NaN (linear interpolation), far faster than nanpercentile."""
    ordered = np.sort(values, axis=1)                    # NaN sorts last
    count = (~np.isnan(values)).sum(1)
    rows = np.arange(len(values))
    out = []
    for q in qs:
        pos = q / 100.0 * np.maximum(count - 1, 0)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, np.maximum(count - 1, 0))
        frac = pos - lo
        v = ordered[rows, lo] * (1.0 - frac) + ordered[rows, hi] * frac
        out.append(np.where(count > 0, v, np.nan))
    return out


def _team_block(x, y, vx, vy, speed, prefix, feats):
    """Shape and movement of one team. x, y are already in 'home attacks +x' coordinates."""
    feats[prefix + "n"] = (~np.isnan(x)).sum(1).astype(float)
    feats[prefix + "cx"] = _nan(np.nanmean, x, axis=1)
    feats[prefix + "cy"] = _nan(np.nanmean, y, axis=1)
    feats[prefix + "sx"] = _nan(np.nanstd, x, axis=1)
    feats[prefix + "sy"] = _nan(np.nanstd, y, axis=1)
    for q, v in zip((0, 10, 30, 50, 70, 90, 100), _row_quantiles(x, (0, 10, 30, 50, 70, 90, 100))):
        feats["{}x_q{}".format(prefix, q)] = v
    for q, v in zip((10, 50, 90), _row_quantiles(y, (10, 50, 90))):
        feats["{}y_q{}".format(prefix, q)] = v
    feats[prefix + "vx"] = _nan(np.nanmean, vx, axis=1)
    feats[prefix + "vy"] = _nan(np.nanmean, vy, axis=1)
    feats[prefix + "speed"] = _nan(np.nanmean, speed, axis=1)
    feats[prefix + "speed_q80"] = _row_quantiles(speed, (80,))[0]
    feats[prefix + "slow"] = _nan(np.nanmean, np.where(np.isnan(speed), np.nan, speed < SLOW), axis=1)
    cx, cy = feats[prefix + "cx"][:, None], feats[prefix + "cy"][:, None]
    feats[prefix + "compact"] = _nan(np.nanmean, np.hypot(x - cx, y - cy), axis=1)


def _look_at(matrices, valid, centre):
    """Pitch point (metres) at the image centre and metres spanned by 200 px there."""
    n = len(matrices)
    lx = np.full(n, np.nan)
    ly = np.full(n, np.nan)
    zoom = np.full(n, np.nan)
    idx = np.flatnonzero(valid)
    if not len(idx):
        return lx, ly, zoom
    with np.errstate(all="ignore"):
        try:
            inv = np.linalg.inv(matrices[idx])
        except np.linalg.LinAlgError:
            inv = np.stack([np.linalg.pinv(m) for m in matrices[idx]])

        def unproject(px, py):
            v = inv @ np.array([px, py, 1.0])
            return v[:, 0] / v[:, 2] * D.PITCH_LENGTH, v[:, 1] / v[:, 2] * D.PITCH_WIDTH

        cx, cy = unproject(centre[0], centre[1])
        ax, ay = unproject(centre[0] - 100.0, centre[1])
        bx, by = unproject(centre[0] + 100.0, centre[1])
    lx[idx], ly[idx] = cx, cy
    zoom[idx] = np.hypot(bx - ax, by - ay)
    # a look-at point far outside the stadium is a degenerate matrix, not a camera direction
    bad = (np.abs(lx - D.PITCH_LENGTH / 2) > 150) | (np.abs(ly - D.PITCH_WIDTH / 2) > 150) | (zoom > 300)
    lx[bad] = ly[bad] = zoom[bad] = np.nan
    return lx, ly, zoom


def _image_centre(matrices, valid, px, py):
    """Guess the image size from where tracked players land in it; return its centre."""
    idx = np.flatnonzero(valid)[::50]
    xs = []
    for i in idx:
        m = matrices[i]
        x, y = px[i] / D.PITCH_LENGTH, py[i] / D.PITCH_WIDTH
        ok = ~np.isnan(x)
        if not ok.any():
            continue
        w = m[2, 0] * x[ok] + m[2, 1] * y[ok] + m[2, 2]
        xs.append((m[0, 0] * x[ok] + m[0, 1] * y[ok] + m[0, 2]) / w)
    top = np.percentile(np.concatenate(xs), 99.5) if xs else 1920.0
    width = min((1280.0, 1920.0, 3840.0), key=lambda w: abs(w - top))
    return (width / 2.0, width * 9.0 / 32.0)


def broadcast_on_tactical(mapping, bro, bro_h, bro_valid):
    """Per tactical frame: is the TV feed on the live wide shot, and where is it pointing?"""
    n_b = len(mapping["target"])
    trusted = mapping["trusted"]
    offset = (mapping["target"] - np.arange(n_b)).astype(float)
    idx = np.where(trusted, np.arange(n_b), -1)
    prev = np.maximum.accumulate(idx)
    nxt = np.minimum.accumulate(np.where(trusted, np.arange(n_b), n_b)[::-1])[::-1]
    use_prev = (prev >= 0) & ((nxt >= n_b) | (np.arange(n_b) - prev <= nxt - np.arange(n_b)))
    src = np.where(use_prev, prev, np.minimum(nxt, n_b - 1))
    nominal = (np.arange(n_b) + offset[src]).astype(int)
    ok = (nominal >= 0) & (nominal < D.N_FRAMES) & (trusted[src])

    centre = _image_centre(bro_h, bro_valid, bro["px"], bro["py"])
    lx, ly, zoom = _look_at(bro_h, bro_valid, centre)
    out = {k: np.full(D.N_FRAMES, np.nan) for k in ("bc_live", "bc_valid", "bc_n", "bc_lx", "bc_ly", "bc_zoom", "bc_cost")}
    t = nominal[ok]
    out["bc_live"][t] = trusted[ok].astype(float)
    out["bc_valid"][t] = bro_valid[ok].astype(float)
    out["bc_n"][t] = (~np.isnan(bro["px"][ok])).sum(1)
    out["bc_cost"][t] = np.minimum(mapping["cost"][ok], 8.0)
    live = ok & trusted
    t = nominal[live]
    out["bc_lx"][t] = lx[live] + mapping["shift"][live, 0]
    out["bc_ly"][t] = ly[live] + mapping["shift"][live, 1]
    out["bc_zoom"][t] = zoom[live]
    return out


def _rolling_mean(v, half):
    ok = ~np.isnan(v)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, v, 0.0))))
    k = np.concatenate(([0.0], np.cumsum(ok.astype(float))))
    lo = np.maximum(0, np.arange(len(v)) - half)
    hi = np.minimum(len(v), np.arange(len(v)) + half + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (c[hi] - c[lo]) / (k[hi] - k[lo])


def build_features(tr, sign, pvx, pvy, tags=(), tac_h=None, tac_valid=None, bc=None):
    """(matrix, names). One row per tactical frame, no ball information anywhere.

    Cameras (tac_h, bc) and tagging are optional: the stand-in matches have neither, and then
    only the team-shape features are built.
    """
    flip = sign < 0
    x = np.where(flip[:, None], D.PITCH_LENGTH - tr["px"], tr["px"])
    y = np.where(flip[:, None], D.PITCH_WIDTH - tr["py"], tr["py"])
    vx = pvx * sign[:, None]
    vy = pvy * sign[:, None]
    speed = np.hypot(pvx, pvy)

    f = {}
    home, away = tr["team"] == D.HOME, tr["team"] == D.AWAY
    _team_block(x[:, home], y[:, home], vx[:, home], vy[:, home], speed[:, home], "h_", f)
    _team_block(x[:, away], y[:, away], vx[:, away], vy[:, away], speed[:, away], "a_", f)
    f["d_cx"] = f["h_cx"] - f["a_cx"]
    f["d_cy"] = f["h_cy"] - f["a_cy"]
    f["d_vx"] = f["h_vx"] - f["a_vx"]
    f["d_sx"] = f["h_sx"] - f["a_sx"]
    f["d_sy"] = f["h_sy"] - f["a_sy"]
    f["d_compact"] = f["h_compact"] - f["a_compact"]
    f["d_front"] = f["h_x_q100"] - (D.PITCH_LENGTH - f["a_x_q0"])
    f["all_speed"] = _nan(np.nanmean, speed, axis=1)
    f["all_slow"] = _nan(np.nanmean, np.where(np.isnan(speed), np.nan, speed < SLOW), axis=1)
    f["all_n"] = (~np.isnan(x)).sum(1).astype(float)
    f["all_vx"] = _nan(np.nanmean, vx, axis=1)
    f["all_speed_2s"] = _rolling_mean(f["all_speed"], 50)
    f["all_slow_2s"] = _rolling_mean(f["all_slow"], 50)
    f["all_speed_6s"] = _rolling_mean(f["all_speed"], 150)

    n = D.N_FRAMES
    if tac_h is None or bc is None:
        return _finish(f, n)

    # cameras
    lx, ly, zoom = _look_at(tac_h, tac_valid, (960.0, 540.0))
    f["tac_lx"] = np.where(flip, D.PITCH_LENGTH - lx, lx)
    f["tac_ly"] = np.where(flip, D.PITCH_WIDTH - ly, ly)
    f["tac_zoom"] = zoom
    f["tac_pan"] = np.concatenate((np.zeros(25), f["tac_lx"][25:] - f["tac_lx"][:-25]))
    for key in ("bc_live", "bc_valid", "bc_n", "bc_zoom", "bc_cost"):
        f[key] = bc[key]
    f["bc_live"] = np.where(np.isnan(f["bc_live"]), 0.0, f["bc_live"])
    f["bc_live_2s"] = _rolling_mean(f["bc_live"], 50)
    f["bc_live_6s"] = _rolling_mean(f["bc_live"], 150)
    blx = np.where(flip, D.PITCH_LENGTH - bc["bc_lx"], bc["bc_lx"])
    bly = np.where(flip, D.PITCH_WIDTH - bc["bc_ly"], bc["bc_ly"])
    f["bc_lx"], f["bc_ly"] = blx, bly
    f["bc_pan"] = np.concatenate((np.zeros(25), blx[25:] - blx[:-25]))
    for side, cols in (("h", home), ("a", away)):
        d = np.hypot(x[:, cols] - blx[:, None], y[:, cols] - bly[:, None])
        f["bc_near_{}".format(side)] = _nan(np.nanmin, d, axis=1)
        f["bc_cnt_{}".format(side)] = np.where(np.isnan(blx), np.nan, _nan(np.nansum, d < NEAR_CAMERA, axis=1))
    f["bc_d_near"] = f["bc_near_h"] - f["bc_near_a"]
    f["bc_rel_h"] = blx - f["h_cx"]
    f["bc_rel_a"] = blx - f["a_cx"]
    f["tac_rel_h"] = f["tac_lx"] - f["h_cx"]
    f["tac_rel_a"] = f["tac_lx"] - f["a_cx"]

    # smart tagging: phases signed by team (+1 home, -1 away), set pieces as time-to/time-since
    attack_sum = np.zeros(n)
    for code in ATTACK_CODES + DEFEND_CODES:
        v = np.zeros(n)
        for tag in tags:
            if tag["code"] == code and tag["team"] is not None:
                v[max(0, tag["start_frame"]):tag["end_frame"]] += 1.0 if tag["team"] == D.HOME else -1.0
        f["tag_" + code.lower().replace(" ", "_")] = v
        attack_sum += v if code in ATTACK_CODES else -v
    f["tag_attack_sum"] = attack_sum
    goals = np.zeros(n)
    for tag in tags:
        if tag["code"] == "GOALS":
            goals[max(0, tag["start_frame"]):tag["end_frame"]] = 1.0
    f["tag_goal"] = goals
    starts = np.array(sorted(t["start_frame"] for t in tags if t["code"] == "SET PIECES"))
    frames = np.arange(n)
    pos = np.searchsorted(starts, frames, side="right")
    since = np.where(pos > 0, frames - starts[np.maximum(pos - 1, 0)], np.inf) / D.FPS
    until = np.where(pos < len(starts), starts[np.minimum(pos, len(starts) - 1)] - frames, np.inf) / D.FPS
    f["sp_since"] = np.minimum(since, CAP_SECONDS)
    f["sp_until"] = np.minimum(until, CAP_SECONDS)
    for kind in SET_PIECE_TYPES:
        v = np.zeros(n)
        for tag in tags:
            if tag["code"] == "SET PIECES" and tag["type"] == kind:
                v[max(0, tag["start_frame"] - 250):tag["start_frame"] + 250] = 1.0
        f["sp_" + kind.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")] = v

    return _finish(f, n)


def _finish(f, n):
    """Add the temporal-context columns and stack everything into a matrix."""
    # temporal context: how a core set of features differs a few seconds either side
    core = ("h_cx", "a_cx", "h_vx", "a_vx", "all_speed", "all_slow", "bc_live", "tac_lx", "h_sx", "a_sx")
    for name in core:
        if name not in f:
            continue
        base = f[name]
        for shift in SHIFTS:
            moved = np.full(n, np.nan)
            if shift > 0:
                moved[:-shift] = base[shift:]
            else:
                moved[-shift:] = base[:shift]
            f["{}_d{:+d}".format(name, shift)] = moved - base

    names = sorted(f)
    return np.column_stack([f[k] for k in names]).astype(np.float32), names


def pseudo_labels(post, bv, tags, kickoff_dead, confident, speed_6s, tv_live_6s):
    """Per-frame training target in {0: H, 1: A, 2: D}, -1 where nothing is sure.

    Sources, none of them human:
      * the ball-view HMM where its posterior is at least `confident` and the ball is seen
      * "everybody is walking and the TV director has cut away" -> dead. A stoppage with the ball
        still in view (a free kick being set up) looks like possession to the ball view; it does
        not look like play to twenty players and a director. Such frames are never taken as live.
      * a few seconds around every tagged set-piece start, and the wait before a kick-off -> dead
    The "walking" threshold is a quantile of this match's own speed distribution, not a constant.
    """
    graded = np.zeros(D.N_FRAMES, dtype=bool)
    for first, stop in D.HALVES:
        graded[first:stop] = True
    y = np.full(D.N_FRAMES, -1, dtype=np.int8)
    with np.errstate(invalid="ignore"):
        best = np.where(np.isnan(post).any(1), -1, np.nanargmax(np.where(np.isnan(post), -1, post), axis=1))
        sure = (np.nanmax(np.where(np.isnan(post), -1, post), axis=1) >= confident) & bv["has"]
        if tv_live_6s is None:                      # no TV feed (stand-in matches): this rule is off
            stopped = clearly_stopped = np.zeros(D.N_FRAMES, dtype=bool)
        else:
            walking = speed_6s < np.nanquantile(speed_6s[graded], WALKING_QUANTILE)
            stopped = walking & (tv_live_6s < TV_MOSTLY_AWAY)
            clearly_stopped = walking & (tv_live_6s < TV_AWAY)
    y[sure & ~(stopped & (best != 2))] = best[sure & ~(stopped & (best != 2))]
    y[clearly_stopped] = 2
    for tag in tags:
        if tag["code"] == "SET PIECES":
            y[max(0, tag["start_frame"] + SET_PIECE_DEAD[0]):tag["start_frame"] + SET_PIECE_DEAD[1]] = 2
    y[kickoff_dead] = 2
    y[~graded] = -1
    return y


def cross_fitted_probabilities(X, y, seed=0, verbose=True):
    """(n, 3) class probabilities for every graded frame, each from a model that did not train on
    that frame's one-minute block. Also returns the held-out agreement with the pseudo-labels."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    n = len(y)
    fold = (np.arange(n) // BLOCK) % N_FOLDS
    graded = np.zeros(n, dtype=bool)
    for first, stop in D.HALVES:
        graded[first:stop] = True
    proba = np.full((n, 3), np.nan)
    labelled = np.flatnonzero(y >= 0)
    labelled = labelled[::TRAIN_STRIDE]
    for k in range(N_FOLDS):
        train = labelled[fold[labelled] != k]
        counts = np.bincount(y[train], minlength=3).astype(float)
        weight = (len(train) / (3.0 * np.maximum(counts, 1.0)))[y[train]]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                             min_samples_leaf=40, l2_regularization=1.0,
                                             random_state=seed)
        clf.fit(X[train], y[train], sample_weight=weight)
        test = np.flatnonzero(graded & (fold == k))
        p = clf.predict_proba(X[test])
        full = np.zeros((len(test), 3))
        full[:, clf.classes_] = p
        proba[test] = full

    held = np.flatnonzero((y >= 0) & graded)
    pred = np.argmax(proba[held], axis=1)
    report = {"n_pseudo_labels": int(len(held)),
              "pseudo_label_share": [float(np.mean(y[held] == c)) for c in range(3)],
              "heldout_agreement": float(np.mean(pred == y[held])),
              "heldout_recall": [float(np.mean(pred[y[held] == c] == c)) if np.any(y[held] == c) else None
                                 for c in range(3)]}
    if verbose:
        for key, value in report.items():
            print("  player view {:<22} {}".format(key, value))
    return proba, report
