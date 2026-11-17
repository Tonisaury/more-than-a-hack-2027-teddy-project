"""Recover ball positions the tactical tracker lost, from the unsynchronised broadcast tracking.

The broadcast recording has its own timeline and no mapping to the tactical video is provided.
Both trackings, however, live in the same pitch coordinates, so the mapping can be *measured*:

  1. coarse  -- cross-correlate the two ball trajectories (FFT, masked) chunk by chunk. Every chunk
                votes for an offset `tactical_frame - broadcast_frame`; the votes fall into a few
                tight clusters (one per uninterrupted stretch of the broadcast recording).
  2. fine    -- around every candidate offset, pick the per-window shift that makes the broadcast
                players land on the tactical players (mean nearest same-team neighbour distance).
  3. verify  -- a broadcast frame is only trusted if its players really do match the tactical ones
                at the mapped frame. Replays and cutaways show another moment of the match, their
                players do not line up, and they are dropped here without any special casing.
  4. correct -- the residual between matched players (median vector) is the broadcast homography's
                local error; the same shift is applied to the broadcast ball before it is used.

Nothing is aligned by hand and no constant in here refers to a frame of this match.
"""

import numpy as np

from . import data as D

# Params
CHUNK = 4000              # frames per coarse cross-correlation chunk (160 s)
MIN_BALL_FRAMES = 400     # a chunk needs this many ball observations to vote
CLUSTER_TOL = 40          # frames; coarse votes closer than this are the same offset
MIN_CLUSTER_VOTES = 3
FINE_RANGE = 15           # +/- frames searched around each coarse offset
FINE_WINDOW = 1500        # frames per fine-alignment window (60 s)
MIN_PLAYERS = 5           # broadcast players needed to verify a frame
MAX_MATCH_COST = 1.5      # metres; mean nearest-neighbour distance for a frame to be trusted
CLIP = 8.0                # metres; cap on a single player's mismatch, so one stray track cannot veto
BATCH = 4000


def _xcorr(a, b, n):
    return np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)


def _coarse_votes(tac, bro):
    """Offset voted by each broadcast chunk: argmin of the masked mean squared ball distance."""
    t_mask = ~np.isnan(tac["bx"])
    t_xy = [np.where(t_mask, tac["bx"], 0.0), np.where(t_mask, tac["by"], 0.0)]
    t_maskf = t_mask.astype(float)
    n_b = len(bro["bx"])
    votes = []
    for start in range(0, n_b - CHUNK + 1, CHUNK):
        b_mask = ~np.isnan(bro["bx"][start:start + CHUNK])
        if b_mask.sum() < MIN_BALL_FRAMES:
            continue
        n = 1 << int(np.ceil(np.log2(len(t_mask) + CHUNK)))
        b_maskf = b_mask.astype(float)
        count = _xcorr(t_maskf, b_maskf, n)
        ssd = np.zeros(n)
        for k, key in enumerate(("bx", "by")):
            b_val = np.where(b_mask, bro[key][start:start + CHUNK], 0.0)
            ssd += (_xcorr(t_xy[k] ** 2, b_maskf, n) - 2.0 * _xcorr(t_xy[k], b_val, n)
                    + _xcorr(t_maskf, b_val ** 2, n))
        msd = np.where(count > 0.3 * b_mask.sum(), ssd / np.maximum(count, 1.0), np.inf)
        lag = int(np.argmin(msd))
        if lag > len(t_mask):          # circular correlation: large lags are negative ones
            lag -= n
        votes.append(lag - start)
    return votes


def _cluster(votes):
    votes = sorted(votes)
    clusters, current = [], [votes[0]]
    for value in votes[1:]:
        if value - current[-1] <= CLUSTER_TOL:
            current.append(value)
        else:
            clusters.append(current)
            current = [value]
    clusters.append(current)
    return [int(round(float(np.median(c)))) for c in clusters if len(c) >= MIN_CLUSTER_VOTES]


def _match(tac, bro, b_frames, t_frames):
    """For broadcast frames mapped onto tactical frames: (cost, n_players, shift_x, shift_y).

    cost is the mean distance from each broadcast player to the nearest tactical player of the
    same team, each capped at CLIP. shift is the median residual of the well-matched players.
    """
    n = len(b_frames)
    cost = np.full(n, np.inf)
    count = np.zeros(n, dtype=int)
    shift = np.zeros((n, 2))
    valid = (t_frames >= 0) & (t_frames < len(tac["bx"]))
    same_team = bro["team"][:, None] == tac["team"][None, :]
    for lo in range(0, n, BATCH):
        sel = np.flatnonzero(valid[lo:lo + BATCH]) + lo
        if not len(sel):
            continue
        bx, by = bro["px"][b_frames[sel]], bro["py"][b_frames[sel]]
        tx, ty = tac["px"][t_frames[sel]], tac["py"][t_frames[sel]]
        dx = tx[:, None, :] - bx[:, :, None]
        dy = ty[:, None, :] - by[:, :, None]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.where(same_team[None] & ~np.isnan(dist), dist, np.inf)
        nearest = np.argmin(dist, axis=2)
        best = np.take_along_axis(dist, nearest[:, :, None], 2)[:, :, 0]
        present = ~np.isnan(bx) & np.isfinite(best)
        k = present.sum(1)
        clipped = np.where(present, np.minimum(best, CLIP), 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            cost[sel] = np.where(k > 0, clipped.sum(1) / np.maximum(k, 1), np.inf)
        count[sel] = k
        good = present & (best < CLIP / 2)
        rx = np.where(good, np.take_along_axis(dx, nearest[:, :, None], 2)[:, :, 0], np.nan)
        ry = np.where(good, np.take_along_axis(dy, nearest[:, :, None], 2)[:, :, 0], np.nan)
        any_good = good.any(1)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shift[sel, 0] = np.where(any_good, np.nanmedian(rx, axis=1), 0.0)
                shift[sel, 1] = np.where(any_good, np.nanmedian(ry, axis=1), 0.0)
    return cost, count, shift


def broadcast_ball_on_tactical_timeline(tac, bro, verbose=True):
    """(bx, by, report, mapping): broadcast ball observations placed on tactical frames.

    mapping holds, per broadcast frame, the tactical frame it shows (`target`), how well its
    players match there (`cost`, metres) and whether it passed verification (`trusted`).
    """
    offsets = _cluster(_coarse_votes(tac, bro))
    n_b, n_t = len(bro["bx"]), len(tac["bx"])
    b_all = np.arange(n_b)

    best_cost = np.full(n_b, np.inf)
    best_target = np.full(n_b, -1)
    best_shift = np.zeros((n_b, 2))
    fine_log = []
    for offset in offsets:
        sub = b_all[::5]                       # fine search on every 5th frame is plenty
        window_cost = {}
        for delta in range(-FINE_RANGE, FINE_RANGE + 1):
            cost, count, _ = _match(tac, bro, sub, sub + offset + delta)
            cost = np.where(count >= MIN_PLAYERS, np.minimum(cost, CLIP), np.nan)
            window_cost[delta] = cost
        deltas = np.array(sorted(window_cost))
        stack = np.vstack([window_cost[d] for d in deltas])           # (n_delta, n_sub)
        per_frame_delta = np.zeros(n_b, dtype=int)
        last = 0
        for lo in range(0, n_b, FINE_WINDOW):
            cols = (sub >= lo) & (sub < lo + FINE_WINDOW)
            block = stack[:, cols]
            usable = (~np.isnan(block)).sum(1)
            if usable.max() >= 20:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    score = np.nanmean(block, axis=1)
                score = np.where(usable >= 20, score, np.inf)
                if np.min(score) < MAX_MATCH_COST:
                    last = int(deltas[int(np.argmin(score))])
            per_frame_delta[lo:lo + FINE_WINDOW] = last
        fine_log.append((offset, int(per_frame_delta.min()), int(per_frame_delta.max())))

        target = b_all + offset + per_frame_delta
        cost, count, shift = _match(tac, bro, b_all, target)
        cost = np.where(count >= MIN_PLAYERS, cost, np.inf)
        better = cost < best_cost
        best_cost[better] = cost[better]
        best_target[better] = target[better]
        best_shift[better] = shift[better]

    trusted = (best_cost < MAX_MATCH_COST) & ~np.isnan(bro["bx"])
    out_x = np.full(n_t, np.nan)
    out_y = np.full(n_t, np.nan)
    out_x[best_target[trusted]] = bro["bx"][trusted] + best_shift[trusted, 0]
    out_y[best_target[trusted]] = bro["by"][trusted] + best_shift[trusted, 1]

    report = {
        "coarse_offsets": offsets,
        "fine_delta_range": fine_log,
        "broadcast_frames": int(n_b),
        "broadcast_frames_verified": int((best_cost < MAX_MATCH_COST).sum()),
        "broadcast_ball_frames": int((~np.isnan(bro["bx"])).sum()),
        "broadcast_ball_frames_trusted": int(trusted.sum()),
    }
    both = ~np.isnan(out_x) & ~np.isnan(tac["bx"])
    if both.any():
        err = np.hypot(out_x[both] - tac["bx"][both], out_y[both] - tac["by"][both])
        report["agreement_frames"] = int(both.sum())
        report["agreement_median_m"] = float(np.median(err))
        report["agreement_p90_m"] = float(np.percentile(err, 90))
        report["agreement_within_3m"] = float(np.mean(err < 3.0))
    if verbose:
        for key, value in report.items():
            print("  sync {:<28} {}".format(key, value))
    mapping = {"target": best_target, "cost": best_cost, "shift": best_shift,
               "trusted": best_cost < MAX_MATCH_COST}
    return out_x, out_y, report, mapping
