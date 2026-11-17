"""Where could the decoded possessions be wrong? Aggregate counts only.

    python solution/run.py --diagnostics <MTAH_DATA_DIR>/_cache/diag
    python solution/diagnose.py <MTAH_DATA_DIR>/_cache/diag

Reads the arrays run.py saved and prints counts, shares and duration buckets -- never a frame
number, a position or a label sequence. Nothing here uses ground truth of any kind: it describes
the model's own output so that we know which part of it rests on the least evidence.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mtah import data as D, hmm, prep, refine  # noqa: E402

# Params
FPS = D.FPS
DURATION_BUCKETS_S = (0, 1, 2, 3, 5, 8, 12, 20, 40, 1e9)
SHIFT_BUCKETS_F = (0, 1, 4, 9, 13, 26, 51, 1e9)     # |frames| a boundary would move
NEAR = 25                                            # frames either side of a boundary = 1 s
NAMES = {hmm.H: "H", hmm.A: "A", hmm.D: "D"}


def runs_of(codes):
    """[(first, stop, state)] inside each half."""
    out = []
    for first, stop in D.HALVES:
        w = codes[first:stop]
        cuts = np.flatnonzero(w[1:] != w[:-1]) + 1
        bounds = np.concatenate(([0], cuts, [len(w)]))
        out.append([(int(first + bounds[i]), int(first + bounds[i + 1]), int(w[bounds[i]]))
                    for i in range(len(bounds) - 1)])
    return out


def bucket_line(values, edges, unit="s"):
    values = np.asarray(values, dtype=float)
    parts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        n = int(np.sum((values >= lo) & (values < hi)))
        label = "{:g}+".format(lo) if hi > 1e8 else "{:g}-{:g}".format(lo, hi)
        parts.append("{}{}: {}".format(label, unit, n))
    return "   ".join(parts)


def describe_segments(codes, has, out_flag, still, forced):
    print("\n== segments ==")
    halves = runs_of(codes)
    flat = [r for half in halves for r in half]
    live = [r for r in flat if r[2] != hmm.D]
    dead = [r for r in flat if r[2] == hmm.D]
    print("possessions {}   stoppages {}".format(len(live), len(dead)))
    print("possession durations   ", bucket_line([(b - a) / FPS for a, b, _ in live], DURATION_BUCKETS_S))
    print("stoppage durations     ", bucket_line([(b - a) / FPS for a, b, _ in dead], DURATION_BUCKETS_S))

    # how much of each segment had the ball in the data
    def seen_share(seg):
        return [float(has[a:b].mean()) for a, b, _ in seg]
    edges = (0, 0.05, 0.25, 0.5, 0.75, 1.0001)
    print("possessions by share of frames with a ball  ", bucket_line(seen_share(live), edges, ""))
    print("stoppages   by share of frames with a ball  ", bucket_line(seen_share(dead), edges, ""))

    # what a stoppage rests on
    cue_out = sum(1 for a, b, _ in dead if out_flag[a:b].any())
    cue_still = sum(1 for a, b, _ in dead if still[a:b].any() and not out_flag[a:b].any())
    cue_forced = sum(1 for a, b, _ in dead if forced[a:b].any())
    print("stoppages with the ball seen out {}   only seen placed {}   forced (kick-off wait) {}   no ball cue {}".format(
        cue_out, cue_still, cue_forced, len(dead) - cue_out - cue_still - cue_forced))

    # short stoppages: do they split one side's possession in two?
    for limit in (2, 3, 5):
        same = other = edge = 0
        for half in halves:
            for i, (a, b, k) in enumerate(half):
                if k != hmm.D or (b - a) / FPS >= limit:
                    continue
                if i == 0 or i == len(half) - 1:
                    edge += 1
                elif half[i - 1][2] == half[i + 1][2]:
                    same += 1
                else:
                    other += 1
        print("stoppages under {} s: same side before and after {}   other side after {}   at a half's edge {}".format(
            limit, same, other, edge))

    # short possessions and their neighbours
    for limit in (2, 4):
        sandwiched = after_dead = other = 0
        for half in halves:
            for i, (a, b, k) in enumerate(half):
                if k == hmm.D or (b - a) / FPS >= limit:
                    continue
                prev_k = half[i - 1][2] if i else None
                next_k = half[i + 1][2] if i < len(half) - 1 else None
                if prev_k == next_k and prev_k not in (None, hmm.D):
                    sandwiched += 1
                elif prev_k == hmm.D or next_k == hmm.D:
                    after_dead += 1
                else:
                    other += 1
        print("possessions under {} s: inside the other side's play {}   next to a stoppage {}   other {}".format(
            limit, sandwiched, after_dead, other))
    return halves


def describe_boundaries(halves, has):
    print("\n== boundaries: was the ball in the data? ==")
    kinds = {"team change": [], "live -> dead": [], "dead -> live": []}
    for half in halves:
        for (a0, b0, k0), (a1, b1, k1) in zip(half[:-1], half[1:]):
            t = a1
            kind = "team change" if hmm.D not in (k0, k1) else "live -> dead" if k1 == hmm.D else "dead -> live"
            kinds[kind].append((bool(has[t - 1] and has[t]),
                                float(has[max(0, t - NEAR):t].mean()),
                                float(has[t:t + NEAR].mean())))
    for kind, rows in kinds.items():
        if not rows:
            continue
        at = np.array([r[0] for r in rows])
        before = np.array([r[1] for r in rows])
        after = np.array([r[2] for r in rows])
        print("{:<13} n {:>3}   ball at the boundary {:>3}   ball most of the second before {:>3} / after {:>3}"
              "   no ball at all within 1 s {:>3}".format(
                  kind, len(rows), int(at.sum()), int((before > 0.5).sum()), int((after > 0.5).sum()),
                  int(((before == 0) & (after == 0)).sum())))


def describe_views(arrays):
    print("\n== the two views against the final answer (graded frames) ==")
    final = arrays["final"]
    graded = final >= 0
    has = arrays["ball_has"]
    for key in ("ball_only", "player_only"):
        other = arrays[key]
        agree = final[graded] == other[graded]
        flat = [r for half in runs_of(other) for r in half]
        print("{:<12} agrees on {:.1%} of frames   ({:.1%} with a ball, {:.1%} without)   possessions {}   stoppages {}".format(
            key, agree.mean(), (final[graded & has] == other[graded & has]).mean(),
            (final[graded & ~has] == other[graded & ~has]).mean(),
            sum(1 for r in flat if r[2] != hmm.D), sum(1 for r in flat if r[2] == hmm.D)))
    live = graded & (final != hmm.D)
    print("frames called live: {:.1%} have a ball   |   frames called dead: {:.1%} have a ball".format(
        has[live].mean(), has[graded & (final == hmm.D)].mean()))
    # stretches called live with no ball for a long time
    gaps = []
    for first, stop in D.HALVES:
        blind = (final[first:stop] != hmm.D) & ~has[first:stop]
        edges = np.flatnonzero(np.diff(np.concatenate(([0], blind.astype(np.int8), [0]))))
        gaps += [(b - a) / FPS for a, b in zip(edges[::2], edges[1::2])]
    print("stretches called live with no ball   ", bucket_line(gaps, DURATION_BUCKETS_S))
    proba = arrays["proba"]
    top = np.nanmax(proba[graded], axis=1)
    print("player view's top probability on graded frames: under 0.5 {:.1%}   0.5-0.8 {:.1%}   over 0.8 {:.1%}".format(
        (top < 0.5).mean(), ((top >= 0.5) & (top < 0.8)).mean(), (top >= 0.8).mean()))


def describe_refinement(arrays):
    """How far would the switched-off boundary snapping (refine.py) move each kind of boundary?"""
    print("\n== what refine.py would do (it is switched off) ==")
    tac = D.load_tracking()
    bv = prep.ball_view(tac, arrays["ball_x"].astype(np.float64), arrays["ball_y"].astype(np.float64))
    codes = arrays["final"]
    for label, options in (("team change", dict(team=True, restarts=False, exits=False)),
                           ("dead -> live", dict(team=False, restarts=True, exits=False)),
                           ("live -> dead", dict(team=False, restarts=False, exits=True))):
        moved = refine.refine(codes, bv, **options)
        shifts = []
        n = 0
        for first, stop in D.HALVES:
            old = np.flatnonzero(codes[first + 1:stop] != codes[first:stop - 1])
            new = np.flatnonzero(moved[first + 1:stop] != moved[first:stop - 1])
            if len(old) != len(new):
                print("  {}: boundary count changed ({} -> {})".format(label, len(old), len(new)))
                continue
            for o, m in zip(old, new):
                a, b = int(codes[first + o]), int(codes[first + o + 1])
                kind = "team change" if hmm.D not in (a, b) else "live -> dead" if b == hmm.D else "dead -> live"
                if kind == label:
                    n += 1
                    if m != o:
                        shifts.append(m - o)
        shifts = np.array(shifts)
        print("{:<13} n {:>3}   moved {:>3}   earlier {:>3} / later {:>3}   |shift| ".format(
            label, n, len(shifts), int((shifts < 0).sum()), int((shifts > 0).sum()))
            + bucket_line(np.abs(shifts), SHIFT_BUCKETS_F, "f"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", help="the --diagnostics folder written by run.py")
    args = parser.parse_args()
    arrays = dict(np.load(os.path.join(args.folder, "arrays.npz")))
    halves = describe_segments(arrays["final"], arrays["ball_has"], arrays["ball_out"],
                               arrays["ball_still"], arrays["forced_dead"])
    describe_boundaries(halves, arrays["ball_has"])
    describe_views(arrays)
    describe_refinement(arrays)
