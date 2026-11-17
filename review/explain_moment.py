"""Why did the model say what it said at one doubtful moment? Prints the arithmetic.

    python review/explain_moment.py 66            # the moment shown as "no. 66" in the review page
    python review/explain_moment.py --frame 83110 # or the stretch around a frame

For each stretch of the moment it prints what every lane says, and then, for every stretch where
the model and another lane disagree, how much evidence each view gave to each answer and what a
change of answer would have cost. That is all the decoder ever looks at, so it is the whole reason.

The output is aggregates only -- sums of evidence, shares, medians -- never positions or rows of
the match data, so it is safe to paste where the data itself must not go.

Evidence is recomputed from the stored diagnostics of the last `solution/run.py --diagnostics`
run, i.e. the submitted model. The "previous version" lane came from an older run (cost 12, no
goalkeeper rule, its own classifier); it is shown, and "what cost 12 would have done with today's
evidence" is given as the nearest thing to an explanation of it. The ball view's evidence is the
learned one (model.LEARNED_BALL_VIEW); the grip and distances printed beside it are the formula's
own outputs, which are inputs of the classifier, not the evidence itself.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (os.path.join(ROOT, "solution"), ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")):
    sys.path.insert(0, path)

import possession_metrics as pm  # noqa: E402
from mtah import data as D, model, prep  # noqa: E402
import run as pipeline  # noqa: E402

FPS = D.FPS
NAMES = ["Barcelona (home)", "Athletic (away)", "dead ball", "not graded"]
SHORT = "HADX"


def sequence(codes, first):
    """'H 3.2 s -> A 1.4 s -> H 2.0 s' and the runs as (start, stop, code)."""
    breaks = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(codes)]))
    runs = [(first + int(a), first + int(b), int(codes[a])) for a, b in zip(bounds[:-1], bounds[1:])]
    return " -> ".join("{} {:.1f} s".format(SHORT[c], (b - a) / FPS) for a, b, c in runs), runs


def clock(frame):
    for k, (first, stop) in enumerate(D.HALVES):
        if first <= frame < stop:
            s = (frame - first) // FPS
            return "{} half {}:{:02d}".format("1st" if k == 0 else "2nd", s // 60, s % 60)
    return "outside the halves"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("moment", nargs="?", type=int, help="the moment's number as the review page shows it (its rank)")
    parser.add_argument("--frame", type=int, help="explain the stretch around this frame instead")
    parser.add_argument("--seconds", type=float, default=4.0, help="with --frame: how much either side (default 4)")
    args = parser.parse_args()

    folder = os.path.join(D.DATA_DIR, "_cache", "review", "bar-ath")
    with open(os.path.join(folder, "moments.json"), encoding="utf-8") as handle:
        moments = json.load(handle)
    if args.frame is not None:
        lo, hi = args.frame - int(args.seconds * FPS), args.frame + int(args.seconds * FPS)
        inside = [m for m in moments if m["start"] <= args.frame < m["end"]]
        title = "Around frame {} ({})".format(args.frame, clock(args.frame)) + (
            ", inside doubtful moment no. {}".format(inside[0]["rank"]) if inside else "")
    else:
        found = [m for m in moments if m["rank"] == args.moment]
        if not found:
            sys.exit("there is no moment no. {} (there are {})".format(args.moment, len(moments)))
        m = found[0]
        lo, hi = m["play_start"], m["play_end"]
        title = "Doubtful moment no. {} of {}: {}, {:.1f} s, flagged for: {}".format(
            m["rank"], len(moments), clock(m["start"]), (m["end"] - m["start"]) / FPS, ", ".join(m["reasons"]))
    print(title)
    print("(shown with {:.0f} s of lead-in and {:.0f} s after, as the page plays it)\n".format(
        (m["start"] - lo) / FPS if args.frame is None else args.seconds, (hi - m["end"]) / FPS if args.frame is None else args.seconds))

    arrays = np.load(os.path.join(D.DATA_DIR, "_cache", "workspace", "arrays.npz"))
    final = pm.label_codes(pm.read_label_csv(os.path.join(ROOT, "evaluation", "results", "possession_two_views.csv")))
    stored = arrays["final"].astype(int)
    stored[stored < 0] = 3
    if not np.array_equal(stored, final):
        sys.exit("the stored diagnostics are not from the submitted file: run solution/run.py --diagnostics <data>/_cache/workspace")
    prev_path = os.path.join(D.DATA_DIR, "_cache", "possession_two_views_v1_cost12.csv")
    prev = pm.label_codes(pm.read_label_csv(prev_path)) if os.path.isfile(prev_path) else final
    lanes = {"Model": final, "Previous version": prev}
    for key, name in (("ball_only", "Ball view alone"), ("player_only", "Player view alone")):
        codes = arrays[key].astype(int)
        codes[codes < 0] = 3
        lanes[name] = codes

    tac = D.load_tracking()
    bv = prep.ball_view(tac, arrays["ball_x"].astype(float), arrays["ball_y"].astype(float))
    learned = pipeline.learned_ball_view()                     # who is on the ball, read as run.py reads it
    if learned is not None:
        bv["learned"] = learned(bv, arrays["ball_x"].astype(float), arrays["ball_y"].astype(float))
    has = bv["has"]
    proba = arrays["proba"].astype(float)
    ball_e = model.ball_emissions(bv)
    players_e = model.player_emissions(proba, has)             # as the model weighs it
    players_alone = model.player_emissions(proba)              # as the "player view alone" lane weighs it
    cost = np.where(has, model.TEAM_COST_BALL_SEEN, model.SWITCH_COST)

    print("What each lane says, in order:")
    runs = None
    for name, codes in lanes.items():
        text, these = sequence(codes[lo:hi], lo)
        runs = runs or these
        print("  {:<18} {}".format(name, text))
    print("  (H = Barcelona, A = Athletic, D = dead ball)")
    print("  ball in the data on {:.0f}% of these frames; goalkeeper rule active on {} frames\n".format(
        100 * has[lo:hi].mean(), int((bv["keeper_hold"][lo:hi] != 0).sum())))

    told = False
    for a, b, label in runs:
        others = {name: int(np.bincount(codes[a:b], minlength=4).argmax()) for name, codes in lanes.items() if name != "Model"}
        rivals = sorted({c for c in others.values() if c != label and c < 3})
        if label > 2 or not rivals:
            continue
        told = True
        n = b - a
        print("Stretch {} -> {} ({:.1f} s, {} frames): the model says {}".format(clock(a), clock(b - 1), n / FPS, n, NAMES[label]))
        print("   others: " + "; ".join("{} says {}".format(k.lower(), NAMES[v]) for k, v in others.items()))
        seen = has[a:b]
        print("   ball in the data on {:.0f}% of the stretch".format(100 * seen.mean())
              + ("" if not seen.any() else "; median distance from the ball to the nearest Barcelona player {:.1f} m, Athletic player {:.1f} m".format(
                  float(np.nanmedian(bv["nearest_home"][a:b][seen])), float(np.nanmedian(bv["nearest_away"][a:b][seen])))))
        if seen.any():
            print("   mean grip while the ball is seen: Barcelona {:.2f}, Athletic {:.2f} (1 = firmly on the ball, 0 = nobody within 2.5 m)".format(
                float(bv["control_home"][a:b][seen].mean()), float(bv["control_away"][a:b][seen].mean())))
        p = np.nan_to_num(proba[a:b]).mean(0)
        print("   player view's probabilities, averaged: Barcelona {:.0%}, Athletic {:.0%}, dead ball {:.0%}".format(*p))
        hold = int((bv["keeper_hold"][a:b] != 0).sum())
        if hold:
            print("   goalkeeper rule: active on {} frames, for {}".format(hold, NAMES[0 if bv["keeper_hold"][a:b].max() > 0 else 1]))
        for rival in rivals:
            ball = float((ball_e[a:b, label] - ball_e[a:b, rival]).sum())
            players = float((players_e[a:b, label] - players_e[a:b, rival]).sum())
            alone = float((players_alone[a:b, label] - players_alone[a:b, rival]).sum())
            pay = float(cost[a] + cost[min(b, len(cost) - 1)]) if rival < 2 and label < 2 else 2 * model.SWITCH_COST
            print("   evidence for {} over {}, added up over the stretch:".format(NAMES[label], NAMES[rival]))
            print("      from the ball view   {:+7.1f}".format(ball))
            print("      from the player view {:+7.1f}   (it would be {:+.1f} at the full weight the 'player view alone' lane uses)".format(players, alone))
            print("      together             {:+7.1f}".format(ball + players))
            print("      price of switching to it and back: {:.0f} for the model ({:.0f} in, {:.0f} out); 24 for the ball view alone, "
                  "the player view alone and the previous version (12 each way)".format(pay, cost[a], cost[min(b, len(cost) - 1)]))
            print("      -> model:             {:+.1f} against {:.0f}: {}".format(ball + players, pay, "worth it" if ball + players > pay else "not worth it on its own (the neighbouring stretches decide)"))
            print("      -> ball view alone:   {:+.1f} against 24: {}".format(ball, "would switch" if ball > 24 else "stays where it was"))
            print("      -> player view alone: {:+.1f} against 24: {}".format(alone, "would switch" if alone > 24 else "stays where it was"))
            print("      -> today's evidence at the previous version's price: {:+.1f} against 24: {}".format(
                ball + players, "would switch" if ball + players > 24 else "stays where it was"))
        print()
    if not told:
        print("All four lanes agree throughout this stretch; there is nothing to reconcile.")
    print("How to read it: every frame, each view gives points to H, A and D. The model keeps its current answer unless the "
          "other answer has collected more points than the price of changing. The price of a change of team is {:.0f} while the "
          "ball is in the data and {:.0f} when it is not or for dead ball.".format(model.TEAM_COST_BALL_SEEN, model.SWITCH_COST))


if __name__ == "__main__":
    main()
