"""Score decoder variants on other matches that have answers.

    python solution/standin_eval.py                 # downloads Metrica's open sample games on first run
    python solution/standin_eval.py --sweep         # the grid behind the constants in model.py
    python solution/standin_eval.py --noise         # how the degraded stand-in compares with our tracking
    python solution/standin_eval.py --keeper        # the goalkeeper rule: is its premise true on other matches?

The sample games are hand-corrected tracking, far cleaner than the challenge match's automatic
tracking, so each game is first degraded (SCENARIOS): the ball is given a slow positional wander
sized so that label-free statistics (distance from the ball to the nearest player, how often the
nearest team flips) bracket the challenge match's, and in half of the scenarios the challenge
tracker's own ball-gap pattern is laid over it. `--noise` prints those statistics side by side.

For each game and scenario the expensive stages (ball view, first pass, player view) run once and
are cached. Every decoder variant is then scored against the event-derived truth with the
organisers' own metric code.

Two truths are reported, because "possession" from events can be read two ways:
  loose       every owning event changes hands (median possession about 5 s)
  controlled  a spell of the new team shorter than CONTROLLED_SECONDS, sandwiched inside the other
              team's play, is treated as a touch and not a change (the challenge's "controlled
              possession")
A change is adopted only if it helps under both.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))

import possession_metrics as pm  # noqa: E402
from mtah import data as D, model, prep, standin  # noqa: E402
import run as pipeline  # noqa: E402

# (name, lay the challenge gap pattern over the ball at this shift or None, wander metres, wander seconds)
SCENARIOS = (
    ("own gaps, wander 0.8 m / 1 s", None, 0.8, 1.0),
    ("own gaps, wander 1.2 m / 2 s", None, 1.2, 2.0),
    ("challenge gaps, wander 0.8 m / 1 s", 0.37, 0.8, 1.0),
    ("challenge gaps, wander 1.2 m / 2 s", 0.37, 1.2, 2.0),
)
ORACLE = "[stoppages given]"        # variant-name suffix: dead ball is forced from the truth, so only the
                                    # who-has-it part of the decoder is being measured
CONTROLLED_SECONDS = 2.0


def controlled(truth, halves, seconds=CONTROLLED_SECONDS):
    """Fold short spells that sit between two spells of the other team into that team."""
    out = truth.copy()
    limit = int(seconds * D.FPS)
    for first, stop in halves:
        changed = True
        while changed:
            changed = False
            segs = pm.segments(out, [(first, stop)])
            for i in range(1, len(segs) - 1):
                a, b, k = segs[i]
                if k in (0, 1) and b - a < limit and segs[i - 1][2] == segs[i + 1][2] == 1 - k:
                    out[a:b] = 1 - k
                    changed = True
                    break
    return out


def baseline(views, window=50):
    """The organisers' baseline on the same degraded ball: nearest player, then a majority vote."""
    from baseline import smooth
    bv = views["bv"]
    codes = np.where(~bv["has"], -1, np.where(bv["nearest_home"] <= bv["nearest_away"], 0, 1)).astype(np.int8)
    labels = smooth(codes, window)
    out = np.full(len(codes), 3, dtype=np.int8)
    for k, letter in enumerate("HAD"):
        out[labels == letter] = k
    return out


def degraded_ball(game, masks, scenario):
    _name, shift, offset, seconds = scenario
    bx, by = standin.degrade(game, None if shift is None else masks, shift or 0.0, offset, seconds, seed=int(game["name"][-1]))
    bx, by = prep.clean_ball(bx, by)
    bx, by, _ = prep.interpolate_short_gaps(bx, by, prep.INTERP_MAX_GAP)
    return bx, by


def prepare(game, masks, scenario, cache_dir):
    """Views for one game and one scenario, cached on disk."""
    _name, shift, offset, seconds = scenario
    path = os.path.join(cache_dir, "views2_g{}_{}_{}_{}.npz".format(game["name"][-1], "own" if shift is None else shift, offset, seconds))
    bx, by = degraded_ball(game, masks, scenario)
    forced_dead = np.zeros(game["n_frames"], dtype=bool)
    if os.path.isfile(path):
        stored = np.load(path)
        bv = prep.ball_view(game["tac"], bx, by)
        views = {"bv": bv, "ball_e": model.ball_emissions(bv), "proba": stored["proba"], "bx": bx, "by": by, "tac": game["tac"],
                 "agreement": float(stored["agreement"])}
    else:
        views = pipeline.two_views(game["tac"], bx, by, forced_dead)
        views["agreement"] = views["report"]["heldout_agreement"]
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(path, proba=views["proba"].astype(np.float32), agreement=views["agreement"])
    return views, forced_dead


def to_pm(codes):
    out = codes.astype(np.int8).copy()
    out[out < 0] = 3
    return out


COLUMNS = "{:<44} {:>7}{:>7}{:>8}  {:>7}{:>7}{:>8}  {:>6} {:>7} {:>7}"


def table(title, variants, rows):
    print("\n" + title)
    print(COLUMNS.format("variant", "spell", "macro", "n/true", "spell", "macro", "n/true", "D-F1", "trans1s", "IoU.7"))
    print(COLUMNS.format("", "(loose", "truth)", "", "(contr.", "truth)", "", "", "", ""))
    for name in variants:
        row = []
        for kind in ("loose", "controlled"):
            a = np.array(rows[name][kind])
            row += ["{:.3f}".format(a[:, 0].mean()), "{:.3f}".format(a[:, 1].mean()), "{:.0f}/{:.0f}".format(a[:, 2].mean(), a[:, 3].mean())]
        a = np.array(rows[name]["controlled"])
        print(COLUMNS.format(name, *row, "{:.3f}".format(a[:, 4].mean()), "{:.3f}".format(a[:, 5].mean()), "{:.3f}".format(a[:, 6].mean())))


def evaluate(variants, verbose=True):
    """{scenario: {variant: {truth_kind: [metrics per game]}}} plus printed tables."""
    root = standin.fetch()
    cache_dir = os.path.join(root, "cache")
    masks = standin.challenge_gap_mask()
    results = {}
    for n in standin.GAMES:
        game = standin.load_game(n)
        truths = {"loose": game["truth"], "controlled": controlled(game["truth"], game["halves"])}
        with standin.match_shape(game):
            for scenario in SCENARIOS:
                views, forced_dead = prepare(game, masks, scenario, cache_dir)
                oracle = truths["loose"] == 2            # for variants that are told when play is stopped
                for name, fn in variants.items():
                    codes = to_pm(fn(views, oracle if name.endswith(ORACLE) else forced_dead))
                    for kind, truth in truths.items():
                        r = pm.evaluate(truth, codes)
                        results.setdefault(scenario[0], {}).setdefault(name, {}).setdefault(kind, []).append(
                            (r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"], r["per_class"]["D"]["f1"],
                             r["transitions"][25]["f1"], r["spell_f1"][0.7]["f1"]))
                if verbose:
                    print("  done", game["name"], "|", scenario[0], "| player-view agreement {:.3f}".format(views["agreement"]), flush=True)
    if verbose:
        overall = {}
        for scenario in SCENARIOS:
            table(scenario[0], variants, results[scenario[0]])
            for name in variants:
                for kind in ("loose", "controlled"):
                    overall.setdefault(name, {}).setdefault(kind, []).extend(results[scenario[0]][name][kind])
        table("ALL SCENARIOS", variants, overall)
    return results


def noise_report():
    """Label-free statistics of the ball, challenge match against each scenario (aggregates only)."""
    def stats(tac, bx, by, halves, label):
        bv = prep.ball_view(tac, bx, by)
        graded = np.zeros(len(bx), dtype=bool)
        for first, stop in halves:
            graded[first:stop] = True
        m = graded & bv["has"]
        near = np.minimum(bv["nearest_home"], bv["nearest_away"])[m]
        side = bv["nearest_home"] <= bv["nearest_away"]
        minutes = m.sum() / D.FPS / 60.0
        flips = np.sum((side[1:] != side[:-1]) & m[1:] & m[:-1])
        net = bv["control_home"] - bv["control_away"]
        clear = np.flatnonzero(m & (np.abs(net) > model.MIN_TOUCH))
        clear_flips = np.sum(np.sign(net[clear][1:]) != np.sign(net[clear][:-1]))
        print("{:<52} seen {:.2f} | nearest player q25/50/75 {:.2f} {:.2f} {:.2f} m | nearest-team flips {:5.1f}/min | "
              "clear-control flips {:4.1f}/min | clear control {:.2f} of seen".format(
                  label, m.sum() / graded.sum(), *np.percentile(near, [25, 50, 75]), flips / minutes, clear_flips / minutes,
                  len(clear) / m.sum()))

    tac = D.load_tracking()
    bro = D.load_tracking(source="broadcast")
    from mtah import sync
    bx, by, _report, _mapping = sync.broadcast_ball_on_tactical_timeline(tac, bro, False)
    fused_x, fused_y, _observed = pipeline.fused_ball(tac, bx, by)
    stats(tac, fused_x, fused_y, D.HALVES, "CHALLENGE MATCH (both cameras fused)")
    masks = standin.challenge_gap_mask()
    standin.fetch()
    for n in standin.GAMES:
        game = standin.load_game(n)
        with standin.match_shape(game):
            for scenario in (("as published", None, 0.0, 1.0),) + SCENARIOS:
                bx, by = degraded_ball(game, masks, scenario)
                stats(game["tac"], bx, by, game["halves"], "game {} | {}".format(n, scenario[0]))


VARIANTS = {
    "baseline nearest+vote(2s)": lambda v, fd: baseline(v, 50),
    "ball view only": lambda v, fd: model.decode(v["ball_e"], fd),
    "current two views": lambda v, fd: pipeline.final_pass(v, fd),
    "baseline nearest+vote(2s) " + ORACLE: lambda v, fd: np.where(fd, 2, baseline(v, 50)),
    "current two views " + ORACLE: lambda v, fd: pipeline.final_pass(v, fd),
}


def with_ball_view(views, forced_dead, **constants):
    """The final pass with some ball-view constants (prep.py) changed for this call only."""
    saved = {k: getattr(prep, k) for k in constants}
    try:
        for k, v in constants.items():
            setattr(prep, k, v)
        bv = prep.ball_view(views["tac"], views["bx"], views["by"])
        alt = dict(views, bv=bv, ball_e=model.ball_emissions(bv))
        return pipeline.final_pass(alt, forced_dead, 6, 12, False, None, 0.25)
    finally:
        for k, v in saved.items():
            setattr(prep, k, v)


def sweep():
    """The grid behind the constants in model.py."""
    out = {"baseline nearest+vote(2s)": lambda v, fd: np.where(fd, 2, baseline(v, 50))}
    for discount in (1.0, 0.25, 0.0):
        for seen in (12, 8, 6, 4, 3, 2):
            out["cost {}/12 discount {}".format(seen, discount)] = (
                lambda v, fd, s=seen, d=discount: pipeline.final_pass(v, fd, s, 12, False, None, d))
    for weight in (0.15, 0.40):
        for seen in (8, 4):
            out["cost {}/12 discount 0.25 weight {}".format(seen, weight)] = (
                lambda v, fd, s=seen, w=weight: pipeline.final_pass(v, fd, s, 12, False, w, 0.25))
    for gap in (8, 24):
        out["cost 6/{} discount 0.25".format(gap)] = lambda v, fd, g=gap: pipeline.final_pass(v, fd, 6, g, False, None, 0.25)
    for radius in (1.5, 2.0, 3.0, 3.5, 4.5):
        out["cost 6/12 discount 0.25 radius {} m".format(radius)] = lambda v, fd, r=radius: with_ball_view(v, fd, CONTROL_RADIUS=r)
    for seen in (12, 6):
        out["cost {}/12 discount 0.25 + sharpen".format(seen)] = lambda v, fd, s=seen: pipeline.final_pass(v, fd, s, 12, True, None, 0.25)
    # stoppages are given: these matches have no TV feed or tagging, which is what our stoppage detection
    # leans on, so only the who-has-it part of the decoder can be judged here
    return {name + " " + ORACLE: fn for name, fn in out.items()}


def keeper_report():
    """The goalkeeper rule (prep.keeper_hold) on the stand-in: is its premise true, and does it move the score?"""
    root = standin.fetch()
    cache_dir = os.path.join(root, "cache")
    masks = standin.challenge_gap_mask()
    saved = model.W_KEEPER_HOLD
    premise, scores = np.zeros(5), {}
    try:
        for n in standin.GAMES:
            game = standin.load_game(n)
            truths = {"loose": game["truth"], "controlled": controlled(game["truth"], game["halves"])}
            with standin.match_shape(game):
                for scenario in SCENARIOS:
                    views, _ = prepare(game, masks, scenario, cache_dir)
                    hold = views["bv"]["keeper_hold"]
                    for value, side in ((1, 0), (-1, 1)):
                        m = hold == value
                        t = truths["loose"][m]
                        premise += (m.sum(), (t == side).sum(), (t == 1 - side).sum(), (t == 2).sum(),
                                    np.sum(np.diff(np.concatenate(([0], m.astype(int)))) == 1))
                    for weight in (0.0, 1.0):
                        model.W_KEEPER_HOLD = weight
                        alt = dict(views, ball_e=model.ball_emissions(views["bv"]))
                        codes = to_pm(pipeline.final_pass(alt, truths["loose"] == 2))
                        for kind, truth in truths.items():
                            scores.setdefault((weight, kind), []).append(pm.evaluate(truth, codes, bootstrap_n=0)["spell_f1_mean"])
    finally:
        model.W_KEEPER_HOLD = saved
    print("the rule fires {:.0f} times, {:.0f} s in all; in those frames the event truth says: keeper's side {:.2f}, "
          "other side {:.2f}, dead ball {:.2f}".format(premise[4], premise[0] / D.FPS, *(premise[1:4] / max(premise[0], 1))))
    for kind in ("loose", "controlled"):
        print("spellF1, {} truth: without the rule {:.4f}, with it {:.4f}".format(
            kind, np.mean(scores[(0.0, kind)]), np.mean(scores[(1.0, kind)])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep", action="store_true", help="score the grid of switching costs and refinement options")
    parser.add_argument("--noise", action="store_true", help="compare the ball's label-free statistics: challenge match vs scenarios")
    parser.add_argument("--keeper", action="store_true", help="check the goalkeeper rule: how often its premise holds, and its effect")
    args = parser.parse_args()
    if args.noise:
        noise_report()
    elif args.keeper:
        keeper_report()
    else:
        evaluate(sweep() if args.sweep else VARIANTS)
