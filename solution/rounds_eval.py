"""Two cheap additions to the player view, measured on DFL matches the ball classifier never saw.

    python solution/rounds_eval.py [--targets J03WMX J03WN1 J03WOH] [--seeds 0 1 2]

  seed averaging   the player view's classifier holds out a random tenth of its training frames for
                   early stopping, so its seed decides about 5% of the possessions (seed_stability.py).
                   Averaging the probabilities of several seeds removes that lottery.
  second round     the player view is taught by pseudo-labels from the ball view alone, so it is only
                   ever taught on frames that have a ball. After the first full decode the two views
                   together are sure about many frames without one; a second round teaches the player
                   view on those too, and decodes again.

Leave-one-match-out as in ballmodel_eval.py: the ball classifier used on each match was trained on
the other matches. Stoppages are not given.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ballmodel_eval as B  # noqa: E402
from mtah import ballmodel, dfl, model, playerview, standin  # noqa: E402

E, bench, pipeline = B.E, B.bench, B.pipeline


def second_round_labels(views, proba, forced_dead, confident=None):
    """Pseudo-labels from the full decode: wherever both views together are sure, ball or no ball."""
    confident = model.CONFIDENT if confident is None else confident
    bv = views["bv"]
    emissions = views["ball_e"] + model.player_emissions(proba, bv["has"])
    _codes, post = model.decode(emissions, forced_dead, want_posteriors=True)
    everywhere = dict(bv, has=np.ones(len(bv["has"]), dtype=bool))
    names = views["names"]
    tv = views["X"][:, names.index("bc_live_6s")] if "bc_live_6s" in names else None
    return playerview.pseudo_labels(post, everywhere, views.get("tags", ()), forced_dead, confident,
                                    views["X"][:, names.index("all_speed_6s")], tv)


def run(targets, seeds):
    everything = dfl.available()
    cache_dir = os.path.join(dfl.folder(), "cache")
    masks = standin.challenge_gap_mask()
    overall, cells = {}, {}
    order = ["round 1, one seed (branch ander/work)", "round 1, {} seeds averaged".format(len(seeds)),
             "round 2, one seed", "round 2, {} seeds averaged".format(len(seeds))]
    for match in targets:
        clf, names = B.train_on([m for m in everything if m != match], masks, cache_dir)
        for scenario in E.SCENARIOS:
            game, _controlled, cell = B.build_cell(match, scenario, masks, cache_dir)
            truths = {"loose": game["truth"], "controlled": bench.controlled(game["truth"], game["halves"])}
            seed = list(dfl.FILES).index(match)
            with standin.match_shape(game):
                logp = ballmodel.log_odds(clf, cell["X"], names, cell["has"])
                bx, by = E.degraded_ball(game, masks, scenario, seed)
                forced_dead = np.zeros(game["n_frames"], dtype=bool)
                views = pipeline.two_views(game["tac"], bx, by, forced_dead, learned=lambda _bv, _bx, _by: logp)
                first = {seeds[0]: views["proba"]}
                for s in seeds[1:]:
                    first[s], _ = playerview.cross_fitted_probabilities(views["X"], views["pseudo"], seed=s, verbose=False)
                first_mean = np.mean([first[s] for s in seeds], axis=0)

                y2 = second_round_labels(views, first[seeds[0]], forced_dead)
                second = {s: playerview.cross_fitted_probabilities(views["X"], y2, seed=s, verbose=False)[0] for s in seeds}
                second_mean = np.mean([second[s] for s in seeds], axis=0)

                for name, proba in zip(order, (first[seeds[0]], first_mean, second[seeds[0]], second_mean)):
                    codes = bench.to_pm(pipeline.final_pass(dict(views, proba=proba), forced_dead))
                    B.score(overall, cells, name, codes, truths, game["truth"], game["halves"])
            graded = np.zeros(game["n_frames"], dtype=bool)
            for a, b in game["halves"]:
                graded[a:b] = True
            print("  done {} | {} | pseudo-labelled frames: round 1 {:.0%}, round 2 {:.0%} of the match".format(
                game["name"], scenario[0], np.mean(views["pseudo"][graded] >= 0), np.mean(y2[graded] >= 0)), flush=True)
    B.report("LEAVE ONE MATCH OUT ({} cells)".format(len(next(iter(cells.values())))), order, overall, cells)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    run(args.targets or dfl.available(), args.seeds)
