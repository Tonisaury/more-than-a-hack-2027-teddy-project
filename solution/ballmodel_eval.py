"""Does a learned ball view (mtah/ballmodel.py) beat the hand-written one? Measured on matches it never saw.

    python solution/ballmodel_eval.py                         # every DFL match, each scored by a model trained on the others
    python solution/ballmodel_eval.py --targets J03WMX J03WN1
    python solution/ballmodel_eval.py --metrica               # trained on all DFL matches, scored on Metrica's sample games
    python solution/ballmodel_eval.py --train-final OUT.pkl   # the model that would ship: trained on every DFL match

Leave-one-match-out: for each DFL match, the classifier is trained on the other matches (both
degradation scenarios of dfl_eval.py) and then used on that match, so no score below comes from a
model that saw the match it is scored on. The Metrica sample games are a second, stricter test:
another provider's data, never used for training at all.

Everything downstream of the ball view is the shipped pipeline: the first pass that makes
pseudo-labels, the player view trained on them, the sticky decoder.
"""

import argparse
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))

import possession_metrics as pm  # noqa: E402
from mtah import ballmodel, dfl, model, prep, standin  # noqa: E402
import dfl_eval as E  # noqa: E402
import run as pipeline  # noqa: E402
import standin_eval as bench  # noqa: E402

ORACLE = bench.ORACLE
K_GRID = (0.25, 0.5, 1.0)
COST_GRID = (2, 4, 8, 12)


def cell_key(match, scenario):
    return "{}_{}_{}_{}".format(match, scenario[1], scenario[2], E.BLIND_STOPPAGES)


def build_cell(match, scenario, masks, cache_dir):
    """Features, labels and the degraded ball of one match and scenario (cached)."""
    path = os.path.join(cache_dir, "ballfeat2_{}.npz".format(cell_key(match, scenario)))
    game = dfl.load_game(match)
    seed = list(dfl.FILES).index(match)
    truth = bench.controlled(game["truth"], game["halves"])
    if os.path.isfile(path):
        s = np.load(path)
        return game, truth, {k: s[k] for k in s.files}
    with standin.match_shape(game):
        bx, by = E.degraded_ball(game, masks, scenario, seed)
        bv = prep.ball_view(game["tac"], bx, by)
        X, names = ballmodel.features(bv, bx, by)
        Xt, yt = ballmodel.training_rows(X, names, truth, bv["has"], game["halves"])
    cell = {"X": X, "has": bv["has"], "Xt": Xt, "yt": yt, "names": np.array(names)}
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(path, **cell)
    return game, truth, cell


def train_on(matches, masks, cache_dir, verbose=True):
    Xs, ys, names = [], [], None
    for match in matches:
        for scenario in E.SCENARIOS:
            _game, _truth, cell = build_cell(match, scenario, masks, cache_dir)
            Xs.append(cell["Xt"])
            ys.append(cell["yt"])
            names = list(cell["names"])
    X, y = np.concatenate(Xs), np.concatenate(ys)
    if verbose:
        print("  training on {} matches, {} rows (H/A {})".format(len(matches), len(y), np.bincount(y, minlength=2).tolist()), flush=True)
    return ballmodel.fit(X, y), names


def views_with(game, bx, by, logp, path):
    """The shipped two views, with the learned log-probabilities in place of the hand-written ball evidence."""
    forced_dead = np.zeros(game["n_frames"], dtype=bool)
    bv = prep.ball_view(game["tac"], bx, by)
    bv["learned"] = logp
    if os.path.isfile(path):
        stored = np.load(path)
        proba, agreement = stored["proba"], float(stored["agreement"])
    else:
        views = pipeline.two_views(game["tac"], bx, by, forced_dead, learned=lambda _bv, _bx, _by: logp)
        proba, agreement = views["proba"], views["report"]["heldout_agreement"]
        np.savez_compressed(path, proba=proba.astype(np.float32), agreement=agreement)
    return {"bv": bv, "proba": proba, "bx": bx, "by": by, "tac": game["tac"], "agreement": agreement}, forced_dead


def learned_pass(views, forced_dead, k=None, cost_seen=None, dead_weight=None, **options):
    saved = model.K_LEARNED, model.LEARNED_DEAD_WEIGHT
    try:
        if k is not None:
            model.K_LEARNED = k
        if dead_weight is not None:
            model.LEARNED_DEAD_WEIGHT = dead_weight
        alt = dict(views, ball_e=model.ball_emissions(views["bv"]))
        return pipeline.final_pass(alt, forced_dead, cost_seen, **options)
    finally:
        model.K_LEARNED, model.LEARNED_DEAD_WEIGHT = saved


def variants():
    out = {"shipped two views": ("shipped", lambda v, fd: pipeline.final_pass(v, fd))}
    for k in K_GRID:
        for cost in COST_GRID:
            out["learned ball view k {} cost {}".format(k, cost)] = ("learned", lambda v, fd, k=k, c=cost: learned_pass(v, fd, k, c))
    # how much of the learned evidence also counts against "dead ball" (model.LEARNED_DEAD_WEIGHT). Final pass only:
    # the cached player view was made with whatever weight was in force when the cache was written
    for w in (1.0, 0.6, 0.36):
        out["learned k 0.5 cost 4 dead weight {}".format(w)] = ("learned", lambda v, fd, w=w: learned_pass(v, fd, 0.5, 4, dead_weight=w))
    out["learned k 0.5 cost 4 + sharpen team"] = ("learned", lambda v, fd: learned_pass(
        v, fd, 0.5, 4, sharpen=True, team=True, restarts=False, exits=False))
    out["shipped two views " + ORACLE] = ("shipped", lambda v, fd: pipeline.final_pass(v, fd))
    out["learned ball view k 0.5 cost 4 " + ORACLE] = ("learned", lambda v, fd: learned_pass(v, fd, 0.5, 4))
    return out


def score(overall, cells, name, codes, truths, raw_truth, halves):
    row = overall.setdefault(name, {"loose": [], "controlled": [], "edges": []})
    for kind, truth in truths.items():
        r = pm.evaluate(truth, codes, bootstrap_n=0)
        row[kind].append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"],
                          r["per_class"]["D"]["f1"], r["spell_f1"][0.7]["f1"]))
    into, out = E.edge_errors(raw_truth, codes, halves)
    dead = 2
    row["edges"].append(into + out + (sum(1 for _a, _b, k in pm.segments(codes, halves) if k == dead),
                                      sum(1 for _a, _b, k in pm.segments(raw_truth, halves) if k == dead)))
    cells.setdefault(name, []).append((row["loose"][-1][0], row["controlled"][-1][0]))


def report(title, names, overall, cells):
    E.table(title, names, overall)
    ref = np.array(cells[names[0]])
    print("\ncells where a variant beats '{}' (loose / controlled truth), and the mean difference:".format(names[0]))
    for name in names[1:]:
        c = np.array(cells[name])
        print("  {:<46} {}/{}  {}/{}   {:+.4f} / {:+.4f}".format(
            name, int((c[:, 0] > ref[:, 0]).sum()), len(c), int((c[:, 1] > ref[:, 1]).sum()), len(c),
            (c[:, 0] - ref[:, 0]).mean(), (c[:, 1] - ref[:, 1]).mean()))


def leave_one_out(targets):
    everything = dfl.available()
    cache_dir = os.path.join(dfl.folder(), "cache")
    masks = standin.challenge_gap_mask()
    V = variants()
    overall, cells = {}, {}
    for match in targets:
        clf, names = train_on([m for m in everything if m != match], masks, cache_dir)
        for scenario in E.SCENARIOS:
            game, _controlled, cell = build_cell(match, scenario, masks, cache_dir)
            truths = {"loose": game["truth"], "controlled": bench.controlled(game["truth"], game["halves"])}
            seed = list(dfl.FILES).index(match)
            with standin.match_shape(game):
                logp = ballmodel.log_odds(clf, cell["X"], names, cell["has"])
                shipped, forced_dead = E.prepare(game, masks, scenario, cache_dir, seed)
                learned, _ = views_with(game, shipped["bx"], shipped["by"], logp,
                                        os.path.join(cache_dir, "views_learned2_{}_n{}.npz".format(cell_key(match, scenario), len(everything))))
                oracle = game["truth"] == dfl.DEAD
                for name, (which, fn) in V.items():
                    codes = bench.to_pm(fn(shipped if which == "shipped" else learned, oracle if name.endswith(ORACLE) else forced_dead))
                    score(overall, cells, name, codes, truths, game["truth"], game["halves"])
            held = cell["has"] & (truths["controlled"] <= 1)
            acc = float(np.mean((logp[held] < 0).astype(np.int8) == truths["controlled"][held]))
            print("  done {} | {} | learned ball view names the right side on {:.3f} of live frames with a ball | player-view agreement {:.3f} (shipped {:.3f})".format(
                game["name"], scenario[0], acc, learned["agreement"], shipped["agreement"]), flush=True)
    report("LEAVE ONE MATCH OUT ({} cells)".format(len(next(iter(cells.values())))), list(V), overall, cells)


def metrica():
    """Trained on every DFL match; scored on Metrica's sample games with stoppages given, as standin_eval.py does."""
    cache_dir = os.path.join(dfl.folder(), "cache")
    masks = standin.challenge_gap_mask()
    clf, names = train_on(dfl.available(), masks, cache_dir)
    root = standin.fetch()
    standin_cache = os.path.join(root, "cache")
    V = {name: v for name, v in variants().items() if name.endswith(ORACLE) or "cost" in name or name == "shipped two views"}
    V = {(name if name.endswith(ORACLE) else name + " " + ORACLE): v for name, v in V.items()}
    overall, cells = {}, {}
    for n in standin.GAMES:
        game = standin.load_game(n)
        truths = {"loose": game["truth"], "controlled": bench.controlled(game["truth"], game["halves"])}
        with standin.match_shape(game):
            for scenario in bench.SCENARIOS:
                shipped, _fd = bench.prepare(game, masks, scenario, standin_cache)
                bv = shipped["bv"]
                X, feature_names = ballmodel.features(bv, shipped["bx"], shipped["by"])
                assert feature_names == names
                logp = ballmodel.log_odds(clf, X, names, bv["has"])
                key = "g{}_{}_{}_{}".format(n, "own" if scenario[1] is None else scenario[1], scenario[2], scenario[3])
                learned, _ = views_with(game, shipped["bx"], shipped["by"], logp,
                                        os.path.join(standin_cache, "views_learned2_{}_n{}.npz".format(key, len(dfl.available()))))
                oracle = truths["loose"] == 2
                for name, (which, fn) in V.items():
                    codes = bench.to_pm(fn(shipped if which == "shipped" else learned, oracle))
                    score(overall, cells, name, codes, truths, game["truth"], game["halves"])
                print("  done", game["name"], "|", scenario[0], flush=True)
    report("METRICA SAMPLE GAMES, stoppages given ({} cells)".format(len(next(iter(cells.values())))), list(V), overall, cells)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--targets", nargs="+", default=None, help="DFL matches to score (default: all downloaded)")
    parser.add_argument("--metrica", action="store_true", help="train on all DFL matches, score on Metrica's sample games")
    parser.add_argument("--train-final", default=None, help="write the classifier trained on every DFL match to this file")
    args = parser.parse_args()
    if args.train_final:
        dfl.fetch(list(dfl.FILES))                # the seven public matches, once (about 2.6 GB)
        clf, names = train_on(dfl.available(), standin.challenge_gap_mask(), os.path.join(dfl.folder(), "cache"))
        with open(args.train_final, "wb") as handle:
            pickle.dump({"clf": clf, "names": names}, handle)
        print("wrote", args.train_final)
    elif args.metrica:
        metrica()
    else:
        leave_one_out(args.targets or dfl.available())
