"""Score decoder variants on the DFL open matches, where the answer says when play is stopped.

    python solution/dfl_eval.py                  # the shipped decoder, stoppages NOT given
    python solution/dfl_eval.py --matches J03WMX J03WN1

standin_eval.py measures the who-has-it part of the decoder on Metrica's sample games with the
stoppages handed over, because those games give the decoder nothing to find a stoppage with. The
DFL matches (mtah/dfl.py) carry a per-frame "ball in play" flag, so here the whole decoder runs
with nothing given, and the dead-ball class and the edges of stoppages are scored too.

Making the DFL ball look like ours (label-free statistics of the challenge match only):
  * while the ball is dead the DFL position is a frozen placeholder, on every frame. Left as it is
    it would hand over every stoppage. In the challenge match about half of the stoppages have the
    ball in the data for under a quarter of their length, so the ball is removed for the whole of
    a seeded random half of the stoppages; in the others the placeholder stays and plays the part
    of a ball seen lying out of play or placed for a restart.
  * the challenge tracker's own ball-gap pattern is laid over everything (as in standin_eval.py);
  * the ball in play gets the same slow positional wander as in standin_eval.py.

Two truths, as in standin_eval.py: the flag as published ("loose": a touch changes hands, median
possession 3 s) and with short spells folded into the surrounding side's play ("controlled").
The DFL possession flag is the provider's and is not documented in detail, so these scores compare
two versions of our decoder on the same data. They do not predict the score on the challenge match.
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
from mtah import data as D, dfl, model, prep, standin  # noqa: E402
import run as pipeline  # noqa: E402
import standin_eval as bench  # noqa: E402

# Params
GAP_SHIFT = 0.37              # where the challenge gap pattern is laid over each half (as in standin_eval.py)
BLIND_STOPPAGES = 0.5         # share of stoppages in which the ball is not in the data at all
SCENARIOS = (                 # (name, wander metres, wander seconds)
    ("wander 0.8 m / 1 s", 0.8, 1.0),
    ("wander 1.2 m / 2 s", 1.2, 2.0),
)
ORACLE = bench.ORACLE


def degraded_ball(game, masks, scenario, seed):
    _name, offset, seconds = scenario
    rng = np.random.RandomState(seed)
    bx, by = game["tac"]["bx"].copy(), game["tac"]["by"].copy()
    dead = game["ball_placeholder"]
    live = ~dead
    bx[live] += standin.smooth_noise(len(bx), offset, seconds, rng)[live]
    by[live] += standin.smooth_noise(len(by), offset, seconds, rng)[live]
    for a, b, k in pm.segments(game["truth"], game["halves"]):
        if k == dfl.DEAD and rng.rand() < BLIND_STOPPAGES:
            bx[a:b] = np.nan
            by[a:b] = np.nan
    for (first, stop), mask in zip(game["halves"], masks):
        seen = np.resize(np.roll(mask, int(GAP_SHIFT * len(mask))), stop - first)
        bx[first:stop][~seen] = np.nan
        by[first:stop][~seen] = np.nan
    bx, by = prep.clean_ball(bx, by)
    bx, by, _ = prep.interpolate_short_gaps(bx, by, prep.INTERP_MAX_GAP)
    return bx, by


def prepare(game, masks, scenario, cache_dir, seed):
    """Ball view and player view for one match and scenario; the classifier's output is cached."""
    path = os.path.join(cache_dir, "views_{}_{}_{}_{}.npz".format(game["name"][-6:], scenario[1], scenario[2], BLIND_STOPPAGES))
    bx, by = degraded_ball(game, masks, scenario, seed)
    forced_dead = np.zeros(game["n_frames"], dtype=bool)
    if os.path.isfile(path):
        stored = np.load(path)
        bv = prep.ball_view(game["tac"], bx, by)
        views = {"bv": bv, "ball_e": model.ball_emissions(bv), "proba": stored["proba"], "bx": bx, "by": by, "tac": game["tac"],
                 "agreement": float(stored["agreement"]), "pseudo": stored["pseudo"]}
    else:
        views = pipeline.two_views(game["tac"], bx, by, forced_dead)
        views["agreement"] = views["report"]["heldout_agreement"]
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(path, proba=views["proba"].astype(np.float32), agreement=views["agreement"], pseudo=views["pseudo"])
    return views, forced_dead


def edge_errors(truth, codes, halves):
    """Median |frames| from each true stoppage edge to the nearest predicted edge of the same kind, and the
    share within half a second. Returned for (live -> dead, dead -> live)."""
    out = []
    for into_dead in (True, False):
        gaps = []
        for first, stop in halves:
            def edges(c):
                c = c[first:stop]
                dead = c == dfl.DEAD
                change = np.flatnonzero(dead[1:] != dead[:-1]) + 1
                return change[dead[change] == into_dead]
            t, p = edges(truth), edges(codes)
            if len(p):
                gaps += [int(np.min(np.abs(p - e))) for e in t]
            else:
                gaps += [10 ** 6] * len(t)
        gaps = np.array(gaps)
        out.append((float(np.median(gaps)), float(np.mean(gaps <= 12)), float(np.mean(gaps <= 50))))
    return out


COLUMNS = "{:<46} {:>6} {:>6} {:>8}  {:>6} {:>6} {:>8}  {:>5} {:>5} {:>9}  {:>12} {:>12}"


def table(title, variants, rows):
    print("\n" + title)
    print(COLUMNS.format("variant", "spell", "macro", "n/true", "spell", "macro", "n/true", "D-F1", "IoU.7", "stoppages",
                         "into dead", "out of dead"))
    print(COLUMNS.format("", "(loose", "truth)", "", "(contr.", "truth)", "", "", "", "n/true", "med|<.5s|<2s", "med|<.5s|<2s"))
    for name in variants:
        r = rows[name]
        lo, co = np.array(r["loose"]), np.array(r["controlled"])
        e = np.array(r["edges"])
        print(COLUMNS.format(
            name, "{:.3f}".format(lo[:, 0].mean()), "{:.3f}".format(lo[:, 1].mean()), "{:.0f}/{:.0f}".format(lo[:, 2].mean(), lo[:, 3].mean()),
            "{:.3f}".format(co[:, 0].mean()), "{:.3f}".format(co[:, 1].mean()), "{:.0f}/{:.0f}".format(co[:, 2].mean(), co[:, 3].mean()),
            "{:.3f}".format(co[:, 4].mean()), "{:.3f}".format(co[:, 5].mean()), "{:.0f}/{:.0f}".format(e[:, 6].mean(), e[:, 7].mean()),
            "{:.0f}|{:.2f}|{:.2f}".format(np.median(e[:, 0]), e[:, 1].mean(), e[:, 2].mean()),
            "{:.0f}|{:.2f}|{:.2f}".format(np.median(e[:, 3]), e[:, 4].mean(), e[:, 5].mean())))


def evaluate(variants, matches=None, verbose=True):
    """{variant: {"loose": [...], "controlled": [...], "edges": [...]}} over every match x scenario, plus tables."""
    matches = matches or dfl.available()
    root = dfl.folder()
    cache_dir = os.path.join(root, "cache")
    masks = standin.challenge_gap_mask()
    overall, cells = {}, {}
    for index, match in enumerate(matches):
        game = dfl.load_game(match)
        truths = {"loose": game["truth"], "controlled": bench.controlled(game["truth"], game["halves"])}
        n_dead = sum(1 for _a, _b, k in pm.segments(game["truth"], game["halves"]) if k == dfl.DEAD)
        with standin.match_shape(game):
            for scenario in SCENARIOS:
                views, forced_dead = prepare(game, masks, scenario, cache_dir, seed=index)
                oracle = game["truth"] == dfl.DEAD
                for name, fn in variants.items():
                    codes = bench.to_pm(fn(views, oracle if name.endswith(ORACLE) else forced_dead))
                    row = overall.setdefault(name, {"loose": [], "controlled": [], "edges": []})
                    for kind, truth in truths.items():
                        r = pm.evaluate(truth, codes, bootstrap_n=0)
                        row[kind].append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"],
                                          r["per_class"]["D"]["f1"], r["spell_f1"][0.7]["f1"]))
                    into, out = edge_errors(game["truth"], codes, game["halves"])
                    pred_dead = sum(1 for _a, _b, k in pm.segments(codes, game["halves"]) if k == dfl.DEAD)
                    row["edges"].append(into + out + (pred_dead, n_dead))
                    cells.setdefault(name, []).append((row["loose"][-1][0], row["controlled"][-1][0]))
                if verbose:
                    pseudo = views["pseudo"]
                    print("  done {} | {} | player-view agreement {:.3f} | pseudo-labels H/A/D {}".format(
                        game["name"], scenario[0], views["agreement"],
                        [int(np.sum(pseudo == k)) for k in range(3)]), flush=True)
    if verbose:
        table("ALL MATCHES AND SCENARIOS ({} cells)".format(len(next(iter(cells.values())))), variants, overall)
        names = list(variants)
        ref = np.array(cells[names[0]])
        print("\ncells where a variant beats '{}' (loose / controlled truth), and the mean difference:".format(names[0]))
        for name in names[1:]:
            c = np.array(cells[name])
            print("  {:<46} {}/{}  {}/{}   {:+.4f} / {:+.4f}".format(
                name, int((c[:, 0] > ref[:, 0]).sum()), len(c), int((c[:, 1] > ref[:, 1]).sum()), len(c),
                (c[:, 0] - ref[:, 0]).mean(), (c[:, 1] - ref[:, 1]).mean()))
    return overall, cells


VARIANTS = {
    "shipped two views": lambda v, fd: pipeline.final_pass(v, fd),
    "baseline nearest+vote(2s)": lambda v, fd: bench.baseline(v, 50),
    "ball view only": lambda v, fd: model.decode(v["ball_e"], fd),
    "shipped two views " + ORACLE: lambda v, fd: pipeline.final_pass(v, fd),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matches", nargs="+", default=None, help="DFL match ids; default: every match already downloaded")
    parser.add_argument("--fetch", nargs="+", default=None, help="download these matches first (about 400 MB each)")
    args = parser.parse_args()
    if args.fetch:
        dfl.fetch(args.fetch)
    evaluate(VARIANTS, args.matches)
