"""How much does the answer depend on the classifier's random seed? Aggregate scores only.

    python solution/seed_stability.py [--seeds 0 1 2 3 4]

The player view's gradient-boosted classifier holds out a random tenth of its training frames for
early stopping, so its seed changes the model. This script trains it with several seeds on the
same features and pseudo-labels, decodes each, and scores the decoded sequences AGAINST EACH OTHER
with the organisers' metric (no ground truth involved): if two seeds only agree at spellF1 0.9,
a tenth of the possessions are decided by the seed. It then decodes the average of the seeds'
probabilities and reports how close that sits to every single seed.
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
from mtah import data as D, playerview, sync  # noqa: E402
import load_homography  # noqa: E402
import run as pipeline  # noqa: E402


def to_pm(codes):
    out = codes.astype(np.int8).copy()
    out[out < 0] = 3
    return out


def agreement(a, b):
    r = pm.evaluate(to_pm(a), to_pm(b), bootstrap_n=0)
    return r["spell_f1_mean"], r["macro_f1"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--save", default=None, help="optional .npz for the per-seed probabilities (keep it next to the data)")
    args = parser.parse_args()

    tac = D.load_tracking()
    bro = D.load_tracking(source="broadcast")
    tags = D.load_smart_tagging()
    tac_h, _, tac_valid = load_homography.load()
    bro_h, _, bro_valid = load_homography.load(source="broadcast")
    bro_x, bro_y, _report, mapping = sync.broadcast_ball_on_tactical_timeline(tac, bro, False)
    bx, by, _observed = pipeline.fused_ball(tac, bro_x, bro_y)
    forced_dead = pipeline.kickoff_dead_mask(tags)
    bc = playerview.broadcast_on_tactical(mapping, bro, bro_h, bro_valid)
    views = pipeline.two_views(tac, bx, by, forced_dead, tags, tac_h, tac_valid, bc)

    probas, decoded = {}, {}
    for seed in args.seeds:
        if seed == 0:
            probas[seed] = views["proba"]
        else:
            probas[seed], _ = playerview.cross_fitted_probabilities(views["X"], views["pseudo"], seed=seed, verbose=False)
        decoded[seed] = pipeline.final_pass(dict(views, proba=probas[seed]), forced_dead)
        spells = pm.possession_spells(to_pm(decoded[seed]), D.HALVES)
        print("seed {}: {} possessions".format(seed, len(spells)), flush=True)

    print("\nseed against seed (spellF1 / macroF1 of one decoded sequence scored against the other):")
    pairs = []
    for i, a in enumerate(args.seeds):
        for b in args.seeds[i + 1:]:
            s, m = agreement(decoded[a], decoded[b])
            pairs.append((s, m))
            print("  {} vs {}: {:.3f} / {:.3f}".format(a, b, s, m))
    pairs = np.array(pairs)
    print("  mean {:.3f} / {:.3f}   min {:.3f}".format(pairs[:, 0].mean(), pairs[:, 1].mean(), pairs[:, 0].min()))

    mean_proba = np.mean([probas[s] for s in args.seeds], axis=0)
    averaged = pipeline.final_pass(dict(views, proba=mean_proba), forced_dead)
    print("\naverage of {} seeds: {} possessions".format(len(args.seeds), len(pm.possession_spells(to_pm(averaged), D.HALVES))))
    rows = np.array([agreement(decoded[s], averaged) for s in args.seeds])
    print("  the average against each single seed: mean {:.3f} / {:.3f}   min {:.3f}".format(
        rows[:, 0].mean(), rows[:, 1].mean(), rows[:, 0].min()))
    if args.save:
        np.savez_compressed(args.save, averaged=averaged, **{"proba_{}".format(s): probas[s].astype(np.float32) for s in args.seeds},
                            **{"codes_{}".format(s): decoded[s] for s in args.seeds})
