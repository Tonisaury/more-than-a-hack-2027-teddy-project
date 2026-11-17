"""End to end: provided data in, submission CSV out.

    python solution/run.py                      # writes evaluation/results/possession_two_views.csv
    python solution/run.py --name my_team       # ... evaluation/results/my_team.csv

Deterministic: the same data produces the same file. Takes a few minutes, a good part of it
parsing the two tracking files the first time (they are cached afterwards, next to the data and
never inside the repository).

Stages (each is one module in solution/mtah/):
  1. data.py        parse tracking, tagging and homography
  2. sync.py        measure the broadcast -> tactical time mapping, borrow the broadcast ball
  3. prep.py        clean the ball track; "ball view" evidence for every frame
  4. model.py       sticky HMM on the ball view alone  -> confident pseudo-labels
  5. playerview.py  classifier that never sees the ball, trained on those pseudo-labels
  6. model.py       sticky HMM on both views together  -> the submission
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from mtah import ballmodel, data as D, model, playerview, prep, refine, sync  # noqa: E402
import load_homography  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
BALL_MODEL = os.path.join(HERE, "models", "ball_view.pkl")   # written by ballmodel_eval.py --train-final


def learned_ball_view(path=BALL_MODEL):
    """callable(bv, bx, by) -> (n,) home-against-away log-odds from the classifier trained on other matches, or None."""
    if not model.LEARNED_BALL_VIEW:
        return None
    import pickle
    with open(path, "rb") as handle:
        stored = pickle.load(handle)

    def apply(bv, bx, by):
        X, names = ballmodel.features(bv, bx, by)
        if names != stored["names"]:
            raise ValueError("ball_view.pkl was trained on other features; retrain it (ballmodel_eval.py --train-final)")
        return ballmodel.log_odds(stored["clf"], X, names, bv["has"])
    return apply


def kickoff_dead_mask(tags):
    """Frames between the start of a graded half and its tagged kick-off: the ball is not in play."""
    mask = np.zeros(D.N_FRAMES, dtype=bool)
    starts = [t["start_frame"] for t in tags
              if t["code"] == "SET PIECES" and t["type"] == "kick off (start)"]
    for first, stop in D.HALVES:
        inside = [s for s in starts if first < s < stop]
        if inside:
            mask[first:min(inside)] = True
    return mask


def fused_ball(tac, bro_x, bro_y, use_broadcast=True):
    bx, by = prep.clean_ball(tac["bx"], tac["by"])
    if use_broadcast:
        missing = np.isnan(bx)
        bx[missing] = bro_x[missing]
        by[missing] = bro_y[missing]
        bx, by = prep.clean_ball(bx, by)
    return prep.interpolate_short_gaps(bx, by, prep.INTERP_MAX_GAP)


def two_views(tac, bx, by, forced_dead, tags=(), tac_h=None, tac_valid=None, bc=None, log=lambda *a: None, learned=None):
    """Stages 3-5 for any match: the ball view, the first pass that teaches, and the player view.

    Works without cameras and tagging (stand-in matches); the features then shrink to team shapes.
    """
    bv = prep.ball_view(tac, bx, by)
    if learned is not None:                       # callable(bv, bx, by) -> (n,) log-odds, see ballmodel.py
        bv["learned"] = learned(bv, bx, by)
    ball_e = model.ball_emissions(bv)
    log("4/6 first pass: ball view only")
    first_codes, post = model.decode(ball_e, forced_dead, want_posteriors=True)
    log("5/6 player view (no ball): features, pseudo-labels, cross-fitted classifier")
    sign = prep.attack_signs(tac)
    X, names = playerview.build_features(tac, sign, bv["pvx"], bv["pvy"], tags, tac_h, tac_valid, bc)
    tv = X[:, names.index("bc_live_6s")] if "bc_live_6s" in names else None
    y = playerview.pseudo_labels(post, bv, tags, forced_dead, model.CONFIDENT, X[:, names.index("all_speed_6s")], tv)
    proba, report = playerview.cross_fitted_probabilities(X, y, verbose=log is print)
    return {"bv": bv, "ball_e": ball_e, "proba": proba, "pseudo": y, "X": X, "names": names, "report": report,
            "first_codes": first_codes, "bx": bx, "by": by, "tac": tac, "sign": sign}


def final_pass(views, forced_dead, cost_seen=None, cost_gap=None, sharpen=None, weight=None, discount=None, **refine_options):
    """Stage 6: both views through the sticky HMM, then the boundary refinement.

    cost_seen / cost_gap are the evidence needed for a change of team where the ball is in view and
    where it is not; sharpen switches the refinement on. Defaults come from model.py.
    """
    bv = views["bv"]
    cost_seen = model.TEAM_COST_BALL_SEEN if cost_seen is None else cost_seen
    cost_gap = model.SWITCH_COST if cost_gap is None else cost_gap
    sharpen = model.SHARPEN if sharpen is None else sharpen
    team_cost = np.where(bv["has"], float(cost_seen), float(cost_gap))
    codes = model.decode(views["ball_e"] + model.player_emissions(views["proba"], bv["has"], weight, discount), forced_dead, team_cost=team_cost)
    return refine.refine(codes, bv, **refine_options) if sharpen else codes


def run(verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    log("1/6 loading")
    tac = D.load_tracking()
    bro = D.load_tracking(source="broadcast")
    tags = D.load_smart_tagging()
    tac_h, _, tac_valid = load_homography.load()
    bro_h, _, bro_valid = load_homography.load(source="broadcast")

    log("2/6 synchronising the broadcast recording")
    bro_x, bro_y, sync_report, mapping = sync.broadcast_ball_on_tactical_timeline(tac, bro, verbose)

    log("3/6 ball view")
    bx, by, observed = fused_ball(tac, bro_x, bro_y)
    forced_dead = kickoff_dead_mask(tags)
    bc = playerview.broadcast_on_tactical(mapping, bro, bro_h, bro_valid)
    views = two_views(tac, bx, by, forced_dead, tags, tac_h, tac_valid, bc, log, learned=learned_ball_view())
    bv, proba, y, X, names, pv_report, first_codes, sign = (
        views[k] for k in ("bv", "proba", "pseudo", "X", "names", "report", "first_codes", "sign"))

    log("6/6 final pass: both views")
    codes = final_pass(views, forced_dead)
    player_only = model.decode(model.player_emissions(proba), forced_dead)

    graded = codes >= 0
    diagnostics = {
        "sync": sync_report,
        "player_view": pv_report,
        "n_features": len(names),
        "ball": {
            "tactical_visible_graded": float(np.mean(~np.isnan(tac["bx"][graded]))),
            "after_broadcast_graded": float(np.mean(observed[graded])),
            "after_interpolation_graded": float(np.mean(bv["has"][graded])),
        },
    }
    arrays = {"final": codes, "ball_only": first_codes, "player_only": player_only,
              "proba": proba.astype(np.float32), "ball_has": bv["has"], "ball_observed": observed,
              "tactical_ball": ~np.isnan(tac["bx"]), "pseudo": y, "bc_live": bc["bc_live"],
              "all_speed": X[:, names.index("all_speed")],
              # evidence lanes for the explanation workspace (workspace/); they never leave this machine
              "control_net": (bv["control_home"] - bv["control_away"]).astype(np.float32),
              "ball_out": bv["out"], "ball_still": bv["still"],
              "ball_x": bx.astype(np.float32), "ball_y": by.astype(np.float32),
              "attack_sign": sign.astype(np.int8), "forced_dead": forced_dead}
    return codes, diagnostics, arrays


def write_submission(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the possession submission.")
    parser.add_argument("--name", default="possession_two_views",
                        help="file stem (use your team name for the real submission)")
    parser.add_argument("--out", default=RESULTS_DIR)
    parser.add_argument("--diagnostics", default=None, help="folder for diagnostics (json + npz)")
    args = parser.parse_args()

    codes, diagnostics, arrays = run()
    path = write_submission(os.path.join(args.out, args.name + ".csv"), model.to_labels(codes))
    print("\nwrote", os.path.relpath(path, ROOT))
    if args.diagnostics:
        os.makedirs(args.diagnostics, exist_ok=True)
        with open(os.path.join(args.diagnostics, "diagnostics.json"), "w") as handle:
            json.dump(diagnostics, handle, indent=2)
        np.savez_compressed(os.path.join(args.diagnostics, "arrays.npz"), **arrays)
        for key in ("ball_only", "player_only"):
            write_submission(os.path.join(args.diagnostics, key + ".csv"),
                             model.to_labels(arrays[key]))
    print("now:  python evaluation/validate_submission.py", os.path.relpath(path, ROOT))
