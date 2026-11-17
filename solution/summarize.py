"""Turn a submission (and the baselines) into one small JSON of aggregate results.

    python solution/summarize.py evaluation/results/possession_two_views.csv --out summary.json

Only labels and scores go in -- nothing from the tracking data -- so the output is safe to share.
Scores are against the PUBLIC five minutes: 17 possessions, a pipeline check and not a ranking.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402

GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
RESULTS = os.path.join(ROOT, "evaluation", "results")
BASELINES = ("baseline_raw", "baseline_smooth25", "baseline_smooth50", "baseline_smooth125")
LETTERS = "HADX"
FPS = 25
BOOTSTRAP = 1000


def segment_list(codes, first, stop):
    window = codes[first:stop]
    breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(window)]))
    return [[int(first + bounds[i]), int(first + bounds[i + 1]), LETTERS[int(window[bounds[i]])]]
            for i in range(len(bounds) - 1)]


def score(gt, pred):
    r = pm.evaluate(gt, pred, bootstrap_n=BOOTSTRAP)
    return {
        "macro_f1": r["macro_f1"], "spell_f1": r["spell_f1_mean"],
        "spell_f1_ci": [r["spell_bootstrap_low"], r["spell_bootstrap_high"]],
        "macro_f1_ci": [r["bootstrap_low"], r["bootstrap_high"]],
        "spell_f1_by_iou": {str(k): v["f1"] for k, v in r["spell_f1"].items()},
        "n_pred_spells": r["n_pred_spells"], "n_gt_spells": r["n_gt_spells"],
        "mean_spell_s": r["mean_spell_duration_s"], "gt_mean_spell_s": r["gt_mean_spell_duration_s"],
        "alive_dead_f1": r["alive_dead_f1"], "team_acc": r["team_acc_both_alive"],
        "per_class_f1": {k: v["f1"] for k, v in r["per_class"].items()},
        "home_share": r["pred_home_share"], "gt_home_share": r["gt_home_share"],
        "confusion": r["confusion"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission")
    parser.add_argument("--extra", nargs="*", default=[], help="name=path of other label files to score")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    gt = pm.label_codes(pm.read_label_csv(GT_PUBLIC))
    public = pm.valid_runs(gt)[0]
    halves = [(first, last + 1) for _, first, last in half_video_blocks(GAME)]
    mine = pm.label_codes(pm.read_label_csv(args.submission))

    out = {"public_window": list(map(int, public)), "halves": halves, "fps": FPS, "scores": {},
           "public_segments": {"truth": segment_list(gt, *public), "ours": segment_list(mine, *public)}}
    out["scores"]["ours"] = score(gt, mine)
    for name in BASELINES:
        codes = pm.label_codes(pm.read_label_csv(os.path.join(RESULTS, name + ".csv")))
        out["scores"][name] = score(gt, codes)
        if name == "baseline_smooth50":
            out["public_segments"]["baseline"] = segment_list(codes, *public)
            out["baseline_full_match"] = pm.possession_stats(codes, halves)
    for item in args.extra:
        name, path = item.split("=", 1)
        out["scores"][name] = score(gt, pm.label_codes(pm.read_label_csv(path)))

    graded = np.zeros(len(mine), dtype=bool)
    for first, stop in halves:
        graded[first:stop] = True
    spells = pm.possession_spells(mine, halves)
    durations = [(b - a) / float(FPS) for a, b, _ in spells]
    dead = [(b - a) / float(FPS) for a, b, k in pm.segments(mine, halves) if k == pm.D]
    out["full_match"] = {
        "graded_frames": int(graded.sum()),
        "share": {LETTERS[k]: float(np.mean(mine[graded] == k)) for k in range(3)},
        "home_share_of_live": float(np.sum(mine[graded] == 0)) / float(np.sum(mine[graded] < 2)),
        "n_possessions": len(spells),
        "n_possessions_by_team": {"H": sum(1 for s in spells if s[2] == 0), "A": sum(1 for s in spells if s[2] == 1)},
        "mean_possession_s": float(np.mean(durations)), "median_possession_s": float(np.median(durations)),
        "possession_durations_s": [round(d, 1) for d in durations],
        "n_stoppages": len(dead), "mean_stoppage_s": float(np.mean(dead)),
        "live_minutes": float(np.sum(mine[graded] < 2)) / FPS / 60.0,
        "dead_minutes": float(np.sum(mine[graded] == 2)) / FPS / 60.0,
        "segments": [segment_list(mine, first, stop) for first, stop in halves],
    }
    with open(args.out, "w") as handle:
        json.dump(out, handle)
    print("wrote", args.out)
    print(json.dumps({k: {m: (round(v, 4) if isinstance(v, float) else v) for m, v in s.items()
                          if m in ("macro_f1", "spell_f1", "n_pred_spells", "mean_spell_s")}
                      for k, s in out["scores"].items()}, indent=1))
    print({k: v for k, v in out["full_match"].items() if k not in ("segments", "possession_durations_s")})
