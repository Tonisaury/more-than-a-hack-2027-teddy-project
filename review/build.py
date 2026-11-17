"""Build the data the review tool shows. Run once per match; the server only reads the result.

    python review/build.py                 # the challenge match and both practice matches
    python review/build.py --only bar-ath  # or metrica-1 / metrica-2

Everything is written next to the match data (<MTAH_DATA_DIR>/_cache/review/<match>/), never into
the repository: a bundle of per-frame arrays for the browser (bundle.bin + meta.json) and the list
of doubtful moments (moments.json).

Two kinds of match:

  bar-ath              the challenge match. WATCH ONLY. The rules forbid hand labels on it, in any
                       amount and for any purpose, including validation -- so the tool shows the
                       video, what the model says and where it is unsure, and stores no labels.
  metrica-1, metrica-2 Metrica Sports' open sample games (see solution/mtah/standin.py). Hand labels
                       on other games are allowed, so these can be labelled; the model shown is the
                       same decoder run on a ball degraded to look like our tracking, with stoppages
                       given from the event data (the stand-in cannot test stoppage detection).
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
from mtah import data as D, hmm, model, prep, standin, sync  # noqa: E402
import run as pipeline  # noqa: E402
import standin_eval  # noqa: E402

FPS = D.FPS
X = 3

# What counts as doubtful (this tunes the review queue, not the model)
W_LOW_CONFIDENCE = 0.55     # 1 - (best posterior - second best)
W_VERSIONS = 0.30           # the shipped decoder and the previous one disagree
W_VIEWS = 0.20              # ball view and player view disagree while the ball is in view
W_SHORT = 0.25              # a possession under SHORT_SECONDS
W_BLIND = 0.10              # live play called with the ball out of the data for BLIND_SECONDS or more
SHORT_SECONDS = 2.0
BLIND_SECONDS = 3.0
THRESHOLD = 0.30            # smoothed score at which a stretch becomes a "moment"
MERGE_GAP = 2.0             # seconds: moments closer than this are one moment
MIN_MOMENT = 0.4            # seconds
LEAD_IN, LEAD_OUT = 3.0, 2.0  # seconds of context played around a moment
REASONS = {1: "low confidence", 2: "views disagree", 4: "versions disagree", 8: "very short possession", 16: "ball not in the data"}


def review_dir(match_id):
    return os.path.join(D.DATA_DIR, "_cache", "review", match_id)


def to_codes(codes):
    out = np.asarray(codes).astype(np.int16).copy()
    out[(out < 0) | (out > 2)] = X
    return out.astype(np.uint8)


def runs(mask):
    """[(start, stop)] of the True stretches."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def run_length(codes):
    """For every frame, the length of the constant-label run it sits in."""
    breaks = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(codes)]))
    out = np.zeros(len(codes), dtype=np.int32)
    for a, b in zip(bounds[:-1], bounds[1:]):
        out[a:b] = b - a
    return out


def posteriors(emissions, team_cost, forced_dead, halves):
    """Forward-backward with the same per-frame team-change cost the decoder uses. (n, 3), 0 outside."""
    base = hmm.transition_matrix(model.SWITCH_COST, model.SWITCH_COST, model.SWITCH_COST)
    post = np.zeros((len(emissions), 3))
    for first, stop in halves:
        e = emissions[first:stop].copy()
        e[forced_dead[first:stop], hmm.H] -= 1e3
        e[forced_dead[first:stop], hmm.A] -= 1e3
        tc = -np.asarray(team_cost[first:stop], dtype=float)
        n = len(e)
        fwd, bwd = np.zeros((n, 3)), np.zeros((n, 3))
        fwd[0] = e[0]
        t = base.copy()
        for i in range(1, n):
            t[0, 1] = t[1, 0] = tc[i]
            m = fwd[i - 1][:, None] + t
            top = m.max(axis=0)
            fwd[i] = top + np.log(np.exp(m - top).sum(axis=0)) + e[i]
        for i in range(n - 2, -1, -1):
            t[0, 1] = t[1, 0] = tc[i + 1]
            m = t + (bwd[i + 1] + e[i + 1])[None, :]
            top = m.max(axis=1)
            bwd[i] = top + np.log(np.exp(m - top[:, None]).sum(axis=1))
        lp = fwd + bwd
        lp -= lp.max(axis=1, keepdims=True)
        p = np.exp(lp)
        post[first:stop] = p / p.sum(axis=1, keepdims=True)
    return post


def doubt_and_moments(final, prev, ball_only, player_only, post, seen, halves):
    """Per-frame doubt (0-1), reason bits, and the ranked list of doubtful moments."""
    n = len(final)
    graded = np.zeros(n, dtype=bool)
    for first, stop in halves:
        graded[first:stop] = True
    ordered = np.sort(post, axis=1)
    low = np.where(graded, 1.0 - (ordered[:, 2] - ordered[:, 1]), 0.0)
    versions = graded & (final != prev)
    views = graded & seen & (ball_only != player_only) & (ball_only < X) & (player_only < X)
    length = run_length(final)
    short = graded & (final < 2) & (length < SHORT_SECONDS * FPS)
    blind = np.zeros(n, dtype=bool)
    for a, b in runs(graded & ~seen):
        if b - a >= BLIND_SECONDS * FPS:
            blind[a:b] = True
    blind &= final < 2
    score = np.clip(W_LOW_CONFIDENCE * low + W_VERSIONS * versions + W_VIEWS * views + W_SHORT * short + W_BLIND * blind, 0, 1)
    score[~graded] = 0
    box = np.ones(FPS) / FPS
    smooth = np.zeros(n)
    for first, stop in halves:
        smooth[first:stop] = np.convolve(score[first:stop], box, mode="same")
    bits = ((low > 0.4) * 1 + views * 2 + versions * 4 + short * 8 + blind * 16).astype(np.uint8)

    # every frame where the two versions disagree, and every very short possession, is in a moment
    # however brief: those are exactly the calls the lower switching cost added
    hot = (smooth >= THRESHOLD) | versions | short
    flagged = versions | short
    stretches = []
    for a, b in runs(hot):
        if stretches and a - stretches[-1][1] <= MERGE_GAP * FPS and half_of(a, halves) == half_of(stretches[-1][0], halves):
            stretches[-1][1] = b
        else:
            stretches.append([a, b])
    moments = []
    for a, b in stretches:
        if b - a < MIN_MOMENT * FPS and not flagged[a:b].any():
            continue
        first, stop = halves[half_of(a, halves)]
        why = [name for bit, name in REASONS.items() if np.mean((bits[a:b] & bit) > 0) >= 0.10]
        moments.append({
            "start": int(a), "end": int(b),
            "play_start": int(max(first, a - LEAD_IN * FPS)), "play_end": int(min(stop, b + LEAD_OUT * FPS)),
            "peak": round(float(smooth[a:b].max()), 3), "weight": round(float(smooth[a:b].sum() / FPS), 2),
            "reasons": why or ["low confidence"],
            "says": sequence(final[a:b]), "said_before": sequence(prev[a:b]),
        })
    order = sorted(range(len(moments)), key=lambda i: -moments[i]["weight"])
    for rank, i in enumerate(order):
        moments[i]["rank"] = rank + 1
    for i, m in enumerate(moments):
        m["id"] = i + 1
    return smooth, bits, moments


def half_of(frame, halves):
    for k, (first, stop) in enumerate(halves):
        if first <= frame < stop:
            return k
    return 0


def sequence(codes):
    """'HAH' for a stretch that goes home, away, home."""
    keep = np.concatenate(([True], codes[1:] != codes[:-1]))
    return "".join("HADX"[c] for c in codes[keep])


def centimetres(values):
    out = np.round(np.asarray(values, dtype=np.float64) * 100.0)
    out[~np.isfinite(out)] = -32768
    return np.clip(out, -32768, 32767).astype(np.int16)


def write_bundle(match_id, meta, arrays, moments):
    folder = review_dir(match_id)
    os.makedirs(folder, exist_ok=True)
    index, offset = [], 0
    with open(os.path.join(folder, "bundle.bin"), "wb") as handle:
        for name, array in arrays.items():
            array = np.ascontiguousarray(array)
            kind = {"uint8": "u1", "int8": "i1", "int16": "i2", "int32": "i4", "float32": "f4"}[array.dtype.name]
            pad = (-offset) % 8                      # typed arrays need aligned offsets
            handle.write(b"\0" * pad)
            offset += pad
            data = array.tobytes()
            handle.write(data)
            index.append({"name": name, "dtype": kind, "shape": list(array.shape), "offset": offset, "bytes": len(data)})
            offset += len(data)
    meta = dict(meta, id=match_id, fps=FPS, arrays=index, n_moments=len(moments))
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle)
    with open(os.path.join(folder, "moments.json"), "w", encoding="utf-8") as handle:
        json.dump(moments, handle)
    return folder, offset


def summary(match_id, final, prev, doubt, moments, halves, post):
    graded = np.zeros(len(final), dtype=bool)
    for first, stop in halves:
        graded[first:stop] = True
    covered = sum(m["end"] - m["start"] for m in moments)
    by_reason = {}
    for m in moments:
        for r in m["reasons"]:
            by_reason[r] = by_reason.get(r, 0) + 1
    print("{}: {} moments covering {:.1f}% of graded time ({:.1f} min); median length {:.1f} s; reasons {}".format(
        match_id, len(moments), 100.0 * covered / graded.sum(), covered / FPS / 60,
        float(np.median([m["end"] - m["start"] for m in moments])) / FPS if moments else 0, by_reason))
    print("   versions differ on {:.1f}% of graded frames; posterior argmax equals the decoded label on {:.1f}%".format(
        100.0 * np.mean(final[graded] != prev[graded]), 100.0 * np.mean(post[graded].argmax(1) == final[graded])))


# ---------------------------------------------------------------- the challenge match (watch only)

def build_challenge(submission, previous):
    import load_homography
    tac = D.load_tracking()
    final = pm.label_codes(pm.read_label_csv(submission)).astype(np.uint8)
    diag_path = os.path.join(D.DATA_DIR, "_cache", "workspace", "arrays.npz")
    arrays = dict(np.load(diag_path)) if os.path.isfile(diag_path) else None
    if arrays is None or not np.array_equal(to_codes(arrays["final"]), final):
        print("bar-ath: stored diagnostics do not match the submission; running the pipeline (a few minutes)")
        codes, _diagnostics, arrays = pipeline.run(verbose=False)
        if not np.array_equal(to_codes(codes), final):
            raise SystemExit("the pipeline no longer reproduces {}; rebuild the submission first".format(submission))
    has_prev = bool(previous and os.path.isfile(previous))
    prev = pm.label_codes(pm.read_label_csv(previous)).astype(np.uint8) if has_prev else final.copy()

    bx, by = arrays["ball_x"].astype(np.float64), arrays["ball_y"].astype(np.float64)
    bv = prep.ball_view(tac, bx, by)
    learned = pipeline.learned_ball_view()          # the same reading of who is on the ball that run.py used
    if learned is not None:
        bv["learned"] = learned(bv, bx, by)
    forced_dead = arrays["forced_dead"].astype(bool)
    emissions = model.ball_emissions(bv) + model.player_emissions(arrays["proba"].astype(np.float64), bv["has"])
    team_cost = np.where(bv["has"], model.TEAM_COST_BALL_SEEN, model.SWITCH_COST)
    post = posteriors(emissions, team_cost, forced_dead, D.HALVES)
    ball_only, player_only = to_codes(arrays["ball_only"]), to_codes(arrays["player_only"])
    doubt, bits, moments = doubt_and_moments(final, prev, ball_only, player_only, post, bv["has"], D.HALVES)

    seen = np.where(arrays["tactical_ball"], 1, np.where(bv["has"], 2, 0)).astype(np.uint8)

    matrices, _error, valid = load_homography.load()
    inside = homography_check(matrices, valid, tac)

    bro = D.load_tracking(source="broadcast")
    _bx, _by, _report, mapping = sync.broadcast_ball_on_tactical_timeline(tac, bro, False)
    b = np.flatnonzero(mapping["trusted"])
    t = mapping["target"][b]
    order = np.argsort(t, kind="stable")
    t, offset = t[order], (t - b)[order]
    frames = np.arange(D.N_FRAMES)
    pos = np.clip(np.searchsorted(t, frames), 1, len(t) - 1)
    nearest = np.where(np.abs(t[pos - 1] - frames) <= np.abs(t[pos] - frames), pos - 1, pos)
    bc_frame = frames - offset[nearest]
    # The TV timeline is continuous between two verified frames that agree on the offset (a replay in
    # between does not break it). Beyond the last verified frame of a half, trust it for five seconds only:
    # the recording skips most of half-time, and nothing says where the cut is.
    between = (t[pos - 1] <= frames) & (frames <= t[pos]) & (np.abs(offset[pos - 1] - offset[pos]) <= 5)
    ok = (between | (np.abs(t[nearest] - frames) <= 5 * FPS)) & (bc_frame >= 0) & (bc_frame < len(bro["bx"]))
    bc_frame = np.where(ok, bc_frame, -1).astype(np.int32)

    tags = [{"frame": int(tag["start_frame"]), "type": tag["type"], "team": tag["team"]}
            for tag in D.load_smart_tagging() if tag["code"] == "SET PIECES"]
    names = ["Barcelona", "Athletic Club"]
    meta = {
        "title": "Barcelona – Athletic Club", "subtitle": "The challenge match", "kind": "challenge", "annotatable": False,
        "n_frames": int(D.N_FRAMES), "halves": [list(h) for h in D.HALVES], "pitch": [D.PITCH_LENGTH, D.PITCH_WIDTH],
        "teams": names, "slots": tac["team"].astype(int).tolist(), "image": [1920, 1080],
        "has_previous": has_prev, "has_video": True, "has_tv": True, "stoppages_given": False, "tags": tags,
        "model_note": "The submitted file. “Previous version” is the earlier decoder (team-change cost 12).",
    }
    out = {
        "final": final, "prev": prev, "ball_view": ball_only, "player_view": player_only,
        "post": np.round(post * 255).astype(np.uint8), "pview": np.round(np.clip(np.nan_to_num(arrays["proba"]), 0, 1) * 255).astype(np.uint8),
        "seen": seen, "doubt": np.round(doubt * 255).astype(np.uint8), "reasons": bits,
        "ball": np.stack((centimetres(bx), centimetres(by)), axis=1),
        "players": np.stack((centimetres(tac["px"]), centimetres(tac["py"])), axis=2),
        "homography": np.nan_to_num(matrices.reshape(-1, 9)).astype(np.float32), "homography_ok": valid.astype(np.uint8),
        "bc_frame": bc_frame, "bc_live": (np.nan_to_num(arrays["bc_live"]) > 0.5).astype(np.uint8),
    }
    folder, size = write_bundle("bar-ath", meta, out, moments)
    summary("bar-ath", final, prev, doubt, moments, D.HALVES, post)
    print("   players projected through the homography land inside the picture on {:.1f}% of checks; TV picture mapped for {:.1f}% "
          "of graded frames; bundle {:.0f} MB in {}".format(100 * inside, 100 * np.mean(bc_frame[final < X] >= 0), size / 1e6, folder))


def homography_check(matrices, valid, tac):
    """Share of player positions that project inside a 1920x1080 picture, on a sample of frames."""
    import load_homography
    hits = total = 0
    for first, stop in D.HALVES:
        for frame in range(first, stop, 997):
            if not valid[frame]:
                continue
            x, y = tac["px"][frame] / D.PITCH_LENGTH, tac["py"][frame] / D.PITCH_WIDTH
            keep = ~np.isnan(x)
            u, v = load_homography.project(matrices[frame], x[keep], y[keep])
            hits += int(np.sum((u >= 0) & (u < 1920) & (v >= 0) & (v < 1080)))
            total += int(keep.sum())
    return hits / max(total, 1)


# ---------------------------------------------------------------- practice matches (can be labelled)

def build_practice(n):
    match_id = "metrica-{}".format(n)
    standin.fetch()
    game = standin.load_game(n)
    scenario = standin_eval.SCENARIOS[0]
    truth = to_codes(game["truth"])
    with standin.match_shape(game):
        views, _none = standin_eval.prepare(game, None, scenario, os.path.join(standin.folder(), "cache"))
        bv = views["bv"]
        stopped = truth == 2
        final = to_codes(pipeline.final_pass(views, stopped))
        prev = to_codes(pipeline.final_pass(views, stopped, 12, 12))
        ball_only = to_codes(model.decode(views["ball_e"], stopped))
        player_only = to_codes(model.decode(model.player_emissions(views["proba"]), stopped))
        emissions = views["ball_e"] + model.player_emissions(views["proba"], bv["has"])
        team_cost = np.where(bv["has"], model.TEAM_COST_BALL_SEEN, model.SWITCH_COST)
        post = posteriors(emissions, team_cost, stopped, game["halves"])
        baseline = to_codes(standin_eval.baseline(views, 50))
    doubt, bits, moments = doubt_and_moments(final, prev, ball_only, player_only, post, bv["has"], game["halves"])
    tac = game["tac"]
    meta = {
        "title": "Practice match {}".format(n), "subtitle": "Metrica Sports open sample game {}".format(n), "kind": "practice",
        "annotatable": True, "n_frames": int(game["n_frames"]), "halves": [list(h) for h in game["halves"]],
        "pitch": list(standin.PITCH), "teams": ["Home", "Away"], "slots": tac["team"].astype(int).tolist(),
        "has_previous": True, "has_video": False, "has_tv": False, "stoppages_given": True, "tags": [],
        "model_note": "Same decoder as the submission, run on a ball degraded to look like our tracking ({}). Stoppages are "
                      "taken from the event data, because these games have no TV feed or tagging.".format(scenario[0]),
    }
    out = {
        "final": final, "prev": prev, "ball_view": ball_only, "player_view": player_only,
        "post": np.round(post * 255).astype(np.uint8), "pview": np.round(np.clip(np.nan_to_num(views["proba"]), 0, 1) * 255).astype(np.uint8),
        "seen": bv["has"].astype(np.uint8), "doubt": np.round(doubt * 255).astype(np.uint8), "reasons": bits,
        "ball": np.stack((centimetres(tac["bx"]), centimetres(tac["by"])), axis=1),
        "ball_model": np.stack((centimetres(views["bx"]), centimetres(views["by"])), axis=1),
        "players": np.stack((centimetres(tac["px"]), centimetres(tac["py"])), axis=2),
        "events": truth,
    }
    folder, size = write_bundle(match_id, meta, out, moments)
    # what the scoring endpoint needs, so it never has to parse the tracking again
    np.savez_compressed(os.path.join(folder, "score.npz"), ball_e=views["ball_e"].astype(np.float32),
                        proba=np.asarray(views["proba"], dtype=np.float32), has=bv["has"], events=truth, baseline=baseline,
                        halves=np.array(game["halves"]), n_frames=game["n_frames"])
    summary(match_id, final, prev, doubt, moments, game["halves"], post)
    print("   bundle {:.0f} MB in {}".format(size / 1e6, folder))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=("bar-ath", "metrica-1", "metrica-2"))
    parser.add_argument("--submission", default=os.path.join(pipeline.RESULTS_DIR, "possession_two_views.csv"))
    parser.add_argument("--previous", default=os.path.join(D.DATA_DIR, "_cache", "possession_two_views_v1_cost12.csv"),
                        help="an earlier submission to show as a second lane (optional)")
    args = parser.parse_args()
    if args.only in (None, "metrica-1"):
        build_practice(1)
    if args.only in (None, "metrica-2"):
        build_practice(2)
    if args.only in (None, "bar-ath"):
        build_challenge(args.submission, args.previous)
