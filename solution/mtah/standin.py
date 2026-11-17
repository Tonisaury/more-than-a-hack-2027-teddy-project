"""Stand-in matches: other games that come with answers, made to look like ours.

The rules forbid tuning on the provided match's public five minutes, and there is no other answer
key for it. So every constant that is not fixed by physics is chosen on *other* matches:
Metrica Sports' open sample data (github.com/metrica-sports/sample-data, Sample Games 1 and 2),
which is the same provider and the same coordinate convention as the challenge data, and comes
with hand-made event data from which per-frame possession can be derived.

What this module does:
  fetch()        download the six CSV files once (public GitHub, about 190 MB)
  load_game(n)   tracking as the same dict the pipeline uses, plus per-frame truth codes
  degrade()      remove the ball with the gap pattern of the challenge match's own tracker, so the
                 stand-in has the same "the ball is missing for ten seconds at a time" problem

Truth, derived mechanically from the events (H = 0, A = 1, D = 2, X = 3 outside play):
  * dead ball runs from the end of the last event before each SET PIECE to that set piece
  * otherwise the ball belongs to the team of the latest PASS, SHOT, SET PIECE, RECOVERY,
    BALL LOST or BALL OUT -- i.e. a loose ball stays with the side that last had it
This is the challenge's own definition as closely as events allow. It is not the organisers'
ground truth (theirs comes from corrected tracking), so absolute scores are not comparable;
differences between two versions of the decoder on the same stand-in are what it is for.

These games have no broadcast recording, no homography and no Smart Tagging, so the stand-in
exercises the ball view, a shape-only player view and the decoder -- not the camera features.
"""

import os
import urllib.request

import numpy as np

from . import data as D

BASE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_{n}/"
FILES = ("Sample_Game_{n}_RawTrackingData_Home_Team.csv", "Sample_Game_{n}_RawTrackingData_Away_Team.csv",
         "Sample_Game_{n}_RawEventsData.csv")
GAMES = (1, 2)
PITCH = (105.0, 68.0)                 # the sample data's pitch, metres
H, A, DEAD, X = 0, 1, 2, 3
OWNING = ("PASS", "SHOT", "SET PIECE", "RECOVERY", "BALL LOST", "BALL OUT")


def folder(root=None):
    return root or os.environ.get("MTAH_STANDIN_DIR") or os.path.join(D.DATA_DIR, "_standin")


def fetch(root=None, verbose=True):
    root = folder(root)
    os.makedirs(root, exist_ok=True)
    for n in GAMES:
        for name in FILES:
            name = name.format(n=n)
            path = os.path.join(root, name)
            if os.path.isfile(path) and os.path.getsize(path) > 1000:
                continue
            if verbose:
                print("  downloading", name, flush=True)
            urllib.request.urlretrieve(BASE_URL.format(n=n) + name, path + ".part")
            os.replace(path + ".part", path)
    return root


def _tracking(path):
    """(period, frame, xy) with xy shaped (n_frames, n_objects, 2); the ball is the last object."""
    raw = np.genfromtxt(path, delimiter=",", skip_header=3, filling_values=np.nan)
    xy = raw[:, 3:]
    xy = xy[:, : (xy.shape[1] // 2) * 2].reshape(len(raw), -1, 2)
    return raw[:, 0].astype(int), raw[:, 1].astype(int), xy


def _events(path):
    import csv
    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({"team": 0 if row["Team"].strip().lower() == "home" else 1, "type": row["Type"].strip().upper(),
                         "start": int(row["Start Frame"]) - 1, "end": int(row["End Frame"]) - 1, "period": int(row["Period"])})
    rows.sort(key=lambda r: (r["start"], r["end"]))
    return rows


def truth_from_events(events, period, n_frames):
    """Per-frame codes and the graded [first, stop) span of each half."""
    codes = np.full(n_frames, X, dtype=np.int8)
    halves = []
    for half in (1, 2):
        ev = [e for e in events if e["period"] == half and e["type"] != "CARD"]
        if not ev:
            continue
        first = min(e["start"] for e in ev if e["type"] == "SET PIECE")          # the kick-off
        stop = max(e["end"] for e in ev) + 1
        owner = np.full(stop - first, -1, dtype=np.int8)
        for e in ev:
            if e["type"] in OWNING and e["start"] >= first:
                owner[e["start"] - first:] = e["team"]
        owner[owner < 0] = ev[0]["team"]
        dead = np.zeros(stop - first, dtype=bool)
        for i, e in enumerate(ev):
            if e["type"] == "SET PIECE" and i > 0 and e["start"] > first:
                prev = ev[i - 1]
                if prev["type"] == "SET PIECE":
                    continue
                dead[max(0, prev["end"] - first):e["start"] - first] = True
        codes[first:stop] = np.where(dead, DEAD, owner)
        halves.append((int(first), int(stop)))
    return codes, halves


def load_game(n, root=None):
    """dict(tac=..., bx, by, truth, halves, n_frames) with positions in metres."""
    root = folder(root)
    period, _frame, home = _tracking(os.path.join(root, FILES[0].format(n=n)))
    _p, _f, away = _tracking(os.path.join(root, FILES[1].format(n=n)))
    n_frames = min(len(home), len(away))
    home, away, period = home[:n_frames], away[:n_frames], period[:n_frames]
    players = np.concatenate((home[:, :-1], away[:, :-1]), axis=1)
    ball = home[:, -1]
    team = np.array([0] * (home.shape[1] - 1) + [1] * (away.shape[1] - 1), dtype=np.int8)
    truth, halves = truth_from_events(_events(os.path.join(root, FILES[2].format(n=n))), period, n_frames)
    px, py = players[:, :, 0] * PITCH[0], players[:, :, 1] * PITCH[1]
    speed = np.full_like(px, np.nan)
    speed[1:] = np.hypot(np.diff(px, axis=0), np.diff(py, axis=0)) * D.FPS
    tac = {"px": px, "py": py, "ps": speed, "bx": ball[:, 0] * PITCH[0], "by": ball[:, 1] * PITCH[1], "team": team,
           "team_names": ["Home", "Away"]}
    return {"tac": tac, "truth": truth, "halves": halves, "n_frames": n_frames, "name": "Metrica sample game {}".format(n)}


def challenge_gap_mask():
    """True where the challenge match's own tracker saw the ball, one array per half."""
    tac = D.load_tracking()
    seen = ~np.isnan(tac["bx"])
    return [seen[first:stop] for first, stop in D.HALVES]


def smooth_noise(n, sigma, seconds, rng):
    """Zero-mean wander of standard deviation sigma that changes over about `seconds`."""
    k = max(1, int(seconds * D.FPS))
    white = rng.normal(0.0, 1.0, n + k)
    c = np.cumsum(white)
    return (c[k:] - c[:-k]) / np.sqrt(k) * sigma


def degrade(game, masks=None, shift=0.0, offset=0.0, offset_seconds=1.0, seed=0):
    """A copy of the game's ball made to look like an automatic tracker's.

    masks   if given, the challenge tracker's ball-seen pattern is laid over each half (rolled by
            `shift` in [0, 1), so one match gives several tests). This is the pessimistic case: in
            the challenge match most gaps fall in dead time, here they fall anywhere.
            If None, the sample data's own gaps are kept (the ball is present on ~87% of live frames,
            close to our estimate for the challenge match).
    offset  metres of slow positional wander added to the ball. In the challenge tracking the ball
            sits a median 1.9 m from the nearest player; in the clean sample data 0.3 m. The wander
            is sized so the two distributions match (see standin_eval.py --noise).
    """
    bx, by = game["tac"]["bx"].copy(), game["tac"]["by"].copy()
    if masks is not None:
        for (first, stop), mask in zip(game["halves"], masks):
            rolled = np.roll(mask, int(shift * len(mask)))
            seen = np.resize(rolled, stop - first)
            bx[first:stop][~seen] = np.nan
            by[first:stop][~seen] = np.nan
    if offset > 0:
        rng = np.random.RandomState(seed)
        bx = bx + smooth_noise(len(bx), offset, offset_seconds, rng)
        by = by + smooth_noise(len(by), offset, offset_seconds, rng)
    return bx, by


class match_shape:
    """Temporarily make the pipeline's match constants describe a stand-in game."""

    def __init__(self, game):
        pitch = game.get("pitch", PITCH)
        self.new = {"N_FRAMES": game["n_frames"], "HALVES": list(game["halves"]), "PITCH_LENGTH": pitch[0], "PITCH_WIDTH": pitch[1]}

    def __enter__(self):
        self.old = {k: getattr(D, k) for k in self.new}
        for k, v in self.new.items():
            setattr(D, k, v)
        return self

    def __exit__(self, *exc):
        for k, v in self.old.items():
            setattr(D, k, v)
        return False
