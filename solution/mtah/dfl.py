"""A second set of stand-in matches: the DFL open dataset, which has the answer on every frame.

Bassek, Rein, Weber & Memmert (2025), "An integrated dataset of spatiotemporal and event data in
elite soccer", Scientific Data. Seven matches of the 2022/23 Bundesliga and 2. Bundesliga, TRACAB
optical tracking at 25 Hz, CC BY 4.0, figshare DOI 10.6084/m9.figshare.28196177.

Why it is worth having next to Metrica's sample games (standin.py): each ball frame carries
`BallStatus` (1 in play, 0 interrupted) and `BallPossession` (1 home, 2 away). That is the same
kind of answer the organisers describe -- dead where the tracking marks the ball dead, otherwise
the side the tracking names -- and, unlike a truth rebuilt from events, it says when play is
stopped. So stoppage detection can be measured on other matches, which the sample games cannot do.

What it does not have: a broadcast recording, homographies or Smart Tagging. As with the sample
games, the test exercises the ball view, a shape-only player view and the decoder.

Nothing here is used by run.py, and nothing is trained on these matches.
"""

import os
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

from . import data as D
from . import standin

# figshare file ids: match -> (match information, positions)
FILES = {
    "J03WMX": (51643475, 51643514),
    "J03WN1": (51643472, 51643517),
    "J03WOH": (51643478, 51643520),
    "J03WOY": (51643481, 51643523),
    "J03WPY": (51643487, 51643526),
    "J03WQQ": (51643484, 51643529),
    "J03WR9": (51643490, 51643532),
}
URL = "https://ndownloader.figshare.com/files/{}"
SECTIONS = ("firstHalf", "secondHalf")
H, A, DEAD = 0, 1, 2


def folder(root=None):
    return root or os.path.join(standin.folder(), "dfl")


def _find(root, match, kind):
    for name in os.listdir(root):
        if match in name and kind in name and name.endswith(".xml"):
            return os.path.join(root, name)
    return None


def fetch(matches, root=None, verbose=True):
    """Download the two files of each match once (about 400 MB per match)."""
    root = folder(root)
    os.makedirs(root, exist_ok=True)
    for match in matches:
        for kind, file_id in zip(("matchinformation", "positions_raw"), FILES[match]):
            if _find(root, match, kind):
                continue
            if verbose:
                print("  downloading", match, kind, flush=True)
            path = os.path.join(root, "DFL_{}_DFL-MAT-{}.xml".format(kind, match))
            urllib.request.urlretrieve(URL.format(file_id), path + ".part")
            os.replace(path + ".part", path)
    return root


def available(root=None):
    root = folder(root)
    if not os.path.isdir(root):
        return []
    return [m for m in FILES if _find(root, m, "matchinformation") and _find(root, m, "positions_raw")]


def _home_guest(info_path):
    roles = {}
    for team in ET.parse(info_path).getroot().iter("Team"):
        role = (team.get("Role") or "").lower()
        if role in ("home", "guest"):
            roles[team.get("TeamId")] = 0 if role == "home" else 1
    if len(roles) != 2:
        raise ValueError("{}: expected a home and a guest team, found {}".format(info_path, roles))
    return roles


def _parse(positions_path, roles):
    """One pass over the position XML. Returns per-section dicts of {object: (N, X, Y)} and the labels."""
    players = {s: {} for s in SECTIONS}        # section -> (person, side) -> [n, x, y]
    ball = {}                                  # section -> [n, x, y, possession, status]
    pitch = None
    current = None
    for event, elem in ET.iterparse(positions_path, events=("start", "end")):
        tag = elem.tag
        if event == "start":
            if tag == "FrameSet":
                section, team = elem.get("GameSection"), elem.get("TeamId")
                if section not in SECTIONS:
                    current = None
                elif team == "BALL":
                    current = ball.setdefault(section, ([], [], [], [], []))
                elif team in roles:
                    current = players[section].setdefault((elem.get("PersonId"), roles[team]), ([], [], []))
                else:                          # the referee
                    current = None
            elif tag == "PitchSize":
                pitch = (float(elem.get("X")), float(elem.get("Y")))
            continue
        if tag == "Frame":
            if current is not None:
                current[0].append(int(elem.get("N")))
                current[1].append(float(elem.get("X")))
                current[2].append(float(elem.get("Y")))
                if len(current) == 5:
                    current[3].append(int(elem.get("BallPossession")))
                    current[4].append(int(elem.get("BallStatus")))
            elem.clear()
        elif tag == "FrameSet":
            current = None
            elem.clear()
    return players, ball, pitch


def load_game(match, root=None, use_cache=True):
    """The same dict standin.load_game returns, plus `pitch` and `ball_placeholder`.

    truth: 0 home, 1 away, 2 dead, on every frame of the two halves (which are laid end to end).
    ball_placeholder: True where the ball is marked dead; its position there is not a measurement
    we rely on (it is often frozen), so the test bench decides what to do with it.
    """
    root = folder(root)
    cache = os.path.join(root, "cache_{}.npz".format(match))
    if use_cache and os.path.isfile(cache):
        s = np.load(cache)
        px, py, team, bx, by, truth = s["px"], s["py"], s["team"], s["bx"], s["by"], s["truth"]
        halves, pitch = [tuple(map(int, h)) for h in s["halves"]], tuple(map(float, s["pitch"]))
    else:
        roles = _home_guest(_find(root, match, "matchinformation"))
        players, ball, pitch = _parse(_find(root, match, "positions_raw"), roles)
        if pitch is None:
            pitch = standin.PITCH
        keys = sorted({k for s in SECTIONS for k in players[s]}, key=lambda k: (k[1], k[0]))
        column = {k: i for i, k in enumerate(keys)}
        team = np.array([k[1] for k in keys], dtype=np.int8)
        lengths, firsts = [], []
        for s in SECTIONS:
            n = np.array(ball[s][0])
            firsts.append(int(n.min()))
            lengths.append(int(n.max() - n.min() + 1))
        total = sum(lengths)
        px = np.full((total, len(keys)), np.nan, dtype=np.float32)
        py = np.full((total, len(keys)), np.nan, dtype=np.float32)
        bx = np.full(total, np.nan)
        by = np.full(total, np.nan)
        truth = np.full(total, DEAD, dtype=np.int8)
        halves, offset = [], 0
        for s, first, length in zip(SECTIONS, firsts, lengths):
            n, x, y, possession, status = (np.array(v) for v in ball[s])
            row = n - first + offset
            bx[row] = x + pitch[0] / 2.0
            by[row] = y + pitch[1] / 2.0
            truth[row] = np.where(status == 1, np.where(possession == 1, H, A), DEAD)
            for key, (pn, x, y) in players[s].items():
                prow = np.array(pn) - first + offset
                ok = (prow >= offset) & (prow < offset + length)
                px[prow[ok], column[key]] = np.array(x)[ok] + pitch[0] / 2.0
                py[prow[ok], column[key]] = np.array(y)[ok] + pitch[1] / 2.0
            halves.append((offset, offset + length))
            offset += length
        if use_cache:
            np.savez_compressed(cache, px=px, py=py, team=team, bx=bx, by=by, truth=truth,
                                halves=np.array(halves), pitch=np.array(pitch))
    px, py = px.astype(np.float64), py.astype(np.float64)
    speed = np.full_like(px, np.nan)
    speed[1:] = np.hypot(np.diff(px, axis=0), np.diff(py, axis=0)) * D.FPS
    tac = {"px": px, "py": py, "ps": speed, "bx": bx.copy(), "by": by.copy(), "team": team, "team_names": ["Home", "Away"]}
    return {"tac": tac, "truth": truth, "halves": halves, "n_frames": len(truth), "pitch": pitch,
            "ball_placeholder": truth == DEAD, "name": "DFL " + match}
