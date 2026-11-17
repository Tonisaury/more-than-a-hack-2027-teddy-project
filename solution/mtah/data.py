"""Load the provided match assets into plain numpy arrays.

Everything downstream works on video-frame-indexed arrays: row i is frame i of the tactical video,
which is also row i of the submission. The ATD counts from 1, so `video_frame = frame_id - 1`.

The text parse takes ~30 s, so the result is cached as an .npz next to the data (never inside the
repository -- the match data is not redistributable).
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES, DATA_DIR, half_video_blocks, require, source_frames  # noqa: E402

HOME, AWAY = 0, 1
FPS = GAMES[GAME]["fps"]
PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]["pitch"]
N_FRAMES = GAMES[GAME]["n_video_frames"]
HALVES = [(first, last + 1) for _, first, last in half_video_blocks(GAME)]  # half-open


def team_of_each_track(meta_path):
    """0 (home) / 1 (away) for every track slot. The first team in the metadata is the home side."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    names = [team.findtext("Name") for team in root.iter("Team")]
    slots = [player.get("teamId") for player in root.iter("Player")]
    team = np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)
    return team, names


def _parse_raw(raw_path, n_frames, n_slots):
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1
            values = np.array(parts[1].replace(";", ",").split(","), dtype=np.float32)
            players[row] = values.reshape(n_slots, 3)
            ball[row] = np.array(parts[2].split(","), dtype=np.float32)
    return players, ball


def load_tracking(data_dir=None, source="tactical", use_cache=True):
    """dict with, in METRES and video-frame order:

    px, py   (n_frames, n_slots) player positions, NaN where the slot is empty
    ps       (n_frames, n_slots) provider speed, m/s
    bx, by   (n_frames,) ball position, NaN where the tracker lost it
    team     (n_slots,) 0 home / 1 away
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir, source)
    cache_dir = os.path.join(data_dir or DATA_DIR, "_cache")
    cache = os.path.join(cache_dir, "{}_{}_atd.npz".format(GAME, source))
    team, names = team_of_each_track(meta_path)

    if use_cache and os.path.isfile(cache) and os.path.getmtime(cache) > os.path.getmtime(raw_path):
        stored = np.load(cache)
        players, ball = stored["players"], stored["ball"]
    else:
        players, ball = _parse_raw(raw_path, source_frames(source, GAME), len(team))
        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(cache, players=players, ball=ball)

    return {
        "px": players[:, :, 0].astype(np.float64) * PITCH_LENGTH,
        "py": players[:, :, 1].astype(np.float64) * PITCH_WIDTH,
        "ps": players[:, :, 2].astype(np.float64),
        "bx": ball[:, 0].astype(np.float64) * PITCH_LENGTH,
        "by": ball[:, 1].astype(np.float64) * PITCH_WIDTH,
        "team": team,
        "team_names": names,
    }


def load_smart_tagging(data_dir=None):
    """[{code, start_frame, end_frame, team (0/1/None), tags}] -- intervals in video frames."""
    path = require(["smart_tagging"], GAME, data_dir)[0]
    meta_path = require(["atd_meta"], GAME, data_dir)[0]
    _, names = team_of_each_track(meta_path)
    home_name = names[0].strip().lower()
    away_name = names[1].strip().lower()

    out = []
    for instance in ET.parse(path).iter("instance"):
        tags = {label.findtext("group"): label.findtext("text")
                for label in instance.findall("label")}
        team_tag = (tags.get("Team") or "").strip().lower()
        team = HOME if team_tag == home_name else AWAY if team_tag == away_name else None
        out.append({
            "code": (instance.findtext("code") or "").strip().upper(),
            "start_frame": int(round(float(instance.findtext("start")) * FPS)),
            "end_frame": int(round(float(instance.findtext("end")) * FPS)),
            "team": team,
            "type": (tags.get("Type") or "").strip().lower(),
            "tags": tags,
        })
    return out
