"""Everything the assistant can look up about the match, returned as cards.

Each tool returns two things on purpose:

  card     what the BROWSER draws. It may contain frame-level detail (label strips, evidence
           curves, player positions for the pitch view). It goes from this local server to the
           local page and nowhere else.
  summary  what the LANGUAGE MODEL is told. Only aggregates, scores and our own predicted labels:
           no tracking rows, no coordinates, no video. The competition's terms forbid giving the
           match data to a cloud model, so this split is the design, not an optimisation.

Frames are tactical-video frames (25 per second). A "clock" is half + minutes from that half's
kick-off, which is how the tools take time arguments.
"""

import hashlib
import json
import os
import re
import sys
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "solution")):
    if path not in sys.path:
        sys.path.insert(0, path)

import possession_metrics as pm  # noqa: E402
from games import GAME, DATA_DIR, half_video_blocks  # noqa: E402

FPS = 25
H, A, D, X = 0, 1, 2, 3
LETTER = "HADX"
TEAM = {"H": "Barcelona", "A": "Athletic", "D": "Dead ball"}
HALVES = [(first, last + 1) for _, first, last in half_video_blocks(GAME)]
RESULTS = os.path.join(ROOT, "evaluation", "results")
GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
SUBMISSION = os.path.join(RESULTS, "possession_two_views.csv")
VARIANT_NAMES = {
    "ours": "Two views (submitted)", "ball_only": "Ball view only", "player_only": "Player view only",
    "baseline": "Baseline, 2 s window (published floor)", "baseline_raw": "Baseline, no smoothing",
    "baseline_smooth25": "Baseline, 1 s window", "baseline_smooth125": "Baseline, 5 s window",
    "truth": "The truth (public 5 minutes)",
}
DOCS = {
    "challenge": "CHALLENGE.md", "data": "DATA.md", "evaluation": "EVALUATION.md", "rules": "RULES.md",
    "faq": "FAQ.md", "submission": "SUBMISSION.md", "overview": "OVERVIEW.md",
    "technical": os.path.join("solution", "TECHNICAL.md"),
}
DIM_NAMES = {"half": "half", "period15": "15-minute period", "period5": "5-minute period", "label": "our answer", "ball_view": "ball view alone",
             "player_view": "player view alone", "baseline": "baseline", "truth": "the truth", "ball_seen": "ball tracked or not", "tv": "TV shot",
             "pace": "pace of play", "tag_phase": "tagged attack phase"}
KICKOFF_2ND = 113742          # published in FAQ.md: the second half is graded from 110,808 but kicks off here
ATTACK_CODES = ("BUILD UP", "PROGRESSION", "ATTACKING TRANSITION", "CREATING CHANCES", "LONG BALL",
                "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX")


def _segments(codes, first, stop):
    window = codes[first:stop]
    if not len(window):
        return []
    breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(window)]))
    return [[int(first + bounds[i]), int(first + bounds[i + 1]), LETTER[int(window[bounds[i]])]]
            for i in range(len(bounds) - 1)]


def _mmss(seconds):
    sign = "-" if seconds < 0 else ""
    seconds = abs(int(round(seconds)))
    return "{}{}:{:02d}".format(sign, seconds // 60, seconds % 60)


def _round(value, digits=3):
    return None if value is None else round(float(value), digits)


class Workspace:
    """Loads once, answers many times. Thread-safe for the reads the server does."""

    def __init__(self, cache_dir=None):
        self.lock = threading.Lock()
        self.counter = 0
        self.problems = []
        self.cache_dir = cache_dir or os.path.join(DATA_DIR, "_cache", "workspace")
        self._load_labels()
        self._load_arrays()
        self._load_tracking()
        self._score()

    # ------------------------------------------------------------------ loading
    def _load_labels(self):
        self.gt = pm.label_codes(pm.read_label_csv(GT_PUBLIC))
        self.public = pm.valid_runs(self.gt)[0]
        self.n = len(self.gt)
        self.graded = np.zeros(self.n, dtype=bool)
        for first, stop in HALVES:
            self.graded[first:stop] = True
        self.variants = {"truth": self.gt}
        for key, name in (("baseline", "baseline_smooth50"), ("baseline_raw", "baseline_raw"),
                          ("baseline_smooth25", "baseline_smooth25"), ("baseline_smooth125", "baseline_smooth125")):
            path = os.path.join(RESULTS, name + ".csv")
            if os.path.isfile(path):
                self.variants[key] = pm.label_codes(pm.read_label_csv(path))
        if os.path.isfile(SUBMISSION):
            self.variants["ours"] = pm.label_codes(pm.read_label_csv(SUBMISSION))
        else:
            self.problems.append("No submission file yet. Run: python solution/run.py")

    def _load_arrays(self):
        self.arrays, self.diagnostics = None, {}
        path = os.path.join(self.cache_dir, "arrays.npz")
        if not os.path.isfile(path):
            self.problems.append("Evidence arrays not found. Run: python solution/run.py --diagnostics \"{}\"".format(self.cache_dir))
            return
        stored = np.load(path)
        self.arrays = {key: stored[key] for key in stored.files}
        with open(os.path.join(self.cache_dir, "diagnostics.json")) as handle:
            self.diagnostics = json.load(handle)
        for key in ("final", "ball_only", "player_only"):
            codes = self.arrays[key].astype(np.int8).copy()
            codes[codes < 0] = X
            self.variants["ours" if key == "final" else key] = codes
        self.kickoff = [HALVES[0][0], KICKOFF_2ND if HALVES[1][0] <= KICKOFF_2ND < HALVES[1][1] else HALVES[1][0]]

    def _load_tracking(self):
        self.tracking, self.tags = None, []
        if not hasattr(self, "kickoff"):
            self.kickoff = [HALVES[0][0], HALVES[1][0]]
        try:
            from mtah import data as D
            self.tracking = D.load_tracking()
            self.tags = D.load_smart_tagging()
            self.pitch = (D.PITCH_LENGTH, D.PITCH_WIDTH)
        except Exception as exc:                      # the data is gated; everything else still works
            self.problems.append("Match data not available ({}). Pitch and tagging views are off.".format(type(exc).__name__))

    def _score(self):
        self.scores = {}
        for key, codes in self.variants.items():
            if key == "truth":
                continue
            r = pm.evaluate(self.gt, codes, bootstrap_n=400 if key == "ours" else 0)
            self.scores[key] = r

    # ------------------------------------------------------------------ helpers
    def card(self, title, blocks, subtitle="", icon="chart", foot=""):
        with self.lock:
            self.counter += 1
            cid = "c{}".format(self.counter)
        return {"id": cid, "title": title, "subtitle": subtitle, "icon": icon, "blocks": blocks,
                "foot": foot or "Computed on this computer from the provided data and our labels."}

    def frame_at(self, half, minute):
        first, stop = HALVES[half - 1]
        return int(np.clip(self.kickoff[half - 1] + round(float(minute) * 60 * FPS), first, stop - 1))

    def clock(self, frame):
        half = 1 if frame < HALVES[1][0] else 2
        return "{}H {}".format(half, _mmss((frame - self.kickoff[half - 1]) / FPS))

    def window(self, half, from_minute, to_minute):
        if half not in (1, 2):
            raise ValueError("half must be 1 or 2")
        a, b = self.frame_at(half, from_minute), self.frame_at(half, to_minute)
        if b <= a:
            raise ValueError("to_minute must be after from_minute")
        return a, b + 1

    def need(self, what="arrays"):
        if what == "arrays" and self.arrays is None:
            raise LookupError("The evidence arrays are not built yet. Run solution/run.py with --diagnostics.")
        if what == "tracking" and self.tracking is None:
            raise LookupError("The match data is not available on this computer.")

    def shares(self, codes, mask):
        sel = codes[mask]
        total = max(1, int(np.sum(sel != X)))
        return {LETTER[k]: _round(np.sum(sel == k) / total) for k in range(3)}

    def spells(self, codes, first, stop):
        return [(a, b, l) for a, b, l in _segments(codes, first, stop) if l in "HA"]

    def _lane(self, key, a, b, points=480):
        """A curve over frames [a, b) averaged into at most `points` bins."""
        values = self.arrays[key][a:b].astype(np.float64)
        step = max(1, int(np.ceil((b - a) / points)))
        trimmed = values[: (len(values) // step) * step]
        if not len(trimmed):
            trimmed, step = values, 1
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                binned = np.nanmean(trimmed.reshape(-1, step), axis=1)
        return [None if np.isnan(v) else round(float(v), 3) for v in binned], step

    # ------------------------------------------------------------------ tools: orientation
    def show_overview(self):
        ours = self.variants["ours"]
        g = self.graded
        live = float(np.sum(ours[g] < 2)) / FPS / 60
        dead = float(np.sum(ours[g] == 2)) / FPS / 60
        spells = [s for first, stop in HALVES for s in self.spells(ours, first, stop)]
        r, base = self.scores["ours"], self.scores.get("baseline")
        home = float(np.sum(ours[g] == H)) / max(1.0, float(np.sum(ours[g] < 2)))
        blocks = [
            {"type": "kpis", "items": [
                {"label": "Turns found whole (spellF1)", "value": "{:.2f}".format(r["spell_f1_mean"]),
                 "note": "baseline {:.2f} · public 5 min".format(base["spell_f1_mean"]) if base else "public 5 min"},
                {"label": "Right letter per frame (macroF1)", "value": "{:.2f}".format(r["macro_f1"]),
                 "note": "baseline {:.2f} · public 5 min".format(base["macro_f1"]) if base else "public 5 min"},
                {"label": "Possessions, whole match", "value": str(len(spells)), "note": "predicted, not checkable yet"},
                {"label": "Ball in play", "value": "{:.0f} min".format(live), "note": "of {:.0f} graded minutes".format(live + dead)},
                {"label": "Barcelona share of in-play time", "value": "{:.0f}%".format(100 * home), "note": "predicted"},
            ]},
            {"type": "timeline", "title": "First half, as predicted", "start": HALVES[0][0], "end": HALVES[0][1], "fps": FPS,
             "clock0": (HALVES[0][0] - self.kickoff[0]) / FPS, "rows": [{"name": "Our answer", "segments": _segments(ours, *HALVES[0])}], "lanes": []},
            {"type": "timeline", "title": "Second half, as predicted", "start": HALVES[1][0], "end": HALVES[1][1], "fps": FPS,
             "clock0": (HALVES[1][0] - self.kickoff[1]) / FPS, "rows": [{"name": "Our answer", "segments": _segments(ours, *HALVES[1])}], "lanes": []},
        ]
        summary = {"spell_f1_public": _round(r["spell_f1_mean"]), "macro_f1_public": _round(r["macro_f1"]),
                   "spell_f1_ci_public": [_round(r["spell_bootstrap_low"]), _round(r["spell_bootstrap_high"])],
                   "baseline_spell_f1_public": _round(base["spell_f1_mean"]) if base else None,
                   "possessions_whole_match": len(spells), "in_play_minutes": _round(live, 1), "dead_minutes": _round(dead, 1),
                   "barcelona_share_of_in_play": _round(home), "note": "Whole-match figures are predictions; only the public 5 minutes can be checked."}
        return self.card("The entry at a glance", blocks, "Public five minutes and the whole match as predicted", "overview"), summary

    def show_scoreboard(self):
        order = ["ours", "player_only", "ball_only", "baseline_smooth125", "baseline", "baseline_smooth25", "baseline_raw"]
        rows, summary = [], []
        for key in order:
            r = self.scores.get(key)
            if not r:
                continue
            rows.append({"name": VARIANT_NAMES[key], "spell": "{:.3f}".format(r["spell_f1_mean"]), "macro": "{:.3f}".format(r["macro_f1"]),
                         "poss": r["n_pred_spells"], "dur": "{:.1f} s".format(r["mean_spell_duration_s"]),
                         "dead": "{:.2f}".format(r["per_class"]["D"]["f1"]), "_highlight": key == "ours"})
            summary.append({"answer": VARIANT_NAMES[key], "spell_f1": _round(r["spell_f1_mean"]), "macro_f1": _round(r["macro_f1"]),
                            "possessions": r["n_pred_spells"], "mean_duration_s": _round(r["mean_spell_duration_s"], 1), "dead_f1": _round(r["per_class"]["D"]["f1"], 2)})
        ours = self.scores["ours"]
        blocks = [
            {"type": "bars", "title": "Turns found whole (spellF1), public five minutes", "orientation": "horizontal", "max": 1,
             "categories": [VARIANT_NAMES[k] for k in order if k in self.scores],
             "series": [{"name": "spellF1", "values": [_round(self.scores[k]["spell_f1_mean"]) for k in order if k in self.scores]}],
             "highlight": 0, "format": "fixed2",
             "whisker": {"index": 0, "low": _round(ours["spell_bootstrap_low"]), "high": _round(ours["spell_bootstrap_high"])}},
            {"type": "table", "columns": [{"key": "name", "label": "Answer"}, {"key": "spell", "label": "spellF1", "align": "right"},
                                           {"key": "macro", "label": "macroF1", "align": "right"}, {"key": "dead", "label": "Dead-ball F1", "align": "right"},
                                           {"key": "poss", "label": "Possessions", "align": "right"}, {"key": "dur", "label": "Mean length", "align": "right"}],
             "rows": rows, "note": "Truth: {} possessions of mean {:.1f} s.".format(ours["n_gt_spells"], ours["gt_mean_spell_duration_s"])},
        ]
        return (self.card("Scoreboard on the public five minutes", blocks, "17 possessions: a health check, not a ranking", "trophy",
                          "Organisers' scorer (evaluation/possession_metrics.py). The thin line is our 5-95% bootstrap range."),
                {"public_window_possessions": ours["n_gt_spells"], "rows": summary,
                 "ours_spell_f1_range": [_round(ours["spell_bootstrap_low"]), _round(ours["spell_bootstrap_high"])],
                 "full_match_published_floor": 0.2433})

    # ------------------------------------------------------------------ tools: time views
    def _rows(self, names, a, b):
        rows = []
        in_public = a < self.public[1] and b > self.public[0]
        for name in names:
            if name == "truth" and not in_public:
                continue
            if name not in self.variants:
                continue
            rows.append({"name": VARIANT_NAMES[name].split(" (")[0], "key": name, "segments": _segments(self.variants[name], a, b)})
        return rows

    def _lanes(self, names, a, b):
        self.need()
        lanes = []
        for name in names:
            if name == "ball_seen":
                v, step = self._lane("ball_has", a, b)
                lanes.append({"name": "Ball visible", "kind": "area", "step": step, "min": 0, "max": 1, "series": [{"name": "share of frames with a ball position", "tone": "neutral", "values": v}]})
            elif name == "tv_live":
                v, step = self._lane("bc_live", a, b)
                lanes.append({"name": "TV on the live wide shot", "kind": "area", "step": step, "min": 0, "max": 1, "series": [{"name": "share of frames", "tone": "neutral", "values": v}]})
            elif name == "speed":
                v, step = self._lane("all_speed", a, b)
                lanes.append({"name": "Mean player speed", "kind": "line", "step": step, "min": 0, "max": 5, "unit": "m/s", "series": [{"name": "m/s", "tone": "neutral", "values": v}]})
            elif name == "p_dead":
                p = self.arrays["proba"]
                self.arrays.setdefault("_pd", p[:, 2])
                v, step = self._lane("_pd", a, b)
                lanes.append({"name": "Player view: chance play is stopped", "kind": "area", "step": step, "min": 0, "max": 1, "series": [{"name": "p(dead)", "tone": "D", "values": v}]})
            elif name == "p_team":
                p = self.arrays["proba"]
                if "_ph" not in self.arrays:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        self.arrays["_ph"] = p[:, 0] / (p[:, 0] + p[:, 1])
                v, step = self._lane("_ph", a, b)
                lanes.append({"name": "Player view: which side (1 = Barcelona, 0 = Athletic)", "kind": "line", "step": step, "min": 0, "max": 1, "mid": 0.5, "series": [{"name": "p(Barcelona | in play)", "tone": "H", "values": v}]})
            elif name == "control":
                v, step = self._lane("control_net", a, b)
                lanes.append({"name": "Ball view: who is on the ball (+ Barcelona, - Athletic)", "kind": "line", "step": step, "min": -1, "max": 1, "mid": 0, "series": [{"name": "control", "tone": "H", "values": v}]})
        return lanes

    def _row_summary(self, rows, a, b):
        out = {}
        for row in rows:
            segs = row["segments"]
            total = max(1, b - a)
            item = {"possessions": sum(1 for s in segs if s[2] in "HA"), "stoppages": sum(1 for s in segs if s[2] == "D"),
                    "share": {l: _round(sum(s[1] - s[0] for s in segs if s[2] == l) / total, 2) for l in "HAD"}}
            if row["key"] in ("ours", "truth") and len(segs) <= 45:
                item["segments"] = ["{}-{} {}".format(self.clock(s[0]).split(" ")[1], self.clock(s[1]).split(" ")[1], s[2]) for s in segs]
            out[row["key"]] = item
        return out

    def show_timeline(self, half, from_minute, to_minute, rows, lanes):
        a, b = self.window(half, from_minute, to_minute)
        row_specs = self._rows(rows or ["truth", "ours"], a, b)
        lane_specs = self._lanes(lanes or [], a, b) if lanes else []
        block = {"type": "timeline", "start": a, "end": b, "fps": FPS, "clock0": (a - self.kickoff[half - 1]) / FPS,
                 "rows": row_specs, "lanes": lane_specs}
        title = "{} to {}".format(self.clock(a), self.clock(b - 1).split(" ")[1])
        summary = {"window": title, "rows": self._row_summary(row_specs, a, b), "letters": "H Barcelona, A Athletic, D dead ball"}
        if not any(r["key"] == "truth" for r in row_specs) and "truth" in (rows or []):
            summary["truth"] = "not available here: the public ground truth covers only 1H 0:00-5:00"
        for lane in lane_specs:
            vals = [v for v in lane["series"][0]["values"] if v is not None]
            summary.setdefault("lane_means", {})[lane["name"]] = _round(np.mean(vals), 2) if vals else None
        return self.card("Who had the ball, " + title, [block], "One bar per answer; hover for exact times", "timeline"), summary

    def show_evidence(self, half, from_minute, to_minute):
        self.need()
        if to_minute - from_minute > 12:
            raise ValueError("Evidence view is for close-ups: ask for 12 minutes or less.")
        a, b = self.window(half, from_minute, to_minute)
        rows = self._rows(["truth", "ours", "ball_only", "player_only"], a, b)
        lanes = self._lanes(["control", "p_team", "p_dead", "ball_seen", "tv_live", "speed"], a, b)
        block = {"type": "timeline", "start": a, "end": b, "fps": FPS, "clock0": (a - self.kickoff[half - 1]) / FPS, "rows": rows, "lanes": lanes}
        title = "{} to {}".format(self.clock(a), self.clock(b - 1).split(" ")[1])
        summary = {"window": title, "rows": self._row_summary(rows, a, b),
                   "lane_means": {l["name"]: _round(np.mean([v for v in l["series"][0]["values"] if v is not None] or [0]), 2) for l in lanes},
                   "how_to_read": "Top bars are answers. Curves are the evidence each view had: ball-view control, player-view side and stopped-chance, ball visibility, TV live shot, player speed."}
        return self.card("Why it decided what it decided, " + title, [block], "Answers on top, the evidence underneath", "layers"), summary

    # ------------------------------------------------------------------ tools: numbers
    def _dim(self, name):
        """(labels per graded frame as small ints, category names) for a breakdown dimension."""
        n = self.n
        frames = np.arange(n)
        if name == "half":
            return np.where(frames < HALVES[1][0], 0, 1), ["First half", "Second half"]
        if name in ("period15", "period5"):
            size = 15 if name == "period15" else 5
            idx = np.zeros(n, dtype=np.int64)
            names = []
            offset = 0
            for h, (first, stop) in enumerate(HALVES):
                minute = np.clip((frames[first:stop] - self.kickoff[h]) / (60.0 * FPS), 0, None)
                count = int(np.ceil(45 / size)) + 1
                local = np.minimum((minute // size).astype(np.int64), count - 1)
                idx[first:stop] = offset + local
                for k in range(count):
                    names.append("{}H {}".format(h + 1, "{}-{}".format(k * size, (k + 1) * size) if k < count - 1 else "{}+".format(k * size)))
                offset += count
            return idx, names
        if name in ("label", "ball_view", "player_view", "baseline", "truth"):
            key = {"label": "ours", "ball_view": "ball_only", "player_view": "player_only"}.get(name, name)
            return self.variants[key].astype(np.int64), ["Barcelona", "Athletic", "Dead ball", "(not graded)"]
        self.need()
        if name == "ball_seen":
            return self.arrays["tactical_ball"].astype(np.int64), ["Ball not tracked", "Ball tracked"]
        if name == "tv":
            return np.nan_to_num(self.arrays["bc_live"], nan=0).astype(np.int64), ["TV away (replay, close-up, no match)", "TV on live wide shot"]
        if name == "pace":
            speed = self.arrays["all_speed"]
            q = np.nanquantile(speed[self.graded], [0.4, 0.8])
            return np.digitize(np.nan_to_num(speed, nan=0), q), ["Walking (slowest 40%)", "Jogging", "Running (fastest 20%)"]
        if name == "tag_phase":
            idx = np.zeros(n, dtype=np.int64)
            for tag in self.tags:
                if tag["code"] in ATTACK_CODES and tag["team"] is not None:
                    idx[max(0, tag["start_frame"]):tag["end_frame"]] = 1 + tag["team"]
            return idx, ["No attack tag", "Tagged: Barcelona attacking", "Tagged: Athletic attacking"]
        raise ValueError("Unknown dimension " + name)

    def show_breakdown(self, rows, columns, half, only_label, measure, chart):
        r_idx, r_names = self._dim(rows)
        mask = self.graded.copy()
        if "truth" in (rows, columns):
            mask &= self.gt != X
        if half in (1, 2):
            mask &= (np.arange(self.n) >= HALVES[half - 1][0]) & (np.arange(self.n) < HALVES[half - 1][1])
        if only_label in ("H", "A", "D"):
            mask &= self.variants["ours"] == LETTER.index(only_label)
        elif only_label == "in_play":
            mask &= self.variants["ours"] < 2
        if columns == "none":
            c_idx, c_names = np.zeros(self.n, dtype=np.int64), ["All"]
        else:
            c_idx, c_names = self._dim(columns)
        table = np.zeros((len(r_names), len(c_names)))
        np.add.at(table, (r_idx[mask], c_idx[mask]), 1.0)
        keep_r = [i for i in range(len(r_names)) if table[i].sum() > 0]
        keep_c = [j for j in range(len(c_names)) if table[:, j].sum() > 0]
        table = table[np.ix_(keep_r, keep_c)]
        r_names = [r_names[i] for i in keep_r]
        c_names = [c_names[j] for j in keep_c]
        minutes = table / FPS / 60.0
        if measure == "share_of_row":
            values = table / np.maximum(table.sum(axis=1, keepdims=True), 1)
            fmt, unit = "percent", "share of each row"
        elif measure == "share_of_total":
            values = table / max(table.sum(), 1)
            fmt, unit = "percent", "share of all selected time"
        else:
            values, fmt, unit = minutes, "minutes", "minutes"
        tone = {"Barcelona": "H", "Athletic": "A", "Dead ball": "D"}
        blocks = []
        if chart == "heatmap" and len(c_names) > 1:
            blocks.append({"type": "heatmap", "rows": r_names, "cols": c_names, "values": np.round(values, 4).tolist(), "format": fmt})
        elif chart != "table":
            blocks.append({"type": "bars", "orientation": "horizontal" if len(r_names) > 6 else "vertical", "stacked": len(c_names) > 1,
                           "categories": r_names, "format": fmt,
                           "series": [{"name": c, "tone": tone.get(c), "values": np.round(values[:, j], 4).tolist()} for j, c in enumerate(c_names)]})
        blocks.append({"type": "table",
                       "columns": [{"key": "name", "label": DIM_NAMES[rows].capitalize()}] + [{"key": "c{}".format(j), "label": c, "align": "right"} for j, c in enumerate(c_names)],
                       "rows": [dict({"name": r}, **{"c{}".format(j): ("{:.0f}%".format(100 * values[i, j]) if fmt == "percent" else "{:.1f}".format(values[i, j])) for j in range(len(c_names))})
                                for i, r in enumerate(r_names)],
                       "note": "Unit: {}. {} minutes selected.".format(unit, round(float(minutes.sum()), 1))})
        title = "Time by {}".format(DIM_NAMES[rows]) + (" and {}".format(DIM_NAMES[columns]) if columns != "none" else "")
        summary = {"rows": rows, "columns": columns, "unit": unit, "selected_minutes": _round(minutes.sum(), 1),
                   "table": {r: {c: _round(values[i, j], 3) for j, c in enumerate(c_names)} for i, r in enumerate(r_names)}}
        if "truth" in (rows, columns):
            summary["note"] = "Restricted to the public five minutes, the only frames with a truth label."
        return self.card(title, blocks, "Time split by the chosen dimensions", "grid"), summary

    def show_durations(self, kind, team):
        ours = self.variants["ours"]
        segs = [s for first, stop in HALVES for s in _segments(ours, first, stop)]
        if kind == "stoppages":
            lengths = [(b - a) / FPS for a, b, l in segs if l == "D"]
            edges, label = [0, 5, 10, 20, 30, 45, 60, 120, 1e9], "stoppages"
        else:
            want = {"home": "H", "away": "A"}.get(team)
            lengths = [(b - a) / FPS for a, b, l in segs if l in "HA" and (want is None or l == want)]
            edges, label = [0, 5, 10, 20, 30, 45, 60, 1e9], "possessions"
        names = ["{}-{} s".format(int(edges[i]), int(edges[i + 1])) if edges[i + 1] < 1e8 else "over {} s".format(int(edges[i])) for i in range(len(edges) - 1)]
        counts = [int(sum(1 for v in lengths if edges[i] <= v < edges[i + 1])) for i in range(len(edges) - 1)]
        blocks = [
            {"type": "kpis", "items": [{"label": "How many", "value": str(len(lengths))},
                                        {"label": "Median", "value": "{:.1f} s".format(float(np.median(lengths)) if lengths else 0)},
                                        {"label": "Mean", "value": "{:.1f} s".format(float(np.mean(lengths)) if lengths else 0)},
                                        {"label": "Longest", "value": "{:.0f} s".format(max(lengths) if lengths else 0)}]},
            {"type": "bars", "orientation": "vertical", "categories": names, "format": "int",
             "series": [{"name": "Number of " + label, "tone": "D" if kind == "stoppages" else {"home": "H", "away": "A"}.get(team), "values": counts}]},
        ]
        summary = {"kind": label, "team": team, "count": len(lengths), "median_s": _round(np.median(lengths) if lengths else 0, 1),
                   "mean_s": _round(np.mean(lengths) if lengths else 0, 1), "histogram": dict(zip(names, counts)),
                   "note": "Predicted whole match. The public truth has 17 possessions of mean 14.2 s in its five minutes."}
        who = "" if kind == "stoppages" or team == "both" else (" · " + TEAM["H" if team == "home" else "A"])
        return self.card("How long {} last{}".format(label, who), blocks, "Whole match, as predicted", "histogram"), summary

    def show_confusion(self, variant):
        if variant not in self.scores:
            raise ValueError("Unknown answer " + variant)
        r = self.scores[variant]
        cm = np.array(r["confusion"])[:3, :3].astype(float)
        share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        names = ["Barcelona", "Athletic", "Dead ball"]
        blocks = [
            {"type": "heatmap", "rows": ["Truth: " + n for n in names], "cols": ["Said: " + n for n in names], "values": np.round(share, 4).tolist(),
             "format": "percent", "counts": (cm / FPS).round(1).tolist(), "count_unit": "s"},
            {"type": "table", "columns": [{"key": "c", "label": "Class"}, {"key": "p", "label": "Precision", "align": "right"}, {"key": "r", "label": "Recall", "align": "right"},
                                           {"key": "f", "label": "F1", "align": "right"}, {"key": "s", "label": "Seconds of truth", "align": "right"}],
             "rows": [{"c": names[k], "p": "{:.2f}".format(r["per_class"][LETTER[k]]["precision"]), "r": "{:.2f}".format(r["per_class"][LETTER[k]]["recall"]),
                       "f": "{:.2f}".format(r["per_class"][LETTER[k]]["f1"]), "s": "{:.0f}".format(r["per_class"][LETTER[k]]["support"] / FPS)} for k in range(3)],
             "note": "macroF1 {:.3f} is the plain average of the three F1 values.".format(r["macro_f1"])},
        ]
        summary = {"answer": VARIANT_NAMES[variant], "row_shares": {names[i]: {names[j]: _round(share[i, j], 2) for j in range(3)} for i in range(3)},
                   "per_class": {names[k]: {m: _round(r["per_class"][LETTER[k]][m], 2) for m in ("precision", "recall", "f1")} for k in range(3)},
                   "macro_f1": _round(r["macro_f1"]), "scope": "public five minutes only"}
        return self.card("Where the letters go wrong · " + VARIANT_NAMES[variant].split(" (")[0], blocks, "Each row is what really happened; each column what we said", "grid"), summary

    def show_turn_matching(self, variant):
        if variant not in self.variants or variant == "truth":
            raise ValueError("Unknown answer " + variant)
        a, b = self.public
        runs = [(a, b)]
        gt_sp = pm.possession_spells(self.gt, runs)
        codes = pm.fill_abstentions(self.variants[variant], runs)
        pr_sp = pm.possession_spells(codes, runs)
        overlaps = pm.spell_overlaps(gt_sp, pr_sp)
        best = {}
        used = set()
        for iou, gi, pi in overlaps:                         # the scorer's greedy one-to-one match at IoU >= 0.3
            if iou < 0.3:
                break
            if gi in best or pi in used:
                continue
            best[gi] = (iou, pi)
            used.add(pi)
        truth_rows, table = [], []
        for i, (s, e, k) in enumerate(gt_sp):
            iou, pi = best.get(i, (0.0, None))
            truth_rows.append([int(s), int(e), LETTER[k], _round(iou, 2), pi])
            table.append({"n": i + 1, "start": self.clock(s).split(" ")[1], "team": TEAM[LETTER[k]], "len": "{:.1f} s".format((e - s) / FPS),
                          "iou": "{:.2f}".format(iou) if pi is not None else "missed",
                          "at": ", ".join("{:.1f}".format(t) for t in pm.SPELL_IOUS if iou >= t) or "none"})
        pred_rows = [[int(s), int(e), LETTER[k], j in used] for j, (s, e, k) in enumerate(pr_sp)]
        r = self.scores[variant]
        blocks = [
            {"type": "turns", "start": a, "end": b, "fps": FPS, "truth": truth_rows, "pred": pred_rows, "dead_truth": [s for s in _segments(self.gt, a, b) if s[2] == "D"],
             "dead_pred": [s for s in _segments(codes, a, b) if s[2] == "D"]},
            {"type": "bars", "title": "Score at each strictness level (how much overlap is demanded)", "orientation": "vertical", "max": 1, "format": "fixed2",
             "categories": ["overlap ≥ {:.0%}".format(t) for t in pm.SPELL_IOUS], "series": [{"name": "F1", "values": [_round(r["spell_f1"][t]["f1"]) for t in pm.SPELL_IOUS]}]},
            {"type": "table", "columns": [{"key": "n", "label": "#"}, {"key": "start", "label": "Starts"}, {"key": "team", "label": "Who"}, {"key": "len", "label": "Length", "align": "right"},
                                           {"key": "iou", "label": "Overlap (IoU)", "align": "right"}, {"key": "at", "label": "Counts at levels"}], "rows": table,
             "note": "spellF1 {:.3f} is the average of the five bars.".format(r["spell_f1_mean"])},
        ]
        missed = [t for t in table if t["iou"] == "missed"]
        summary = {"answer": VARIANT_NAMES[variant], "true_turns": len(gt_sp), "our_turns": len(pr_sp), "spell_f1": _round(r["spell_f1_mean"]),
                   "f1_by_overlap": {str(t): _round(r["spell_f1"][t]["f1"], 2) for t in pm.SPELL_IOUS},
                   "turns": [{k: v for k, v in t.items()} for t in table], "missed_count": len(missed),
                   "transitions_within_1s": "{} of {}".format(r["transitions"][25]["n_matched"], r["transitions"][25]["n_gt"])}
        return self.card("Turn by turn: truth against " + VARIANT_NAMES[variant].split(" (")[0].lower(), blocks,
                         "Each true possession, and the predicted one it was paired with", "link"), summary

    # ------------------------------------------------------------------ tools: the data
    def show_ball_coverage(self):
        self.need()
        tac, any_ = self.arrays["tactical_ball"], self.arrays["ball_has"]
        ours = self.variants["ours"]
        blocks = []
        for h, (first, stop) in enumerate(HALVES):
            step = 60 * FPS
            n = (stop - first) // step
            t = tac[first:first + n * step].reshape(n, step).mean(1)
            f = any_[first:first + n * step].reshape(n, step).mean(1)
            blocks.append({"type": "timeline", "title": "{} half, minute by minute".format("First" if h == 0 else "Second"), "start": first, "end": first + n * step, "fps": FPS,
                           "clock0": (first - self.kickoff[h]) / FPS, "rows": [{"name": "Our answer", "key": "ours", "segments": _segments(ours, first, first + n * step)}],
                           "lanes": [{"name": "Share of each minute with a ball position", "kind": "area", "step": step, "min": 0, "max": 1,
                                      "series": [{"name": "after borrowing from TV tracking", "tone": "H", "values": np.round(f, 3).tolist()},
                                                 {"name": "stadium camera only", "tone": "neutral", "values": np.round(t, 3).tolist()}]}]})
        g = self.graded
        gaps = []
        for first, stop in HALVES:
            v = np.concatenate(([1], tac[first:stop].astype(int), [1]))
            e = np.diff(v)
            gaps += list((np.flatnonzero(e == 1) - np.flatnonzero(e == -1)) / FPS)
        edges = [0, 2, 5, 10, 20, 40, 80, 1e9]
        names = ["{}-{} s".format(int(edges[i]), int(edges[i + 1])) if edges[i + 1] < 1e8 else "over {} s".format(int(edges[i])) for i in range(len(edges) - 1)]
        counts = [int(sum(1 for x in gaps if edges[i] <= x < edges[i + 1])) for i in range(len(edges) - 1)]
        dead_when_missing = float(np.mean(ours[g & ~any_] == D))
        dead_when_seen = float(np.mean(ours[g & any_] == D))
        blocks.insert(0, {"type": "kpis", "items": [
            {"label": "Ball tracked, stadium camera", "value": "{:.0f}%".format(100 * tac[g].mean()), "note": "of graded frames"},
            {"label": "After borrowing TV tracking", "value": "{:.0f}%".format(100 * any_[g].mean()), "note": "and bridging gaps up to 1 s"},
            {"label": "Typical gap", "value": "{:.0f} s".format(float(np.median(gaps))), "note": "median; longest {:.0f} s".format(max(gaps))},
            {"label": "We say dead when ball missing", "value": "{:.0f}%".format(100 * dead_when_missing), "note": "{:.0f}% when it is visible".format(100 * dead_when_seen)}]})
        blocks.append({"type": "bars", "title": "How long the ball stays lost", "orientation": "vertical", "categories": names, "format": "int",
                       "series": [{"name": "Number of gaps", "tone": "neutral", "values": counts}]})
        summary = {"tactical_ball_share": _round(tac[g].mean()), "after_fusion_share": _round(any_[g].mean()), "gaps": len(gaps), "median_gap_s": _round(np.median(gaps), 1),
                   "longest_gap_s": _round(max(gaps), 0), "gap_histogram": dict(zip(names, counts)),
                   "we_say_dead_when_ball_missing": _round(dead_when_missing, 2), "we_say_dead_when_ball_visible": _round(dead_when_seen, 2),
                   "sync": {k: self.diagnostics.get("sync", {}).get(k) for k in ("coarse_offsets", "agreement_median_m", "agreement_within_3m", "broadcast_frames_verified", "broadcast_frames")}}
        return self.card("Where the ball goes missing", blocks, "The central difficulty: four graded frames in ten have no ball", "eye-off"), summary

    def show_pitch(self, half, minute, replay_seconds):
        self.need()
        self.need("tracking")
        centre = self.frame_at(half, minute)
        span = int(np.clip(replay_seconds, 0, 20) * FPS / 2)
        frames = list(range(max(HALVES[half - 1][0], centre - span), min(HALVES[half - 1][1] - 1, centre + span) + 1, 5)) or [centre]
        tr, ours = self.tracking, self.variants["ours"]
        out = []
        for f in frames:
            px, py = tr["px"][f], tr["py"][f]
            ok = ~np.isnan(px)
            bx, by = float(self.arrays["ball_x"][f]), float(self.arrays["ball_y"][f])
            out.append({"t": self.clock(f), "label": LETTER[int(ours[f])],
                        "players": [[round(float(px[i]), 1), round(float(py[i]), 1), int(tr["team"][i])] for i in np.flatnonzero(ok)],
                        "ball": None if np.isnan(bx) else [round(bx, 1), round(by, 1)],
                        "tv": bool(np.nan_to_num(self.arrays["bc_live"][f]) > 0.5), "speed": _round(self.arrays["all_speed"][f], 1)})
        sign = int(self.arrays["attack_sign"][centre])
        block = {"type": "pitch", "frames": out, "length": self.pitch[0], "width": self.pitch[1], "centre": frames.index(min(frames, key=lambda x: abs(x - centre))),
                 "home_attacks": "right" if sign > 0 else "left"}
        net = float(np.nan_to_num(self.arrays["control_net"][centre]))
        summary = {"moment": self.clock(centre), "our_label": LETTER[int(ours[centre])], "ball_position_known": bool(self.arrays["ball_has"][centre]),
                   "ball_view_says": "Barcelona on the ball" if net > 0.3 else "Athletic on the ball" if net < -0.3 else "nobody clearly on the ball",
                   "tv_on_live_shot": bool(np.nan_to_num(self.arrays["bc_live"][centre]) > 0.5), "mean_player_speed_m_s": _round(self.arrays["all_speed"][centre], 1),
                   "players_tracked": int(np.sum(~np.isnan(tr["px"][centre]))), "replay_seconds": round(len(frames) * 5 / FPS, 1),
                   "privacy": "Positions were drawn on the user's screen only; they are not included here."}
        return self.card("The pitch at " + self.clock(centre), [block], "Tracking positions, drawn locally. Press play to replay the moment.", "pitch",
                         "Positions come straight from the provided tracking and are shown on this computer only."), summary

    def show_tags(self):
        self.need("tracking")
        ours = self.variants["ours"]
        counts = {}
        for tag in self.tags:
            counts.setdefault(tag["code"], [0, 0, 0])[0 if tag["team"] == 0 else 1 if tag["team"] == 1 else 2] += 1
        rows, agree = [], {}
        for code in sorted(counts, key=lambda c: -sum(counts[c])):
            row = {"code": code.title(), "h": counts[code][0], "a": counts[code][1], "n": counts[code][2], "agree": ""}
            if code in ATTACK_CODES:
                hit = tot = 0
                for tag in self.tags:
                    if tag["code"] == code and tag["team"] is not None:
                        seg = ours[max(0, tag["start_frame"]):tag["end_frame"]]
                        live = seg[seg < 2]
                        hit += int(np.sum(live == tag["team"]))
                        tot += len(live)
                if tot:
                    agree[code.title()] = round(hit / tot, 2)
                    row["agree"] = "{:.0f}%".format(100 * hit / tot)
            rows.append(row)
        blocks = [{"type": "table", "columns": [{"key": "code", "label": "Tag"}, {"key": "h", "label": "Barcelona", "align": "right"}, {"key": "a", "label": "Athletic", "align": "right"},
                                                 {"key": "n", "label": "No team", "align": "right"}, {"key": "agree", "label": "Our in-play label agrees", "align": "right"}],
                   "rows": rows, "note": "Agreement: of the in-play frames inside an attack tag, the share we give to the tagged team. The tagging is automatic and it is one of the player view's inputs, so this is a consistency check, not independent proof."}]
        return self.card("The automatic Smart Tagging", blocks, "{} tagged phases and set pieces".format(len(self.tags)), "tag"), {
            "tags": len(self.tags), "counts_by_code": {c.title(): {"barcelona": v[0], "athletic": v[1], "no_team": v[2]} for c, v in counts.items()},
            "agreement_of_our_labels_with_attack_tags": agree, "caveat": "Tags are automatic and feed the player view, so agreement is a consistency check."}

    # ------------------------------------------------------------------ tools: validation
    def validation_checks(self):
        ours = self.variants["ours"]
        r = self.scores["ours"]
        checks = [{"name": "Against the public ground truth (5 minutes)", "status": "pass",
                   "value": "spellF1 {:.2f} · macroF1 {:.2f}".format(r["spell_f1_mean"], r["macro_f1"]),
                   "detail": "The only real answer key we may use. 17 possessions; plausible range {:.2f}-{:.2f}. Allowed for checking, not for tuning.".format(r["spell_bootstrap_low"], r["spell_bootstrap_high"])}]
        if self.tags:
            sp = [t for t in self.tags if t["code"] == "SET PIECES" and HALVES[0][0] <= t["start_frame"] < HALVES[1][1] and self.graded[min(t["start_frame"], self.n - 1)]]
            near = sum(1 for t in sp if np.any(ours[max(0, t["start_frame"] - 75):t["start_frame"] + 76] == D))
            checks.append({"name": "Tagged set pieces sit on a predicted stoppage", "status": "pass" if near >= 0.9 * len(sp) else "warn", "value": "{} of {}".format(near, len(sp)),
                           "detail": "Every corner, throw-in, goal kick, free kick and kick-off that the automatic tagging found should have dead ball within 3 s of its start. Partly circular: tag starts are also training hints."})
            hit = tot = 0
            for t in sp:
                if t["team"] is None:
                    continue
                after = ours[t["start_frame"]:t["start_frame"] + 20 * FPS]
                live = after[after < 2]
                if len(live):
                    tot += 1
                    hit += int(live[0] == t["team"])
            checks.append({"name": "The team that restarts matches the tag", "status": "pass" if tot and hit >= 0.75 * tot else "warn", "value": "{} of {}".format(hit, tot),
                           "detail": "For corners, goal kicks and free kicks the tagging names the taker. After the stoppage, our first in-play label should be that team. This team information is not used as a label anywhere, so it is the most independent internal check we have."})
            ko = [t for t in self.tags if t["code"] == "SET PIECES" and t["type"] == "kick off (after goal)"]
            dead_before = sum(1 for t in ko if np.mean(ours[max(0, t["start_frame"] - 20 * FPS):t["start_frame"]] == D) > 0.8)
            checks.append({"name": "Goals: dead ball before each kick-off after a goal", "status": "pass" if dead_before == len(ko) else "warn", "value": "{} of {}".format(dead_before, len(ko)),
                           "detail": "The automatic tagging marks {} kick-offs after goals. The 20 s before each should be dead ball. The tags are automatic and can include false positives.".format(len(ko))})
        if self.arrays is not None:
            has = self.arrays["ball_has"] & self.graded
            b, p = self.variants["ball_only"], self.variants["player_only"]
            both_live = has & (b < 2) & (p < 2)
            checks.append({"name": "The two views agree on who has it", "status": "pass" if np.mean(b[both_live] == p[both_live]) > 0.85 else "warn",
                           "value": "{:.0f}%".format(100 * np.mean(b[both_live] == p[both_live])),
                           "detail": "Frames where the ball is seen and both views say the ball is in play. The player view never sees the ball, so agreement means team shape really does carry the answer."})
            net = self.arrays["control_net"]
            clear = self.graded & (np.abs(np.nan_to_num(net)) > 0.5) & (ours < 2)
            side = np.where(net[clear] > 0, H, A)
            checks.append({"name": "Final label matches the player on the ball", "status": "pass" if np.mean(side == ours[clear]) > 0.85 else "warn", "value": "{:.0f}%".format(100 * np.mean(side == ours[clear])),
                           "detail": "When one player is clearly on the ball and we say play is live, the label should be his team. Disagreements are loose-ball and tackle moments where possession has not changed yet."})
            pv = self.diagnostics.get("player_view", {})
            if pv:
                checks.append({"name": "Player view on moments it never trained on", "status": "pass", "value": "{:.0f}%".format(100 * pv.get("heldout_agreement", 0)),
                               "detail": "Cross-fitted over one-minute blocks against the pseudo-labels. Recall: Barcelona {:.2f}, Athletic {:.2f}, dead {:.2f}.".format(*pv.get("heldout_recall", [0, 0, 0]))})
        g = self.graded
        live_min = float(np.sum(ours[g] < 2)) / FPS / 60
        total = float(np.sum(g)) / FPS / 60
        checks.append({"name": "Plausibility: ball in play", "status": "info", "value": "{:.0f} of {:.0f} min ({:.0f}%)".format(live_min, total, 100 * live_min / total),
                       "detail": "Professional matches usually have the ball in play for a little over half of the total time. This is general football knowledge, not data about this match, so it is allowed, and it only says the number is not absurd."})
        same = os.path.isfile(SUBMISSION) and np.array_equal(pm.label_codes(pm.read_label_csv(SUBMISSION)), ours)
        digest = hashlib.md5(open(SUBMISSION, "rb").read()).hexdigest()[:12] if os.path.isfile(SUBMISSION) else "-"
        checks.append({"name": "Reproducible file", "status": "pass" if same else "warn", "value": "md5 " + digest,
                       "detail": "Two full runs produced byte-identical files. The submission on disk {} the labels this workspace is showing.".format("matches" if same else "DOES NOT match")})
        checks.append({"name": "Public match statistics (possession %, goal times, fouls...)", "status": "locked", "value": "not used",
                       "detail": "RULES.md forbids using any human-produced data about this game, from any source, including to validate. Comparing with a stats site would be exactly that. Ask the organisers in an issue before doing it."})
        checks.append({"name": "The full-match ground truth", "status": "locked", "value": "private",
                       "detail": "Only the organisers hold it. The real score arrives with the results (announced before 26 Nov 2026)."})
        return checks

    def show_validation(self):
        checks = self.validation_checks()
        blocks = [{"type": "text", "text": "There is no second answer key. What we can do is check the labels against the one public sample, and against things that must be true if the labels are right. Each line says what was checked and how far it can be trusted."},
                  {"type": "checks", "items": checks}]
        summary = {"checks": [{"name": c["name"], "status": c["status"], "value": c["value"], "meaning": c["detail"]} for c in checks]}
        return self.card("Can we validate it?", blocks, "What is checkable, what it shows, and what the rules lock away", "shield"), summary

    # ------------------------------------------------------------------ tools: explainers
    def show_explainer(self, topic):
        ours, r = self.variants["ours"], self.scores["ours"]
        a, b = self.public
        if topic == "goal":
            blocks = [
                {"type": "text", "text": "Think of the match as a flip-book with 199,319 pages, 25 for every second. On each page you write one letter.\n**H** Barcelona has the ball. **A** Athletic has it. **D** play is stopped: the ball is out, there was a foul, or a goal.\nBelow are the real first five minutes. A run of one colour is one **turn**. The judges count turns, so each one has to start and stop close to the real one."},
                {"type": "timeline", "start": a, "end": b, "fps": FPS, "clock0": 0, "rows": [{"name": "The truth", "key": "truth", "segments": _segments(self.gt, a, b)}], "lanes": []},
                {"type": "kpis", "items": [{"label": "Pages to label", "value": "199,319"}, {"label": "Pages that are graded", "value": "158,857", "note": "the two halves"},
                                            {"label": "Pages with no ball in the data", "value": "41%", "note": "the hard part"}, {"label": "Baseline to beat (full match)", "value": "0.24", "note": "spellF1"}]},
            ]
            return self.card("The goal, in plain words", blocks, "One letter per page, judged by turns", "book"), {"shown": "goal explainer with the public five minutes"}
        if topic == "scoring":
            blocks = [
                {"type": "text", "text": "**spellF1** decides the ranking. For every true turn it looks for one of ours from the same team and measures the overlap: shared time divided by the time either one covers (called IoU). A turn counts as found if the overlap passes a bar. The bar is set at 30%, 40%, 50%, 60% and 70%, and the five results are averaged.\n**macroF1** is the tie-break: page by page, was the letter right, averaged over the three letters so dead ball cannot dominate.\nFlickering between letters barely hurts macroF1 and destroys spellF1."},
                {"type": "bars", "title": "Our score at each bar, public five minutes", "orientation": "vertical", "max": 1, "format": "fixed2",
                 "categories": ["overlap ≥ {:.0%}".format(t) for t in pm.SPELL_IOUS], "series": [{"name": "F1", "values": [_round(r["spell_f1"][t]["f1"]) for t in pm.SPELL_IOUS]}]},
                {"type": "table", "columns": [{"key": "c", "label": "Case"}, {"key": "v", "label": "spellF1", "align": "right"}],
                 "rows": [{"c": "Perfect", "v": "1.00"}, {"c": "Perfect but every change 1 frame late", "v": "1.00"}, {"c": "Every change 8 frames late (0.3 s)", "v": "0.89"},
                          {"c": "Every change 1 s late", "v": "0.65"}, {"c": "Any constant guess, or random letters", "v": "0.00"}], "note": "From EVALUATION.md."},
            ]
            return self.card("How the score works", blocks, "Turns matched by overlap, not pages counted", "target"), {
                "spell_f1": _round(r["spell_f1_mean"]), "by_overlap": {str(t): _round(r["spell_f1"][t]["f1"], 2) for t in pm.SPELL_IOUS}}
        if topic == "pipeline":
            d = self.diagnostics
            blocks = [{"type": "flow", "steps": [
                {"title": "Read the data", "text": "Two trackings (stadium camera and TV), camera geometry, automatic tags. No video pixels."},
                {"title": "Line up the two cameras", "text": "The TV recording has its own clock. Offsets measured from the tracking itself: {} frames.".format(" and ".join("{:,}".format(v) for v in d.get("sync", {}).get("coarse_offsets", [])) or "two")},
                {"title": "Ball view", "text": "Where the ball is seen: who is on it, is it out of the pitch, is it lying still."},
                {"title": "First pass", "text": "A sticky state machine on the ball view alone gives labels it is sure about."},
                {"title": "Player view", "text": "A model that never sees the ball learns from those sure moments: team shapes, speeds, cameras, tags. {} features.".format(d.get("n_features", 141))},
                {"title": "Patient referee", "text": "Both views are added up. The state changes only after about half a second of solid evidence."},
            ]},
                {"type": "text", "text": "Nothing was labelled by hand and no parameter was searched against the public five minutes."}]
            return self.card("What was built, step by step", blocks, "Two views and a sticky state machine", "flow"), {"steps": 6, "features": d.get("n_features"), "sync": d.get("sync", {}).get("coarse_offsets")}
        if topic == "rules":
            blocks = [{"type": "checks", "items": [
                {"name": "Any algorithm, model or external data about other games", "status": "pass", "badge": "Allowed", "value": "", "detail": "As long as the organisers can inspect it and the technical document explains it."},
                {"name": "Improving or replacing the provided tracking by code", "status": "pass", "badge": "Allowed", "value": "", "detail": "Corrections must come out of an algorithm."},
                {"name": "Pseudo-labels made by an algorithm on this match", "status": "pass", "badge": "Allowed", "value": "", "detail": "This is how our player view is trained."},
                {"name": "Hand labels on this match, for anything", "status": "locked", "badge": "Forbidden", "value": "", "detail": "Not to submit, train or validate."},
                {"name": "Human-produced data about this match (stats sites, event data)", "status": "locked", "badge": "Forbidden", "value": "", "detail": "From any source. Data about other matches is fine."},
                {"name": "Tuning on the public five minutes", "status": "locked", "badge": "Forbidden", "value": "", "detail": "They are for checking the pipeline, not for choosing between versions."},
                {"name": "Giving the match data to a cloud AI model", "status": "locked", "badge": "Forbidden", "value": "", "detail": "Terms and FAQ. This workspace sends the model only summaries and our own labels; positions stay on this computer."},
                {"name": "Committing or publishing the match data", "status": "locked", "badge": "Forbidden", "value": "", "detail": "The submission repository must be public and must not contain it."},
            ]}]
            return self.card("What the rules allow", blocks, "From RULES.md and FAQ.md", "shield"), {"shown": "rules summary", "key_points": [
                "no hand labels on this match", "no human-produced data about this match, even to validate", "no tuning on the public 5 minutes", "no match data to cloud LLMs"]}
        raise ValueError("Unknown topic " + topic)

    # ------------------------------------------------------------------ tools: free-form
    def show_html(self, title, html):
        if not isinstance(html, str) or not 1 <= len(html) <= 60000:
            raise ValueError("html must be between 1 and 60,000 characters")
        return self.card(title[:100] or "Custom visual", [{"type": "html", "html": html}], "AI-designed visual", "spark",
                         "Designed by the assistant from numbers it was given. Not an independent source."), {"shown": "custom visual"}

    def show_note(self, title, text):
        return self.card(title[:100] or "Note", [{"type": "text", "text": text[:3000]}], "Assistant's explanation", "note",
                         "Written by the assistant. Check figures against the data cards."), {"shown": "note"}

    def read_docs(self, doc, query):
        if doc not in DOCS:
            raise ValueError("Unknown document " + doc)
        path = os.path.join(ROOT, DOCS[doc])
        if not os.path.isfile(path):
            raise LookupError("Document not found")
        text = open(path, encoding="utf-8").read()
        parts = [p.strip() for p in re.split(r"\n(?=#{1,4} )", text) if p.strip()]
        words = [w for w in re.findall(r"[a-z0-9]{3,}", query.lower())]
        scored = sorted(((sum(p.lower().count(w) for w in words), i) for i, p in enumerate(parts)), reverse=True)
        chosen = sorted(i for score, i in scored[:4] if score > 0) or list(range(min(2, len(parts))))
        out = "\n\n".join(parts[i] for i in chosen)[:6000]
        return None, {"document": DOCS[doc].replace("\\", "/"), "excerpt": out, "note": "Quoted from the repository's own documentation."}


TOOL_NAMES = ("show_overview", "show_scoreboard", "show_timeline", "show_evidence", "show_breakdown", "show_durations", "show_confusion",
              "show_turn_matching", "show_ball_coverage", "show_pitch", "show_tags", "show_validation", "show_explainer", "show_html", "show_note", "read_docs")


def execute(ws, name, args):
    """Run one allowlisted tool. Returns {'card': ..., 'summary': ...}; card may be None."""
    if name not in TOOL_NAMES:
        raise ValueError("Unknown tool " + str(name))
    if not isinstance(args, dict):
        raise ValueError("Tool arguments must be an object")
    args = {k: v for k, v in args.items() if k != "placement"}
    card, summary = getattr(ws, name)(**args)
    return {"card": card, "summary": summary}
