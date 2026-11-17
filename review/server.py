"""Possession Review: watch the match with the model's answer on top, and label practice matches.

    python review/build.py      # once: prepares the data (a few minutes)
    python review/server.py     # http://127.0.0.1:8778

Local by default. It listens on 127.0.0.1, refuses other origins, and sends nothing anywhere: the
page, the per-frame data and the video all travel from this process to your browser and no further.

Hosted for a team: with REVIEW_USERS set, every request needs a team
login, and REVIEW_PUBLIC_HOSTS names the host the page is served on. The match data is licensed to
enrolled participants only, so the server refuses to listen beyond this machine without logins.

The challenge match is watch-only by design. The competition rules forbid hand labels on it "in
any amount and for any purpose" -- not to train, not to validate. So for that match this server
has no endpoint that stores a label, and refuses if asked. Labelling is for the practice matches
(other games, where the rules allow it); those labels are written to review/labels/, inside the
repository, because the organisers must be able to see any hand labels a team used.
"""

import argparse
import base64
import csv
import datetime
import glob
import hashlib
import hmac
import json
import mimetypes
import os
import re
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STATIC = os.path.join(HERE, "static")
FONTS = os.path.join(ROOT, "workspace", "static", "fonts")
LABELS = os.path.join(HERE, "labels")
CSP = ("default-src 'self'; img-src 'self' data:; media-src 'self'; style-src 'self' 'unsafe-inline'; "
       "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
MATCH_IDS = ("bar-ath", "metrica-1", "metrica-2")
LETTERS = "HADX"
NOTE_TOPICS = ("goalkeeper", "wrong team or identity", "ball lost by the tracker", "restart or dead ball",
               "loose or contested ball", "short touch", "camera or TV", "other")
STATE = {"data_dir": None, "videos": {}, "score_lock": threading.Lock(), "notes_lock": threading.Lock(),
         "users": {}, "hosts": (), "secret": b"", "cookie_path": "/"}
SESSION_DAYS = 14


def make_session(user):
    """A signed 'user.expiry.signature' cookie value, so the video element need not resend the password."""
    stem = "{}.{}".format(user, int(time.time()) + SESSION_DAYS * 86400)
    return stem + "." + hmac.new(STATE["secret"], stem.encode(), hashlib.sha256).hexdigest()


def check_session(token):
    stem, _, signature = token.rpartition(".")
    user, _, expiry = stem.partition(".")
    good = hmac.new(STATE["secret"], stem.encode(), hashlib.sha256).hexdigest()
    return (hmac.compare_digest(good, signature) and user in STATE["users"]
            and expiry.isdigit() and int(expiry) > time.time())


def session_user(token):
    return token.split(".", 1)[0]


def notes_dir(match_id):
    """Notes and their pictures live beside the match data, outside _cache/review (which is replaced
    wholesale when the review data is rebuilt) and outside the repository (a picture is a video frame)."""
    return os.path.join(STATE["data_dir"], "review_notes", match_id)


def load_notes(match_id):
    path = os.path.join(notes_dir(match_id), "notes.json")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_notes(match_id, notes):
    folder = notes_dir(match_id)
    os.makedirs(folder, exist_ok=True)
    tmp = os.path.join(folder, "notes.json.part")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(notes, handle, indent=1, ensure_ascii=False)
    os.replace(tmp, os.path.join(folder, "notes.json"))


def save_note(match_id, meta, body, user):
    """Create or update one note. A note is an observation about a moment -- text, a topic and optionally a
    picture of the frame with drawings on it. It has no field for "the right answer", on purpose."""
    with STATE["notes_lock"]:
        notes = load_notes(match_id)
        note_id = str(body.get("id") or "")
        note = next((n for n in notes if n["id"] == note_id), None)
        now = datetime.datetime.now().isoformat(timespec="seconds")
        if note is None:
            note = {"id": secrets.token_hex(6), "created": now, "author": user, "image": None}
            notes.append(note)
        frame = max(0, min(int(body.get("frame", 0)), meta["n_frames"] - 1))
        moment = body.get("moment")
        topic = str(body.get("topic") or "other")
        note.update({
            "frame": frame, "moment": int(moment) if moment is not None else None,
            "topic": topic if topic in NOTE_TOPICS else "other",
            "text": str(body.get("text") or "")[:8000], "updated": now, "edited_by": user,
        })
        folder = notes_dir(match_id)
        image = body.get("image")
        if image == "remove":
            if note["image"] and os.path.isfile(os.path.join(folder, note["image"])):
                os.remove(os.path.join(folder, note["image"]))
            note["image"] = None
        elif isinstance(image, str) and image.startswith("data:image/jpeg;base64,"):
            data = base64.b64decode(image.split(",", 1)[1], validate=True)
            if len(data) > 4_000_000 or data[:3] != b"\xff\xd8\xff":
                raise ValueError("the picture is not a JPEG under 4 MB")
            os.makedirs(folder, exist_ok=True)
            note["image"] = note["id"] + ".jpg"
            with open(os.path.join(folder, note["image"]), "wb") as handle:
                handle.write(data)
            note["image_version"] = int(time.time())
        notes.sort(key=lambda n: n["frame"])
        write_notes(match_id, notes)
        return note


def delete_note(match_id, note_id):
    with STATE["notes_lock"]:
        notes = load_notes(match_id)
        keep = [n for n in notes if n["id"] != note_id]
        for note in notes:
            if note["id"] == note_id and note.get("image"):
                path = os.path.join(notes_dir(match_id), note["image"])
                if os.path.isfile(path):
                    os.remove(path)
        write_notes(match_id, keep)
        return len(notes) - len(keep)


def review_dir(match_id):
    return os.path.join(STATE["data_dir"], "_cache", "review", match_id)


def read_meta(match_id):
    path = os.path.join(review_dir(match_id), "meta.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        meta = json.load(handle)
    if match_id == "bar-ath":
        meta["has_video"] = "tactical" in STATE["videos"]
        meta["has_tv"] = "tv" in STATE["videos"]
    return meta


def find_video(kind, given):
    """The match video is not always where the bundle's README puts it; look in the usual places."""
    if given:
        return given if os.path.isfile(given) else None
    data = STATE["data_dir"]
    for folder in (os.path.join(data, kind), data, os.path.dirname(data), os.path.dirname(os.path.dirname(data))):
        hits = sorted(glob.glob(os.path.join(folder, "BAR-ATH_{}*.mp4".format(kind))))
        if hits:
            return hits[0]
    return None


def clean_points(points, n_frames):
    """[[frame, 'H'|'A'|'D'|'X']] sorted, in range, with no two neighbours alike."""
    out = {}
    for item in points:
        frame, label = int(item[0]), str(item[1]).upper()
        if label in LETTERS and 0 <= frame < n_frames:
            out[frame] = label
    cleaned, last = [], "X"
    for frame in sorted(out):
        if out[frame] != last:
            cleaned.append([frame, out[frame]])
            last = out[frame]
    return cleaned


def labels_path(match_id):
    return os.path.join(LABELS, match_id + ".json")


def load_labels(match_id):
    path = labels_path(match_id)
    if not os.path.isfile(path):
        return {"points": []}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_labels(match_id, meta, body):
    os.makedirs(LABELS, exist_ok=True)
    record = {
        "match": match_id, "source": meta["subtitle"], "fps": meta["fps"], "n_frames": meta["n_frames"],
        "format": "points are [frame, label]: the label holds from that frame until the next point; X = not labelled",
        "definition": "H / A = that team has the ball and it is in play, D = dead ball, as in CHALLENGE.md",
        "seen_while_labelling": "2D positions of the players and ball from the published tracking"
                                + ("" if body.get("blind") else "; the model's answer was visible"),
        "labelled_by": str(body.get("labelled_by", ""))[:80],
        "updated": datetime.datetime.now().isoformat(timespec="seconds"),
        "points": clean_points(body.get("points", []), meta["n_frames"]),
    }
    tmp = labels_path(match_id) + ".part"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1)
    os.replace(tmp, labels_path(match_id))
    return record


def points_to_codes(points, n_frames):
    import numpy as np
    codes = np.full(n_frames, 3, dtype=np.int8)
    for i, (frame, label) in enumerate(points):
        stop = points[i + 1][0] if i + 1 < len(points) else n_frames
        codes[frame:stop] = LETTERS.index(label)
    return codes


def export_csv(match_id, meta):
    record = load_labels(match_id)
    codes = points_to_codes(record["points"], meta["n_frames"])
    path = os.path.join(LABELS, match_id + "_labels.csv")
    with open(path, "w", newline="\n") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "label"])
        writer.writerows((i, LETTERS[c]) for i, c in enumerate(codes.tolist()))
    return path, int((codes < 3).sum())


def score(match_id, meta, points):
    """The competition's own metrics for each decoder against the hand labels (practice matches only)."""
    for path in (os.path.join(ROOT, "solution"), ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")):
        if path not in sys.path:
            sys.path.insert(0, path)
    import numpy as np
    import possession_metrics as pm
    from mtah import model, standin
    import run as pipeline

    stored = np.load(os.path.join(review_dir(match_id), "score.npz"))
    n = int(stored["n_frames"])
    truth = points_to_codes(clean_points(points, n), n)
    labelled = truth < 3
    if int(labelled.sum()) < 20 * meta["fps"]:
        return {"error": "Label at least 20 seconds before scoring."}
    events = stored["events"].astype(np.int8)
    stopped = np.where(labelled, truth == 2, events == 2)
    views = {"bv": {"has": stored["has"]}, "ball_e": stored["ball_e"].astype(np.float64), "proba": stored["proba"].astype(np.float64)}
    game = {"n_frames": n, "halves": [tuple(int(v) for v in h) for h in stored["halves"]]}
    with STATE["score_lock"], standin.match_shape(game):
        variants = [
            ("Submitted decoder", "team-change cost 4", pipeline.final_pass(views, stopped)),
            ("Previous decoder", "team-change cost 12", pipeline.final_pass(views, stopped, 12, 12)),
            ("Ball view alone", "", model.decode(views["ball_e"], stopped)),
            ("Player view alone", "never sees the ball", model.decode(model.player_emissions(views["proba"]), stopped)),
            ("Organisers' baseline", "nearest player, 2 s vote", np.where(stopped, 2, stored["baseline"])),
            ("Event data", "the automatic truth used so far", events),
        ]
    rows = []
    for name, note, codes in variants:
        codes = np.asarray(codes).astype(np.int8).copy()
        codes[codes < 0] = 3
        r = pm.evaluate(truth, codes, bootstrap_n=300)
        rows.append({
            "name": name, "note": note, "spell_f1": r["spell_f1_mean"], "spell_low": r["spell_bootstrap_low"],
            "spell_high": r["spell_bootstrap_high"], "macro_f1": r["macro_f1"], "found": r["n_pred_spells"],
            "true": r["n_gt_spells"], "team_accuracy": r["team_acc_both_alive"],
            "transitions_1s": r["transitions"][25]["f1"],
        })
    return {"rows": rows, "labelled_seconds": float(labelled.sum()) / meta["fps"], "possessions": rows[0]["true"],
            "note": "Stoppages are given to every decoder from your labels, so this measures who-has-it only."}


class Handler(BaseHTTPRequestHandler):
    server_version = "PossessionReview/1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if self.command != "GET":
            sys.stderr.write("%s %s\n" % (self.command, self.path))

    def send(self, code, body, ctype="application/json", cache="no-store", headers=()):
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(code)
        text = ctype.startswith(("text/", "application/json", "application/javascript"))
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if text else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Robots-Tag", "noindex, nofollow")
        self.send_header("Cache-Control", cache)
        for name, value in headers:
            self.send_header(name, value)
        if getattr(self, "new_session", None):
            self.send_header("Set-Cookie", "review_session={}; Path={}; Max-Age={}; HttpOnly; Secure; SameSite=Strict".format(
                self.new_session, STATE["cookie_path"], SESSION_DAYS * 86400))
            self.new_session = None
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def local(self):
        """This machine, or (hosted) one of the named public hosts over https -- nothing else."""
        host = (self.headers.get("Host") or "").split(":")[0].lower()
        origin = self.headers.get("Origin")
        if host in ("127.0.0.1", "localhost"):
            return origin is None or re.fullmatch(r"http://(127\.0\.0\.1|localhost)(:\d+)?", origin) is not None
        return host in STATE["hosts"] and (origin is None or origin.lower() == "https://" + host)

    def signed_in(self):
        """Hosted only: a team login, by HTTP Basic or by the session cookie a login sets."""
        self.user = "local"
        if not STATE["users"]:
            return True
        cookie = re.search(r"(?:^|;\s*)review_session=([^;\s]+)", self.headers.get("Cookie") or "")
        if cookie and check_session(cookie.group(1)):
            self.user = session_user(cookie.group(1))
            return True
        header = self.headers.get("Authorization") or ""
        if header.startswith("Basic "):
            try:
                user, _, password = base64.b64decode(header[6:]).decode("utf-8").partition(":")
            except ValueError:
                return False
            user = user.strip().lower()
            expected = STATE["users"].get(user)
            if expected and hmac.compare_digest(expected.encode(), password.encode()):
                self.new_session = make_session(user)
                self.user = user
                return True
            time.sleep(1)                                   # a wrong password costs a second
        return False

    def gate(self):
        """True when the request may go on; otherwise the refusal has already been sent."""
        self.new_session = None
        if not self.local():
            self.send(403, {"error": "Unknown host" if STATE["hosts"] else "Local use only"})
            return False
        if not self.signed_in():
            self.send(401, {"error": "Team login needed"},
                      headers=[("WWW-Authenticate", 'Basic realm="Teddy Project - Possession Review", charset="UTF-8"')])
            return False
        return True

    def send_file(self, path, ctype):
        """A file with Range support, so the browser can seek inside a 14 GB video."""
        size = os.path.getsize(path)
        first, last, code = 0, size - 1, 200
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
        if match and (match.group(1) or match.group(2)):
            if match.group(1):
                first = int(match.group(1))
                last = min(int(match.group(2)), size - 1) if match.group(2) else size - 1
            else:
                first = max(0, size - int(match.group(2)))
            if first > last or first >= size:
                self.send_response(416)
                self.send_header("Content-Range", "bytes */{}".format(size))
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            code = 206
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(last - first + 1))
        if code == 206:
            self.send_header("Content-Range", "bytes {}-{}/{}".format(first, last, size))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command == "HEAD":
            return
        remaining = last - first + 1
        try:
            with open(path, "rb") as handle:
                handle.seek(first)
                while remaining > 0:
                    chunk = handle.read(min(1 << 18, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True          # the browser moved on; that is how seeking works

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/health":                              # for the container's health check: says nothing
            self.new_session = None
            return self.send(200, {"ok": True})
        if not self.gate():
            return
        if path == "/api/matches":
            found = [m for m in (read_meta(i) for i in MATCH_IDS) if m]
            for meta in found:
                meta.pop("arrays", None)
                meta.pop("tags", None)
                if meta["annotatable"]:
                    meta["n_points"] = len(load_labels(meta["id"])["points"])
            return self.send(200, {"matches": found})
        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[:2] == ["api", "match"] and parts[2] in MATCH_IDS:
            match_id, what = parts[2], parts[3]
            meta = read_meta(match_id)
            if not meta:
                return self.send(404, {"error": "Run review/build.py first"})
            if what == "meta":
                return self.send(200, meta)
            if what == "moments":
                with open(os.path.join(review_dir(match_id), "moments.json"), "rb") as handle:
                    return self.send(200, handle.read())
            if what == "bundle":
                return self.send_file(os.path.join(review_dir(match_id), "bundle.bin"), "application/octet-stream")
            if what == "labels":
                if not meta["annotatable"]:
                    return self.send(403, {"error": "Watch only: the rules forbid hand labels on the challenge match."})
                return self.send(200, load_labels(match_id))
            if what == "notes":
                return self.send(200, {"notes": load_notes(match_id), "topics": NOTE_TOPICS, "user": self.user})
        if len(parts) == 5 and parts[:2] == ["api", "match"] and parts[2] in MATCH_IDS and parts[3] == "note-image":
            if not re.fullmatch(r"[0-9a-f]{12}\.jpg", parts[4]):
                return self.send(404, {"error": "Not found"})
            target = os.path.join(notes_dir(parts[2]), parts[4])
            if not os.path.isfile(target):
                return self.send(404, {"error": "Not found"})
            with open(target, "rb") as handle:
                return self.send(200, handle.read(), "image/jpeg")
        if len(parts) == 2 and parts[0] == "video" and parts[1] in STATE["videos"]:
            return self.send_file(STATE["videos"][parts[1]], "video/mp4")
        if path == "/api/notebook":
            target = os.path.join(STATE["data_dir"], "_cache", "review", "notebook.txt")
            text = open(target, encoding="utf-8").read() if os.path.isfile(target) else ""
            return self.send(200, {"text": text})
        if path in ("/", "/index.html"):
            path = "/static/index.html"
        base, rest = (FONTS, path[len("/fonts/"):]) if path.startswith("/fonts/") else (STATIC, path[len("/static/"):])
        if not path.startswith(("/static/", "/fonts/")):
            return self.send(404, {"error": "Not found"})
        target = os.path.normpath(os.path.join(base, rest))
        if not target.startswith(base + os.sep) or not os.path.isfile(target):
            return self.send(404, {"error": "Not found"})
        ctype = "application/javascript" if target.endswith(".js") else (mimetypes.guess_type(target)[0] or "application/octet-stream")
        with open(target, "rb") as handle:
            return self.send(200, handle.read(), ctype)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}") if length < 20_000_000 else {}
        except ValueError:
            self.new_session = None
            return self.send(400, {"error": "Bad request"})
        if not self.gate():                                 # after the body is read, so the connection stays in step
            return
        path = self.path.split("?")[0]
        if path == "/api/notebook":
            folder = os.path.join(STATE["data_dir"], "_cache", "review")
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "notebook.txt"), "w", encoding="utf-8") as handle:
                handle.write(str(body.get("text", ""))[:200_000])
            return self.send(200, {"ok": True})
        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[:2] == ["api", "match"] and parts[2] in MATCH_IDS and parts[3] in ("notes", "notes-delete"):
            meta = read_meta(parts[2])
            if not meta:
                return self.send(404, {"error": "Run review/build.py first"})
            try:
                if parts[3] == "notes":
                    return self.send(200, {"ok": True, "note": save_note(parts[2], meta, body, self.user)})
                return self.send(200, {"ok": True, "removed": delete_note(parts[2], str(body.get("id") or ""))})
            except (ValueError, TypeError) as error:
                return self.send(400, {"error": str(error)})
        if len(parts) == 4 and parts[:2] == ["api", "match"] and parts[2] in MATCH_IDS and parts[3] in ("labels", "score", "export"):
            meta = read_meta(parts[2])
            if not meta:
                return self.send(404, {"error": "Run review/build.py first"})
            if not meta["annotatable"]:
                return self.send(403, {"error": "Watch only: the rules forbid hand labels on the challenge match."})
            try:
                if parts[3] == "labels":
                    record = save_labels(parts[2], meta, body)
                    return self.send(200, {"ok": True, "updated": record["updated"], "n_points": len(record["points"])})
                if parts[3] == "export":
                    target, n = export_csv(parts[2], meta)
                    return self.send(200, {"ok": True, "path": os.path.relpath(target, ROOT).replace(os.sep, "/"), "labelled_frames": n})
                return self.send(200, score(parts[2], meta, body.get("points", [])))
            except Exception as error:                                   # report it in the page, not only the console
                sys.stderr.write("error: {!r}\n".format(error))
                return self.send(500, {"error": "{}: {}".format(type(error).__name__, error)})
        return self.send(404, {"error": "Not found"})


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8778)
    parser.add_argument("--host", default=os.environ.get("REVIEW_HOST") or "127.0.0.1",
                        help="address to listen on; anything but 127.0.0.1 needs REVIEW_USERS (hosted mode)")
    parser.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "data"))
    parser.add_argument("--tactical-video", help="path to BAR-ATH_tactical.mp4 if it is not next to the data")
    parser.add_argument("--broadcast-video", help="path to BAR-ATH_broadcast.mp4 (optional)")
    args = parser.parse_args()
    # Hosted mode, all from the environment so no password is ever on a command line:
    #   REVIEW_USERS         name:password,name:password   (a login is then needed for everything)
    #   REVIEW_PUBLIC_HOSTS  example.org,www.example.org  (the Host the reverse proxy passes on)
    #   REVIEW_COOKIE_PATH   /mtah/                         (the public path the proxy strips)
    for pair in (os.environ.get("REVIEW_USERS") or "").split(","):
        name, _, password = pair.strip().partition(":")
        if name and password:
            STATE["users"][name.lower()] = password
    STATE["hosts"] = tuple(h.strip().lower() for h in (os.environ.get("REVIEW_PUBLIC_HOSTS") or "").split(",") if h.strip())
    STATE["secret"] = (os.environ.get("REVIEW_SESSION_SECRET") or secrets.token_hex(32)).encode()
    STATE["cookie_path"] = os.environ.get("REVIEW_COOKIE_PATH") or "/"
    if args.host not in ("127.0.0.1", "localhost") and not STATE["users"]:
        sys.exit("Refusing to listen on {} without REVIEW_USERS: the match data is for the team only.".format(args.host))
    STATE["data_dir"] = os.path.abspath(args.data_dir)
    os.environ["MTAH_DATA_DIR"] = STATE["data_dir"]
    for kind, key, given in (("tactical", "tactical", args.tactical_video), ("broadcast", "tv", args.broadcast_video)):
        found = find_video(kind, given)
        if found:
            STATE["videos"][key] = found
    ready = [i for i in MATCH_IDS if read_meta(i)]
    print("Possession Review ready at http://{}:{}{}".format(
        args.host, args.port, "   (team login needed: {})".format(", ".join(sorted(STATE["users"]))) if STATE["users"] else ""), flush=True)
    print("  matches: {}   videos: {}".format(", ".join(ready) or "none -- run review/build.py",
                                             ", ".join(sorted(STATE["videos"])) or "none found"), flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
