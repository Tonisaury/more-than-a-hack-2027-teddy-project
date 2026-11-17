"""Possession Workspace: a local page where you can talk to an assistant about the entry.

    python workspace/server.py                      # http://127.0.0.1:8777
    python workspace/server.py --env-file path/.env # where OPENAI_API_KEY lives, if not in the environment
    python workspace/server.py --data-dir path      # the unpacked match data, if MTAH_DATA_DIR is not set

Three jobs, all on this computer:
  * serve the page (workspace/static)
  * run the analysis tools (workspace/analysis.py) on the local data
  * hold the OpenAI key: the browser's WebRTC offer is exchanged here with the realtime model, and
    typed questions go through the Responses API here. The key never reaches the page.

What leaves the machine: your voice or typed questions, and the tools' *summaries* (scores, counts,
shares, our own labels). What does not: tracking rows, player positions, evidence curves, video.
The server listens on 127.0.0.1 only and refuses requests from other origins.
"""

import argparse
import json
import mimetypes
import os
import re
import sys
import threading
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

if "--data-dir" in sys.argv[1:-1]:                     # must be set before games.py reads it at import
    os.environ["MTAH_DATA_DIR"] = sys.argv[sys.argv.index("--data-dir") + 1]

import analysis  # noqa: E402
import briefing  # noqa: E402

STATIC = os.path.join(HERE, "static")
REALTIME_MODEL = os.environ.get("MTAH_REALTIME_MODEL", "gpt-live-1")
TEXT_MODEL = os.environ.get("MTAH_TEXT_MODEL", "gpt-6-luna")
VOICE = os.environ.get("MTAH_VOICE", "marin")
TRANSCRIBE_MODEL = os.environ.get("MTAH_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
# gpt-live-* models speak through the Live API (/v1/live/sessions): the live model is the voice, and a
# Responses model behind it ("delegation") does the thinking and calls our tools. gpt-realtime-* models use
# the Realtime API (/v1/realtime/calls), where one model does both. The page speaks both protocols.
PROTOCOL = "live" if REALTIME_MODEL.startswith("gpt-live") else "realtime"
SENSITIVITY = {"balanced": 0.55, "soft": 0.40, "noisy": 0.70}
MAX_BODY = 256 * 1024
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; "
       "connect-src 'self'; media-src 'self' blob:; frame-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'; object-src 'none'")

STATE = {"key": None, "ws": None, "call": None, "lock": threading.Lock()}


class AIError(Exception):
    pass


def load_key(env_file):
    key = os.environ.get("OPENAI_API_KEY")
    for path in [env_file, os.path.join(ROOT, ".env")]:
        if key or not path or not os.path.isfile(path):
            continue
        for line in open(path, encoding="utf-8"):
            m = re.match(r"\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*(.+)", line)
            if m:
                key = m.group(1).strip().strip('"').strip("'")
    return key or None


def openai(path, payload=None, multipart=None, timeout=45):
    """POST to the OpenAI API. Returns (status, headers, body bytes)."""
    if not STATE["key"]:
        raise AIError("OpenAI is not configured. Start the server with OPENAI_API_KEY set or --env-file.")
    headers = {"Authorization": "Bearer " + STATE["key"]}
    if multipart is not None:
        boundary = "----mtah" + uuid.uuid4().hex
        chunks = []
        for name, (content, ctype) in multipart.items():
            chunks.append("--{}\r\nContent-Disposition: form-data; name=\"{}\"\r\nContent-Type: {}\r\n\r\n{}\r\n".format(boundary, name, ctype, content))
        body = ("".join(chunks) + "--{}--\r\n".format(boundary)).encode("utf-8")
        headers["Content-Type"] = "multipart/form-data; boundary=" + boundary
    else:
        body = json.dumps(payload or {}).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request("https://api.openai.com" + path, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        raise AIError("Could not reach OpenAI. Check the connection and try again.") from None


def provider_message(body):
    try:
        return str(json.loads(body.decode("utf-8", "replace")).get("error", {}).get("message", ""))[:300]
    except ValueError:
        return ""


def audio_input(sensitivity, microphone):
    if sensitivity not in SENSITIVITY or microphone not in ("near_field", "far_field"):
        raise ValueError("Invalid microphone settings")
    return {"noise_reduction": {"type": microphone},
            "turn_detection": {"type": "server_vad", "threshold": SENSITIVITY[sensitivity], "prefix_padding_ms": 300,
                               "silence_duration_ms": 700, "create_response": True, "interrupt_response": True}}


def realtime_session(sensitivity="balanced", microphone="far_field"):
    return {"type": "realtime", "model": REALTIME_MODEL, "instructions": briefing.prompt(STATE["ws"]),
            "tools": briefing.REALTIME_TOOLS, "output_modalities": ["audio"], "max_output_tokens": 6000,
            "audio": {"input": dict(transcription={"model": TRANSCRIBE_MODEL}, **audio_input(sensitivity, microphone)),
                      "output": {"voice": VOICE}}}


def live_session():
    ws = STATE["ws"]
    return {"model": REALTIME_MODEL, "instructions": briefing.voice_prompt(ws), "audio": {"output": {"voice": VOICE}},
            "delegation": {"type": "responses", "responses": {
                "model": TEXT_MODEL, "instructions": briefing.prompt(ws), "tools": briefing.TOOLS, "tool_choice": "auto",
                "max_output_tokens": 6000, "reasoning": {"effort": "low"}}}}


def run_tool(name, args):
    """One tool call -> {'status', 'card', 'summary'}; failures are data for the model, not exceptions."""
    if name == "workspace_action":
        return {"status": "ok", "card": None, "summary": {"applied": args}, "action": args}
    try:
        result = analysis.execute(STATE["ws"], name, args)
        return {"status": "ok", "card": result["card"], "summary": result["summary"]}
    except (ValueError, LookupError, TypeError, KeyError) as exc:
        return {"status": "failed", "error": str(exc)[:300] or type(exc).__name__}


def for_model(result):
    """The part of a tool result the language model is allowed to see."""
    if result["status"] != "ok":
        return {"status": "failed", "error": result["error"]}
    out = {"status": "displayed" if result.get("card") else "ok", "result": result["summary"]}
    if result.get("card"):
        out["card_id"] = result["card"]["id"]
        out["card_title"] = result["card"]["title"]
    return out


def typed_turn(message, history, cards):
    """A typed question through the Responses API with the same tools. Returns (reply, actions)."""
    inputs = []
    for item in history[-12:]:
        if isinstance(item, dict) and item.get("role") in ("user", "assistant") and isinstance(item.get("content"), str):
            inputs.append({"role": item["role"], "content": item["content"][:1500]})
    inputs.append({"role": "user", "content": json.dumps({"message": message, "cards_on_canvas": cards})})
    actions, reply = [], ""
    for _ in range(6):
        status, _headers, body = openai("/v1/responses", {
            "model": TEXT_MODEL, "instructions": briefing.prompt(STATE["ws"]), "input": inputs, "tools": briefing.TOOLS,
            "store": False, "max_output_tokens": 6000, "reasoning": {"effort": "low"}, "include": ["reasoning.encrypted_content"]}, timeout=90)
        if status != 200:
            raise AIError("The assistant is unavailable ({}). {}".format(status, provider_message(body)))
        response = json.loads(body)
        if response.get("status") != "completed":
            raise AIError("The assistant did not complete this request.")
        output = response.get("output", [])
        inputs.extend(output)
        reply = "".join(c.get("text", "") for x in output if x.get("type") == "message" for c in x.get("content", []) if c.get("type") == "output_text")
        calls = [x for x in output if x.get("type") == "function_call"]
        if not calls:
            break
        if len(calls) + len(actions) > 10:
            raise ValueError("Too many cards for one question. Ask for fewer at a time.")
        for call in calls:
            try:
                args = json.loads(call.get("arguments") or "{}")
            except ValueError:
                args = None
            result = run_tool(call["name"], args) if isinstance(args, dict) else {"status": "failed", "error": "Invalid arguments"}
            actions.append({"call_id": call["call_id"], "name": call["name"], "status": result["status"], "card": result.get("card"),
                            "action": result.get("action"), "error": result.get("error")})
            inputs.append({"type": "function_call_output", "call_id": call["call_id"], "output": json.dumps(for_model(result))})
    return reply[:5000], actions


class Handler(BaseHTTPRequestHandler):
    server_version = "PossessionWorkspace/1"

    def log_message(self, fmt, *args):                       # quiet: one line per API call only
        if self.path.startswith("/api/"):
            sys.stderr.write("%s %s\n" % (self.command, self.path))

    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith(("text/", "application/json", "application/javascript")) else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("Permissions-Policy", "microphone=(self), camera=(), display-capture=()")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def local(self):
        """Only this machine's own page may call the API (a web page elsewhere must not spend the key)."""
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):
            return False
        origin = self.headers.get("Origin")
        return origin is None or re.fullmatch(r"http://(127\.0\.0\.1|localhost)(:\d+)?", origin) is not None

    def do_GET(self):
        if not self.local():
            return self.send(403, {"error": "Local use only"})
        path = self.path.split("?")[0]
        if path == "/api/state":
            ws = STATE["ws"]
            return self.send(200, {"configured": bool(STATE["key"]), "realtime_model": REALTIME_MODEL, "text_model": TEXT_MODEL, "protocol": PROTOCOL,
                                   "problems": ws.problems, "has_arrays": ws.arrays is not None, "has_tracking": ws.tracking is not None})
        if path in ("/", "/index.html"):
            path = "/static/index.html"
        if not path.startswith("/static/"):
            return self.send(404, {"error": "Not found"})
        target = os.path.normpath(os.path.join(STATIC, path[len("/static/"):]))
        if not target.startswith(STATIC + os.sep) or not os.path.isfile(target):
            return self.send(404, {"error": "Not found"})
        ctype = mimetypes.guess_type(target)[0] or "application/octet-stream"
        if target.endswith(".js"):
            ctype = "application/javascript"
        with open(target, "rb") as handle:
            return self.send(200, handle.read(), ctype)

    def do_POST(self):
        if not self.local() or self.headers.get("X-Workspace") != "1":
            return self.send(403, {"error": "Local use only"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_BODY:
                raise ValueError("Request too large")
            values = json.loads(self.rfile.read(length))
            if not isinstance(values, dict):
                raise ValueError("Expected a JSON object")
            path = self.path.split("?")[0]
            if path == "/api/tools":
                return self.send(200, self.tool(values))
            if path == "/api/voice":
                return self.send(200, self.voice(values))
            if path == "/api/voice/close":
                return self.send(200, self.hangup())
            if path == "/api/message":
                return self.send(200, self.message(values))
            return self.send(404, {"error": "Not found"})
        except AIError as exc:
            return self.send(503, {"error": str(exc)})
        except (ValueError, LookupError) as exc:
            return self.send(400, {"error": str(exc)})
        except Exception as exc:                              # never leak a stack trace to the page
            sys.stderr.write("error: %r\n" % (exc,))
            return self.send(500, {"error": "The workspace hit an internal error. See the server log."})

    def tool(self, values):
        name, args = values.get("name"), values.get("arguments")
        if not isinstance(name, str) or not isinstance(args, dict):
            raise ValueError("Invalid tool call")
        result = run_tool(name, args)
        return {"status": result["status"], "card": result.get("card"), "action": result.get("action"),
                "error": result.get("error"), "model": for_model(result)}

    def voice(self, values):
        offer = values.get("sdp")
        if not isinstance(offer, str) or not offer.startswith("v=0") or len(offer) > 64000:
            raise ValueError("Invalid voice offer")
        if PROTOCOL == "live":
            status, _headers, body = openai("/v1/live/sessions", {"session": live_session(), "transport": {"type": "webrtc", "sdp": offer}})
            try:
                answer = json.loads(body).get("transport", {}).get("sdp", "") if status in (200, 201) else ""
            except ValueError:
                answer = ""
            if not answer.startswith("v=0"):
                raise AIError("Live voice could not connect ({}). {}".format(status, provider_message(body)))
            return {"sdp": answer, "model": REALTIME_MODEL, "protocol": "live"}
        session = realtime_session(values.get("sensitivity", "balanced"), values.get("microphone", "far_field"))
        status, headers, body = openai("/v1/realtime/calls", multipart={"sdp": (offer, "application/sdp"), "session": (json.dumps(session), "application/json")})
        text = body.decode("utf-8", "replace")
        if status not in (200, 201) or not text.startswith("v=0"):
            raise AIError("Realtime voice could not connect ({}). {}".format(status, provider_message(body)))
        call = (headers.get("Location") or "").rsplit("/", 1)[-1]
        with STATE["lock"]:
            STATE["call"] = call if re.fullmatch(r"[a-zA-Z0-9_-]{1,120}", call or "") else None
        return {"sdp": text, "model": REALTIME_MODEL, "protocol": "realtime"}

    def hangup(self):
        with STATE["lock"]:
            call, STATE["call"] = STATE["call"], None
        if call:
            try:
                openai("/v1/realtime/calls/{}/hangup".format(call), {}, timeout=10)
            except AIError:
                pass
        return {"status": "closed"}

    def message(self, values):
        text = values.get("message")
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 3000:
            raise ValueError("Message must be between 1 and 3000 characters")
        history = values.get("history") or []
        cards = values.get("cards") or []
        if not isinstance(history, list) or not isinstance(cards, list) or len(json.dumps(cards)) > 12000:
            raise ValueError("Conversation context too large")
        reply, actions = typed_turn(text.strip(), history, cards)
        return {"reply": reply, "actions": actions}


def main():
    global REALTIME_MODEL, TEXT_MODEL, PROTOCOL
    parser = argparse.ArgumentParser(description="Possession Workspace: talk to an assistant about the entry.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8777)))
    parser.add_argument("--env-file", default=os.environ.get("MTAH_ENV_FILE"), help="file with OPENAI_API_KEY=...")
    parser.add_argument("--data-dir", default=None, help="the unpacked match data (same as MTAH_DATA_DIR)")
    parser.add_argument("--voice-model", default=REALTIME_MODEL, help="gpt-live-* (Live API) or gpt-realtime-* (Realtime API)")
    parser.add_argument("--text-model", default=TEXT_MODEL, help="the Responses model that thinks and calls the tools")
    parser.add_argument("--cache", default=None, help="folder with arrays.npz from solution/run.py --diagnostics")
    args = parser.parse_args()
    REALTIME_MODEL, TEXT_MODEL = args.voice_model, args.text_model
    PROTOCOL = "live" if REALTIME_MODEL.startswith("gpt-live") else "realtime"
    STATE["key"] = load_key(args.env_file)
    print("loading the match and our labels ...", flush=True)
    STATE["ws"] = analysis.Workspace(args.cache)
    for problem in STATE["ws"].problems:
        print("  note:", problem, flush=True)
    print("OpenAI key: {} - voice model {} ({} API) - text model {}".format("found" if STATE["key"] else "MISSING (tour and quick cards still work)", REALTIME_MODEL, PROTOCOL, TEXT_MODEL), flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("Possession Workspace ready at http://127.0.0.1:{}".format(args.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
