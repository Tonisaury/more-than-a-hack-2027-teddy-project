# Possession Workspace

A local page where you can talk (or type) to an assistant about the entry: what the task is, what
was built, what the results mean and how far they can be trusted. It answers by putting cards on a
canvas: possession strips, evidence curves, the pitch, breakdowns, the scoreboard, the validation
checks. Modelled on the PRA Voice Workspace: a slim bar, one canvas, a voice dock.

It is a companion for understanding the work. It is not part of the submission pipeline.

## Start it

```bash
pip install -r solution/requirements.txt
python solution/run.py --diagnostics "<data>/_cache/workspace"      # once: caches the evidence the cards draw
python workspace/server.py --data-dir "<data>" --env-file "<file with OPENAI_API_KEY=...>"
```

Then open http://127.0.0.1:8777. `<data>` is the unpacked bundle (the folder with `tactical/` and
`broadcast/`); `MTAH_DATA_DIR` works too. Without a key the assistant is off, but the quick cards in
the bar and **Take the tour** still work. Useful links: `/?tour` opens with the guided cards,
`/?gallery` shows one of every kind of card, `/?listen` connects without a microphone (you type, it
speaks), `&theme=dark` or `light`.

## How it is put together

| file | job |
|---|---|
| `server.py` | serves the page, runs the tools, holds the OpenAI key, relays the WebRTC offer |
| `analysis.py` | the tools: each returns a **card** for the browser and a **summary** for the model |
| `briefing.py` | the assistant's briefing (task, rules, data, method, results, limits) and the tool schemas |
| `static/app.js` | canvas, voice call, typed fallback |
| `static/charts.js` | the block renderers: timeline with evidence lanes, turn matching, pitch replay, bars, heatmap, table, checks, flow, sandboxed custom HTML |

**Voice.** `gpt-live-1` through OpenAI's Live API (`POST /v1/live/sessions`, WebRTC). The live model is
the voice; a Responses model behind it (`gpt-6-luna`, "delegation") reasons and calls the tools. Tool
calls arrive on the data channel, the page runs them against this server, draws the card and returns
the summary. `--voice-model gpt-realtime-2.1` switches to the Realtime API, where one model does
both; the page speaks both protocols. Typed questions without a call go through the Responses API
with the same tools.

**Seventeen tools**, all allowlisted with strict schemas: overview, scoreboard, timeline, evidence
close-up, breakdown (the query tool: time split by half, period, label, either view, ball tracked,
TV shot, pace, tagged phase), durations, confusion matrix, turn matching, ball coverage, pitch,
tagging, validation, explainers, custom HTML, note, documentation lookup, canvas actions.

## What leaves this computer

The competition terms forbid giving the match data to a cloud model, so the split is built in:

- **Sent to OpenAI:** your voice or typed question, and each tool's *summary*: scores, counts,
  shares, clock times and our own predicted labels.
- **Never sent:** tracking rows, player or ball positions, evidence curves, video. The pitch view and
  the curves are drawn by this page from this server.

The server listens on `127.0.0.1` only, refuses other origins, and keeps the key out of the page.
Custom visuals from the assistant are sanitised and run in a sandboxed frame with no scripts and no
network. Nothing is stored: cards and transcript vanish on reload.

## Validation, and what is off limits

`show_validation` runs every check that needs no second answer key: the public five minutes, tagged
set pieces against predicted stoppages, the restarting team against the tag, agreement between the
two views, the label against the player on the ball, held-out agreement, reproducibility.

It deliberately does **not** compare with public match statistics (possession percentage, goal
minutes, fouls). `RULES.md` forbids using any human-produced data about this game, from any source,
including to validate. Ask the organisers in an issue before doing that.

## Not verified

Speaking into a real microphone, and how the voice sounds, could not be tested from the build
session; the call itself, tool calls, cards and spoken-reply transcripts were tested with typed
questions over a live `gpt-live-1` session. A Live session bills by the minute while connected:
end the conversation when you are done.
