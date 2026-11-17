# Possession Review

A local page for looking at what the model says, frame by frame, next to the picture — and for
hand-labelling the practice matches.

```bash
python review/build.py      # once, and again after the submission changes (a few minutes)
python review/server.py     # then open http://127.0.0.1:8778
```

`server.py` looks for `BAR-ATH_tactical*.mp4` and `BAR-ATH_broadcast*.mp4` in the data folder and
the two folders above it; pass `--tactical-video` / `--broadcast-video` if they live elsewhere.
Nothing leaves the machine: the server listens on 127.0.0.1 only, and the per-frame data it
prepares is written next to the match data (`<MTAH_DATA_DIR>/_cache/review/`), never here.

## Two kinds of match, on purpose

| match | what you can do | why |
|---|---|---|
| Barcelona – Athletic Club | **watch only** | RULES.md: "No hand labels on the provided game, in any amount and for any purpose — not for the submission, not to train a model and not to validate one." Watching to understand and debug is explicitly allowed. So there is no control that marks a moment right or wrong, and the server refuses to store labels for this match. |
| Practice match 1 and 2 | **watch and label** | Metrica Sports' open sample games are other games. "Hand-labelled data from other games is fine, in any amount", provided the organisers can see it and the technical document describes it. |

The practice matches have no video, only the tracked positions, so they are shown as a pitch map.
Their model lane is the submission's decoder run on a ball degraded to look like our tracking,
with stoppages taken from the event data (see `solution/TECHNICAL.md`, section 2.5).

## What is on screen

- **Picture**: tactical camera, TV broadcast (placed on the tactical timeline by our own
  synchronisation, so it follows the same playhead) or a pitch map. On the tactical camera the
  tracked players and ball can be drawn over the video through the provided homography; the
  thick ring is the nearest player within the model's 2.5 m control radius.
- **Head-up display**: what the model says at this frame and how sure it is; what each view says
  on its own; whether the ball is in the data at all. An amber frame and banner mark a doubtful
  moment.
- **Doubtful moments**: stretches where the decoder's confidence is low, the two views disagree,
  the shipped and the previous decoder disagree, a possession is under two seconds, or play is
  called live with the ball out of the data. Ranked by how much doubt they hold. *Only these*
  plays them back to back.
- **Timeline**: one lane per source. Scroll to zoom, click to move.

## Explanations

Everything on screen explains itself. Click a chip on the picture, a lane's name, or an **i**
button for a short pop-up with the numbers behind the current frame; *Full explanation* (or
**How it works**, key `I`) opens the long version with the formulas and diagrams. The text lives
in `review/static/explain.html` and quotes constants from `solution/mtah/`: change them together.

## Notes

`T`, a right-click on the picture or the timeline, or the *Note* button beside a doubtful moment
opens a note: a topic, what you see, and optionally a picture of the frame that you can draw on
(pen, arrow, circle). Notes show in the Notebook tab, as a count on their moment and as a flag on
the timeline.

- They are stored in `<MTAH_DATA_DIR>/review_notes/<match>/` (`notes.json` plus one JPEG per
  picture), next to the match data and never in the repository: a picture is a frame of the video.
  When the page is hosted for a team behind logins, each note carries its author.
- On the challenge match a note describes what is happening and what may be confusing the model.
  It has no field for the right answer, and notes are never used to score the model or to write a
  rule about particular frames: that would be hand-labelling. They are for finding general causes
  (the goalkeeper rule in `solution/mtah/prep.py` started as one).
- An AI assistant helping with the code may read the text of the notes, never the pictures.

## Labelling a practice match

`1` `2` `3` say home, away or dead ball *from this frame on*; the label then grows as you watch.
Step back to the exact frame and press again to move a change, or drag it in the *Your labels*
lane. `0` stops. `Enter` copies the model's answer into the gap behind the playhead so you only
correct it — or switch on **Hide model** to label without seeing it, which is the cleaner test.

Labels are saved as you go to `review/labels/<match>.json` (in the repository, because the
organisers must be able to see them) and can be written out as a `frame,label` CSV. *Score
against my labels* runs the organisers' own metric for the submitted decoder, the previous one,
each view alone, the baseline and the event-derived truth.

If these labels are ever used to choose a constant, say so in the technical document: what was
labelled, how, by whom and how much.
