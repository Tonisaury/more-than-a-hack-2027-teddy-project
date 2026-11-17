# Two views and a sticky state machine

Our Phase 1 entry: one label (`H` / `A` / `D`) for every frame of BAR-ATH.

```bash
pip install -r solution/requirements.txt          # Python 3.12; numpy, pandas, scikit-learn
export MTAH_DATA_DIR=/path/to/the/unpacked/bundle  # or unpack it into ./data
python solution/run.py --name <team_name>          # -> evaluation/results/<team_name>.csv
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
python evaluation/run_evaluation.py                # public five minutes: a pipeline check
```

`run.py` is the single command that regenerates the submission. It is deterministic and takes
three to four minutes (the tracking files are parsed once and cached next to the data, never in
this repository). The videos are not read.

| file | what it does |
|---|---|
| `mtah/data.py` | parses the tracking, Smart Tagging and metadata into numpy arrays |
| `mtah/sync.py` | measures the broadcast → tactical time mapping and verifies it frame by frame |
| `mtah/prep.py` | cleans the ball track; turns it into per-frame evidence (the *ball view*) |
| `mtah/ballmodel.py`, `models/ball_view.pkl` | who is on the ball: a classifier trained on other matches, and the trained file `run.py` reads |
| `mtah/playerview.py` | a classifier that never sees the ball (the *player view*), trained on pseudo-labels |
| `mtah/hmm.py` | three-state sticky HMM: Viterbi and forward–backward |
| `mtah/model.py` | combines the two views into one label sequence |
| `run.py` | end to end |
| `summarize.py` | aggregate results of a submission as JSON (labels and scores only) |
| `mtah/standin.py`, `standin_eval.py` | the test bench on other matches (below); not used by `run.py` |
| `mtah/refine.py` | boundary snapping that was tried and switched off (`model.SHARPEN`) |
| `mtah/dfl.py`, `dfl_eval.py` | a second test bench: the DFL open matches, where stoppages can be scored |
| `ballmodel_eval.py` | tests of the learned ball view on matches it never saw; `--train-final` rebuilds `models/ball_view.pkl` |
| `rounds_eval.py`, `seed_stability.py`, `diagnose.py` | two additions that were measured and not adopted; label-free counts of a run's output |

## How the constants were tested

The provided match's public five minutes are not a tuning set, so decoder choices are tested on
two other matches that come with answers (Metrica Sports' open sample games, about 190 MB,
downloaded on first use into `<MTAH_DATA_DIR>/_standin`, never into this repository):

```bash
python solution/standin_eval.py --noise    # is the degraded stand-in as noisy as our tracking?
python solution/standin_eval.py --sweep    # the grid behind model.TEAM_COST_BALL_SEEN (about 40 min)
```

The DFL matches (seven public Bundesliga matches with the answer on every frame, about 2.6 GB,
downloaded into `<MTAH_DATA_DIR>/_standin/dfl`) are a second bench and the training data of the
ball classifier:

```bash
python solution/dfl_eval.py                 # the decoder with stoppages not given (after the download below)
python solution/ballmodel_eval.py           # learned against hand-written ball view, each match scored by a model that never saw it
python solution/ballmodel_eval.py --metrica # the same on the sample games, which are never used for training
python solution/ballmodel_eval.py --train-final solution/models/ball_view.pkl   # rebuild the shipped classifier (downloads on first use)
```

The method, the rules it implements and its limits are in **[TECHNICAL.md](TECHNICAL.md)**.
