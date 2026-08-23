# Qualifying lap time predictor — notes

Reference doc for picking this back up when we tune the model later. Code
itself has the step-by-step "why" comments (esp. `features.py`, `model.py`,
`train.py`); this is the higher-level summary + a running list of what to
try next.

## The approach

One regression model: predict each driver's own best qualifying lap time
(seconds). Session-level answers fall out for free — predicted fastest time
= min across the session's predictions, predicted pole driver = whoever has
that min. Simpler than training separate regression + classification models.

## Features (see `features.py` for exact derivation)

- Practice pace this weekend: `practice_best_time_s`, `practice_gap_to_best_s` (relative to the weekend's fastest practice lap — this is what makes pace comparable across different circuits), `practice_sessions_completed`
- Form: `recent_quali_gap_to_pole_avg_s` (rolling last-5), `season_quali_gap_to_pole_avg_s` (expanding, current season) — both `.shift(1)`'d so a session never sees its own result
- Weather: air/track temp, humidity, rainfall — taken from the last practice session before qualifying (closest available leak-free proxy; we can't know qualifying-day weather in advance for a real "upcoming race" prediction)
- Tire compound: one-hot, whichever compound the driver ran most (by lap count) in practice
- Embeddings: driver, team (looked up per-session, not fixed — team affiliation changes, confirmed 2 mid-season swaps in the data), circuit

## Current results (first baseline, not tuned)

| split | baseline MAE | NN MAE | baseline pole acc | NN pole acc |
|---|---|---|---|---|
| train | 1.66s | 1.15s | 30.0% | 10.0% |
| val | 2.80s | 5.69s | 55.6% | 0.0% |
| test | 1.21s | 3.18s | 44.4% | 0.0% |

**The NN doesn't beat the baseline yet.** Baseline = practice time + a
constant learned offset. ~800 training rows (58 real Qualifying sessions
total) is thin for an embeddings-based net to beat a strong
domain-informed formula. Expected for a first pass, not a failure.

## Bugs found + fixed while building this (worth knowing before touching the code again)

1. **Untrained "unknown" embedding row.** Every training row got a real
   category index, so index 0 (reserved for unseen drivers/teams/circuits)
   never received a gradient update — stayed at random init. Any val/test
   row hitting it (e.g. 'Audi'/'Cadillac', new 2026 teams never in
   train) got a near-random prediction. Fixed with category dropout:
   `train.py` relabels 10% of training rows' real categories as "unknown"
   too, so that embedding row learns something sensible. Currently a
   **fixed** mask (rolled once, not per epoch) — resampling it every epoch
   would be a stronger, more standard version of this regularization.
2. **One outlier lap wrecked a whole split's average.** A z-score of 51 on
   `practice_gap_to_best_s` (a likely red-flag-affected practice lap at
   Miami) blew up through the Linear layers. Fixed by clipping standardized
   features to `[-5, 5]` in `train.py`'s `_to_tensors`.

## Known simplifications (documented in code, listed here for visibility)

- Missing numeric values (rare — mostly reserve-driver sessions with no
  practice running) are filled with the **whole-table** mean in
  `features.py`, not a train-only mean like the scaler uses elsewhere — a
  small, deliberate amount of leakage. Move into `train.py` and fit on
  train only if it ever matters.
- `practice_sessions_completed` counts practice *sessions* with a valid
  lap (0-3), not total laps run — simpler join, rougher signal.
  `practice_main_compound` uses whichever compound has the most laps, not
  necessarily the compound of the single fastest lap.
- Category-dropout mask (see bug #1) is fixed once at tensor-build time,
  not resampled every epoch.

## Ideas for when we come back to tune this

- More data: only 58 real Qualifying sessions right now. Re-run ingest for
  earlier seasons (2021-2023) once needed — cache/ingest already supports it.
- Try resampling the unknown-dropout mask every epoch instead of once.
- Try a smaller/simpler architecture (fewer hidden units, or drop one
  embedding) given how little data there is — the baseline winning suggests
  the net may be over-parameterized for ~800 rows.
- Add dropout/weight decay to the MLP itself for stronger regularization.
- Try predicting `gap_to_pole_s` directly instead of raw lap time, using a
  separate simple estimate for the session's own pole-time baseline — could
  make the regression target more homogeneous across circuits.
- Widen the practice-compound feature to reflect the compound of the
  driver's actual fastest lap, not just their most-used compound.

## Running this

```
python -m backend.models.train             # one run with the default hyperparameters
python -m backend.models.tune               # 50-trial Optuna sweep, retrains + saves the winner
python -m backend.models.tune --n-trials 20 --no-retrain   # search only, don't touch qualifying_model.pt
```

(Older notes here said PyTorch needed WSL on this machine — that's no longer
true as of this session; it imports natively on plain Windows Python now.)

## MLflow experiment tracking

Every call to `train_model()` — a plain training run, each individual Optuna
trial, and the final retrain of the winning config — is logged as its own
MLflow run: every hyperparameter (hidden layer sizes, dropout, learning
rate, batch size, weight decay, embedding dims), the per-epoch validation
MAE curve, the final MAE/pole-accuracy for train/val/test (both the model
and the baseline, for comparison), and — for a run that actually saves a
checkpoint — the checkpoint file itself as an artifact you can download
straight from the run's page.

It's entirely local: no Azure resource, no server to run, no account —
tracking data lives in `backend/storage/mlruns/`, right next to
`cache.sqlite` and the checkpoint itself.

To browse it:

```
cd f1-race-analytics
mlflow ui
```

Then open the URL it prints (usually `http://127.0.0.1:5000`). A `tune.py`
sweep shows up as one `hyperparameter_sweep` run with every trial nested
underneath it (tagged with its Optuna trial number), rather than dozens of
unrelated runs cluttering the experiment's top-level list — expand it to
compare trials side by side.
