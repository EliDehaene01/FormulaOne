"""
Trains the qualifying lap time predictor and compares it against the
baseline. Run with:

    python -m backend.models.train

Saves the trained model (+ everything needed to run inference on it later)
to backend/storage/qualifying_model.pt.

HYPERPARAMETERS ARE FUNCTION ARGUMENTS, NOT JUST MODULE CONSTANTS
----------------------------------------------------------------------
train_model() takes hidden_sizes/dropout/learning_rate/batch_size/
weight_decay/max_epochs/early_stop_patience as keyword arguments (defaulting
to the module constants below, so `python -m backend.models.train` with no
arguments behaves exactly as before). This is what lets
backend/models/tune.py reuse this exact training loop for every Optuna
trial instead of maintaining a second copy of it — one training
implementation, called either with fixed defaults (this script) or
trial-suggested values (tune.py).
"""

import sqlite3

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from backend.data.cache import DB_PATH
from backend.models import baseline
from backend.models.features import NUMERIC_FEATURE_COLUMNS, build_feature_table
from backend.models.model import QualifyingLapTimePredictor

MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 20  # stop if val MAE hasn't improved in this many epochs
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
CHECKPOINT_PATH = DB_PATH.parent / "qualifying_model.pt"


def _temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by SESSION (not by row — every row in a session shares the same
    date, so splitting rows randomly would leak teammates' data across
    splits) and by TIME, not randomly. A random split would let the model
    train on a driver's Silverstone qualifying and get evaluated on that
    same driver's Silverstone qualifying two weeks earlier — validation
    would look great and mean nothing, because that's not the situation the
    model will actually face (predicting a race that hasn't happened yet).
    Chronological split: oldest 70% of sessions -> train, next 15% -> val,
    most recent 15% -> test.
    """
    session_dates = df.groupby("session_key")["date_start"].first().sort_values()
    session_keys = session_dates.index.to_numpy()

    n = len(session_keys)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train_keys, val_keys, test_keys = np.split(session_keys, [train_end, val_end])

    train_df = df[df["session_key"].isin(train_keys)].reset_index(drop=True)
    val_df = df[df["session_key"].isin(val_keys)].reset_index(drop=True)
    test_df = df[df["session_key"].isin(test_keys)].reset_index(drop=True)
    return train_df, val_df, test_df


def _build_category_index(train_series: pd.Series) -> dict:
    """
    nn.Embedding needs contiguous integer indices starting at 0, but our
    categories are arbitrary values (driver numbers, team name strings,
    circuit name strings). This builds that mapping — fit on the TRAIN
    split only, same reasoning as the temporal split above: the model
    shouldn't get to "see" a circuit or driver that only appears in
    val/test until it's actually being evaluated on it.

    Index 0 is reserved for "unknown" (see _apply_category_index) — a
    category that shows up in val/test (or at real inference time) but
    was never seen during training. Every real category gets 1..N.
    """
    categories = sorted(train_series.unique())
    return {category: idx + 1 for idx, category in enumerate(categories)}


def _apply_category_index(series: pd.Series, mapping: dict) -> np.ndarray:
    """Unseen categories fall back to index 0 (the reserved "unknown" embedding row) instead of crashing."""
    return series.map(mapping).fillna(0).astype(int).to_numpy()


def _to_tensors(df: pd.DataFrame, driver_map: dict, team_map: dict, circuit_map: dict, numeric_mean, numeric_std):
    """
    Converts a feature DataFrame into the tensors QualifyingLapTimePredictor
    expects. Embedding indices must be dtype long (int64) — that's what
    nn.Embedding requires for its lookup. Numeric features and the target
    are float32, PyTorch's default float type.
    """
    driver_idx = torch.tensor(_apply_category_index(df["driver_number"], driver_map), dtype=torch.long)
    team_idx = torch.tensor(_apply_category_index(df["team_name"], team_map), dtype=torch.long)
    circuit_idx = torch.tensor(_apply_category_index(df["circuit_short_name"], circuit_map), dtype=torch.long)

    # Standardization ((x - mean) / std): puts every numeric feature on a
    # comparable scale. Without this, a feature like humidity (values in the
    # 20-90 range) would dominate the network's early learning purely
    # because its raw numbers are bigger than, say, rainfall (0 or small
    # decimals) — not because it's actually more predictive.
    numeric = (df[NUMERIC_FEATURE_COLUMNS] - numeric_mean) / numeric_std

    # Clip extreme standardized values instead of feeding them in raw. Found
    # by debugging a real result: one practice session (a driver's only
    # clean lap being unusually slow, e.g. after a red flag) produced a
    # z-score of 51 on practice_gap_to_best_s — fed through several Linear
    # layers, that single row's activations exploded and wrecked that
    # split's whole average error. Clamping to [-5, 5] keeps a single freak
    # data point from dominating the loss while leaving 99%+ of normal rows
    # completely untouched (a z-score of 5 is already an extreme outlier).
    numeric = numeric.clip(lower=-5, upper=5)
    numeric_tensor = torch.tensor(numeric.to_numpy(), dtype=torch.float32)

    target = torch.tensor(df["target_lap_time_s"].to_numpy(), dtype=torch.float32)
    return driver_idx, team_idx, circuit_idx, numeric_tensor, target


def train_model(
    hidden_sizes: tuple[int, ...] = (128, 64, 32),
    dropout: float = 0.1,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    weight_decay: float = 0.0,
    max_epochs: int = MAX_EPOCHS,
    early_stop_patience: int = EARLY_STOP_PATIENCE,
    save_checkpoint: bool = True,
) -> float:
    """
    Returns the best validation MAE (seconds) achieved — the value
    backend/models/tune.py's Optuna objective optimizes. save_checkpoint
    exists so a tuning trial can train and be scored WITHOUT overwriting
    qualifying_model.pt on every single trial; only the final, winning
    hyperparameter set (via tune.py's retrain_best()) sets it True.
    """
    conn = sqlite3.connect(DB_PATH)
    df = build_feature_table(conn)
    conn.close()

    train_df, val_df, test_df = _temporal_split(df)
    print(f"train: {len(train_df)} rows / {train_df.session_key.nunique()} sessions")
    print(f"val:   {len(val_df)} rows / {val_df.session_key.nunique()} sessions")
    print(f"test:  {len(test_df)} rows / {test_df.session_key.nunique()} sessions")

    # Fit every "learnable" preprocessing step on train only, then apply it
    # unchanged to val/test — the same rule enforced throughout: val/test
    # must look like genuinely unseen future data, not influence anything
    # the model or its inputs were tuned on.
    driver_map = _build_category_index(train_df["driver_number"])
    team_map = _build_category_index(train_df["team_name"])
    circuit_map = _build_category_index(train_df["circuit_short_name"])
    numeric_mean = train_df[NUMERIC_FEATURE_COLUMNS].mean()
    numeric_std = train_df[NUMERIC_FEATURE_COLUMNS].std().replace(0, 1)  # guard against a constant column (std=0 -> divide by zero)

    train_tensors = _to_tensors(train_df, driver_map, team_map, circuit_map, numeric_mean, numeric_std)
    val_tensors = _to_tensors(val_df, driver_map, team_map, circuit_map, numeric_mean, numeric_std)

    # Bug found by debugging a real result: index 0 ("<UNKNOWN>") is only
    # ever fed to the model for categories val/test/live-inference sees
    # that TRAINING never did — which means, without this, that embedding
    # row never receives a single gradient update and stays at its random
    # initialization forever. A driver/team/circuit that's actually new
    # (e.g. a brand-new F1 team) then gets a prediction from a
    # never-trained random vector — exactly what caused test set rows to
    # blow up. Fix: deliberately relabel a small % of TRAINING rows' real
    # categories as "unknown" too, so that embedding row learns a sensible
    # "no specific information" representation instead of staying random.
    # (Simplification: this mask is fixed once here rather than re-rolled
    # every epoch — a real regularization technique would resample it per
    # epoch for more robustness; fine for a first baseline.)
    train_driver_idx, train_team_idx, train_circuit_idx, train_numeric, train_target = train_tensors
    # Unrelated to the `dropout` parameter above (that's regular NN
    # regularization on hidden activations) — this is a fixed 10% chance of
    # relabeling a category as "<UNKNOWN>" during training, always at this
    # rate regardless of tuning, so the embedding's unknown-category row
    # gets real gradient signal (see the comment below for why that matters).
    unknown_category_relabel_p = 0.1
    generator = torch.Generator().manual_seed(0)
    for idx_tensor in (train_driver_idx, train_team_idx, train_circuit_idx):
        drop_mask = torch.rand(idx_tensor.shape, generator=generator) < unknown_category_relabel_p
        idx_tensor[drop_mask] = 0
    train_tensors = (train_driver_idx, train_team_idx, train_circuit_idx, train_numeric, train_target)

    # cuda = NVIDIA GPU. Most laptops/dev machines don't have one available
    # to PyTorch, so this quietly falls back to the CPU — fine at our data
    # size (a few thousand rows), where GPU transfer overhead would dwarf
    # any speedup anyway.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = QualifyingLapTimePredictor(
        num_drivers=len(driver_map),
        num_teams=len(team_map),
        num_circuits=len(circuit_map),
        num_numeric_features=len(NUMERIC_FEATURE_COLUMNS),
        hidden_sizes=hidden_sizes,
        dropout=dropout,
    ).to(device)  # moves every parameter tensor onto the chosen device; must match where the input tensors live

    # DataLoader batches + shuffles the training set each epoch. Shuffling
    # matters: without it, the model would see sessions in strict
    # chronological order every epoch and could pick up on spurious
    # patterns from that ordering rather than the features themselves.
    train_dataset = torch.utils.data.TensorDataset(*train_tensors)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # MSELoss (mean squared error) is what we optimize during training —
    # squaring the error means one 5-second miss hurts the loss as much as
    # twenty-five 1-second misses, which pushes the optimizer hard to avoid
    # large mistakes. L1Loss (mean absolute error) is what we REPORT,
    # because "the model is off by 0.4 seconds on average" is a number a
    # human can sanity-check; MSE's squared units (seconds^2) aren't
    # intuitive. Training loss and the metric you report don't have to be
    # the same thing — they're serving different purposes.
    train_criterion = nn.MSELoss()
    eval_criterion = nn.L1Loss()

    # Adam adapts its own per-parameter learning rate as it goes, which
    # makes it a robust default that rarely needs tuning — a safe choice
    # for a first baseline rather than plain SGD, which is more sensitive
    # to the learning rate you pick. weight_decay adds L2 regularization
    # (shrinks weights toward zero each step) — another overfitting knob
    # alongside dropout; 0.0 (off) unless tune.py finds a better value.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_mae = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()  # switches on training-mode behavior (irrelevant for this architecture since we have no
        # Dropout/BatchNorm layers, but it's the standard first line of every training loop, so it's here for real ones later)
        for driver_idx, team_idx, circuit_idx, numeric, target in train_loader:
            driver_idx, team_idx, circuit_idx, numeric, target = (
                t.to(device) for t in (driver_idx, team_idx, circuit_idx, numeric, target)
            )

            optimizer.zero_grad()  # PyTorch ACCUMULATES gradients by default; without zeroing, each step would add to the last
            predictions = model(driver_idx, team_idx, circuit_idx, numeric)  # the forward pass
            loss = train_criterion(predictions, target)
            loss.backward()  # backpropagation: computes d(loss)/d(parameter) for every parameter in the model
            optimizer.step()  # applies one gradient-descent update to every parameter, using those gradients

        # Validation: no gradient tracking needed since we're not
        # training here. torch.no_grad() turns that tracking off, which
        # both saves memory and is the correct thing to do when you're
        # just measuring, not learning.
        model.eval()  # switches off training-mode behavior for the same layers .train() switched on
        with torch.no_grad():
            val_driver, val_team, val_circuit, val_numeric, val_target = (t.to(device) for t in val_tensors)
            val_predictions = model(val_driver, val_team, val_circuit, val_numeric)
            val_mae = eval_criterion(val_predictions, val_target).item()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epochs_without_improvement == 0:
            print(f"epoch {epoch:3d} | val MAE: {val_mae:.3f}s | best: {best_val_mae:.3f}s")

        # Early stopping: without it, a small network on a small dataset
        # will keep driving TRAINING loss down by memorizing quirks of the
        # training rows specifically (overfitting) long after it's stopped
        # getting any better — or has started getting worse — on data it
        # wasn't trained on. Stopping once val MAE plateaus, and keeping the
        # best weights we saw rather than the final ones, avoids shipping an
        # overfit snapshot just because it happened to be the last epoch.
        if epochs_without_improvement >= early_stop_patience:
            print(f"stopped early at epoch {epoch} (no val improvement in {early_stop_patience} epochs)")
            break

    model.load_state_dict(best_state_dict)  # restore the best-seen weights, discarding any overfitting since then
    model.eval()

    _report_results(model, device, train_df, val_df, test_df, driver_map, team_map, circuit_map, numeric_mean, numeric_std)

    if save_checkpoint:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "driver_map": driver_map,
                "team_map": team_map,
                "circuit_map": circuit_map,
                "numeric_feature_columns": NUMERIC_FEATURE_COLUMNS,
                "numeric_mean": numeric_mean,
                "numeric_std": numeric_std,
                "hidden_sizes": hidden_sizes,
                "dropout": dropout,
            },
            CHECKPOINT_PATH,
        )
        print(f"\nSaved checkpoint to {CHECKPOINT_PATH}")

    return best_val_mae


def _nn_predict(model, device, df, driver_map, team_map, circuit_map, numeric_mean, numeric_std) -> pd.Series:
    driver_idx, team_idx, circuit_idx, numeric, _ = _to_tensors(df, driver_map, team_map, circuit_map, numeric_mean, numeric_std)
    with torch.no_grad():
        predictions = model(driver_idx.to(device), team_idx.to(device), circuit_idx.to(device), numeric.to(device))
    return pd.Series(predictions.cpu().numpy(), index=df.index)


def _report_results(model, device, train_df, val_df, test_df, driver_map, team_map, circuit_map, numeric_mean, numeric_std) -> None:
    baseline_offset = baseline.fit_offset(train_df)

    print("\n=== baseline (practice time + constant offset) vs neural net ===")
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        baseline_preds = baseline.predict(split_df, baseline_offset)
        baseline_result = baseline.evaluate(split_df, baseline_preds)

        nn_preds = _nn_predict(model, device, split_df, driver_map, team_map, circuit_map, numeric_mean, numeric_std)
        nn_result = baseline.evaluate(split_df, nn_preds)  # same scoring function — it only needs target_lap_time_s + a predicted column

        print(f"\n{name} ({baseline_result['n_rows']} rows, {baseline_result['n_sessions']} sessions):")
        print(f"  baseline   MAE: {baseline_result['mae_seconds']:.3f}s | pole accuracy: {baseline_result['pole_accuracy']:.1%}")
        print(f"  neural net MAE: {nn_result['mae_seconds']:.3f}s | pole accuracy: {nn_result['pole_accuracy']:.1%}")


if __name__ == "__main__":
    train_model()
