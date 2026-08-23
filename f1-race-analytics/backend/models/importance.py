"""
Permutation importance for the qualifying lap time predictor: for each
input (driver, team, circuit, and every numeric feature column), shuffle
just that input across the validation set, rerun predictions, and measure
how much validation MAE increases relative to the unshuffled baseline. A
feature the model actually relies on will hurt a lot when scrambled; one it
mostly ignores won't move the score much either way.

WHY SHUFFLE INSTEAD OF DROP-ONE-FEATURE-AND-RETRAIN
----------------------------------------------------
Retraining once per feature (54+ times) to see which one hurts to remove
would take hours and conflates "how much this feature matters" with "how
differently the optimizer happens to converge without it." Permutation
importance uses the ALREADY-TRAINED model exactly as is — only the input
changes — so it isolates one thing: how much THIS model's predictions
depend on this specific input carrying real information, versus being
randomly reordered across rows.

WHY REPEAT EACH SHUFFLE ~10 TIMES
------------------------------------
A single random shuffle is one particular scrambled ordering — by chance it
might land close to the original row order (understating importance) or in
an unusually disruptive one (overstating it). Averaging N_REPEATS
independent shuffles smooths that noise into a stable estimate.

Run with:
    python -m backend.models.importance
"""

import sqlite3

import numpy as np
import pandas as pd
import torch

from backend.data.cache import DB_PATH
from backend.models.features import NUMERIC_FEATURE_COLUMNS, build_feature_table
from backend.models.model import QualifyingLapTimePredictor
from backend.models.train import CHECKPOINT_PATH, _apply_category_index, _temporal_split, _to_tensors

N_REPEATS = 10


def _load_model_and_val_tensors():
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)  # see predict_qualifying_pace in tools.py for why weights_only=False is safe here

    conn = sqlite3.connect(DB_PATH)
    df = build_feature_table(conn)
    conn.close()

    # Same temporal split train.py used — importance must be measured on the
    # same "genuinely unseen future" val rows the model was validated on
    # during training, not on rows it trained on directly.
    _, val_df, _ = _temporal_split(df)

    model = QualifyingLapTimePredictor(
        num_drivers=len(checkpoint["driver_map"]),
        num_teams=len(checkpoint["team_map"]),
        num_circuits=len(checkpoint["circuit_map"]),
        num_numeric_features=len(checkpoint["numeric_feature_columns"]),
        hidden_sizes=tuple(checkpoint.get("hidden_sizes", (128, 64, 32))),
        dropout=checkpoint.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    driver_idx, team_idx, circuit_idx, numeric, target = _to_tensors(
        val_df,
        checkpoint["driver_map"],
        checkpoint["team_map"],
        checkpoint["circuit_map"],
        checkpoint["numeric_mean"],
        checkpoint["numeric_std"],
    )
    return model, driver_idx, team_idx, circuit_idx, numeric, target


def _mae(model, driver_idx, team_idx, circuit_idx, numeric, target) -> float:
    with torch.no_grad():
        predictions = model(driver_idx, team_idx, circuit_idx, numeric)
    return (predictions - target).abs().mean().item()


def compute_permutation_importance(seed: int = 0) -> pd.DataFrame:
    """One row per input (driver/team/circuit + every numeric column), sorted by mae_increase_s descending — the model's own ranking of what it actually relies on."""
    model, driver_idx, team_idx, circuit_idx, numeric, target = _load_model_and_val_tensors()
    baseline_mae = _mae(model, driver_idx, team_idx, circuit_idx, numeric, target)

    generator = torch.Generator().manual_seed(seed)
    n_rows = len(target)

    def _average_mae_increase(shuffled_mae_fn) -> float:
        increases = []
        for _ in range(N_REPEATS):
            perm = torch.randperm(n_rows, generator=generator)
            increases.append(shuffled_mae_fn(perm) - baseline_mae)
        return float(np.mean(increases))

    rows: list[tuple[str, float]] = []

    # Categorical inputs: shuffle which ROW gets which driver/team/circuit,
    # decorrelating that identity from everything else about the row.
    rows.append(("driver", _average_mae_increase(lambda perm: _mae(model, driver_idx[perm], team_idx, circuit_idx, numeric, target))))
    rows.append(("team", _average_mae_increase(lambda perm: _mae(model, driver_idx, team_idx[perm], circuit_idx, numeric, target))))
    rows.append(("circuit", _average_mae_increase(lambda perm: _mae(model, driver_idx, team_idx, circuit_idx[perm], numeric, target))))

    # Numeric features: shuffle one COLUMN's values across rows at a time —
    # standard permutation-importance treatment (same as scikit-learn's
    # permutation_importance), everything else about the row stays intact.
    for col_idx, col_name in enumerate(NUMERIC_FEATURE_COLUMNS):
        def _shuffled_mae(perm, col_idx=col_idx):
            numeric_shuffled = numeric.clone()
            numeric_shuffled[:, col_idx] = numeric[perm, col_idx]
            return _mae(model, driver_idx, team_idx, circuit_idx, numeric_shuffled, target)

        rows.append((col_name, _average_mae_increase(_shuffled_mae)))

    result = pd.DataFrame(rows, columns=["feature", "mae_increase_s"])
    result["baseline_val_mae_s"] = baseline_mae
    return result.sort_values("mae_increase_s", ascending=False).reset_index(drop=True)


def main() -> None:
    result = compute_permutation_importance()
    baseline = result["baseline_val_mae_s"].iloc[0]
    print(f"Baseline validation MAE (unshuffled): {baseline:.4f}s\n")
    print(f"{'feature':45s} {'val MAE increase (s)':>22s}")
    print("-" * 68)
    for _, row in result.iterrows():
        print(f"{row['feature']:45s} {row['mae_increase_s']:>22.4f}")


if __name__ == "__main__":
    main()
