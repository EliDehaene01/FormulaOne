"""
The baseline: a deliberately dumb model to compare the neural net against.

WHY BOTHER WITH A BASELINE
-----------------------------
If we only ever look at the neural net's error, we have no idea whether
that error is *good*. A baseline gives us a floor: if a two-line formula
does nearly as well as our embeddings + feedforward net, that's a sign
either the net needs work or the extra complexity isn't earning its keep.
Best practice is to make the baseline as simple as you can defend, not as
clever as you can make it.

THE BASELINE ITSELF
----------------------
Predict a driver's qualifying time as their own practice best time this
weekend, shifted by a single constant: the average historical difference
between qualifying pace and practice pace (cars are lighter on fuel and on
softer tires in qualifying, so this constant is usually negative — quali is
faster than practice). That constant is the ONE thing this baseline
"learns", and it's fit only on the training split, same rule as everything
else in train.py.
"""

import pandas as pd


def fit_offset(train_df: pd.DataFrame) -> float:
    """The single learned parameter: mean(actual quali time - practice best time) on the training split."""
    return (train_df["target_lap_time_s"] - train_df["practice_best_time_s"]).mean()


def predict(df: pd.DataFrame, offset: float) -> pd.Series:
    return df["practice_best_time_s"] + offset


def evaluate(df: pd.DataFrame, predicted: pd.Series) -> dict:
    """
    Two metrics, matching what the project actually asks a model to do:
      - mae_seconds: average |predicted - actual| lap time error.
      - pole_accuracy: of the sessions in `df`, how often does the row with
        the LOWEST predicted time belong to the driver who actually took
        pole? This is the "which driver sets it" half of the prediction —
        MAE alone wouldn't tell us if we're getting driver rankings right.
    """
    errors = (predicted - df["target_lap_time_s"]).abs()

    scored = df[["session_key", "driver_number", "target_lap_time_s"]].copy()
    scored["predicted"] = predicted.values
    correct = 0
    sessions = scored.groupby("session_key")
    for _, session_rows in sessions:
        predicted_pole = session_rows.loc[session_rows["predicted"].idxmin(), "driver_number"]
        actual_pole = session_rows.loc[session_rows["target_lap_time_s"].idxmin(), "driver_number"]
        correct += int(predicted_pole == actual_pole)

    return {
        "mae_seconds": round(errors.mean(), 3),
        "pole_accuracy": round(correct / sessions.ngroups, 3),
        "n_rows": len(df),
        "n_sessions": sessions.ngroups,
    }
