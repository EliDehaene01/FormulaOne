"""
Per-prediction attribution for the qualifying lap time predictor, using
Captum's Integrated Gradients: for one specific (session, driver) prediction,
how much did each input push the predicted lap time up or down relative to a
neutral baseline?

WHY INTEGRATED GRADIENTS (not just the raw gradient at the input)
--------------------------------------------------------------------
The raw gradient at one point only says "if this input nudged slightly right
now, which way would the prediction move" — misleading wherever the network's
response is non-linear. Integrated Gradients instead walks a straight path
from a neutral BASELINE input to the actual input, accumulating the gradient
along the whole path. That gives attributions satisfying "completeness": they
sum (almost exactly) to (this prediction - the baseline's prediction) — see
`attribution_total_s` / `prediction_minus_baseline_s` below, a built-in sanity
check that this isn't just plausible-looking noise for a given row.

WHY WE ATTRIBUTE ON THE CONCATENATED EMBEDDING+NUMERIC VECTOR, NOT SEPARATE
CAPTUM CALLS PER INPUT GROUP
--------------------------------------------------------------------
driver_number/team_name/circuit_short_name reach the model as integer
embedding-table indices (see model.py) — you can't walk a path between two
integers the way you can between two floats; index 43.7 isn't a category.
The fix is the same one used for word-embedding attribution in NLP: do the
embedding LOOKUP first (a plain, ungraded forward call — driver_idx ->
driver_vec), then run Integrated Gradients on the resulting continuous
vectors, which model.py already concatenates into one tensor
(`[driver_vec | team_vec | circuit_vec | numeric_features]`) before it ever
reaches model.mlp. Attributing that single concatenated tensor in ONE
Captum call — instead of one call per input group, each holding the others
fixed at their real (non-baseline) value — is what makes the completeness
axiom hold: every attribution comes from the SAME straight-line path from
baseline to actual, so they sum to (this prediction - the baseline's
prediction) as the theory promises (this was verified against a real
prediction while building this: separate per-group calls left a ~25%
unexplained gap between the two; this joint version doesn't). Summing the
embedding-dimension slices of that one result gives one combined "how much
did knowing this driver/team/circuit matter" number per entity.

WHY INDEX 0 IS THE BASELINE FOR DRIVER/TEAM/CIRCUIT
--------------------------------------------------------------------
Index 0 isn't an arbitrary zero here — it's the reserved "<UNKNOWN>" category
(see train.py), which train.py deliberately trains on purpose (10% of
training rows relabeled to it) so that embedding row learns a real "no
specific information" representation instead of staying at random init (see
backend/models/README.md, bug #1). That makes it a principled baseline: "the
prediction if we knew nothing about who this driver/team/circuit specifically
was" — not just a convenient zero vector.

WHY ZERO IS THE BASELINE FOR NUMERIC FEATURES
--------------------------------------------------------------------
Numeric features are standardized ((x - train_mean) / train_std) before
reaching the model — so 0 in that space already means "the training set's
average value". "Every numeric feature at its average" is a sensible neutral
baseline for the same reason it's a sensible default for a missing value
elsewhere in this codebase (see features.py's fillna(mean) choice).

Run with:
    python -m backend.models.explain --session-key <key> --driver-number <num>
"""

import argparse
import sqlite3

import torch
from captum.attr import IntegratedGradients

from backend.data.cache import DB_PATH
from backend.models.features import NUMERIC_FEATURE_COLUMNS, build_feature_table
from backend.models.model import QualifyingLapTimePredictor
from backend.models.train import CHECKPOINT_PATH, _apply_category_index

N_STEPS = 50  # Captum's own default: points sampled along the baseline -> input path


def _load_model_and_row(session_key: int, driver_number: int):
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)  # see predict_qualifying_pace in tools.py for why weights_only=False is safe here

    conn = sqlite3.connect(DB_PATH)
    df = build_feature_table(conn)
    conn.close()

    row = df[(df["session_key"] == session_key) & (df["driver_number"] == driver_number)]
    if row.empty:
        raise ValueError(f"no cached practice+qualifying row for session {session_key}, driver {driver_number}")

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

    driver_idx = torch.tensor(_apply_category_index(row["driver_number"], checkpoint["driver_map"]), dtype=torch.long)
    team_idx = torch.tensor(_apply_category_index(row["team_name"], checkpoint["team_map"]), dtype=torch.long)
    circuit_idx = torch.tensor(_apply_category_index(row["circuit_short_name"], checkpoint["circuit_map"]), dtype=torch.long)
    numeric = (row[NUMERIC_FEATURE_COLUMNS] - checkpoint["numeric_mean"]) / checkpoint["numeric_std"]
    numeric_tensor = torch.tensor(numeric.clip(lower=-5, upper=5).to_numpy(), dtype=torch.float32)

    return model, driver_idx, team_idx, circuit_idx, numeric_tensor, float(row["target_lap_time_s"].iloc[0])


def explain_prediction(session_key: int, driver_number: int) -> dict:
    """
    Attributes one (session, driver) prediction back to its inputs: driver
    identity, team identity, circuit identity (each summed across their
    embedding dimensions into one number), and every individual numeric
    feature. Returns them ranked most-to-least influential by absolute
    contribution, alongside enough context to sanity-check the numbers.
    """
    model, driver_idx, team_idx, circuit_idx, numeric, actual_lap_time_s = _load_model_and_row(session_key, driver_number)

    with torch.no_grad():
        predicted_lap_time_s = model(driver_idx, team_idx, circuit_idx, numeric).item()

    driver_baseline = torch.zeros_like(driver_idx)
    team_baseline = torch.zeros_like(team_idx)
    circuit_baseline = torch.zeros_like(circuit_idx)
    numeric_baseline = torch.zeros_like(numeric)

    with torch.no_grad():
        baseline_prediction_s = model(driver_baseline, team_baseline, circuit_baseline, numeric_baseline).item()

    # Do the embedding lookup OURSELVES (a plain forward call, no gradient
    # needed for the lookup itself) so Captum sees one continuous tensor —
    # exactly what model.mlp actually consumes — instead of three
    # non-differentiable index tensors it can't walk a path between.
    with torch.no_grad():
        x = torch.cat([model.driver_embedding(driver_idx), model.team_embedding(team_idx), model.circuit_embedding(circuit_idx), numeric], dim=1)
        x_baseline = torch.cat(
            [model.driver_embedding(driver_baseline), model.team_embedding(team_baseline), model.circuit_embedding(circuit_baseline), numeric_baseline],
            dim=1,
        )

    ig = IntegratedGradients(lambda inp: model.mlp(inp).squeeze(-1))
    attr = ig.attribute(inputs=x, baselines=x_baseline, n_steps=N_STEPS)[0]

    driver_dim = model.driver_embedding.embedding_dim
    team_dim = model.team_embedding.embedding_dim
    circuit_dim = model.circuit_embedding.embedding_dim
    driver_attr, team_attr, circuit_attr, numeric_attr = attr.split([driver_dim, team_dim, circuit_dim, len(NUMERIC_FEATURE_COLUMNS)])

    contributions = [
        {"feature_name": "driver_identity", "contribution_s": driver_attr.sum().item()},
        {"feature_name": "team_identity", "contribution_s": team_attr.sum().item()},
        {"feature_name": "circuit_identity", "contribution_s": circuit_attr.sum().item()},
    ] + [
        {"feature_name": col, "contribution_s": val}
        for col, val in zip(NUMERIC_FEATURE_COLUMNS, numeric_attr.tolist())
    ]
    contributions.sort(key=lambda c: abs(c["contribution_s"]), reverse=True)
    for c in contributions:
        c["contribution_s"] = round(c["contribution_s"], 4)

    attribution_total_s = round(sum(c["contribution_s"] for c in contributions), 4)

    return {
        "session_key": session_key,
        "driver_number": driver_number,
        "predicted_lap_time_s": round(predicted_lap_time_s, 3),
        "actual_lap_time_s": round(actual_lap_time_s, 3),
        "baseline_prediction_s": round(baseline_prediction_s, 3),
        # Completeness check (see module docstring): should be close to
        # prediction_minus_baseline_s. A big gap between the two would mean
        # these attributions have drifted from IG's theoretical guarantee for
        # this particular row and shouldn't be trusted as-is.
        "attribution_total_s": attribution_total_s,
        "prediction_minus_baseline_s": round(predicted_lap_time_s - baseline_prediction_s, 3),
        "contributions": contributions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain one qualifying prediction via Captum Integrated Gradients.")
    parser.add_argument("--session-key", type=int, required=True)
    parser.add_argument("--driver-number", type=int, required=True)
    parser.add_argument("--top", type=int, default=10, help="How many top contributions to print (default 10).")
    args = parser.parse_args()

    result = explain_prediction(args.session_key, args.driver_number)
    print(f"Predicted: {result['predicted_lap_time_s']}s (actual: {result['actual_lap_time_s']}s)")
    print(f"Baseline (unknown driver/team/circuit, average conditions): {result['baseline_prediction_s']}s")
    print(
        f"Sum of attributions: {result['attribution_total_s']}s "
        f"(should be close to prediction - baseline = {result['prediction_minus_baseline_s']}s)\n"
    )
    print(f"{'feature':25s} {'contribution (s)':>18s}")
    print("-" * 45)
    for c in result["contributions"][: args.top]:
        print(f"{c['feature_name']:25s} {c['contribution_s']:>18.4f}")


if __name__ == "__main__":
    main()
