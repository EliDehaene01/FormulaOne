"""
Hyperparameter search over the qualifying lap time predictor, using Optuna.

Run with:
    python -m backend.models.tune                  # 50 trials, retrains + saves the winner
    python -m backend.models.tune --n-trials 20     # fewer trials, faster
    python -m backend.models.tune --no-retrain      # search only, don't touch qualifying_model.pt

WHY OPTUNA (not grid search)
------------------------------
Grid search scales combinatorially with the number of hyperparameters —
with 6 knobs here (number of layers, each layer's width, dropout, learning
rate, batch size, weight decay), even a modest grid becomes thousands of
runs. Optuna's default sampler (TPE, Tree-structured Parzen Estimator)
treats this as Bayesian optimization instead: each trial's result biases
where the next trial samples, so it converges on good regions of the search
space in far fewer trials than an exhaustive grid would need.

WHAT WE OPTIMIZE FOR
----------------------
Best validation MAE (train.py's temporal val split) — never the test split.
Test must stay genuinely unseen future data, unseen by the WHOLE modeling
process including hyperparameter search, not just the final training run —
tuning against test would be the same leakage sin as everything
backend/models/features.py's docstring warns against, just at the
hyperparameter level instead of the feature level.

EACH TRIAL RETRAINS FROM SCRATCH, CHEAPLY
------------------------------------------
Every trial calls train.train_model() with save_checkpoint=False and a
smaller epoch budget/patience (TUNING_MAX_EPOCHS/TUNING_EARLY_STOP_PATIENCE)
than a full run — early stopping still applies within a trial, so a bad
trial fails fast rather than burning the full epoch budget for nothing.
Only the FINAL selected hyperparameters (the best trial found) get
retrained at the full epoch budget and written to qualifying_model.pt, via
retrain_best() — every other trial's weights are discarded, only its score
matters.
"""

import argparse

import optuna

from backend.models.train import EARLY_STOP_PATIENCE, MAX_EPOCHS, train_model

# Deliberately smaller than train.py's full budget (MAX_EPOCHS/
# EARLY_STOP_PATIENCE) — with dozens of trials, each one needs to be cheap.
# A genuinely good hyperparameter set will still separate itself from a bad
# one well before 60 epochs; the final retrain_best() call uses the FULL
# budget, so this shortcut only affects which candidate gets chosen, not
# how well the final saved model is trained.
TUNING_MAX_EPOCHS = min(60, MAX_EPOCHS)
TUNING_EARLY_STOP_PATIENCE = min(10, EARLY_STOP_PATIENCE)

HIDDEN_LAYER_WIDTH_CHOICES = [16, 32, 64, 128, 256]


def _objective(trial: optuna.Trial) -> float:
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_sizes = tuple(
        trial.suggest_categorical(f"hidden_size_{i}", HIDDEN_LAYER_WIDTH_CHOICES) for i in range(n_layers)
    )
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    return train_model(
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        learning_rate=learning_rate,
        batch_size=batch_size,
        weight_decay=weight_decay,
        max_epochs=TUNING_MAX_EPOCHS,
        early_stop_patience=TUNING_EARLY_STOP_PATIENCE,
        save_checkpoint=False,
    )


def retrain_best(best_params: dict) -> None:
    """
    Retrain with the winning trial's hyperparameters at the FULL epoch
    budget and save the checkpoint — the only call in this module that
    actually overwrites qualifying_model.pt.
    """
    n_layers = best_params["n_layers"]
    hidden_sizes = tuple(best_params[f"hidden_size_{i}"] for i in range(n_layers))
    train_model(
        hidden_sizes=hidden_sizes,
        dropout=best_params["dropout"],
        learning_rate=best_params["learning_rate"],
        batch_size=best_params["batch_size"],
        weight_decay=best_params["weight_decay"],
        save_checkpoint=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter search for the qualifying lap time predictor.")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument(
        "--no-retrain",
        action="store_true",
        help="Search only — print the winning hyperparameters but don't retrain/overwrite qualifying_model.pt.",
    )
    args = parser.parse_args()

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(_objective, n_trials=args.n_trials)

    print(f"\nBest val MAE across {args.n_trials} trials: {study.best_value:.3f}s")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    if args.no_retrain:
        print("\n--no-retrain set: qualifying_model.pt left untouched.")
        return

    print("\nRetraining with the winning hyperparameters at the full epoch budget...")
    retrain_best(study.best_params)


if __name__ == "__main__":
    main()
