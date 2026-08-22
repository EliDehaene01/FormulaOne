"""
The PyTorch model: a small feedforward network that predicts a driver's
qualifying lap time (in seconds), given who they are, what team they're in,
which circuit it is, and the numeric features from features.py.

WHAT AN EMBEDDING LAYER ACTUALLY IS
-------------------------------------
`driver_number` and `team_name` are categories, not numbers — Red Bull
isn't "more" than Ferrari. If we fed the model a raw category ID (like
"driver 44"), it would wrongly assume ID 44 means something quantitatively
close to ID 43. An `nn.Embedding` sidesteps this: it's a lookup table of
shape (num_categories, embedding_dim), one learned vector per category,
initialized randomly. There's no hand-picked meaning to the vector's
numbers — during training, backpropagation nudges each driver's vector
wherever makes the model's predictions more accurate. Over enough
qualifying sessions, "Verstappen's vector" and "Hamilton's vector" end up
encoding whatever latent traits (raw pace, quali-specific skill, etc.)
actually help predict lap time — the network figures out what those traits
are, we never label them ourselves. It's the same trick word embeddings use
for "king"/"queen" in NLP, just applied to F1 drivers/teams/circuits.

WHY A FEEDFORWARD NET (not something fancier)
------------------------------------------------
Our input is a fixed-size row of tabular features — no sequence, no image,
no variable-length text. A feedforward net (a.k.a. multi-layer perceptron:
stack of Linear -> activation -> Linear...) is the standard, simplest tool
for "fixed-size numeric input -> numeric output" and is the right amount of
model for ~1,200 training rows. Something like an RNN or Transformer would
be solving a problem we don't have (sequence order) and would badly overfit
this little data.
"""

import torch
import torch.nn as nn


class QualifyingLapTimePredictor(nn.Module):
    """
    forward() takes four inputs (three category-index tensors + one numeric
    features tensor) and returns one predicted lap time (in seconds) per row.
    """

    def __init__(
        self,
        num_drivers: int,
        num_teams: int,
        num_circuits: int,
        num_numeric_features: int,
        driver_embed_dim: int = 8,
        team_embed_dim: int = 4,
        circuit_embed_dim: int = 6,
        hidden_sizes: tuple[int, ...] = (64, 32),
    ):
        super().__init__()  # required boilerplate: lets nn.Module wire up parameter tracking, .to(device), etc.

        # nn.Embedding(N, D): a trainable (N, D) matrix. Given an integer
        # index tensor, it returns the corresponding row(s) — a
        # differentiable "array lookup". +1 on num_* reserves index 0 for
        # an explicit "<UNKNOWN>" category (see train.py) — used for a
        # driver/team/circuit the model never saw during training.
        self.driver_embedding = nn.Embedding(num_drivers + 1, driver_embed_dim)
        self.team_embedding = nn.Embedding(num_teams + 1, team_embed_dim)
        self.circuit_embedding = nn.Embedding(num_circuits + 1, circuit_embed_dim)

        # Everything gets concatenated into one flat vector per row before
        # going through the feedforward stack: [driver_vec | team_vec |
        # circuit_vec | numeric_features].
        input_dim = driver_embed_dim + team_embed_dim + circuit_embed_dim + num_numeric_features

        # nn.Sequential chains layers in order. Linear(in, out) is
        # y = xW^T + b — a learned weighted sum. ReLU (max(0, x)) is what
        # makes the network non-linear; stacking plain Linear layers with no
        # activation between them would collapse into a single big Linear
        # layer no matter how many you stack, since a chain of linear maps
        # is still a linear map. ReLU breaks that, letting the network learn
        # curved, non-linear relationships (e.g. "temperature only matters
        # once it's above X").
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # final layer: hidden features -> a single predicted lap time
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        driver_idx: torch.Tensor,
        team_idx: torch.Tensor,
        circuit_idx: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Every input tensor's first dimension is the batch size — PyTorch
        processes a whole batch of rows at once (as one set of matrix
        multiplications), not row-by-row, which is what makes it fast.

        driver_idx/team_idx/circuit_idx: shape (batch,), dtype long (integer
        indices — nn.Embedding requires integer indices, not floats).
        numeric_features: shape (batch, num_numeric_features), dtype float.
        """
        driver_vec = self.driver_embedding(driver_idx)  # (batch,) -> (batch, driver_embed_dim)
        team_vec = self.team_embedding(team_idx)
        circuit_vec = self.circuit_embedding(circuit_idx)

        # dim=1 concatenates along the feature axis (dim 0 is the batch
        # axis — we're not stacking rows, we're widening each row).
        x = torch.cat([driver_vec, team_vec, circuit_vec, numeric_features], dim=1)

        predicted_time = self.mlp(x)  # shape (batch, 1)
        return predicted_time.squeeze(-1)  # shape (batch,) — drop the trailing size-1 dim to match the target's shape
