"""
Smoke tests for backend/models/model.py.

The second test here is a classic PyTorch debugging technique worth
knowing: before trusting a model on real data, check it can memorize a
handful of made-up examples. If a tiny network can't drive its loss to
~zero on 8 rows it's allowed to memorize, something structural is wrong
(wrong shapes, a frozen layer, a broken loss) — no amount of real data or
training time will fix a bug like that.

Run with:
    python -m backend.tests.test_model
"""

import torch

from backend.models.model import QualifyingLapTimePredictor


def test_forward_pass_shape():
    model = QualifyingLapTimePredictor(num_drivers=20, num_teams=10, num_circuits=24, num_numeric_features=15)
    batch_size = 4
    driver_idx = torch.randint(0, 21, (batch_size,))
    team_idx = torch.randint(0, 11, (batch_size,))
    circuit_idx = torch.randint(0, 25, (batch_size,))
    numeric = torch.randn(batch_size, 15)

    output = model(driver_idx, team_idx, circuit_idx, numeric)

    assert output.shape == (batch_size,), f"expected shape ({batch_size},), got {tuple(output.shape)}"


def test_model_can_overfit_a_tiny_batch():
    torch.manual_seed(0)
    model = QualifyingLapTimePredictor(num_drivers=3, num_teams=2, num_circuits=2, num_numeric_features=4)

    n = 8
    driver_idx = torch.randint(0, 4, (n,))
    team_idx = torch.randint(0, 3, (n,))
    circuit_idx = torch.randint(0, 3, (n,))
    numeric = torch.randn(n, 4)
    target = torch.rand(n) * 10 + 80  # arbitrary "lap times" in a plausible range

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = torch.nn.MSELoss()

    for _ in range(300):
        optimizer.zero_grad()
        predictions = model(driver_idx, team_idx, circuit_idx, numeric)
        loss = criterion(predictions, target)
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.1, f"model should be able to memorize 8 rows; final loss was {loss.item():.4f}"


if __name__ == "__main__":
    test_forward_pass_shape()
    test_model_can_overfit_a_tiny_batch()
    print("All model tests passed.")
