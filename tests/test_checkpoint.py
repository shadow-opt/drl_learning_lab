from __future__ import annotations

import torch

from drl_lab.common.checkpoint import CheckpointMetadata, load_checkpoint, save_checkpoint
from drl_lab.common.networks import MLP
from drl_lab.common.seed import set_global_seed


def test_checkpoint_restores_model_parameters(tmp_path) -> None:  # type: ignore[no-untyped-def]
    set_global_seed(5)
    model = MLP(input_dim=4, hidden_sizes=[8], output_dim=2)
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}

    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(checkpoint_path, model, metadata=CheckpointMetadata(step=10, seed=5))

    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)

    payload = load_checkpoint(checkpoint_path, model)

    for key, value in model.state_dict().items():
        assert torch.equal(value, original[key])
    assert payload["metadata"].step == 10
    assert payload["metadata"].seed == 5
