"""Tiny RNG checks; no dataset loading, training, or inference."""

import random

import numpy as np
import pytest
import torch

from src.config import get_seed
from src.models import AdvSTAD, AdvSTAD_NoSpatial, TranAD, TranAD_Corrected
from src.reproducibility import seed_everything, seeding_metadata


@pytest.mark.parametrize('value', [True, -1, 2**32, 1.5, '1', None])
def test_invalid_common_seed_is_rejected(value):
    with pytest.raises(ValueError, match='seed must be an integer'):
        get_seed({'seed': value})


def test_default_seed_and_all_rng_streams_are_repeatable(monkeypatch):
    assert get_seed({}) == get_seed({'seed': 1}) == 1
    calls = []
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda seed: calls.append(seed))
    seed_everything(12)
    expected = (random.random(), np.random.rand(4), torch.rand(4))
    seed_everything(12)
    actual = (random.random(), np.random.rand(4), torch.rand(4))
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert torch.equal(actual[2], expected[2])
    assert calls == [12, 12]
    metadata = seeding_metadata()
    assert metadata['python_seed'] == metadata['numpy_seed'] == metadata['torch_seed'] == 12
    # No attempt is made to change the process hash seed at runtime.
    assert 'python_hash_seed' in metadata
    seed_everything(1)


@pytest.mark.parametrize('model_class', [TranAD, TranAD_Corrected, AdvSTAD_NoSpatial, AdvSTAD])
def test_original_torch_initialization_and_rng_progress_are_preserved(model_class):
    torch.manual_seed(1)
    original = model_class(3)
    expected_rng = torch.get_rng_state()
    expected_flags = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    seed_everything(1)
    actual = model_class(3)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert original.state_dict().keys() == actual.state_dict().keys()
    for name, value in original.state_dict().items():
        assert torch.equal(value, actual.state_dict()[name]), name
    assert expected_flags == (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
