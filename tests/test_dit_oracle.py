# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the decomposed DiT fidelity oracle."""

import pytest
import torch
import torch.nn as nn

from dit_oracle import discover_blocks, dit_latent_fidelity


def test_identical_zero_signal_is_perfect_fidelity():
    zero = torch.zeros(4)
    result = dit_latent_fidelity([zero], [zero], zero, zero)

    assert result.final_cos == 1.0
    assert result.final_sqnr == float("inf")
    assert result.first_bad_block is None


def test_nonfinite_candidate_localizes_as_divergence():
    reference = torch.ones(4)
    candidate = torch.tensor([1.0, 1.0, float("nan"), 1.0])
    result = dit_latent_fidelity([reference], [candidate], reference, candidate)

    assert result.final_cos == float("-inf")
    assert result.final_sqnr == float("-inf")
    assert result.first_bad_block == 0


def test_block_count_mismatch_is_runtime_validation():
    value = torch.ones(4)
    with pytest.raises(ValueError, match="Block count mismatch"):
        dit_latent_fidelity([value], [], value, value)


def test_discovers_supported_block_container():
    model = nn.Module()
    model.transformer_blocks = nn.ModuleList([nn.Identity()])
    assert discover_blocks(model) is model.transformer_blocks
