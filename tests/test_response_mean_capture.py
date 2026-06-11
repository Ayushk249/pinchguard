"""Unit tests for the response_mean capture contract (Goal 1).

These pin the parts that don't need GPU weights:

* `MockCapturer` accepts `response_mean` and stamps the right meta, so the
  shim wiring is fully exercisable on CPU/Codespaces.
* The averaging rule the `HFCapturer._capture_response_mean` hook implements —
  "drop the prompt-forward fire, average over the generated-token fires, but
  keep the sole fire if generation produced nothing" — is isolated here as a
  pure function so its off-by-one (the thing that silently corrupts a mean) is
  tested without loading a 32B model.

Post-migration: both L32 and L50 are captured response_mean into one npz; the
last_input path remains in the code for the legacy convention but is no longer
the default. response_mean is now DEFAULT_TOKEN_POSITION.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shim.capture import (  # noqa: E402
    VALID_TOKEN_POSITIONS,
    MockCapturer,
)


def _reduce_fires(fires: list[np.ndarray]) -> np.ndarray:
    """Mirror of the HFCapturer hook reducer (numpy stand-in for torch).

    fires[0] is the prompt forward (last prompt token); fires[1:] are the
    generated tokens. We average the generated fires, falling back to the sole
    fire when generation produced none.
    """
    gen = fires[1:] if len(fires) > 1 else fires
    return np.stack(gen, axis=1).mean(axis=1)


def test_mock_accepts_response_mean() -> None:
    cap = MockCapturer(layers=(50,), token_position="response_mean")
    res = cap.capture([{"role": "user", "content": "hi"}], "step-x")
    assert res.activation_meta["token_position"] == "response_mean"
    assert res.activation_meta["layers"] == [50]
    assert "L50" in res.activations


def test_mock_rejects_unknown_position() -> None:
    with pytest.raises(ValueError, match="invalid"):
        MockCapturer(layers=(50,), token_position="middle_token")


def test_response_mean_set_membership() -> None:
    assert "response_mean" in VALID_TOKEN_POSITIONS
    assert "last_input" in VALID_TOKEN_POSITIONS


def test_reducer_drops_prompt_fire() -> None:
    # prompt fire = 100s (should be excluded); two generated fires = 2 and 4.
    h = 4
    prompt = np.full((1, h), 100.0)
    g1 = np.full((1, h), 2.0)
    g2 = np.full((1, h), 4.0)
    out = _reduce_fires([prompt, g1, g2])
    assert np.allclose(out, 3.0), out  # mean(2,4) — prompt 100 excluded


def test_reducer_keeps_sole_fire() -> None:
    # immediate EOS: only the prompt fire exists; must not error / empty-mean.
    prompt = np.full((1, 4), 7.0)
    out = _reduce_fires([prompt])
    assert np.allclose(out, 7.0)


def test_default_position_is_response_mean() -> None:
    # Migration: both layers now default to response_mean.
    from shim.capture import DEFAULT_TOKEN_POSITION, DEFAULT_LAYERS
    assert DEFAULT_TOKEN_POSITION == "response_mean"
    assert DEFAULT_LAYERS == (32, 50)
