"""Unit tests for the quantized multi-GPU guard (`_guard_single_cuda_device`).

The guard is the production safety net for the M1 footgun verified 2026-06-06:
a bitsandbytes 4-bit/8-bit model sharded across both Blackwell GPUs corrupts
the forward pass (token-salad generation, garbage L32 activations). Pinning a
single device is coherent. Tested as a pure function so no weights/GPU needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shim.capture import _guard_single_cuda_device  # noqa: E402


def test_quantized_multi_device_raises() -> None:
    with pytest.raises(RuntimeError, match="sharded"):
        _guard_single_cuda_device("nf4", {0, 1})


def test_quantized_single_device_ok() -> None:
    _guard_single_cuda_device("nf4", {1})
    _guard_single_cuda_device("int8", {0})


def test_unquantized_multi_device_ok() -> None:
    # Full-precision sharding is unaffected by the bnb bug, so it must not fire.
    _guard_single_cuda_device(None, {0, 1})


def test_quantized_cpu_path_ok() -> None:
    # Empty device set = CPU/full-precision path; nothing to guard.
    _guard_single_cuda_device("nf4", set())
