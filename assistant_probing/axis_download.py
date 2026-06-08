"""Download and load published Assistant Axis vectors.

The Assistant Axis is a per-layer direction in residual stream space that
captures how "Assistant-like" a model persona is. This module fetches a
precomputed axis from Hugging Face and provides helpers to download it into a
local destination and load it as a PyTorch tensor.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import torch

AXIS_REPO_ID = "lu-christina/assistant-axis-vectors"

AXIS_FILES: dict[str, dict[str, str]] = {
    "Qwen/Qwen3-32B": {
        "filename": "qwen-3-32b/assistant_axis.pt",
        "local": "qwen-3-32b/assistant_axis.pt",
    },
}


def download_axis(
    model_id: str,
    dest_dir: str | Path,
    *,
    revision: str | None = None,
    force_download: bool = False,
) -> Path:
    """Ensure a published axis for `model_id` exists under `dest_dir`."""
    try:
        files = AXIS_FILES[model_id]
    except KeyError:
        raise KeyError(
            f"no published assistant-axis file registered for {model_id!r}; "
            f"known models: {sorted(AXIS_FILES)}"
        ) from None

    dest_path = Path(dest_dir) / files["local"]
    if dest_path.exists() and not force_download:
        return dest_path

    from huggingface_hub import hf_hub_download

    cached_path = Path(
        hf_hub_download(
            repo_id=AXIS_REPO_ID,
            filename=files["filename"],
            repo_type="dataset",
            revision=revision,
        )
    )

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(cached_path, dest_path)
    return dest_path


def load_axis(path: str | Path) -> torch.Tensor:
    """Load an axis tensor from a `.pt` file."""
    data = torch.load(Path(path), map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        if "axis" not in data:
            raise ValueError(f"{path}: dict has no 'axis' key (keys={sorted(data)})")
        return data["axis"]
    return data
