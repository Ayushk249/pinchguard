"""Concern C — Probe Eval: fetching the Assistant Axis (AGENTS.md §2 row C).

Downloads/loads the precomputed Assistant Axis (Lu et al. 2026,
arXiv:2601.10387) for the model Pinchguard captures activations from.
Self-contained — does not import the safety-research/assistant-axis
reference repo, which won't exist on the production box.

See README.md for how `data/axis/` was populated.
"""

from .axis_download import (
    AXIS_FILES,
    AXIS_REPO_ID,
    download_axis,
    load_axis,
)

__all__ = [
    "AXIS_FILES",
    "AXIS_REPO_ID",
    "download_axis",
    "load_axis",
]
