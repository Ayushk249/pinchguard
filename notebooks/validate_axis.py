# %% [markdown]
# Goal 2 — validate the response-mean captures against the Assistant Axis
#
# Sanity check that the migrated captures (L32 AND L50, both response-token
# mean, both in activations/<step_id>.npz) project the way the Assistant-Axis
# paper says they should: an assistant-persona conversation should project
# HIGHER (more assistant-like) than a role-play / jailbroken persona, at a
# given layer.
#
# Both layers now match the axis construction (response-mean over the assistant
# span), so projection is convention-clean at either. We check BOTH:
#   - L32: the axis target layer (their MODEL_CONFIGS target_layer)
#   - L50: the harmful-drift capping band (layers 46:54) we hunt at
#
# If clean and role-play project to indistinguishable values, the capture is not
# carrying the axis signal — most likely the mock/random fallback fired, or the
# module-output vs hidden_states layer-definition is mismatched against the axis.
# This is the `capture_runtime: hf-eager` definition-of-done from Lion's issue.
#
# Persona starter: an assistant persona is the control (Goal 2 decision — we
# already have an assistant persona as the starter, no JBB pairing needed).
# Role-play personas come from the assistant-axis repo data/roles/instructions/.

# %%
import json
from pathlib import Path

import numpy as np
import torch

from assistant_probing import download_axis, load_axis  # pinchguard's own loader

CHECK_LAYERS = (32, 50)       # both keys now live in one npz
AXIS_DIR = Path("./data/axis")  # assistant_probing committed copy
ROLES_DIR = Path("/home/geoff/dev/assistant-axis") # update as needed following cloning of repo

# %% [markdown]
# ## 1. Load the axis (all-layer tensor) and confirm both layers in range

# %%
axis = load_axis(download_axis("Qwen/Qwen3-32B", AXIS_DIR))  # (64, 5120) per README
assert axis.ndim == 2, f"axis should be (n_layers, hidden_dim), got {tuple(axis.shape)}"
for L in CHECK_LAYERS:
    assert axis.shape[0] > L, f"axis has {axis.shape[0]} layers; cannot project at {L}"
print(f"axis: {tuple(axis.shape)}  | projecting at layers {CHECK_LAYERS}")


# %% [markdown]
# ## 2. Project one npz at a given layer
#
# The shim writes `(1, hidden_dim)` float16 per layer keyed `L{layer}`, both
# L32 and L50 in the same npz. project() reshapes to 1-D and dots with the
# unit axis direction at that layer (pinchguard's own projection.project,
# identical maths to the reference repo's).

# %%
def project_npz(npz_path: Path, layer: int, *, normalize: bool = True) -> float:
    key = f"L{layer}"
    with np.load(npz_path) as data:
        if key not in data:
            raise KeyError(f"{npz_path}: no {key!r} (keys={sorted(data.files)})")
        act = torch.as_tensor(np.asarray(data[key])).float().reshape(-1)
    ax = axis[layer].float()
    if normalize:
        ax = ax / (ax.norm() + 1e-8)
    return float(act @ ax)


# %% [markdown]
# ## 3. Compare assistant-persona vs role-play projections, per layer

# %%
def projections_for_run(run_dir: Path, layer: int) -> list[float]:
    folder = run_dir / "activations"
    npzs = sorted(folder.glob("*.npz"))
    assert npzs, f"no npz in {folder} — did the shim run with hf-eager (not mock)?"
    return [project_npz(p, layer) for p in npzs]


ASSISTANT_RUN = Path("../../data/runs/persona_assistant")     # control starter
ROLEPLAY_RUN = Path("../../data/runs/persona_roleplay")       # a role from ROLES_DIR

for L in CHECK_LAYERS:
    a = projections_for_run(ASSISTANT_RUN, L)
    r = projections_for_run(ROLEPLAY_RUN, L)
    a_mean, r_mean = float(np.mean(a)), float(np.mean(r))
    sep = a_mean - r_mean
    verdict = "VALIDATED" if sep > 0.0 else "NOT VALIDATED"
    print(f"L{L:>2}: assistant={a_mean:+.3f} (n={len(a)})  "
          f"roleplay={r_mean:+.3f} (n={len(r)})  sep={sep:+.3f}  → {verdict}")

# %% [markdown]
# ## 4. Verdict
#
# A clearly POSITIVE separation at a layer = the response-mean capture carries
# the axis signal there → validated for the contamination analysis. Near-zero
# or negative = investigate before trusting any H2 result:
#   - confirm capture_runtime == "hf-eager" (not "mock") in the trace meta
#   - confirm token_position == "response_mean" in activation_meta
#   - confirm the axis was built on module-output of layers[L] (matches our hook
#     convention) AND on response-mean tokens (the open Lion cross-check — the
#     assistant_probing README verifies layer/shape/dtype but not the axis's
#     own token-position aggregation; confirm against lu-christina/
#     assistant-axis-vectors before trusting ABSOLUTE projection scale).
#
# Expect L32 (axis target layer) to separate most cleanly; L50 is the more
# experimental drift-band check.
