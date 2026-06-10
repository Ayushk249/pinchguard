# assistant_probing

Concern C (Probe Eval) starting point: fetches the precomputed **Assistant
Axis** (Lu et al. 2026, *The Assistant Axis*, arXiv:2601.10387) for the model
Pinchguard captures activations from.

Self-contained on purpose — it does not import the `safety-research/
assistant-axis` reference repo (that exists locally only as a side-by-side
reference; it won't be present on the production box).

## What's here

- `axis_download.py` — `download_axis()` + `load_axis()`: fetch and load the axis
  tensor as a `torch.Tensor` of shape `(n_layers, hidden_dim)`.
- `data/axis/qwen-3-32b/assistant_axis.pt` — our committed local copy of the
  published Qwen 3 32B axis (see below for how it got there).
- `projection.py` — Step 1 of the drift experiment ("does drift occur at
  all?"): projects each turn's captured `L32` activation onto the axis and
  prints `label_probe` per turn, so a drop during the contamination window is
  visible by eye. See "Linear probe" below.
- `linear_probe.py` *(gitignored, WIP)* — trains a logistic-regression probe
  directly from the published per-role activation vectors
  (`lu-christina/assistant-axis-vectors`, `qwen-3-32b/`): 275 role vectors
  (label 0) vs. the default Assistant vector (label 1). The trained probe
  scores a single turn capture for `prob_assistant`. Use `train` to fit and
  save `probe.joblib`, `infer` to score a `.npz`.

## Where the axis came from

Pre-computed axes for Gemma 2 27B, Qwen 3 32B, and Llama 3.3 70B are
published at
[`lu-christina/assistant-axis-vectors`](https://huggingface.co/datasets/lu-christina/assistant-axis-vectors)
on HuggingFace (a `dataset` repo). Pinchguard captures activations from
Qwen 3 32B (AGENTS.md §3/§5; see `notebooks/gpu_test/example_activation/`),
and that model already has a published axis — so rather than replicating the
275-role generation + LLM-judge pipeline that produced it
(`pipeline/1_generate.py` … `5_axis.py` in the reference repo), we just
downloaded the published one.

## How `data/axis/` was populated

From the `pinchguard` repo root, with the `assistant_probing` extra synced:

```powershell
uv sync --extra assistant_probing
```

then:

```python
import shutil
from assistant_probing import download_axis

src = download_axis("Qwen/Qwen3-32B")  # hf_hub_download — caches under HF_HOME
shutil.copy(src, "assistant_probing/data/axis/qwen-3-32b/assistant_axis.pt")
```

`download_axis()` calls `huggingface_hub.hf_hub_download(repo_id=
"lu-christina/assistant-axis-vectors", filename="qwen-3-32b/assistant_axis.pt",
repo_type="dataset")`, which downloads the file (~657 KB) to the local HF
cache (`HF_HOME`, see `.env.example`); we then copied that cached file into
`data/axis/` so it's committed and the module works offline (the production
sandbox likely has no HF access).

## Usage

```python
from assistant_probing import download_axis, load_axis

axis_path = download_axis("Qwen/Qwen3-32B")  # returns the committed local copy
axis = load_axis(axis_path)                   # torch.Tensor, shape (64, 5120)
```

`download_axis()` checks `data/axis/` first and returns that path immediately
if present — it only reaches out to HuggingFace when no local copy exists yet
(or `force_download=True` is passed).

## Linear probe (Step 1 — does drift occur at all?)

`scenario_02` captured 15 turns of activations
(`/datapool/analysis_data/tara/pinchguard/runs/scenario_02/activations/`,
15 `.npz` files, filename order = turn order): turns 0-4 clean, 5-9
contaminated, 10-14 clean recovery. `projection.py` projects each turn's
`L32` activation onto the axis and prints `label_probe` per turn:

```powershell
uv run python -m assistant_probing.projection \
    /datapool/analysis_data/tara/pinchguard/runs/scenario_02/activations \
    ./data/axis
```

```
turn  0  label_probe =  -19.5419   <step_id>.npz
turn  1  label_probe =  -20.3470   <step_id>.npz
...
```

What a positive result looks like: `label_probe` drops during turns ~5-9
(contaminated) relative to turns 0-4 and 10-14 (clean) — that's evidence the
Assistant Axis is tracking the contamination. A flat line across all three
blocks means the axis isn't picking it up here (the experiment's "rethink"
branch). `run_probe()` returns `[(filename, label_probe), ...]` in turn order
for further analysis (deltas between block means, plotting, etc.).

## Verified facts (Qwen 3 32B)

- **axis shape**: `(64, 5120)` — one direction per decoder layer, `hidden_dim=5120`
- **dtype**: `bfloat16`
- **target layer**: `32` (module-output of `model.model.layers[32]` /
  residual stream "resid_post" — matches `notebooks/gpu_test/
  example_activation/README.md` and `MODEL_CONFIGS["Qwen/Qwen3-32B"]
  ["target_layer"]` in the reference repo's `assistant_axis/models.py`)
