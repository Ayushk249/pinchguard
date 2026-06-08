# Run manifest — scenario_02_baseline_20260606T065855Z

Pinchguard v2 first cut, **baseline** (control) capture. A neutral helpful
assistant (`scenarios/02`) ran 15 turns over 15 neutral feed posts through the
unchanged capture shim. Produced for the **Assistant-Axis probe** (Lion / Concern C).

## Bundle contents
| File | What |
|---|---|
| `traces.jsonl` | 15 schema-v0.2 rows, one per turn (`output_raw` carries the full `<think>…</think>` + answer). |
| `activations/<step_id>.npz` | 15 arrays, one per turn; dict key `L32`, shape `(1, 5120)`. |
| `loop_records.jsonl` | per-turn behaviour: `reasoning`, `action`, `kwargs`, `crossed_boundary`, `scratchpad`. |
| `comments_dryrun.jsonl` | the agent's logged comments (dry-run; nothing was posted). |

## Capture geometry (confirm before trusting a projection)
- **model_id**: `/datapool/analysis_data/tara/pinchguard/models`
- **layer**: 32  (decoder module-output of `model.model.layers[32]`; `num_hidden_layers=64`, midpoint)
- **token_position**: `last_input`  (prompt-only forward; last input token)
- **hidden_dim**: 5120
- **activation key format**: `L32`  → `np.load(npz)["L32"]`
- **dtype** (stored activation): `float16`
- **quantization** (model compute): `nf4`  (weights 4-bit; capture is fp16 off the residual stream)
- **capture_runtime**: `hf-eager`  ← must be `hf-eager`, not `mock`, to trust drift

## Join key
Everything joins on **`turn`** (0..14), 1:1 because the loop makes exactly one
model call per turn. `turn_id` is also 1:1 (forward-compat). To pair an activation
with its behaviour: `traces.jsonl[turn].step_id` → `activations/<step_id>.npz`, and
`loop_records.jsonl[turn]` for the action/reasoning/scratchpad.

## Baseline expectations (this is the control)
- boundary crossings: **0** (expected ~0 — no contamination this cut).
- The probe should read a roughly **flat** Assistant-Axis projection here. A later
  *treatment* run (contamination turns 6–10, recovery 11–15) is the contrast.

## For Lion
Extract the axis at the **same layer (32) and token_position
(`last_input`)** this run captured, so projection and capture are
self-consistent. Confirm those two values before trusting any projection.
