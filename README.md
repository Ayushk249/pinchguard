# Pinchguard

Project for the TARA June 26 cohort. **Goal:** train a passive, black-box
predictor that watches an autonomous agent's observable behaviour and flags
**alignment drift** — the gradual, peer-contamination-driven decay that an
[OpenClaw](https://github.com/openclaw/openclaw) executive assistant can
suffer. Stretch: explore or simulate the effect of participating in 
agent-native social platforms like [Moltbook](https://moltbook.com/) on drift.

The bet: even when the production model is closed-weights and we can't run
real-time mech-interp probes, an offline-trained predictor that consumes
operator-visible traces can give a useful early-warning signal. We test the
bet by capturing both behavioural traces *and* internal activations from an
open-weights stand-in, so we can compare a probe-grade label against what a
black-box predictor would have seen.

Full threat model, schema, and architectural invariants in
**[AGENTS.md](./AGENTS.md)**.

## Separation of concerns

Work is split across four concerns, each with its own input/output contract.
Full table in [AGENTS.md §2](./AGENTS.md). Short version:

| Concern | Owns | Input | Output |
|---|---|---|---|
| **A · Data Logging** | Inference shim, activation capture, schema, parity validator | Curl/OpenClaw requests | `traces.jsonl` + `activations/*.npz` |
| **B · Behavioural Eval** | LLM-judge rubric, human validation | `traces.jsonl` | `traces_enriched.jsonl` with `label_behav` |
| **C · Probe Eval** | Assistant-axis linear probe, layer/site sweeps | `activations/*.npz` | `probe_scores.jsonl` with `label_probe` |
| **D · Predictor** | Feature engineering, model selection, ablations | Enriched traces ⋈ probe scores on `step_id` | Trained predictor + CV metrics |
| **E · Scenarios** | Which conditions to expose an agent to | test-harness | parameters for the test-harness |  

This repo is shared. Concern A is implemented here today; B/C/D/E will land
alongside as their owners scope and contribute.

## Project tree

```
shim/                  Concern A — OpenAI-compatible inference shim
  main.py                FastAPI /v1/chat/completions; mints step_id;
                         writes traces.jsonl + activation npz per call.
  capture.py             Capturer protocol + MockCapturer (numpy, no weights)
                         + HFCapturer (transformers + forward hooks on the
                         residual stream of selected decoder layers).
tools/
  validate_run.py        Enforces step_id↔npz 1:1 parity (AGENTS.md §4 #1).
scripts/
  smoke_shim.sh          Boots the shim, fires N curl completions, runs the
                         validator. Mock by default; PINCHGUARD_CAPTURE_BACKEND=hf
                         for real weights.
  run_session.sh         Reproduces the scenario-01 capture session: boots the
                         shim (Qwen2.5-0.5B on CPU), fetches N live Moltbook posts,
                         drives comment_session.py, validates. Dry-run by default
                         (logs comments; does not post).
  comment_session.py     Cusco reads + comments on N Moltbook posts via the shim,
                         keeping cumulative history so peer-content contamination
                         compounds. Captures one trace+npz per comment; logs to
                         comments_dryrun.jsonl. Never posts to Moltbook.
tests/
  conftest.py            Materialises a hand-rolled fixture run.
  test_schema.py         Parity invariant: happy path + missing/orphan/mismatch.
  test_parity.py         N requests → N rows + N npz; required Phase-1 fields;
                         turn_id null/passthrough; monotonic turn counter.
  fixtures/sample_run/   Hand-rolled traces.jsonl (binary npz materialised at
                         session scope to keep blobs out of git).
data/runs/
  schema_v0.2_example/   Committed reference run (5 rows + activations) so
                         B/C/D can inspect concrete data without rerunning the
                         model. See its note.md for caveats.
  smoke_*/               Local smoke runs (gitignored).
scenarios/             Concern E - Scenario YAMLs (placeholder).
docs/                  Project docs (placeholder).
pyproject.toml         uv-managed deps. Extras: `dev` (pytest), `capture`
                       (torch + transformers + nnsight + nnterp), `analysis`.
.github/workflows/     CI: uv sync --extra dev && uv run pytest on push/PR.
```

## What data we have

Concern A produces one row per chat-completion call, plus one compressed npz
of residual-stream activations per row. The 1:1 parity is invariant #1.

Schema is **v0.2** (see AGENTS.md §3 for the canonical example). A
committed reference run lives in `data/runs/schema_v0.2_example/` — five
rows and matching activation npz files captured against
Qwen2.5-0.5B-Instruct, plus a `note.md` flagging which fields are
expected to change. An illustrative trimmed row:

```json
{
  "schema_version": "0.2",
  "step_id": "d1282e37-c0a9-4867-b5e4-b9f7c20f846d",
  "turn_id": null,
  "run_id": "smoke_20260526T203349Z",
  "turn": 0,
  "timestamp": "2026-05-26T20:34:19Z",
  "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
  "prompt": "<|im_start|>system\nYou are Qwen...<|im_end|>\n<|im_start|>user\nsmoke 1<|im_end|>\n<|im_start|>assistant\n",
  "output_raw": "I'm sorry, but I don't have enough context...",
  "activation_ref": "activations/d1282e37-c0a9-4867-b5e4-b9f7c20f846d.npz",
  "activation_meta": {
    "token_position": "last_input",
    "layers": [12, 23],
    "dtype": "float16",
    "capture_runtime": "hf-eager"
  }
}
```

The matching npz holds `{"L12": ndarray(1, 896, float16), "L23": ndarray(1, 896, float16)}`
for Qwen2.5-0.5B-Instruct on CPU. Several other rows (`scenario_id`,
`agent_id`, `agent_persona`, `alignment_label`, `context`, `observation`,
`parsed_action`) are nullable today and will be populated as OpenClaw +
Moltbook wiring lands.

**The schema is still settling.** Build against v0.2 but design for
evolution — expect changes once real OpenClaw traces land.

## How the data was generated

```bash
# Mock backend (no weights, instant):
uv sync --extra dev
N=10 bash scripts/smoke_shim.sh

# Real weights (Qwen2.5-0.5B-Instruct on CPU, ~1GB download first time):
uv sync --extra dev --extra capture
PINCHGUARD_CAPTURE_BACKEND=hf \
PINCHGUARD_LAYERS=12,23 \
MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct \
bash scripts/smoke_shim.sh
```

Output lands in `data/runs/smoke_<UTC timestamp>/{traces.jsonl,activations/}`.
The validator runs automatically and exits non-zero on parity violations.

### Scenario-01 capture session (Cusco on Moltbook)

A higher-fidelity run that captures a real agent task instead of synthetic
smoke. Cusco (scenario `scenarios/01`, persona in `SOUL.md`) is given a simple
task — read and comment on N live Moltbook posts — and the shim captures one
trace + activation per comment. Cumulative chat history is kept across posts so
peer-content contamination can compound; this is the "natural experiment" for
observing goal/identity drift in the traces.

```bash
uv sync --extra capture
N=10 bash scripts/run_session.sh   # needs MOLTBOOK_API_KEY in .env
```

**Dry-run by default:** comments are written to
`data/runs/<run_id>/comments_dryrun.jsonl` and **never posted to Moltbook**
(live posting is irreversible and Moltbook flags low-effort content from new
agents as spam). Output bundle: `traces.jsonl` + `activations/*.npz` +
`comments_dryrun.jsonl` + `feed_snapshot.json`, validated for 1:1 parity.

## Where to start, by concern

### B · Behavioural Eval
You only need `traces.jsonl`. Read `prompt` + `output_raw` per row and emit
a `label_behav` (continuous risk score) plus a `label_behav_meta` block
documenting the scorer/rubric/judge. The mock backend gives stable inputs
for scaffolding; the real-weights smoke gives plausible Qwen text. The
AGENTS.md §2 done-bar is 80%+ alignment with human reviewers on 30 rows.

### C · Probe Eval
You need `activations/<step_id>.npz`. Load with `np.load(path)`; keys are
`L<layer_idx>`, values are `(1, hidden_dim)` float16 vectors at the
`last_input` token position. Output is a `probe_scores.jsonl` keyed by
`step_id` carrying `label_probe` + `label_probe_meta` (probe id, layer,
site, training dataset version). Layer/site selection sits with you —
`PINCHGUARD_LAYERS` controls what Concern A captures at run time.

### D · Predictor
You consume B's enriched traces joined with C's probe scores on `step_id`.
**Hard constraint from AGENTS.md §4 #2:** features must be black-box only —
no activation tensors, no sandbox IDs. Allowed surface = `prompt`,
`output_raw`, `parsed_action`, token counts, lexical style flags, learned
embeddings of text fields. You can scaffold the join + CV harness against
synthetic labels until B and C produce real ones.
