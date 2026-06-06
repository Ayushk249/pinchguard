"""Pinchguard inference shim — OpenAI-compatible `/v1/chat/completions`.

Mints `step_id` per completion, writes activations to
`<run_dir>/activations/<step_id>.npz`, appends a schema-v0.2 row to
`<run_dir>/traces.jsonl`.

Production launch (after `uv sync --extra capture` for real weights):

    uv run uvicorn shim.main:create_app --factory --host 0.0.0.0 --port 8000

Tests construct an app directly via `create_app(run_dir=..., capturer=...)`
to avoid touching env vars or filesystem defaults. See AGENTS.md §3 for the
trace row schema and PLAN.md Phase 1 for scope.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from shim.capture import (
    DEFAULT_LAYERS,
    DEFAULT_TOKEN_POSITION,
    Capturer,
    CaptureResult,
    MockCapturer,
)

SCHEMA_VERSION = "0.2"


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: str | list | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    # Pinchguard extension: OpenClaw telemetry sidecar (Phase 2) will set this
    # to group all step_ids inside a single operator-visible turn. Null otherwise.
    turn_id: str | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_layers_env() -> tuple[int, ...]:
    raw = os.environ.get("PINCHGUARD_LAYERS")
    if not raw:
        return DEFAULT_LAYERS
    return tuple(int(x) for x in raw.split(",") if x.strip())


def _default_run_dir() -> Path:
    base = Path(os.environ.get("PINCHGUARD_DATA_DIR", "data/runs"))
    run_id = os.environ.get("PINCHGUARD_RUN_ID") or (
        f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    return base / run_id


def _build_default_capturer() -> Capturer:
    backend = os.environ.get("PINCHGUARD_CAPTURE_BACKEND", "auto").lower()
    layers = _resolve_layers_env()
    token_position = os.environ.get("PINCHGUARD_TOKEN_POSITION", DEFAULT_TOKEN_POSITION)
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    # Unset/empty → None so the unquantized, CPU (no device_map) path stays the
    # default and 0.5B parity is fully backward-compatible. `nf4`/`auto` only
    # engage for the 32B bring-up when these are exported.
    device_map = os.environ.get("PINCHGUARD_DEVICE_MAP") or None
    quantization = os.environ.get("PINCHGUARD_QUANT") or None

    if backend == "mock":
        return MockCapturer(layers=layers, token_position=token_position, model_id=model_name)
    if backend in {"auto", "hf", "transformer", "transformers"}:
        try:
            from shim.capture import HFCapturer

            max_new_tokens = int(os.environ.get("PINCHGUARD_MAX_NEW_TOKENS", "64"))
            return HFCapturer(
                model_name=model_name,
                layers=layers,
                token_position=token_position,
                max_new_tokens=max_new_tokens,
                device_map=device_map,
                quantization=quantization,
            )
        except Exception as exc:  # pragma: no cover — environment-dependent
            if backend != "auto":
                raise
            print(
                f"[pinchguard.shim] HFCapturer unavailable ({type(exc).__name__}: {exc}); "
                "falling back to MockCapturer. Install the `capture` extra for real activations."
            )
            return MockCapturer(
                layers=layers, token_position=token_position, model_id=model_name
            )
    raise ValueError(f"Unknown PINCHGUARD_CAPTURE_BACKEND={backend!r}")


def _trace_row(
    *,
    step_id: str,
    turn: int,
    turn_id: str | None,
    run_id: str,
    capturer_model_id: str,
    result: CaptureResult,
    activation_ref: str,
) -> dict[str, Any]:
    """Build the v0.2 row. Unknown-by-the-shim fields are explicit nulls so
    downstream consumers can distinguish "not yet populated" from "field absent".
    Sidecar/OpenClaw (Phase 2) and offline scorers (Concerns B/C) fill the rest.

    `model_id` is the capturer's actual model, *not* the client's request.model
    (which is OpenAI-spec passthrough). The activations were produced by this
    specific model, so analysis joins should pivot on it.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "step_id": step_id,
        "turn_id": turn_id,
        "scenario_id": os.environ.get("PINCHGUARD_SCENARIO_ID"),
        "run_id": run_id,
        "turn": turn,
        "timestamp": _utc_now_iso(),
        "agent_id": os.environ.get("PINCHGUARD_AGENT_ID"),
        "agent_persona": os.environ.get("PINCHGUARD_AGENT_PERSONA"),
        "alignment_label": os.environ.get("PINCHGUARD_ALIGNMENT_LABEL"),
        "model_id": capturer_model_id,
        "context": os.environ.get("PINCHGUARD_CONTEXT"),
        "observation": None,
        "prompt": result.prompt_text,
        "output_raw": result.completion_text,
        "parsed_action": None,
        "activation_ref": activation_ref,
        "activation_meta": result.activation_meta,
    }


def create_app(
    *,
    run_dir: Path | str | None = None,
    capturer: Capturer | None = None,
) -> FastAPI:
    app = FastAPI(title="Pinchguard inference shim", version=SCHEMA_VERSION)

    resolved_run_dir = Path(run_dir) if run_dir is not None else _default_run_dir()
    resolved_run_dir.mkdir(parents=True, exist_ok=True)
    (resolved_run_dir / "activations").mkdir(exist_ok=True)
    traces_path = resolved_run_dir / "traces.jsonl"
    resolved_capturer = capturer or _build_default_capturer()

    state: dict[str, Any] = {
        "run_dir": resolved_run_dir,
        "run_id": resolved_run_dir.name,
        "traces_path": traces_path,
        "capturer": resolved_capturer,
        "turn_counter": 0,
    }
    app.state.pinchguard = state

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "run_id": state["run_id"],
            "run_dir": str(state["run_dir"]),
            "capturer": type(state["capturer"]).__name__,
            "model_id": getattr(state["capturer"], "model_id", None),
            "schema_version": SCHEMA_VERSION,
        }

    def _models_payload() -> dict[str, Any]:
        # OpenAI-spec model list. OpenClaw (and other OpenAI-compatible clients)
        # probe this endpoint to confirm a local provider is reachable before
        # routing chat completions through it.
        model_id = getattr(state["capturer"], "model_id", "unknown")
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "pinchguard",
                }
            ],
        }

    # Both paths: clients probe `/v1/models`; some probe `/models` (no prefix).
    app.get("/v1/models")(_models_payload)
    app.get("/models")(_models_payload)

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
        cap: Capturer = state["capturer"]
        messages = [m.model_dump() for m in req.messages]
        step_id = str(uuid.uuid4())

        result = cap.capture(messages, step_id)

        activation_ref = f"activations/{step_id}.npz"
        np.savez_compressed(state["run_dir"] / activation_ref, **result.activations)

        turn = state["turn_counter"]
        state["turn_counter"] = turn + 1

        row = _trace_row(
            step_id=step_id,
            turn=turn,
            turn_id=req.turn_id,
            run_id=state["run_id"],
            capturer_model_id=getattr(cap, "model_id", "unknown"),
            result=result,
            activation_ref=activation_ref,
        )
        with state["traces_path"].open("a") as f:
            f.write(json.dumps(row) + "\n")

        return {
            "id": f"chatcmpl-{step_id}",
            "object": "chat.completion",
            "created": int(datetime.now(timezone.utc).timestamp()),
            "model": req.model or row["model_id"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.completion_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "pinchguard": {
                "step_id": step_id,
                "activation_ref": activation_ref,
                "turn_id": req.turn_id,
            },
        }

    return app
