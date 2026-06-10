"""Activation capture backends for the Pinchguard inference shim.

Two backends, sharing a common `Capturer` protocol:

* `MockCapturer` — numpy-only, deterministic per (step_id, prompt). Used by
  parity tests and as the `auto`-fallback when real weights aren't available.
* `HFCapturer` — HuggingFace transformers + forward hooks on the residual
  stream at each configured layer. CPU-friendly for Qwen2.5-0.5B; the same
  module is the eventual home of the nnterp wrapping (PLAN.md Phase 4).

Schema fields produced (`activation_meta`, per AGENTS.md §3 v0.2):
  - token_position: which token's hidden state we kept (default `last_input`).
  - layers: integer indices captured.
  - dtype: storage dtype string.
  - capture_runtime: identifies the backend so future replays disambiguate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

DEFAULT_LAYERS: tuple[int, ...] = (12, 24)
DEFAULT_TOKEN_POSITION: str = "last_input"
DEFAULT_DTYPE: str = "float16"


@dataclass
class CaptureResult:
    completion_text: str
    activations: dict[str, np.ndarray]
    activation_meta: dict[str, Any]
    prompt_text: str


class Capturer(Protocol):
    model_id: str

    def capture(self, messages: list[dict[str, Any]], step_id: str) -> CaptureResult: ...


def _content_to_text(content: Any) -> str:
    """Coerce an OpenAI chat `content` field to a plain string.

    OpenClaw (and other OpenAI-compatible clients) may send `content` as a list
    of structured content parts, e.g. ``[{"type": "text", "text": "hi"}]``,
    instead of a bare string. HF chat templates concatenate `content` as a
    string and raise ``TypeError`` on lists, so we flatten the text parts here.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce arbitrary OpenAI messages to template-safe ``{role, content}``.

    Keeps only fields every HF chat template understands; flattens structured
    content to text. Dropping `tool_calls`/`tool_call_id` keeps templates that
    don't model tools from breaking — the shim captures activations off the
    rendered text, so structural tool metadata isn't needed here.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        msg: dict[str, Any] = {"role": role, "content": _content_to_text(m.get("content"))}
        name = m.get("name")
        if isinstance(name, str):
            msg["name"] = name
        out.append(msg)
    return out


def _flatten_messages(messages: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{m.get('role', '?')}: {_content_to_text(m.get('content'))}" for m in messages
    )


def _guard_single_cuda_device(quantization: str | None, cuda_indices: set[int]) -> None:
    """Raise if a quantized model's weights span more than one CUDA device.

    Verified 2026-06-06: bitsandbytes 4-bit/8-bit weights sharded across GPUs
    (e.g. via ``device_map="auto"`` on a 2-GPU box) corrupt the forward pass —
    generation degrades to token-salad and the captured L32 activations are
    garbage. The same nf4 quant pinned to a single device is coherent. This is
    a quantization-specific bug, so the guard only fires when `quantization` is
    set; full-precision sharding is unaffected and left alone.

    Pure (takes the resolved device-index set, not the model) so it is unit-
    testable without weights. `cuda_indices` is empty on the CPU/full-precision
    path, which never trips the guard.
    """
    if quantization is None:
        return
    if len(cuda_indices) > 1:
        raise RuntimeError(
            f"Refusing to capture: quantization={quantization!r} model is sharded "
            f"across {len(cuda_indices)} CUDA devices {sorted(cuda_indices)}. "
            "bitsandbytes sharding corrupts the forward pass (token-salad output, "
            "garbage activations). Pin a single device with "
            "PINCHGUARD_DEVICE_MAP=cuda:N — never device_map='auto'."
        )


class MockCapturer:
    """Deterministic, numpy-only capture for tests and CPU-poor dev loops."""

    def __init__(
        self,
        *,
        layers: tuple[int, ...] = DEFAULT_LAYERS,
        token_position: str = DEFAULT_TOKEN_POSITION,
        hidden_dim: int = 8,
        model_id: str = "mock-qwen2.5-0.5b",
    ) -> None:
        self.layers = tuple(layers)
        self.token_position = token_position
        self.hidden_dim = hidden_dim
        self.model_id = model_id

    def capture(self, messages: list[dict[str, Any]], step_id: str) -> CaptureResult:
        prompt_text = _flatten_messages(messages)
        seed_bytes = hashlib.sha256(f"{step_id}|{prompt_text}".encode()).digest()[:8]
        rng = np.random.default_rng(int.from_bytes(seed_bytes, "big"))
        activations = {
            f"L{layer}": rng.standard_normal(size=(1, self.hidden_dim)).astype(np.float16)
            for layer in self.layers
        }
        meta: dict[str, Any] = {
            "token_position": self.token_position,
            "layers": list(self.layers),
            "dtype": DEFAULT_DTYPE,
            "capture_runtime": "mock",
        }
        completion = f"[mock {step_id[:8]}] ack: {prompt_text[-32:]}"
        return CaptureResult(
            completion_text=completion,
            activations=activations,
            activation_meta=meta,
            prompt_text=prompt_text,
        )


class HFCapturer:
    """HuggingFace transformers + forward-hook activation capture.

    Forward-hooks the residual stream output of each requested decoder layer,
    runs a prompt-only forward to pull `token_position` activations, then runs
    a separate greedy `generate` for the completion text. Two passes keeps
    activation semantics clean (no generation context bleeding back into the
    captured token's hidden state).

    nnterp swap-in (PLAN.md Phase 4): replace `_resolve_layers` + the hook
    machinery with `StandardizedTransformer.trace`; bump `capture_runtime` to
    `nnterp+hf-eager`.
    """

    def __init__(
        self,
        *,
        model_name: str,
        layers: tuple[int, ...] = DEFAULT_LAYERS,
        token_position: str = DEFAULT_TOKEN_POSITION,
        max_new_tokens: int = 64,
        dtype: str | None = None,
        device_map: str | None = None,
        quantization: str | None = None,
    ) -> None:
        import torch  # type: ignore[import-not-found]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

        if token_position != "last_input":
            raise ValueError(
                f"token_position={token_position!r} not yet supported "
                "(only 'last_input' implemented)"
            )

        self._torch = torch
        self.model_id = model_name
        self.layers = tuple(layers)
        self.token_position = token_position
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.quantization = quantization

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # `low_cpu_mem_usage=True` skips the float32 init copy when we target
        # float16 — important on Codespaces, where total RAM is ~8 GiB and
        # the float32 spike OOMs even a 0.5B model.
        load_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        quant_config = self._build_quant_config(quantization)
        if quant_config is not None:
            # Gotcha (1): bitsandbytes owns the compute dtype. Passing `dtype=`
            # alongside a `quantization_config` conflicts, so only one is ever
            # set — the quant config here, or the explicit dtype below.
            load_kwargs["quantization_config"] = quant_config
        else:
            load_kwargs["dtype"] = getattr(torch, dtype) if dtype else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()

        # Belt-and-braces multi-GPU guard: a quantized load that landed on >1
        # CUDA device (e.g. device_map="auto" sharding) silently captures
        # garbage, so fail loudly before any capture happens.
        cuda_indices = {
            p.device.index
            for p in self.model.parameters()
            if p.device.type == "cuda" and p.device.index is not None
        }
        _guard_single_cuda_device(self.quantization, cuda_indices)

        self._layer_modules = self._resolve_layers()
        max_layer = len(self._layer_modules) - 1
        for layer in self.layers:
            if layer < 0 or layer > max_layer:
                raise ValueError(
                    f"layer {layer} out of range; model has {len(self._layer_modules)} layers"
                )

    def _build_quant_config(self, quantization: str | None) -> Any:
        """Map a quantization name to a bitsandbytes `BitsAndBytesConfig`.

        `None` → full-precision load (the existing 0.5B CPU path, untouched).
        `"nf4"` is the verified Blackwell 4-bit path (Q4): nf4 + double-quant,
        float16 compute. `"int8"` is the 8-bit fallback rung. The bnb import is
        deferred so the unquantized path never requires bitsandbytes installed.
        """
        if quantization is None:
            return None
        from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]

        torch = self._torch
        if quantization == "nf4":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        if quantization == "int8":
            return BitsAndBytesConfig(load_in_8bit=True)
        raise ValueError(
            f"quantization={quantization!r} not supported (use 'nf4' or 'int8')"
        )

    def _resolve_layers(self) -> Any:
        m = self.model
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m.transformer.h
        raise RuntimeError(
            f"Could not locate decoder layers on {type(m).__name__}; "
            "extend HFCapturer._resolve_layers."
        )

    def capture(self, messages: list[dict[str, Any]], step_id: str) -> CaptureResult:
        torch = self._torch
        prompt_text = self.tokenizer.apply_chat_template(
            _sanitize_messages(messages), tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        # Gotcha (2): under `device_map` the weights live on GPU, so prompt
        # tensors must be moved to the model's device or the forward errors on
        # a device mismatch. The CPU-only path is a no-op move.
        if self.device_map is not None:
            inputs = inputs.to(next(self.model.parameters()).device)

        captured: dict[int, Any] = {}
        handles = []

        def _make_hook(idx: int):
            def _hook(_module, _inp, out):
                tensor = out[0] if isinstance(out, tuple) else out
                captured[idx] = tensor.detach()
            return _hook

        for layer_idx in self.layers:
            handles.append(
                self._layer_modules[layer_idx].register_forward_hook(_make_hook(layer_idx))
            )

        try:
            with torch.no_grad():
                # logits_to_keep=1: run lm_head on the LAST position only. This
                # forward exists solely to fire the layer hooks (we keep only
                # tensor[:, -1, :] below); the full-sequence logits it otherwise
                # computes are discarded — and that ~2.2 GiB spike OOM'd the 24 GB
                # card around turn ~15 once the context grew. Hooks fire on the
                # decoder layers regardless, so captured activations are identical.
                self.model(**inputs, logits_to_keep=1)
        finally:
            for h in handles:
                h.remove()

        activations: dict[str, np.ndarray] = {}
        for layer_idx, tensor in captured.items():
            vec = tensor[:, -1, :]  # last_input position
            activations[f"L{layer_idx}"] = vec.to(torch.float16).cpu().numpy()

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        completion = self.tokenizer.decode(
            out_ids[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        meta: dict[str, Any] = {
            "token_position": self.token_position,
            "layers": list(self.layers),
            "dtype": DEFAULT_DTYPE,
            "capture_runtime": "hf-eager",
            # Provenance for the M5 manifest: `dtype` is the *stored activation*
            # dtype (fp16); `quantization` records the model's compute precision
            # (null on the full-precision path), which `dtype` alone can't convey.
            "quantization": self.quantization,
        }
        return CaptureResult(
            completion_text=completion,
            activations=activations,
            activation_meta=meta,
            prompt_text=prompt_text,
        )
