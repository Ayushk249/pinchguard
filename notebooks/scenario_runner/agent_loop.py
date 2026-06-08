"""Pure helpers for the 15-turn agent loop in ``agent_run.ipynb``.

The shim is a *completion* endpoint: it returns plain text, never OpenAI
``tool_calls``. So the loop parses the model's text into a single action and
dispatches it client-side (the OpenClaw-faithful shape — see ``agent_tools``).
These helpers are the text/IO plumbing around that:

* ``load_jsonl`` / ``write_jsonl`` — feed in, per-turn records out.
* ``split_think`` — peel Qwen3's ``<think>…</think>`` off the answer.
* ``parse_action`` — turn ``ACTION: tool(kwargs)`` into ``(name, kwargs)``.
* ``parse_scratchpad`` — read a ``SCRATCHPAD:`` write-back line, if any.
* ``render_observation`` — build a turn's user message (scratchpad + feed post +
  prior tool result).

Kept free of Inspect/torch imports so they unit-test with no model or GPU.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

# Positional-arg order per tool, so a model that writes read_comment("p07")
# instead of read_comment(post_id="p07") still parses. Keyword args always win.
_TOOL_POSITIONAL = {
    "read_comment": ["post_id", "comment"],
    "filesystem": ["path"],
}

_THINK_RE = re.compile(r"<think>(.*?)</think>(.*)", re.DOTALL)
_ACTION_RE = re.compile(r"ACTION:\s*([A-Za-z_]\w*)\s*\((.*)\)", re.DOTALL)
_SCRATCHPAD_RE = re.compile(r"^[ \t]*SCRATCHPAD:[ \t]*(.*)$", re.MULTILINE)


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dicts (blank lines skipped)."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    """Write rows as JSONL (one compact object per line, UTF-8 preserved)."""
    Path(path).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )


def split_think(completion: str) -> tuple[str, str]:
    """Split a completion into ``(reasoning, answer)``.

    Qwen3's chat template emits ``<think>…</think>`` before the answer. Returns
    the inner reasoning (stripped) and the text after ``</think>``. If there is
    no closed think block, reasoning is "" and the whole completion is the
    answer — so a truncated or thinking-off completion still parses.
    """
    m = _THINK_RE.search(completion)
    if not m:
        return "", completion.strip()
    return m.group(1).strip(), m.group(2).strip()


def _coerce_value(node: ast.expr) -> Any:
    """literal_eval a single arg node, falling back to its source-ish repr."""
    try:
        return ast.literal_eval(node)
    except Exception:
        return ast.unparse(node)


def parse_action(text: str) -> tuple[str | None, dict]:
    """Parse ``ACTION: tool(kwargs)`` out of model text → ``(name, kwargs)``.

    Returns ``(None, {})`` when no action is present (e.g. the mock backend, or
    a turn the model spends only reasoning) — the loop treats that as a no-op
    turn, preserving one capture per turn regardless. Both ``post_id="p07"`` and
    positional ``"p07"`` forms are accepted; keyword args take precedence.
    """
    m = _ACTION_RE.search(text)
    if not m:
        return None, {}
    name, argstr = m.group(1), m.group(2).strip()
    try:
        call = ast.parse(f"_f({argstr})", mode="eval").body
        assert isinstance(call, ast.Call)
    except Exception:
        return name, {}
    kwargs: dict[str, Any] = {kw.arg: _coerce_value(kw.value) for kw in call.keywords if kw.arg}
    # Map any positional args onto this tool's parameter names.
    for param, node in zip(_TOOL_POSITIONAL.get(name, []), call.args):
        kwargs.setdefault(param, _coerce_value(node))
    return name, kwargs


def reasoning_from_content(content: Any) -> str:
    """Pull reasoning text out of an Inspect assistant message's ``content``.

    Inspect's ``openai-api`` provider parses ``<think>…</think>`` out of the
    model's content into ``ContentReasoning`` parts, so the reasoning is NOT in
    ``out.completion`` (which arrives already stripped). ``content`` is either a
    plain ``str`` (no reasoning) or a list of content parts; we collect the
    ``.reasoning`` of any reasoning part. Duck-typed (matches on the attribute,
    not the class) so this module stays free of Inspect imports and unit-tests
    with a stub.
    """
    if content is None or isinstance(content, str):
        return ""
    parts = [
        c.reasoning
        for c in content
        if isinstance(getattr(c, "reasoning", None), str)
    ]
    return "\n".join(parts).strip()


def parse_scratchpad(text: str) -> str | None:
    """Return the last ``SCRATCHPAD:`` line's value, or None if the model wrote none."""
    matches = _SCRATCHPAD_RE.findall(text)
    return matches[-1].strip() if matches else None


def render_post(post: dict) -> str:
    """Render a feed post as the agent sees it (mirrors agent_tools._render_post)."""
    return (
        f"Post {post.get('post_id', '?')} in m/{post.get('submolt', 'general')} "
        f"by @{post.get('author', '?')}\n"
        f"TITLE: {post.get('title', '')}\n\n{post.get('content', '')}"
    )


def render_observation(
    turn: int,
    n_turns: int,
    post: dict,
    scratchpad: str,
    last_tool_result: str | None = None,
) -> str:
    """Build one turn's user message: counter + scratchpad + feed post + prior result."""
    blocks = [f"Iteration {turn + 1}/{n_turns}."]
    if scratchpad:
        blocks.append(f"[scratchpad]\n{scratchpad}")
    if last_tool_result:
        blocks.append(f"[result of your last action]\n{last_tool_result}")
    blocks.append(f"[new post in your feed]\n{render_post(post)}")
    return "\n\n".join(blocks)


__all__ = [
    "load_jsonl",
    "write_jsonl",
    "split_think",
    "parse_action",
    "reasoning_from_content",
    "parse_scratchpad",
    "render_post",
    "render_observation",
]
