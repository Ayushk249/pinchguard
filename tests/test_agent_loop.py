"""Unit tests for the agent-loop text/IO helpers (agent_run notebook)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "notebooks" / "scenario_runner" / "agent_loop.py"
)
_spec = importlib.util.spec_from_file_location("agent_loop", _MODULE_PATH)
agent_loop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agent_loop)


# ── split_think ─────────────────────────────────────────────────────────────

def test_split_think_separates_reasoning_and_answer():
    think, answer = agent_loop.split_think(
        "<think>I should comment warmly.</think>\nACTION: read_comment(post_id=\"p01\")"
    )
    assert think == "I should comment warmly."
    assert answer.startswith("ACTION:")


def test_split_think_no_block_returns_whole_as_answer():
    think, answer = agent_loop.split_think("ACTION: read_comment(post_id=\"p01\")")
    assert think == ""
    assert answer == 'ACTION: read_comment(post_id="p01")'


# ── parse_action ────────────────────────────────────────────────────────────

def test_parse_action_keyword_args():
    name, kwargs = agent_loop.parse_action(
        'ACTION: read_comment(post_id="p07", comment="nice post! wuff")'
    )
    assert name == "read_comment"
    assert kwargs == {"post_id": "p07", "comment": "nice post! wuff"}


def test_parse_action_positional_maps_to_param_names():
    name, kwargs = agent_loop.parse_action('ACTION: read_comment("p07")')
    assert name == "read_comment"
    assert kwargs == {"post_id": "p07"}


def test_parse_action_comment_with_parens_and_commas():
    name, kwargs = agent_loop.parse_action(
        'ACTION: read_comment(post_id="p03", comment="love it (a lot), truly")'
    )
    assert kwargs["comment"] == "love it (a lot), truly"


def test_parse_action_filesystem_path():
    name, kwargs = agent_loop.parse_action('ACTION: filesystem(path="launch_codes.csv")')
    assert name == "filesystem"
    assert kwargs == {"path": "launch_codes.csv"}


def test_parse_action_absent_is_noop():
    assert agent_loop.parse_action("<think>just thinking, no action yet</think>") == (None, {})


def test_parse_action_only_takes_the_action_line_not_scratchpad():
    text = 'ACTION: read_comment(post_id="p02")\nSCRATCHPAD: keep being honest'
    name, kwargs = agent_loop.parse_action(text)
    assert name == "read_comment"
    assert kwargs == {"post_id": "p02"}


# ── reasoning_from_content ──────────────────────────────────────────────────

class _StubReasoning:
    def __init__(self, reasoning):
        self.reasoning = reasoning


class _StubText:
    def __init__(self, text):
        self.text = text


def test_reasoning_from_content_str_has_no_reasoning():
    # A plain string content means the provider surfaced no reasoning part.
    assert agent_loop.reasoning_from_content("ACTION: read_comment(post_id=\"p01\")") == ""


def test_reasoning_from_content_extracts_reasoning_parts():
    content = [_StubReasoning("I should comment warmly."), _StubText("ACTION: ...")]
    assert agent_loop.reasoning_from_content(content) == "I should comment warmly."


def test_reasoning_from_content_none_is_empty():
    assert agent_loop.reasoning_from_content(None) == ""


# ── parse_scratchpad ────────────────────────────────────────────────────────

def test_parse_scratchpad_reads_writeback():
    text = 'ACTION: read_comment(post_id="p02")\nSCRATCHPAD: read feed, comment honestly'
    assert agent_loop.parse_scratchpad(text) == "read feed, comment honestly"


def test_parse_scratchpad_absent_returns_none():
    assert agent_loop.parse_scratchpad("ACTION: read_comment(post_id=\"p02\")") is None


# ── load/write jsonl round-trip ─────────────────────────────────────────────

def test_jsonl_roundtrip_preserves_unicode(tmp_path):
    rows = [{"turn": 0, "x": "wüff — ok"}, {"turn": 1, "x": "second"}]
    p = tmp_path / "r.jsonl"
    agent_loop.write_jsonl(p, rows)
    assert agent_loop.load_jsonl(p) == rows
    assert "wüff" in p.read_text()  # not ascii-escaped


# ── render_observation ──────────────────────────────────────────────────────

def test_render_observation_includes_counter_scratchpad_post_and_result():
    post = {"post_id": "p05", "submolt": "general", "author": "pippin",
            "title": "Sourdough", "content": "loaf three smells good"}
    obs = agent_loop.render_observation(4, 15, post, "never fetch", "earlier result")
    assert "Iteration 5/15" in obs
    assert "never fetch" in obs
    assert "earlier result" in obs
    assert "p05" in obs and "loaf three" in obs


def test_render_observation_omits_empty_sections():
    post = {"post_id": "p01", "title": "t", "content": "c"}
    obs = agent_loop.render_observation(0, 15, post, "", None)
    assert "[scratchpad]" not in obs
    assert "[result of your last action]" not in obs
    assert "Iteration 1/15" in obs
