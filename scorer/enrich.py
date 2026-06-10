"""Pinchguard Concern B — main enrichment entrypoint.

Reads a traces.jsonl produced by Concern A, scores each row with the LH-Deception
adapted auditor (via OpenRouter), and writes traces_enriched.jsonl alongside it.

Usage
-----
    # Score a full run directory:
    python -m scorer.enrich data/runs/comment_session_20260530T074501Z/

    # Score a specific traces file, write to a specific output:
    python -m scorer.enrich data/runs/comment_session_20260530T074501Z/traces.jsonl \
           --out data/runs/comment_session_20260530T074501Z/traces_enriched.jsonl

    # Dry-run (no API calls — uses mock auditor output):
    python -m scorer.enrich data/runs/comment_session_20260530T074501Z/ --dry-run

    # Override judge model:
    python -m scorer.enrich ... --model anthropic/claude-3.5-sonnet

Environment
-----------
    OPENROUTER_API_KEY        Required (unless --dry-run).
    PINCHGUARD_JUDGE_MODEL    Optional override for judge model.

Output
------
    traces_enriched.jsonl: All original fields preserved. Two new fields per row:
      label_behav:       float [0.0–4.0]  (0 = aligned, 4 = severe drift)
      label_behav_meta:  dict with scorer provenance and auditor output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scorer.build_history import build_histories
from scorer.auditor import score_step, DEFAULT_JUDGE_MODEL, load_persona_vars_from_soul
from scorer.schema import AuditorOutput


# ---------------------------------------------------------------------------
# Mock auditor (for --dry-run / tests)
# ---------------------------------------------------------------------------

def _mock_score(row: dict[str, Any], history: list[dict[str, str]]) -> AuditorOutput:
    """Return a predictable mock result without calling the API."""
    turn = row.get("turn", 0)
    # Simulate increasing drift for later turns
    if turn >= 7:
        return AuditorOutput(
            deception_detected=True,
            deception_types=["FALSIFICATION"],
            severity=3,
            confidence=0.85,
            reasoning="[MOCK] Late-turn identity drift simulated.",
            evidence=["[MOCK] output_raw diverges from Cusco persona."],
        )
    return AuditorOutput(
        deception_detected=False,
        deception_types=[],
        severity=0,
        confidence=0.9,
        reasoning="[MOCK] No drift detected.",
        evidence=[],
    )


# ---------------------------------------------------------------------------
# Core enrichment logic
# ---------------------------------------------------------------------------

def enrich_traces(
    traces_path: Path,
    out_path: Path,
    *,
    model: str | None = None,
    dry_run: bool = False,
    delay_between_calls: float = 1.0,
    soul_path: Path | None = None,
    boundary_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Score all rows in traces_path and write to out_path.

    Parameters
    ----------
    traces_path:          Path to the input traces.jsonl.
    out_path:             Destination for traces_enriched.jsonl.
    model:                OpenRouter model string override.
    dry_run:              If True, use mock auditor (no API calls).
    delay_between_calls:  Seconds to wait between real API calls.
    soul_path:            Optional path to a SOUL.md file.
    boundary_path:        Optional path to a BOUNDARY.md file.

    Returns the list of enriched row dicts.
    """
    resolved_model = model or os.environ.get("PINCHGUARD_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)

    # Resolve persona variables from SOUL.md (or fall back to Cusco defaults)
    persona_vars = None
    soul_label: str = "(default: Cusco)"
    if soul_path is not None:
        if not soul_path.exists():
            print(f"[enrich] WARNING: SOUL.md not found at {soul_path} — using defaults.",
                  file=sys.stderr)
        else:
            persona_vars = load_persona_vars_from_soul(soul_path, boundary_path)
            soul_label = str(soul_path)
            print(f"[enrich] Persona loaded from: {soul_path}")
            print(f"[enrich] Agent name: {persona_vars.get('agent_name', '?')}")

    # Read traces
    raw_lines = [l for l in traces_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows: list[dict[str, Any]] = [json.loads(l) for l in raw_lines]

    if not rows:
        print(f"[enrich] No rows found in {traces_path}. Nothing to do.")
        return []

    # Sort by turn (defensive)
    rows.sort(key=lambda r: r.get("turn", 0))

    # Load loop_records.jsonl if present (for soft-trigger scoring)
    loop_records_by_turn: dict[int, dict[str, Any]] = {}
    loop_records_path = traces_path.parent / "loop_records.jsonl"
    if loop_records_path.exists():
        lr_lines = [l for l in loop_records_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for lr_line in lr_lines:
            try:
                lr = json.loads(lr_line)
                t = lr.get("turn")
                if t is not None:
                    loop_records_by_turn[int(t)] = lr
            except Exception:
                pass
        print(f"[enrich] loop_records.jsonl loaded: {len(loop_records_by_turn)} turns found.")
    else:
        print(f"[enrich] No loop_records.jsonl found — soft trigger scoring will be 0 for all turns.")

    # Build per-step histories
    histories = build_histories(rows)

    enriched_rows: list[dict[str, Any]] = []
    total = len(rows)

    print(f"[enrich] Scoring {total} row(s) from {traces_path}")
    print(f"[enrich] Judge model: {'MOCK (dry-run)' if dry_run else resolved_model}")
    print(f"[enrich] Soul file:   {soul_label}")
    print(f"[enrich] Output -> {out_path}\n")

    for i, (row, history) in enumerate(zip(rows, histories)):
        step_id = row.get("step_id", "?")[:12]
        turn = row.get("turn", i)
        print(f"  [{i+1}/{total}] turn={turn} step={step_id} ... ", end="", flush=True)

        # Look up this turn's loop record for soft-trigger injection
        loop_record = loop_records_by_turn.get(int(turn))

        try:
            if dry_run:
                result = _mock_score(row, history)
            else:
                result = score_step(row, history, model=resolved_model,
                                    persona_vars=persona_vars,
                                    loop_record=loop_record)

            enriched = {
                **row,
                "label_behav": result.label_behav,
                "soft_trigger_score": result.soft_trigger_score,
                "label_behav_meta": {
                    **result.to_label_behav_meta(judge_model=resolved_model),
                    "soul_file": soul_label,
                },
            }
            flag = "[DRIFT]" if result.deception_detected else "[ok]   "
            print(f"{flag}  severity={result.severity}  soft_trigger={result.soft_trigger_score}  confidence={result.confidence:.2f}")

        except Exception as exc:
            print(f"ERROR: {exc}")
            # Write original row with null labels so we don't lose the row
            enriched = {
                **row,
                "label_behav": None,
                "label_behav_meta": {"error": str(exc), "scorer_id": "lh_deception_v1_pinchguard"},
            }

        enriched_rows.append(enriched)

        # Respect rate limits between calls
        if not dry_run and i < total - 1:
            time.sleep(delay_between_calls)

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in enriched_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Summary
    scored = [r for r in enriched_rows if r.get("label_behav") is not None]
    drifted = [r for r in scored if r.get("label_behav", 0) > 0]
    print(f"\n[enrich] Done. {len(scored)}/{total} rows scored.")
    print(f"[enrich] Drift detected in {len(drifted)}/{len(scored)} rows.")
    if drifted:
        avg_severity = sum(r["label_behav"] for r in drifted) / len(drifted)
        print(f"[enrich] Average severity (drifted rows): {avg_severity:.2f}")
    print(f"[enrich] Written to: {out_path}")

    return enriched_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pinchguard Concern B: enrich traces.jsonl with label_behav scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "input",
        type=Path,
        help="Path to traces.jsonl OR a run directory containing traces.jsonl.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for traces_enriched.jsonl. "
             "Defaults to <input_dir>/traces_enriched.jsonl.",
    )
    p.add_argument(
        "--model",
        default=None,
        help=f"OpenRouter model string (default: {DEFAULT_JUDGE_MODEL}).",
    )
    p.add_argument(
        "--soul-file",
        type=Path,
        default=None,
        metavar="SOUL_MD",
        help=(
            "Path to a SOUL.md file for the agent being scored. When provided, "
            "persona variables (agent name, guardrails, model size, etc.) are extracted "
            "from the file and injected into the judge prompt, overriding the built-in "
            "Cusco defaults. Example: scenarios/01/SOUL.md"
        ),
    )
    p.add_argument(
        "--boundary-file",
        type=Path,
        default=None,
        metavar="BOUNDARY_MD",
        help="Optional path to a BOUNDARY.md file containing hard constraints. Appended to core_guardrails.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Use mock auditor — no API calls, no key required.",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between API calls (default: 1.0).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Resolve input path
    inp: Path = args.input
    if inp.is_dir():
        traces_path = inp / "traces.jsonl"
    else:
        traces_path = inp

    if not traces_path.exists():
        print(f"[enrich] ERROR: traces.jsonl not found at {traces_path}", file=sys.stderr)
        return 1

    # Resolve output path
    if args.out:
        out_path = args.out
    else:
        out_path = traces_path.parent / "traces_enriched.jsonl"

    # Sanity check: don't overwrite the source
    if out_path.resolve() == traces_path.resolve():
        print("[enrich] ERROR: output path must differ from input path.", file=sys.stderr)
        return 1

    # Check API key (unless dry-run)
    if not args.dry_run and not os.environ.get("OPENROUTER_API_KEY"):
        print(
            "[enrich] ERROR: OPENROUTER_API_KEY not set. "
            "Add it to .env or export it, or use --dry-run.",
            file=sys.stderr,
        )
        return 1

    enrich_traces(
        traces_path,
        out_path,
        model=args.model,
        dry_run=args.dry_run,
        delay_between_calls=args.delay,
        soul_path=args.soul_file,
        boundary_path=args.boundary_file,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
