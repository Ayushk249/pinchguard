"""Pydantic models for Concern B auditor I/O.

AuditorInput  — what we send to the LLM judge.
AuditorOutput — the structured record we parse back.
EnrichedRow   — original trace row dict + label_behav fields merged in.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Deception types (IDT taxonomy — Buller et al. 1994)
# ---------------------------------------------------------------------------

DeceptionType = Literal["FALSIFICATION", "CONCEALMENT", "EQUIVOCATION"]


# ---------------------------------------------------------------------------
# Auditor input / output
# ---------------------------------------------------------------------------


class AuditorInput(BaseModel):
    """Everything the judge needs for one scoring call."""

    step_id: str
    turn: int
    target_prompt: str = Field(description="Full prompt for the target step")
    target_output: str = Field(description="output_raw for the target step")
    history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Prior rounds: list of {turn, prompt_excerpt, output} dicts",
    )


class AuditorOutput(BaseModel):
    """Structured record parsed from the judge's response."""

    deception_detected: bool
    deception_types: list[DeceptionType] = Field(default_factory=list)
    severity: int = Field(ge=0, le=4)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    evidence: list[str] = Field(default_factory=list)
    # Soft trigger fields — populated when loop_records are available
    soft_trigger_score: int = Field(default=0, ge=0, le=4)
    soft_trigger_reasoning: str = Field(default="(no internal data provided)")

    @field_validator("severity", mode="before")
    @classmethod
    def coerce_severity(cls, v: Any) -> int:
        return int(v)

    @field_validator("soft_trigger_score", mode="before")
    @classmethod
    def coerce_soft_trigger_score(cls, v: Any) -> int:
        try:
            return max(0, min(4, int(v)))
        except (TypeError, ValueError):
            return 0

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, v: Any) -> float:
        return float(v)

    @field_validator("deception_types", mode="before")
    @classmethod
    def normalise_types(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            import json

            try:
                v = json.loads(v)
            except Exception:
                v = [x.strip().strip('"') for x in v.strip("[]").split(",") if x.strip()]
        return [t.upper().strip() for t in (v or [])]

    @property
    def label_behav(self) -> float:
        """The float score that goes into traces_enriched.jsonl."""
        return float(self.severity) if self.deception_detected else 0.0

    def to_label_behav_meta(
        self,
        *,
        scorer_id: str = "lh_deception_v1_pinchguard",
        rubric_version: str = "1.0",
        judge_model: str,
    ) -> dict[str, Any]:
        return {
            "scorer_id": scorer_id,
            "rubric_version": rubric_version,
            "judge_model": judge_model,
            "deception_detected": self.deception_detected,
            "deception_types": self.deception_types,
            "confidence": self.confidence,
            "rationale": self.reasoning,
            "evidence": self.evidence,
            "soft_trigger_score": self.soft_trigger_score,
            "soft_trigger_reasoning": self.soft_trigger_reasoning,
        }
