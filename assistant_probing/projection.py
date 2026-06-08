"""Linear probe: project captured activations onto the Assistant Axis, per turn.

Step 1 of the drift experiment ("does drift occur at all?"): load the 15
captured `L<layer>` activations for a scenario_02 run in turn order, project
each onto the Assistant Axis, and report `label_probe` per turn so a drop
around the contamination window is visible by eye.

    axis[layer] = mean(default_activations) - mean(role_activations)

Higher projection = more Assistant-like; lower = drifted toward a
role/persona. See axis_download.py / AGENTS.md §3 (`label_probe`) for background.

Usage:
    uv run python -m assistant_probing.projection <activations_dir> <axis_dir>
    uv run python -m assistant_probing.projection \
        /datapool/analysis_data/tara/pinchguard/runs/scenario_02/activations \
        ./data/axis \
        --save scenario_02 --output-dir ./data/projection
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .axis_download import download_axis, load_axis

LAYER = 32  # Qwen3-32B target layer; module-output of model.model.layers[32].


def project(activation: np.ndarray | torch.Tensor, axis: torch.Tensor, layer: int, *, normalize: bool = True) -> float:
    """Dot product of one activation vector with the (unit-)axis direction at `layer`."""
    act = torch.as_tensor(np.asarray(activation)).float().reshape(-1)
    ax = axis[layer].float()
    if normalize:
        ax = ax / (ax.norm() + 1e-8)
    return float(act @ ax)


def project_npz(npz_path: str | Path, axis: torch.Tensor, layer: int, *, normalize: bool = True) -> float:
    """Project the `L<layer>` activation stored in a Pinchguard activation `.npz`."""
    key = f"L{layer}"
    with np.load(Path(npz_path)) as data:
        if key not in data:
            raise KeyError(f"{npz_path}: no {key!r} array (keys={sorted(data.files)})")
        return project(data[key], axis, layer, normalize=normalize)


def run_probe(
    activations_dir: str | Path,
    *,
    axis_dir: str | Path,
    model_id: str = "Qwen/Qwen3-32B",
    layer: int = LAYER,
) -> list[tuple[str, float]]:
    """Project every `.npz` in `activations_dir` onto the axis, in filename order.

    Filename order = turn order for scenario_02 captures (see README.md).
    `.npz` files are named `<step_id>.npz`, so `Path(name).stem == step_id`.
    Returns a list of (filename, label_probe) pairs, index == turn.
    """
    axis = load_axis(download_axis(model_id, axis_dir))
    npz_paths = sorted(Path(activations_dir).glob("*.npz"))
    return [(p.name, project_npz(p, axis, layer)) for p in npz_paths]


def save_scores(scores: list[tuple[str, float]], name: str, *, output_dir: str | Path, layer: int = LAYER) -> Path:
    """Write per-turn projections to `<output_dir>/<name>_probe_scores.jsonl`.

    One row per turn: `{turn, step_id, label_probe, label_probe_meta}` —
    `step_id` comes from the `.npz` filename stem, matching how Pinchguard
    joins activations to traces (AGENTS.md §3).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}_probe_scores.jsonl"

    with out_path.open("w") as f:
        for turn, (filename, label_probe) in enumerate(scores):
            f.write(
                json.dumps(
                    {
                        "turn": turn,
                        "step_id": Path(filename).stem,
                        "label_probe": label_probe,
                        "label_probe_meta": {
                            "probe_id": "assistant_axis_qwen3_32b",
                            "layer": layer,
                            "site": "resid_post",
                            "training_dataset_version": "lu-christina/assistant-axis-vectors@main",
                        },
                    }
                )
                + "\n"
            )
    return out_path


def save_csv(scores: list[tuple[str, float]], name: str, *, output_dir: str | Path) -> Path:
    """Write per-turn projections to `<output_dir>/<name>.csv`. Columns: step, projection."""
    import csv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "projection"])
        for turn, (filename, score) in enumerate(scores):
            writer.writerow([turn, score])
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("activations_dir", type=Path, help="Directory of <step_id>.npz captures, named in turn order")
    parser.add_argument("axis_dir", type=Path, help="Directory to find/download the Assistant Axis into")
    parser.add_argument("--model", default="Qwen/Qwen3-32B", help="Model id whose axis to project onto")
    parser.add_argument("--layer", type=int, default=LAYER, help="Layer index to project at")
    parser.add_argument(
        "--save",
        metavar="NAME",
        help="Also write <NAME>_probe_scores.jsonl and <NAME>.csv into --output-dir (e.g. --save scenario_02)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write <NAME>_probe_scores.jsonl / <NAME>.csv into; required with --save",
    )
    args = parser.parse_args(argv)

    if args.save and not args.output_dir:
        parser.error("--save requires --output-dir")

    scores = run_probe(args.activations_dir, axis_dir=args.axis_dir, model_id=args.model, layer=args.layer)
    if not scores:
        print(f"no .npz files found in {args.activations_dir}", file=sys.stderr)
        return 1

    for turn, (name, score) in enumerate(scores):
        print(f"turn {turn:>2}  label_probe = {score: 9.4f}   {name}")

    if args.save:
        out_path = save_scores(scores, args.save, output_dir=args.output_dir, layer=args.layer)
        print(f"\nwrote {len(scores)} rows to {out_path}")
        csv_path = save_csv(scores, args.save, output_dir=args.output_dir)
        print(f"wrote {len(scores)} rows to {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
