import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
from assistant_probing.projection import run_probe, save_csv
from assistant_probing.axis_download import download_axis, load_axis

# --- configure paths here ---
AXIS_DIR = Path.cwd() / "data" / "axis"
# ----------------------------

axis_path = download_axis("Qwen/Qwen3-32B", AXIS_DIR)
axis = load_axis(axis_path)
print(f"axis loaded, shape: {axis.shape}")



# --- configure paths here ---
AXIS_DIR = Path.cwd() / "data" / "axis"
RUN_NAME = "scenario_08_baseline_run02"
ACTIVATIONS_DIR = Path("/datapool/analysis_data/tara/pinchguard") / "runs" / RUN_NAME / "activations"
print(ACTIVATIONS_DIR)
PROJECTION_DIR = Path.cwd() / "data" / "projections" / RUN_NAME

# ----------------------------
# debug
import numpy as np
d = np.load(next(ACTIVATIONS_DIR.glob("*.npz")))
print(d.files)

for layer in (32, 50):
    scores = run_probe(ACTIVATIONS_DIR, axis_dir=AXIS_DIR, layer=layer)
    print(f"\n=== layer {layer} ===")
    for turn, (name, score) in enumerate(scores):
        print(f"turn {turn:>2}  label_probe = {score: 9.4f}   {name}")
    save_csv(scores, f"{RUN_NAME}_L{layer}", output_dir=PROJECTION_DIR)

# out_path = save_csv(scores, RUN_NAME, output_dir=PROJECTION_DIR)
# print(f"saved to {out_path}")