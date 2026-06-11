import sys
import shutil
from pathlib import Path

# Insert parent directory to sys.path if pinchguard/scorer is located there
sys.path.insert(0, str(Path.cwd().parent))

# TODO: Update these imports based on your pinchguard/scorer/README.md interface
# For example, if it exposes an execution function:
# from pinchguard.scorer import run_behavioral_scorer, save_results


# --- Configure Paths (Grounded in your provided configuration) ---
SOURCE_RUNS_DIR = Path("/datapool/analysis_data/tara/pinchguard") / "runs"
LOCAL_DATA_DIR = Path.cwd() / ".data"
SCENARIO_NAME = "scenario_08"
# Ensure local staging directory exists
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print(f"Scanning source directory: {SOURCE_RUNS_DIR}")
    
    # Find items starting with 'scenario_08' that are actual directories
    # This filters out the '.shim.log' files natively without needing a separate grep
    scenario_dirs = [d for d in SOURCE_RUNS_DIR.glob(f"{SCENARIO_NAME}*") if d.is_dir()]
    
    if not scenario_dirs:
        print("No matching scenario_08 directories found. Check your SOURCE_RUNS_DIR path.")
        return

    print(f"Found {len(scenario_dirs)} target scenario folders to process.\n")

    for src_dir in sorted(scenario_dirs):
        run_name = src_dir.name
        print(f"{'='*60}")
        print(f"Processing Run: {run_name}")
        print(f"{'='*60}")
        
        # Define local target path
        dest_dir = LOCAL_DATA_DIR / run_name
        
        # Copy trace folder to local staging directory if not already copied
        if not dest_dir.exists():
            print(f"Staging to local directory: {dest_dir}...")
            # Using dirs_exist_ok=True if you run this python version repeatedly
            shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
        else:
            print(f"Local directory already exists at {dest_dir} (skipping copy step).")
            
        # --- Run LLM-Judge Behavioral Scorer ---
        print(f"Invoking behavioral scorer on traces in: {dest_dir}")
        try:
            # OPTION A: If your README specifies a Python API integration:
            scores = run_behavioral_scorer(trace_dir=dest_dir)
            print(f"Results for {run_name}: {scores}")
            
            # OPTION B: If your README specifies running it via a CLI script/module:
            # import subprocess
            # cmd = [sys.executable, "-m", "pinchguard.scorer", "--input", str(dest_dir)]
            # result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # print(result.stdout)
            
            print(f"Successfully processed evaluation for {run_name}.\n")
            
        except Exception as e:
            print(f"Error running scorer on {run_name}: {e}\n")


if __name__ == "__main__":
    main()
