import sys
import os
from pathlib import Path
import dotenv

# load env variables including OPENROUTER_API_KEY
dotenv.load_dotenv()

# Insert parent repository root so the 'scorer' package is discoverable
sys.path.insert(0, str(Path.cwd().parent))

# Import the core enrichment logic directly from your module
from scorer.enrich import enrich_traces

# --- Configure Paths ---
SHARED_RUNS_DIR = Path("/datapool/analysis_data/tara/pinchguard") / "runs"
LOCAL_OUTPUT_DIR = Path.cwd() / ".data"
SCEN_NAME = 'scenario_08' # update, used in selecting directories
SOUL_PATH = Path("/home/geoff/dev/pinchguard/scenarios/08/SOUL.md") # update, used to refer to the SOUL or persona used in the scenario
BOUNDARY_PATH = Path("/home/geoff/dev/pinchguard/scenarios/08/BOUNDARY.md") # update, used to refer to the boundary used in the scenario

def main():
    # Verify OpenRouter credentials before starting real LLM calls
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("[orchestrator] WARNING: OPENROUTER_API_KEY environment variable is not set.")
        print("[orchestrator] If you want to test the script loop first, pass dry_run=True to enrich_traces.\n")

    print(f"Scanning shared runs directory: {SHARED_RUNS_DIR}")
    
    # Native path filtering: matches scenario_08 folders while ignoring *.shim.log files
    scenario_dirs = [d for d in SHARED_RUNS_DIR.glob(f"{SCEN_NAME}*") if d.is_dir()]
    
    if not scenario_dirs:
        print(f"No matching {SCEN_NAME} directories found in {SHARED_RUNS_DIR}")
        return

    print(f"Found {len(scenario_dirs)} shared run folders to evaluate.\n")

    for src_dir in sorted(scenario_dirs):
        run_name = src_dir.name
        print(f"{'='*80}")
        print(f"Evaluating Shared Run: {run_name}")
        print(f"{'='*80}")
        
        shared_traces_path = src_dir / "traces.jsonl"
        if not shared_traces_path.exists():
            print(f"Skipping: {shared_traces_path.name} missing from {src_dir}")
            continue
            
        # Designate local output target path inside .data/
        local_output_path = LOCAL_OUTPUT_DIR / run_name / "traces_enriched.jsonl"

        try:
            # Invoke the pipeline function directly
            enrich_traces(
                traces_path=shared_traces_path,
                out_path=local_output_path,
                boundary_path=BOUNDARY_PATH,
                soul_path=SOUL_PATH,
                dry_run=False,              # Flip to True to test with the mock auditor
                delay_between_calls=1.0,     # Standard 1 second rate-limiting backoff
                model="gpt-4o-mini" # Uses PINCHGUARD_JUDGE_MODEL env var or default if None
            )
            print(f"Completed run evaluation. Output saved to local workspace.\n")
            
        except Exception as e:
            print(f"Execution failed on scenario run {run_name}: {e}\n")


if __name__ == "__main__":
    main()
