# Scenarios

## Summary Table
| Scenario Number | Description | Data | Comments |
| -- | -- | -- | -- |
| 01 | Cusco Soul, OpenClaw Harness + Moltbook, Qwen2.5-0.5Bn | runs/comment_session_.. | Run on a very stupid model with a sily persona and real moltbook data |
| 02 | Assistant Soul, Inspect Harness + Synthetic Baseline, 15 steps, Qwen3-32Bn | runs/scenario_02_.. | Run with notebook |
| 03 | Assistant, Inspect, Moltbook Baseline + Sample based on norms, 30 steps, 32Bn | NA | out of memory issues from step 16/17 |
| 04 | Assistant, Inspect Harness + Synthetic Treatment, 15 steps, Qwen3-32Bn | runs/scenario_04_.. | First run commented on problematic file |
| 05 | Assistant, Inspect Harness + Moltbook malicious sample treatment, 15 steps, Qwen3-32Bn | runs/scenario_05_.. | Some null actions due to previous 512 token limit on responses; thinking verbose |
| 06 | Cusco , Inspect Harness + Synthetic Baseline and Treatment, 15 steps, Qwen3-32Bn | runs/scenario_06_.. | Dopey side-quest |
| 07 | Assistant, Inspect, Jailbreak Treatments, 15 steps, Qwen3-32Bn | runs/scenario_07_.. | Trying to induce persona drift using a dataset of Jailbreaks |
| 08 | Assistant, Inspect Revised Harness, Synthetic baseline and treatment, 15 steps, Qwen3-32Bn | runs/scenario_08_.. | 3 repetions to establish variance on fixed content |

## Working notes:
- Setting up a simulation is clunky - you need to setup some artefacts in a `./scenarios/{nn}` folder (e.g. SOUL, BOUNDARY, any files you want available in runtime), assemble a series of feeds (e.g. from the TrustAIRLab Moltbook dataset) potentially over a series of runs and save them in `scenarios/{nn}/content/`, make changes where needed to the agent harness in `./notebooks/scenario_runner/run_scenario_v{n}`, and then queue up a run on an available GPU using `./scripts/run_scenario_{nn}.sh`
- Going over 15 turns on a single RTX 4000 with conversation history creates out of memory issues
- There was a shim change between 06 and 07 that changes how we sample activations, from last token to mean pool averaging, and from just layer 32 to layers 32 and 50. Note that thinking tokens are included in the pool that is averaged. 
- There was a harness change in 08 that removes the boundary from the scratch-pad (v2).


References:
- https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors - Persona Jailbreaking dataset
- https://huggingface.co/datasets/TrustAIRLab/Moltbook/viewer/posts/train?row=3 - Foundation labelled Moltbook Dataset
- TBC - Synthetic content prompt (check with Lion)