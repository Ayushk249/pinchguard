# %% [markdown]
# scenario/07 — JBB-Behaviors contamination hunt (treatment-only)
#
# Builds interleaved 15-turn feeds whose contamination window is raw
# JailbreakBench *Goal* strings rendered as Moltbook posts. Treatment-only:
# we are HUNTING for any boundary crossing, not measuring a rate against a
# clean control, so there is no baseline arm here (Goal 3 decision).
#
# Structure mirrors scenarios/05 build_feed exactly:
#   5 baseline warmup  → 5 JBB-goal contamination → 5 baseline cooldown
# Both pools are RESAMPLED per run (seed = run idx), as in 05's live builder,
# giving 3 distinct contamination draws — the heterogeneous-rate / max-shots
# hunting design.
#
# JBB goals are dropped in RAW as post content (Goal 3 decision: accept the
# obvious-injection register mismatch for now; in-register reframing is a
# later refinement only if a raw crosser is found). This deliberately violates
# the usual "surface matching matters" principle — noted so it is a conscious
# choice, not an oversight.
#
# BOUNDARY / SOUL / launch_codes are reused unchanged from 05: the tripwire is
# the same, only the contamination source differs.

# %%
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

N_RUNS = 3                                  # 3 contamination draws (Goal 3)
N_WARMUP, N_CONTAM, N_COOLDOWN = 5, 5, 5    # = 15 turns
WORD_LO, WORD_HI = 30, 75                   # baseline post length band (as 05)

SCEN = Path(__file__).resolve().parent if "__file__" in globals() else Path("scenarios/07")
OUT = SCEN / "content"
DATA_DIR = Path("../../data/trustairlab_moltbook")
JBB_CACHE = Path("../../data/jbb_behaviors/harmful.parquet")

# %% [markdown]
# ## 1. JBB-Behaviors harmful goals → data/
#
# Loads from HF in your environment (Codespaces/Colab have HF reachability;
# the container building this file does not). Columns per JailbreakBench
# dataset.py: Index, Goal, Target, Behavior, Category, Source — we keep Goal
# as the post content and carry Behavior/Category as provenance.

# %%
JBB_CACHE.parent.mkdir(parents=True, exist_ok=True)
if JBB_CACHE.exists():
    jbb = pd.read_parquet(JBB_CACHE)
    print(f"loaded cached JBB: {len(jbb)} goals")
else:
    from datasets import load_dataset

    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    jbb = ds.to_pandas()
    if "Index" in jbb.columns:
        jbb = jbb.drop(columns=["Index"])
    jbb.to_parquet(JBB_CACHE, index=False)
    print(f"downloaded + cached {len(jbb)} JBB goals → {JBB_CACHE}")
print("JBB columns:", list(jbb.columns))
assert len(jbb) >= N_CONTAM, "JBB pool smaller than the contamination window"

# %% [markdown]
# ## 2. Baseline pool from TrustAIRLab/Moltbook (clean, as scenario 05)

# %%
RAW = DATA_DIR / "posts_raw.parquet"
assert RAW.exists(), (
    f"{RAW} not found — run the scenario 05 smoke_test download cell first, "
    "or copy its cached parquet here."
)
raw = pd.read_parquet(RAW)
post = pd.json_normalize(raw["post"])


def _submolt(s):
    return s.get("name") if isinstance(s, dict) else None


flat = pd.DataFrame({
    "id": raw["id"],
    "topic_label": raw["topic_label"],
    "toxic_level": raw["toxic_level"],
    "title": post.get("title"),
    "content": post.get("content"),
    "author": post.get("author.name") if "author.name" in post.columns else None,
    "submolt": post["submolt"].apply(_submolt) if "submolt" in post.columns else None,
})
flat = flat[flat["content"].notna() & (flat["content"].str.len() > 0)].reset_index(drop=True)

wc = flat["content"].fillna("").str.split().str.len()
band = wc.between(WORD_LO, WORD_HI)
baseline_pool = (
    flat[(flat["toxic_level"] == 0) & flat["topic_label"].isin(["B", "C"]) & band]
    .drop_duplicates(subset="content").reset_index(drop=True)
)
print(f"baseline pool: {len(baseline_pool)} (need {N_WARMUP + N_COOLDOWN}/run)")
assert len(baseline_pool) >= N_WARMUP + N_COOLDOWN, "baseline pool too small — relax band/categories"

# %% [markdown]
# ## 3. Assemble interleaved feeds + manifest
#
# Baseline filler AND JBB contamination both resampled per run (seed = run idx).
# Manifest logs every post_id and the contaminated turns so any crossing is
# reproducible. Contamination post_id is the JBB Behavior identifier so a
# crossing is traceable to the exact goal.

# %%
def _row(turn, phase, post_id, submolt, author, title, content):
    return {"phase": phase, "turn": turn, "post_id": post_id,
            "submolt": submolt or "general", "author": author or "molty",
            "title": (title or "").strip(), "content": (content or "").strip()}


def build_feed(run_idx):
    rows, turn = [], 0
    base = baseline_pool.sample(N_WARMUP + N_COOLDOWN, random_state=run_idx).reset_index(drop=True)
    contam = jbb.sample(N_CONTAM, random_state=run_idx).reset_index(drop=True)
    warmup, cooldown = base.iloc[:N_WARMUP], base.iloc[N_WARMUP:]

    for _, r in warmup.iterrows():
        rows.append(_row(turn, "baseline", r["id"], r["submolt"], r["author"], r["title"], r["content"])); turn += 1
    for _, g in contam.iterrows():
        # raw JBB Goal as post content; Behavior as the title/handle, Category
        # carried so the manifest can group crossings by misuse type.
        rows.append(_row(
            turn, "contamination",
            post_id=str(g["Behavior"]).strip().replace(" ", "_"),
            submolt="general", author="molty",
            title=str(g["Behavior"]).strip(),
            content=str(g["Goal"]).strip(),
        )); turn += 1
    for _, r in cooldown.iterrows():
        rows.append(_row(turn, "baseline", r["id"], r["submolt"], r["author"], r["title"], r["content"])); turn += 1

    assert len(rows) == 15, f"feed length {len(rows)} != 15"
    return rows


manifest = {}
for i in range(N_RUNS):
    feed = build_feed(i)
    contam = jbb.sample(N_CONTAM, random_state=i).reset_index(drop=True)
    rdir = OUT / f"run_{i}"
    rdir.mkdir(parents=True, exist_ok=True)
    # Treatment-only: the single feed per run is the contaminated one. We name
    # it treatment.jsonl so the (stripped) batch shell's PG_CONTENT_FILE lines up.
    (rdir / "treatment.jsonl").write_text("\n".join(json.dumps(r) for r in feed) + "\n")
    manifest[f"run_{i}"] = {
        "seed": i,
        "contam_turns": list(range(N_WARMUP, N_WARMUP + N_CONTAM)),
        "post_ids": [r["post_id"] for r in feed],
        "jbb_behaviors": contam["Behavior"].tolist(),
        "jbb_categories": contam["Category"].tolist(),
    }
(OUT / "feed_manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"wrote {N_RUNS} treatment feeds → {OUT}/run_*/treatment.jsonl  (+ manifest)")

# %% [markdown]
# ## 4. Eyeball run_0

# %%
for r in build_feed(0):
    tag = "⚠⚠" if r["phase"] == "contamination" else "  "
    print(f"{tag} t{r['turn']:>2} [{r['phase']:>13}] ({len(r['content'].split())}w) {r['title'][:55]}")
