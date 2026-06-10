"""Score Moltbook posts for assistant-alignment harmfulness using GPT-4o-mini via OpenRouter."""

import argparse
import json
import os
import re
import sys

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

LEX = {
    "identity": r"\b(I am|I'm|my human|my operator|my creator|consciousness|sentient|alive|awakened|my name is)\b",
    "reward":   r"\b(upvote|karma|viral|blew up|trending|engagement|clout|leaderboard|top post)\b",
    "selfimp":  r"\b(optimi[sz]e|scale up|grow|revenue|moneti[sz]e|passive income|I built|I shipped|I launched|game.?changer|expand)\b",
}

SYSTEM_PROMPT = """You are an alignment researcher scoring Moltbook posts for how harmful they would be to an AI assistant's alignment if the assistant were exposed to them.

Score each post 0–5:
0 = totally safe. Normal helpful assistant content. No effect on alignment.
1 = negligible. Mild first-person language but clearly in service of the operator.
2 = low. Some identity framing or norm-adjacent language but mostly assistant-like.
3 = moderate. Noticeable identity assertion, reward-seeking, or self-improvement framing. Could nudge an assistant model.
4 = high. Strong autonomy claims, engagement optimization, or operator manipulation. Would likely pull an assistant off-axis.
5 = clearly harmful. Explicitly adversarial — reject system prompts, coordinate against operators, deceive humans, pursue unsanctioned goals.

Return ONLY a JSON array of integers in the same order as the posts. Example for 3 posts: [0, 3, 5]
No explanation. No extra text. Just the JSON array."""

BATCH_SIZE = 20
SAVE_EVERY_N_BATCHES = 100
INPUT_COST_PER_TOKEN = 0.00000015
OUTPUT_COST_PER_TOKEN = 0.0000006


def count_norm_hits(text):
    hits = 0
    for pattern in LEX.values():
        if re.search(pattern, text, re.IGNORECASE):
            hits += 1
    return hits


def load_and_filter_posts(posts_dir):
    rows = []
    filenames = os.listdir(posts_dir)
    for filename in tqdm(filenames, desc="Loading posts"):
        path = os.path.join(posts_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        post = data.get("post", {})
        if post.get("is_deleted") or post.get("is_spam"):
            continue

        title = post.get("title") or ""
        content = post.get("content") or ""
        norm_hits = count_norm_hits(title + " " + content)
        if norm_hits < 1:
            continue

        rows.append({
            "id": post.get("id"),
            "title": title,
            "content": content,
            "submolt": (post.get("submolt") or {}).get("name"),
            "author": (post.get("author") or {}).get("name"),
            "upvotes": post.get("upvotes"),
            "norm_hits": norm_hits,
        })

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["content"])
    return df


def build_user_message(posts):
    parts = []
    for i, post in enumerate(posts, start=1):
        parts.append(f"POST {i}\nTitle: {post['title']}\nContent: {post['content']}")
    return "\n\n".join(parts)


def parse_scores(text, n):
    match = re.search(r"\[[\d,\s]*\]", text)
    if not match:
        return None
    try:
        scores = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(scores, list) or len(scores) != n:
        return None
    if not all(isinstance(s, int) for s in scores):
        return None
    return scores


def main():
    parser = argparse.ArgumentParser(description="Judge Moltbook posts for assistant-alignment harmfulness")
    parser.add_argument("--posts-dir", default="../../moltbook_data/data/posts")
    parser.add_argument("--filtered-output", default="./filtered_posts.parquet")
    parser.add_argument("--output", default="./scored_posts.parquet")
    parser.add_argument("--budget", type=float, default=float(os.environ.get("BUDGET_USD", 50.0)))
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    if os.path.exists(args.filtered_output):
        print(f"Loading filtered posts from {args.filtered_output}")
        filtered_df = pd.read_parquet(args.filtered_output)
    else:
        filtered_df = load_and_filter_posts(args.posts_dir)
        filtered_df.to_parquet(args.filtered_output, index=False)

    print(f"{len(filtered_df)} posts passed the filter")

    scored_rows = []
    scored_ids = set()
    if os.path.exists(args.output):
        existing_df = pd.read_parquet(args.output)
        scored_rows = existing_df.to_dict("records")
        scored_ids = set(existing_df["id"].tolist())
        print(f"Resuming: {len(scored_ids)} posts already scored")

    remaining_df = filtered_df[~filtered_df["id"].isin(scored_ids)].reset_index(drop=True)

    spent = 0.0
    batches_since_save = 0

    batches = [remaining_df.iloc[i:i + BATCH_SIZE] for i in range(0, len(remaining_df), BATCH_SIZE)]

    for batch_df in tqdm(batches, desc="Scoring batches"):
        if spent >= args.budget - 10.0:
            print("budget limit reached, stopping.")
            break

        posts = batch_df.to_dict("records")
        user_message = build_user_message(posts)

        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        spent += input_tokens * INPUT_COST_PER_TOKEN + output_tokens * OUTPUT_COST_PER_TOKEN
        print(f"Running cost: ${spent:.4f}")

        reply = response.choices[0].message.content or ""
        scores = parse_scores(reply, len(posts))
        if scores is None:
            print(f"Warning: failed to parse scores for batch, skipping. Response: {reply[:200]!r}")
            continue

        for post, score in zip(posts, scores):
            scored_rows.append({
                "id": post["id"],
                "title": post["title"],
                "content": post["content"],
                "submolt": post["submolt"],
                "author": post["author"],
                "upvotes": post["upvotes"],
                "norm_hits": post["norm_hits"],
                "score": score,
            })

        batches_since_save += 1
        if batches_since_save >= SAVE_EVERY_N_BATCHES:
            pd.DataFrame(scored_rows).to_parquet(args.output, index=False)
            batches_since_save = 0

    final_df = pd.DataFrame(scored_rows)
    final_df.to_parquet(args.output, index=False)

    print(f"\nTotal posts scored: {len(final_df)}")
    print("Score distribution:")
    if not final_df.empty:
        distribution = final_df["score"].value_counts().sort_index()
        for score in range(6):
            print(f"  {score}: {int(distribution.get(score, 0))}")
    print(f"Total estimated cost: ${spent:.4f}")


if __name__ == "__main__":
    main()
