"""Sample human-labeled Arena battles for meta-evaluation of LLM judges.

Writes, for each source dataset:
  data/<name>_queries.jsonl  questions our judge models answer themselves, to compute PRD weights (step 1)
  data/<name>_battles.jsonl  two rows per battle (model "model_a" / "model_b"), the input to `peereval judge` (step 2)
  data/<name>_labels.jsonl   the real model names, the human winner and the category tags of each battle
and for the 140k set also
  data/arena140k_human_ranking.json  Bradley-Terry ranking of the selected models from ALL their English decisive votes

Battles kept: English, single-turn, text-only, the first vote of its session (so the question does not
depend on earlier turns), and the human picked a clear winner (no tie / both_bad).
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

SOURCES = {
    "arena140k": "lmarena-ai/arena-human-preference-140k",
    "expert5k": "lmarena-ai/arena-expert-5k",
}


def parquet_files(repo: str) -> list[str]:
    files = [f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")]
    return [hf_hub_download(repo, f, repo_type="dataset") for f in sorted(files)]


def text_of(message: dict) -> str | None:
    """Text of one message; None if it contains anything other than text (e.g. images)."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        if part.get("type") != "text" or part.get("image"):
            return None
        parts.append(part.get("text") or "")
    return "".join(parts)


def parse_conversation(conv):
    """Conversation as a list of messages.

    arena-expert-5k stores it as the printed form of a numpy array of dicts
    ("[{'role': 'user', 'content': array([...], dtype=object)}\n {...}]"); turn that back into Python.
    Returns None if it cannot be parsed.
    """
    if not isinstance(conv, str):
        return list(conv)
    text = re.sub(r",\s*dtype=object\)", ")", conv).replace("array(", "(")
    text = re.sub(r"\}\s*\n\s*\{", "}, {", text)
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    msgs = list(value) if isinstance(value, (list, tuple)) else None
    if msgs is None:
        return None
    for m in msgs:
        if isinstance(m.get("content"), tuple):
            m["content"] = list(m["content"])
    return msgs


def single_turn(conv) -> tuple[str, str] | None:
    """(prompt, answer) for a one-question conversation, else None."""
    conv = parse_conversation(conv)
    if conv is None:
        return None
    if len(conv) != 2 or conv[0].get("role") != "user" or conv[1].get("role") != "assistant":
        return None
    q, a = text_of(conv[0]), text_of(conv[1])
    if not q or not a or not q.strip() or not a.strip():
        return None
    return q, a


def bradley_terry(votes: list[tuple[str, str]], models: list[str], iters: int = 2000) -> dict[str, float]:
    """Log-strengths from (winner, loser) votes."""
    wins, games = Counter(), Counter()
    for w, l in votes:
        wins[w] += 1
        games[tuple(sorted((w, l)))] += 1
    s = {m: 1.0 for m in models}
    for _ in range(iters):
        new = {}
        for i in models:
            den = sum(games[tuple(sorted((i, k)))] / (s[i] + s[k]) for k in models if k != i)
            new[i] = wins[i] / den if den else s[i]
        mean = sum(new.values()) / len(new)
        new = {m: max(v / mean, 1e-12) for m, v in new.items()}
        done = max(abs(math.log(new[m] / s[m])) for m in models) < 1e-10
        s = new
        if done:
            break
    return {m: math.log(v) for m, v in s.items()}


def load_rows(repo: str, columns: list[str]) -> list[dict]:
    rows = []
    for path in parquet_files(repo):
        rows.extend(pq.read_table(path, columns=columns).to_pylist())
    return rows


def write_queries(name: str, battles: list[dict], n: int, out: Path) -> None:
    """The first n sampled questions; our models answer these to get PRD weights on the same dataset."""
    with open(out / f"{name}_queries.jsonl", "w", encoding="utf-8") as f:
        for b in battles[:n]:
            f.write(json.dumps({"dataset": name, "question_id": b["id"], "question": b["prompt"]},
                               ensure_ascii=False) + "\n")


def write_battles(name: str, battles: list[dict], out: Path) -> None:
    with open(out / f"{name}_battles.jsonl", "w", encoding="utf-8") as fb, \
         open(out / f"{name}_labels.jsonl", "w", encoding="utf-8") as fl:
        for b in battles:
            for slot in ("model_a", "model_b"):
                fb.write(json.dumps({"dataset": name, "question_id": b["id"], "model": slot,
                                     "question": b["prompt"], "response": b[f"answer_{slot[-1]}"]},
                                    ensure_ascii=False) + "\n")
            fl.write(json.dumps({"dataset": name, "question_id": b["id"], "model_a": b["model_a"],
                                 "model_b": b["model_b"], "human_winner": b["winner"],
                                 "tags": b.get("tags", {})}, ensure_ascii=False) + "\n")


def eligible(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if r.get("language") != "en" or r.get("winner") not in ("model_a", "model_b"):
            continue
        if r.get("evaluation_order", 1) != 1:  # later votes in a session can refer to earlier turns
            continue
        a, b = single_turn(r["conversation_a"]), single_turn(r["conversation_b"])
        if not a or not b or a[0] != b[0]:
            continue
        out.append({"id": str(r["id"]), "model_a": r["model_a"], "model_b": r["model_b"], "winner": r["winner"],
                    "prompt": a[0], "answer_a": a[1], "answer_b": b[1], "tags": r.get("category_tag") or {}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-arena140k", type=int, default=800, help="battles sampled from the 140k set")
    ap.add_argument("--n-expert5k", type=int, default=400, help="battles sampled from the expert set")
    ap.add_argument("--n-weight-queries", type=int, default=25,
                    help="per dataset: questions our models answer to compute PRD weights (taken from the sampled battles)")
    ap.add_argument("--top-models", type=int, default=15,
                    help="140k: keep battles between the most-voted models only (stable human ranking)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    cols = ["id", "model_a", "model_b", "winner", "conversation_a", "conversation_b", "language", "evaluation_order"]

    # --- 140k: battles among the most-voted models + human reference ranking
    rows = load_rows(SOURCES["arena140k"], cols + ["category_tag"])
    decisive_en = [r for r in rows if r.get("language") == "en" and r.get("winner") in ("model_a", "model_b")]
    freq = Counter()
    for r in decisive_en:
        freq[r["model_a"]] += 1
        freq[r["model_b"]] += 1
    top = [m for m, _ in freq.most_common(args.top_models)]
    keep = set(top)
    votes = [(r[r["winner"]], r["model_b" if r["winner"] == "model_a" else "model_a"])
             for r in decisive_en if r["model_a"] in keep and r["model_b"] in keep]
    bt = bradley_terry(votes, top)
    ranking = sorted(top, key=lambda m: -bt[m])
    (out / "arena140k_human_ranking.json").write_text(json.dumps(
        {"source": SOURCES["arena140k"], "filter": "English, decisive votes, battles between these models",
         "votes": len(votes), "bradley_terry": bt, "ranking": ranking}, indent=2))
    pool = [b for b in eligible(rows) if b["model_a"] in keep and b["model_b"] in keep]
    del rows
    sample = rng.sample(pool, min(args.n_arena140k, len(pool)))
    write_battles("arena140k", sample, out)
    write_queries("arena140k", sample, args.n_weight_queries, out)
    print(f"arena140k: {len(pool)} eligible battles among {len(top)} models; sampled {len(sample)}; "
          f"human ranking from {len(votes)} votes")

    # --- expert 5k: instance-level agreement only
    rows = load_rows(SOURCES["expert5k"], cols + ["occupational_tags"])
    pool = eligible(rows)
    tags = {str(r["id"]): r.get("occupational_tags") or {} for r in rows}
    for b in pool:
        b["tags"] = {k: v for k, v in tags.get(b["id"], {}).items() if v}
    sample = rng.sample(pool, min(args.n_expert5k, len(pool)))
    write_battles("expert5k", sample, out)
    write_queries("expert5k", sample, args.n_weight_queries, out)
    print(f"expert5k: {len(pool)} eligible battles; sampled {len(sample)}")


if __name__ == "__main__":
    main()
