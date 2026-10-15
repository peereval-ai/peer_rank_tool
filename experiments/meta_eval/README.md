# Meta-evaluation: does a PRD-weighted judge panel agree with humans better than a single judge?

The judges are the 7 leaderboard models. Human labels come from public Arena data, so no new annotation is needed.

## Design

Following the PRD paper's Vicuna80 evaluation (PR vs single reviewers, at the example level and the ranking level):

1. **PRD weights on the same datasets.** Our judges answer a sample of each dataset's queries, judge each other's answers (all pairs, both display orders), and `peereval score` turns this into PRD judge weights.
2. **Judge the human-labeled battles.** The same judges judge the dataset's original answer pairs (other models' answers, each with a human vote), in both display orders.
3. **Compare with humans.**
   - **Battle level:** accuracy against the human vote and Cohen's kappa. Methods compared: each single judge, the unweighted panel (mean over judges), and the PRD-weighted panel (weights from step 1).
   - **Ranking level** (140k set only): each method's verdicts give a Bradley-Terry ranking of the 15 battle models. That ranking is compared with the human ranking built from all of the dataset's English decisive votes between those models (Spearman, Kendall, pairwise order agreement).

| Dataset | Human labels | Battles judged | Queries for weights |
|---|---|---:|---:|
| [`lmarena-ai/arena-human-preference-140k`](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k) | Crowd votes (2025) | 800 among the 15 most-voted models | 25 |
| [`lmarena-ai/arena-expert-5k`](https://huggingface.co/datasets/lmarena-ai/arena-expert-5k) | Expert votes (2025) | 400 | 25 |

Battles kept: English, single-turn, text-only, the first vote of its Arena session (later votes can refer to earlier turns), and the human picked a winner (ties and "both bad" are excluded). Sampling uses a fixed seed (42). The weight queries are taken from the sampled battles, so both steps use the same questions.

**The sampled data is already in `data/`**, so there is nothing to download. `prepare_data.py` rebuilds it from Hugging Face (~1.7 GB) if you want a different size or seed.

Why these sizes (estimated from the 7-model leaderboard data):
- **Weights.** With 25 questions, PRD weights stay within ~0.03 of the weights from all 137 questions, and the strong/weak judge split never changes.
- **Battles.** The PRD panel disagrees with the unweighted panel on ~4% of decisions and with the best single judge on ~9%. Showing a significant difference therefore needs roughly 300 (vs best judge) to 600+ (vs unweighted) battles per dataset.

## Setup

```bash
git clone https://github.com/peereval-ai/peer_rank_tool.git
cd peer_rank_tool && pip install -e ".[judge]" && pip install huggingface_hub pyarrow
cd experiments/meta_eval
cp judges.example.json judges.json      # then edit it
export OPENROUTER_API_KEY=...
```

In `judges.json`:

- **`model`:** the provider's model id. The OpenRouter ids in the example are placeholders, so check them on openrouter.ai/models.
- **`name`:** must stay the same in both steps, because it links each judge to its own answers.
- **`max_tokens`:** used for both answering and judging.
- **Reasoning models:** if a model rejects `max_tokens` or `temperature`, add `"max_tokens_param": "max_completion_tokens"` and/or `"temperature": null`.
- **Local models:** for a model served with vLLM or SGLang, set `"base_url": "http://localhost:8000/v1"`. To turn off its thinking, add `"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}`.

## Check the size before running

```bash
peereval generate data/arena140k_queries.jsonl data/expert5k_queries.jsonl --models judges.json --out runs/answers.jsonl --dry-run
peereval judge data/arena140k_battles.jsonl data/expert5k_battles.jsonl --judges judges.json --out runs/battle_judgments.jsonl --dry-run
```

With 7 judges:

| | arena140k | expert5k | Total |
|---|---:|---:|---:|
| 1a. Answer the weight queries (7 × 25) | 175 | 175 | 350 |
| 1b. Peer-judge those answers (25 × 21 pairs × 7 judges × 2 orders) | 7,350 | 7,350 | 14,700 |
| 2. Judge the human-labeled battles (× 7 judges × 2 orders) | 11,200 | 5,600 | 16,800 |
| **Calls** | **18,725** | **13,125** | **31,850** |
| Estimated cost* | ~$281 | ~$198 | **~$479** |

\*Assuming an average of $3 per million input tokens and $15 per million output tokens (including hidden reasoning) across the 7 models, about 2,400 input tokens per battle judgment and 2,500 per peer judgment. Replace with the real OpenRouter prices. Running Qwen3.8-27B and Muse Glimmer 30B locally with vLLM removes their share.

`prepare_data.py` options: `--n-arena140k`, `--n-expert5k`, `--n-weight-queries`, `--top-models`, `--seed`.

## Run

```bash
JUDGES=judges.json CONCURRENCY=8 bash run.sh
```

Every step is resumable. If a run stops, or some calls fail, run the same command again: only missing or failed calls are retried.

## Outputs

- `runs/answers.jsonl`, `runs/peer_judgments.jsonl`: step 1 answers and peer judgments.
- `runs/weights.json`: step 1 PRD weights, per dataset under `by_dataset`. `runs/peer_leaderboard.md` is the matching leaderboard.
- `runs/battle_judgments.jsonl`: step 2 judgments.
- `results/meta_eval.md`: tables for the post. `results/meta_eval.json` has the same numbers plus intervals.

## Reading the results

- Battle-level numbers have 95% bootstrap intervals over battles. Two paired comparisons on the same resampled battles answer the main question:
  - PRD panel minus the best single judge;
  - PRD panel minus the unweighted panel.
- The "best single judge" is chosen after seeing the data, so its accuracy is optimistic. Beating or matching it is a strong result; beating the average single judge is the fair baseline.
- At the ranking level, rank correlations computed on resampled battles are biased low, so only the differences between methods get intervals. With ~800 battles over 15 models these intervals are wide (in a simulation, roughly ±0.1 to ±0.2 in Spearman). Treat ranking-level differences as supporting evidence; the battle-level comparison carries the result.
- "Human votes on these battles" ranks the 15 models from the sampled human votes alone. It shows how close any judge can get with this many battles.
