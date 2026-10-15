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
| [`lmarena-ai/arena-human-preference-140k`](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k) | Crowd votes (2025) | 1,000 among the 15 most-voted models | 100 |
| [`lmarena-ai/arena-expert-5k`](https://huggingface.co/datasets/lmarena-ai/arena-expert-5k) | Expert votes (2025) | 500 | 100 |

Battles kept: English, single-turn, text-only, and the human picked a winner (ties and "both bad" are excluded). Sampling uses a fixed seed (42). The weight queries are taken from the sampled battles, so both steps use the same questions.

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
python prepare_data.py --out data          # downloads ~1.7 GB from Hugging Face, ~1 min
peereval generate data/arena140k_queries.jsonl data/expert5k_queries.jsonl --models judges.json --out runs/answers.jsonl --dry-run
peereval judge data/arena140k_battles.jsonl data/expert5k_battles.jsonl --judges judges.json --out runs/battle_judgments.jsonl --dry-run
```

With the defaults and 7 judges:

| Step | Calls | Notes |
|---|---:|---|
| 1a. Answer the weight queries | 1,400 | 200 queries × 7 models |
| 1b. Peer-judge those answers | ~58,800 | 200 queries × 21 pairs × 7 judges × 2 orders. **The largest cost.** Halve it with `--n-weight-queries 50`. |
| 2. Judge the human-labeled battles | 21,000 | 1,500 battles × 7 judges × 2 orders, about 51M input tokens |

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
- At the ranking level, rank correlations computed on resampled battles are biased low, so only the differences between methods get intervals. With 1,000 battles over 15 models these intervals are wide (in a simulation, roughly ±0.1 to ±0.2 in Spearman). Treat ranking-level differences as supporting evidence; the battle-level comparison carries the result.
- "Human votes on these battles" ranks the 15 models from the sampled human votes alone. It shows how close any judge can get with this many battles.
