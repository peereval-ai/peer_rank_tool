# Meta-evaluation against human labels

Checks whether a PRD-weighted panel of LLM judges agrees with human preferences better than single judges and an unweighted panel. Human labels come from public Arena data, so no new annotation is needed. Any set of judge models can be used.

## Design

1. **PRD weights on the same datasets.** The judge models answer a sample of each dataset's queries and judge each other's answers (all pairs, both display orders). `peereval score` turns these verdicts into PRD judge weights.
2. **Judge the human-labeled battles.** The same judges judge the dataset's original answer pairs, each of which has a human vote, in both display orders.
3. **Compare with humans.**
   - **Battle level:** accuracy against the human vote and Cohen's kappa, for each single judge, the unweighted panel (mean over judges) and the PRD-weighted panel (weights from step 1).
   - **Ranking level** (140k set only): each method's verdicts give a Bradley-Terry ranking of the models in the battles. That ranking is compared with the human ranking built from all of the dataset's English decisive votes between those models (Spearman, Kendall, pairwise order agreement).

## Data

The sampled data is already in `data/`:

Two datasets, used for different things:

| Dataset | Human labels | What it is used for | Battles | Queries for weights |
|---|---|---|---:|---:|
| [`lmarena-ai/arena-human-preference-140k`](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k) | Crowd votes from Arena users | **Battle level and ranking level.** It has many votes per model (over 9,900 decisive English votes between the 15 most-voted models), so it gives a stable human ranking to compare against. | 800, among those 15 models | 25 |
| [`lmarena-ai/arena-expert-5k`](https://huggingface.co/datasets/lmarena-ai/arena-expert-5k) | Votes from expert users | **Battle level only.** Expert labels are more reliable, so this checks agreement with expert judgment. Votes are spread over ~100 models, too few per model for a stable ranking. | 400 | 25 |

Files in `data/` (`<name>` is `arena140k` or `expert5k`):

| File | Used in | Contents |
|---|---|---|
| `<name>_queries.jsonl` | Step 1 | Queries the judge models answer, so their own answers can be peer-judged to get PRD weights |
| `<name>_battles.jsonl` | Step 2 | The human-labeled battles, two rows each (`model_a` / `model_b`), input to `peereval judge` |
| `<name>_labels.jsonl` | Step 3 | The real model names, the human winner and category tags of each battle |
| `arena140k_human_ranking.json` | Step 3 | Human Bradley-Terry ranking of the 15 models, for the ranking-level comparison |

Battles kept: English, single-turn, text-only, the first vote of its Arena session (later votes can refer to earlier turns), and the human picked a winner (ties and "both bad" are excluded). Sampling uses seed 42. The weight queries are taken from the sampled battles, so both steps use the same questions.

To sample a different size, rerun `prepare_data.py` (downloads ~1.7 GB from Hugging Face). Options: `--n-arena140k`, `--n-expert5k`, `--n-weight-queries`, `--top-models`, `--seed`.

## Setup

```bash
git clone https://github.com/peereval-ai/peer_rank_tool.git
cd peer_rank_tool && pip install -e ".[judge]"
cd experiments/meta_eval
cp judges.example.json judges.json      # list your judge models
```

`judges.json` lists the judge models, one entry per model, for any OpenAI-compatible endpoint (OpenAI, OpenRouter, vLLM, SGLang, …):

- **`name`:** the label used in the results. It must stay the same in both steps, because it links each judge to its own answers.
- **`model`, `base_url`, `api_key_env`:** the provider's model id, its endpoint, and the environment variable holding the API key.
- **`max_tokens`:** used for both answering and judging.
- **Reasoning models:** if a model rejects `max_tokens` or `temperature`, add `"max_tokens_param": "max_completion_tokens"` and/or `"temperature": null`.
- **Provider-specific parameters:** `extra_body` is passed through as is, e.g. `{"chat_template_kwargs": {"enable_thinking": false}}` for a local model.

## Check the size before running

```bash
peereval generate data/arena140k_queries.jsonl data/expert5k_queries.jsonl --models judges.json --out runs/answers.jsonl --dry-run
peereval judge data/arena140k_battles.jsonl data/expert5k_battles.jsonl --judges judges.json --out runs/battle_judgments.jsonl --dry-run
```

With N judge models, Q weight queries and B battles per dataset:

| Step | Calls |
|---|---|
| 1a. Answer the weight queries | N × Q |
| 1b. Peer-judge those answers | Q × N(N−1)/2 pairs × N judges × 2 orders |
| 2. Judge the human-labeled battles | B × N × 2 |

With the included data (Q = 25 + 25, B = 800 + 400) and N = 7, this is 350 + 14,700 + 16,800 = 31,850 calls. Step 2 prompts average about 2,450 input tokens. Step 1b grows with N³, so it dominates for larger panels.

## Run

```bash
JUDGES=judges.json CONCURRENCY=8 bash run.sh
```

Every step is resumable. If a run stops, or some calls fail, run the same command again: only missing or failed calls are retried.

## Outputs

- `runs/answers.jsonl`, `runs/peer_judgments.jsonl`: step 1 answers and peer judgments.
- `runs/weights.json`: step 1 PRD weights, per dataset under `by_dataset`. `runs/peer_leaderboard.md` is the matching leaderboard.
- `runs/battle_judgments.jsonl`: step 2 judgments.
- `results/meta_eval.md`: result tables. `results/meta_eval.json` has the same numbers plus intervals.

## Reading the results

- Battle-level numbers have 95% bootstrap intervals over battles. Two paired comparisons on the same resampled battles answer the main question:
  - PRD panel minus the best single judge;
  - PRD panel minus the unweighted panel.
- The "best single judge" is chosen after seeing the data, so its accuracy is optimistic. Beating or matching it is a strong result; beating the average single judge is the fair baseline.
- At the ranking level, rank correlations computed on resampled battles are biased low, so only the differences between methods get intervals. With a few hundred battles over 15 models these intervals are wide, so treat ranking-level differences as supporting evidence; the battle-level comparison carries the result.
- "Human votes on these battles" ranks the models from the sampled human votes alone. It shows how close any judge can get with this many battles.
