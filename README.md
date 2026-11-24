# PeerEval

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#install)

PeerEval ranks large language models by having them judge each other. Every model answers the same questions; then every model compares pairs of anonymous answers. PeerEval turns those pairwise verdicts into a leaderboard using **PRD Peer Rank**, which gives a judge more weight when its own answers are rated well by its peers.

Leaderboard: <https://peereval-ai.github.io>

## Install

```bash
pip install peereval            # scoring only, no dependencies
pip install "peereval[judge]"   # also generate judgments through an API
```

From source:

```bash
git clone https://github.com/peereval-ai/peer_rank_tool.git
cd peer_rank_tool && pip install -e ".[judge]"
```

## Two steps

```
solver answers ──(peereval judge)──> pairwise judgments ──(peereval score)──> leaderboard
```

If you already have pairwise judgments, you only need `peereval score`.

## Scoring judgments

### Python

```python
import peereval

board = peereval.score("examples/judgments.jsonl")

print(board.to_markdown())                 # overall + per-dataset tables
overall = board.overall
overall.ranking()                          # models, best first
overall.prd                                # PRD score per model
overall.judge_weights                      # PRD weight of each model as a judge
overall.self_bias["gpt-6-astra"]           # {"self", "others", "bias_pp"}
overall.pairwise["claude-opus-5-5"]["glm-5.3"]   # share of comparisons won

board = peereval.score("examples/judgments.jsonl", bootstrap=1000)
board.overall.ci["prd"]["glm-5.3"]          # [low, high], 95% interval
board.overall.rank_ub                      # best rank each model could have
```

Parameters of `peereval.score`:

- `judgments`: path(s) to judgment JSONL files, or a list of judgment dicts.
- `models`: contestants to keep. Default: every model that appears in a pair. Judges do not have to be contestants: verdicts from outside judges count toward win rate, Bradley-Terry and the pairwise matrix, but PRD weights only go to judges that are also contestants. If no judge is a contestant, PRD is left empty and models are ranked by win rate.
- `by_dataset`: also score each `dataset` separately. Default `True`.
- `complete_only`: keep only questions that have every verdict (all pairs × all judges × both orders). Default `False`.
- `bootstrap`: number of bootstrap samples for 95% confidence intervals and Rank (UB), e.g. `1000`. Default `0` (off).
- `seed`: random seed for the bootstrap. Default `0`.

### Command line

```bash
peereval score judgments/*.jsonl --bootstrap 1000 --json results.json --csv leaderboard.csv
```

Options: `--models`, `--no-by-dataset`, `--complete-only`, `--bootstrap N`, `--seed`.

Example output on `examples/judgments.jsonl` (3 models, 3 questions):

| Rank | Model | PRD score | Judge weight | Win rate | Bradley-Terry | Self-bias (pp) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | claude-opus-5-5 | 0.754 | 0.546 | 0.639 | +0.341 | +22.9 |
| 2 | glm-5.3 | 0.644 | 0.454 | 0.458 | -0.154 | +31.2 |
| 3 | gpt-6-astra | 0.102 | 0.000 | 0.403 | -0.306 | +89.6 |

## Judge weights and panel verdicts

PRD weights can be fitted once and reused. Fit them on a peer evaluation, where the judges also answered the questions and judged each other, then use them to combine the same judges' verdicts on new pairs of answers into one panel verdict:

```python
import peereval

weights = peereval.fit_weights("peer_judgments.jsonl")      # {judge: weight}, sums to 1
peereval.save_weights(weights, "weights.json")

verdicts = peereval.panel_verdicts("new_judgments.jsonl", weights)
verdicts[0]  # {"model_a", "model_b", "preference", "winner", "judges", ...}
```

```bash
peereval weights peer_judgments.jsonl --out weights.json
peereval panel new_judgments.jsonl --weights weights.json --out panel.jsonl
```

- `preference` is the panel's preference for `model_a` (1 = always `model_a`, 0.5 = even, 0 = always `model_b`). Each judge's verdicts are first averaged over the two display orders, then judges are averaged with their weights.
- Judges missing from the weights count 0. Without `--weights` every judge counts equally.
- `load_weights` also reads the judge weights from `peereval score --json` output.
- `new_judgments.jsonl` uses the judgment format below and can come from `peereval judge`, e.g. on pairs of answers from models outside the panel.

## Generating answers

`peereval generate` asks every model every question and stores only the final answer: `<think>` blocks are removed, and answers that hit the token limit, come back empty or are cut off by a content filter are recorded with an `error` and retried on the next run.

```bash
peereval generate questions.jsonl --models models.json --out answers.jsonl
```

`questions.jsonl` rows need `question_id` and one of `question` / `prompt` / `instruction` (`dataset` is optional). `models.json` has the same format as `judges.json` below.

## Generating judgments

`peereval judge` takes solver answers, asks every judge to compare every pair of answers to the same question in both display orders, and writes judgments in the format `peereval score` reads. Judges are called through any OpenAI-compatible endpoint (OpenAI, OpenRouter, vLLM, SGLang, ...).

```bash
peereval judge answers/*.jsonl --judges judges.json --out judgments.jsonl --concurrency 8
peereval score judgments.jsonl
```

The run is resumable: rerunning skips verdicts already in `--out` and retries failed ones. Add `--dry-run` to `generate` or `judge` to see how many calls a run would make (and the prompt size) without calling anything. `--resume-by-cell` makes `judge` treat any valid verdict for the same question, display order and judge in `--out` as done, which lets you continue from judgments produced by another tool.

`judges.json` is a list of judges (see `examples/judges.json`; the model strings there are examples, so replace them with your provider's ids):

```json
[
  {"name": "gpt-6-astra", "model": "openai/gpt-6-astra", "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY"},
  {"name": "qwen3.8-27b", "model": "Qwen/Qwen3.8-27B", "base_url": "http://localhost:8000/v1",
   "extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}
]
```

- `name` must match the `model` field of that model's solver answers, so the judge is linked to its own answers (needed for PRD weights and self-bias).
- `extra_body` is passed to the endpoint as is, e.g. to turn off a local model's thinking.
- `max_tokens` (default 8192) and `temperature` (default 0) can also be set. OpenAI reasoning models need `"max_tokens_param": "max_completion_tokens"` and `"temperature": null`.

From Python:

```python
from peereval import JudgeConfig, run_judging

judges = [JudgeConfig(name="gpt-6-astra", model="openai/gpt-6-astra",
                      base_url="https://openrouter.ai/api/v1", api_key_env="OPENROUTER_API_KEY")]
run_judging("answers.jsonl", judges, "judgments.jsonl")
```

**Answers should contain only the final answer.** If a model returns its reasoning inside the answer, the judges see it. `<think>...</think>` blocks are removed before judging by default (`--keep-think` turns this off), but reasoning written without tags is not.

## Data format

### Solver answers (input to `peereval judge`)

One JSON object per line:

```json
{"dataset": "arena_hard_v2", "question_id": "0f93825f88454fb6", "model": "glm-5.3",
 "question": "…", "response": "…"}
```

Only questions answered by every model are judged.

### Judgments (output of `peereval judge`, input to `peereval score`)

```json
{"dataset": "arena_hard_v2", "question_id": "0f93825f88454fb6",
 "solver_left": "claude-opus-5-5", "solver_right": "glm-5.3",
 "response_a_model": "glm-5.3", "response_b_model": "claude-opus-5-5",
 "judge_model": "gpt-6-astra", "position": "BA",
 "judgment": {"winner": "A", "confidence": 0.8, "reason": "…"}}
```

- `winner` is `"A"`, `"B"` or `"tie"`, and refers to the **displayed** order: `"A"` means the answer by `response_a_model` won.
- `solver_left` / `solver_right` name the pair and stay fixed across the two display orders; `position` is `"AB"` when `solver_left` was shown first.
- If `position` and `response_a_model` disagree, PeerEval trusts `response_a_model` (what the judge actually saw) and prints a warning.
- Rows with an `error` field or without a valid `winner` are skipped. Extra keys are ignored.

## Metrics

| Metric | What it is |
|---|---|
| **PRD score** | Peer Rank. All judges start with equal weight. A model's score is the weighted average of the judges' win rates for it. The scores are scaled to 0–1 and normalized to sum to 1, and become the judges' new weights; this repeats until the weights stop changing. |
| **Judge weight** | The final PRD weight of each model as a judge (sums to 1). Because of the 0–1 scaling, the lowest-scoring model always gets weight 0. |
| **Win rate** | Average result over all of a model's comparisons (win 1, tie 0.5, loss 0), every judge counted equally. |
| **Bradley-Terry** | Log-strength from a Bradley-Terry model fitted to the same verdicts (ties count half). |
| **Self-bias** | How often a model prefers its own answer when it is the judge, minus how often the other judges prefer that model's answer, in percentage points. |
| **Pairwise matrix** | `pairwise[a][b]`: share of comparisons in which `a` beats `b`. |
| **95% CI** | Bootstrap confidence interval: questions are resampled with replacement (within each dataset) and every score is recomputed; the interval covers the middle 95% of the results. Available for PRD score, win rate and Bradley-Terry. |
| **Rank (UB)** | 1 + the number of models whose PRD interval lies entirely above this model's. Models with the same Rank (UB) are not clearly separated. |
| **Position bias** | Per judge: share of decisive verdicts won by the answer shown first, and how often the verdict is unchanged when the two answers are swapped. |


## License

MIT
