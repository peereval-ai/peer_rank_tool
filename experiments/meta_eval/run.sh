#!/usr/bin/env bash
# Meta-evaluation: PRD-weighted judge panel vs single judges, against human labels.
# Usage: JUDGES=judges.json CONCURRENCY=8 bash run.sh
set -euo pipefail
cd "$(dirname "$0")"
JUDGES=${JUDGES:-judges.json}
CONC=${CONCURRENCY:-8}
mkdir -p data runs results

# 0. Sample human-labeled battles and the queries used for the weights
[ -f data/arena140k_labels.jsonl ] || python prepare_data.py --out data

# 1. PRD weights on the same datasets: our judges answer the queries and judge each other
peereval generate data/arena140k_queries.jsonl data/expert5k_queries.jsonl \
  --models "$JUDGES" --out runs/answers.jsonl --concurrency "$CONC"
peereval judge runs/answers.jsonl --judges "$JUDGES" --out runs/peer_judgments.jsonl --concurrency "$CONC"
peereval score runs/peer_judgments.jsonl --json runs/weights.json > runs/peer_leaderboard.md

# 2. The same judges judge the human-labeled battles
peereval judge data/arena140k_battles.jsonl data/expert5k_battles.jsonl \
  --judges "$JUDGES" --out runs/battle_judgments.jsonl --concurrency "$CONC"

# 3. Compare single judges, the unweighted panel and the PRD-weighted panel with humans
python analyze.py --labels data/arena140k_labels.jsonl data/expert5k_labels.jsonl \
  --judgments runs/battle_judgments.jsonl --weights runs/weights.json \
  --human-ranking data/arena140k_human_ranking.json --out results
