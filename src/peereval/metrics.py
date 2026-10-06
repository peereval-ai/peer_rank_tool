"""Ranking metrics over pairwise judgments: win rate, Bradley-Terry and PRD Peer Rank."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence

from .data import Judgment


def models_in(judgments: Iterable[Judgment]) -> list[str]:
    return sorted({m for j in judgments for m in (j.model_a, j.model_b)})


def win_rates(judgments: Sequence[Judgment], models: Sequence[str]) -> dict[str, float]:
    """Average score (win 1, tie 0.5, loss 0) of each model over all its comparisons."""
    total, count = defaultdict(float), defaultdict(int)
    for j in judgments:
        for m in (j.model_a, j.model_b):
            total[m] += j.score(m)
            count[m] += 1
    return {m: total[m] / count[m] if count[m] else math.nan for m in models}


def judge_win_rates(judgments: Sequence[Judgment], models: Sequence[str]) -> dict[str, dict[str, float]]:
    """``[judge][model]`` = that judge's average score for that model."""
    total = defaultdict(lambda: defaultdict(float))
    count = defaultdict(lambda: defaultdict(int))
    for j in judgments:
        for m in (j.model_a, j.model_b):
            total[j.judge][m] += j.score(m)
            count[j.judge][m] += 1
    return {
        judge: {m: total[judge][m] / count[judge][m] for m in models if count[judge][m]}
        for judge in total
    }


def peer_rank(
    judgments: Sequence[Judgment],
    models: Sequence[str],
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> tuple[dict[str, float], dict[str, float], int]:
    """PRD Peer Rank (Li et al., 2023, arXiv:2307.02762).

    Every judge starts with equal weight. A model's score is the weighted average
    of the judges' win rates for it; the scores are min-max scaled and normalized
    to sum to 1, and become the judges' new weights. This repeats until the weights
    stop changing.

    Only judges that are also contestants take part; verdicts from other judges
    are ignored here. If no judge is a contestant, every score is NaN. Returns
    ``(scores, judge_weights, iterations)``.
    """
    per_judge = judge_win_rates(judgments, models)
    judges = [m for m in models if m in per_judge]
    if not judges:
        return {m: math.nan for m in models}, {}, 0
    weights = {j: 1.0 / len(judges) for j in judges}

    def scores_for(w: dict[str, float]) -> dict[str, float]:
        out = {}
        for m in models:
            num = sum(w[j] * per_judge[j][m] for j in judges if m in per_judge[j])
            den = sum(w[j] for j in judges if m in per_judge[j])
            out[m] = num / den if den else math.nan
        return out

    iteration = 0
    for iteration in range(1, max_iter + 1):
        scores = scores_for(weights)
        lo = min(scores[j] for j in judges)
        hi = max(scores[j] for j in judges)
        if math.isclose(lo, hi):
            new = {j: 1.0 / len(judges) for j in judges}
        else:
            scaled = {j: (scores[j] - lo) / (hi - lo) for j in judges}
            s = sum(scaled.values())
            new = {j: scaled[j] / s for j in judges}
        delta = max(abs(new[j] - weights[j]) for j in judges)
        weights = new
        if delta < tol:
            break
    return scores_for(weights), weights, iteration


def bradley_terry(judgments: Sequence[Judgment], models: Sequence[str], max_iter: int = 2000) -> dict[str, float]:
    """Bradley-Terry log-strengths (MM algorithm; a tie counts as half a win for each side)."""
    wins = defaultdict(float)
    games = defaultdict(int)
    for j in judgments:
        wins[j.model_a] += j.score(j.model_a)
        wins[j.model_b] += j.score(j.model_b)
        games[j.pair] += 1
    strength = {m: 1.0 for m in models}
    for _ in range(max_iter):
        new = {}
        for i in models:
            denom = sum(
                games[tuple(sorted((i, k)))] / (strength[i] + strength[k]) for k in models if k != i
            )
            new[i] = wins[i] / denom if denom else strength[i]
        mean = sum(new.values()) / len(new)
        new = {m: max(v / mean, 1e-12) for m, v in new.items()}
        done = max(abs(math.log(new[m] / strength[m])) for m in models) < 1e-9
        strength = new
        if done:
            break
    return {m: math.log(strength[m]) for m in models}


def self_bias(judgments: Sequence[Judgment], models: Sequence[str]) -> dict[str, dict[str, float]]:
    """How much more a model favors its own answers as a judge than the other judges do.

    ``bias_pp`` = 100 * (P(model prefers itself) - P(other judges prefer that model)).
    """
    own, other = defaultdict(list), defaultdict(list)
    for j in judgments:
        for m in (j.model_a, j.model_b):
            (own if j.judge == m else other)[m].append(j.score(m))
    out = {}
    for m in models:
        o = sum(own[m]) / len(own[m]) if own[m] else math.nan
        t = sum(other[m]) / len(other[m]) if other[m] else math.nan
        out[m] = {"self": o, "others": t, "bias_pp": 100 * (o - t)}
    return out


def pairwise_matrix(judgments: Sequence[Judgment], models: Sequence[str]) -> dict[str, dict[str, float]]:
    """``[row][col]`` = share of comparisons in which ``row`` beats ``col`` (ties count half)."""
    total, count = defaultdict(float), defaultdict(int)
    for j in judgments:
        for m, o in ((j.model_a, j.model_b), (j.model_b, j.model_a)):
            total[(m, o)] += j.score(m)
            count[(m, o)] += 1
    return {
        r: {c: (total[(r, c)] / count[(r, c)] if count[(r, c)] else math.nan) for c in models if c != r}
        for r in models
    }


def position_bias(judgments: Sequence[Judgment]) -> dict[str, dict[str, float]]:
    """Per judge: share of decisive verdicts won by the first answer shown, and how often the
    verdict stays the same when the two answers are swapped."""
    first, decisive = defaultdict(int), defaultdict(int)
    verdicts = defaultdict(dict)
    for j in judgments:
        if j.winner != "tie":
            decisive[j.judge] += 1
            first[j.judge] += j.winner == "A"
        winner = "tie" if j.winner == "tie" else (j.model_a if j.winner == "A" else j.model_b)
        verdicts[(j.dataset, j.question_id, j.pair, j.judge)][j.model_a] = winner
    same, both = defaultdict(int), defaultdict(int)
    for (_, _, _, judge), v in verdicts.items():
        if len(v) == 2:
            both[judge] += 1
            same[judge] += len(set(v.values())) == 1
    judges = sorted(set(decisive) | set(both))
    return {
        judge: {
            "first_answer_wins": first[judge] / decisive[judge] if decisive[judge] else math.nan,
            "order_consistency": same[judge] / both[judge] if both[judge] else math.nan,
        }
        for judge in judges
    }
