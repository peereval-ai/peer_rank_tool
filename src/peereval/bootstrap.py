"""Bootstrap confidence intervals, resampling questions with replacement."""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Sequence

from . import metrics
from .data import Judgment


def _question_stats(judgments: Sequence[Judgment], contestant_judges: set[str]):
    """Per-question sums so each bootstrap sample is a cheap weighted total."""
    stats = defaultdict(lambda: {"judge": defaultdict(lambda: [0.0, 0]), "model": defaultdict(lambda: [0.0, 0]),
                                 "wins": defaultdict(float), "games": defaultdict(int)})
    for j in judgments:
        q = stats[(j.dataset, j.question_id)]
        for m in (j.model_a, j.model_b):
            s = j.score(m)
            q["model"][m][0] += s
            q["model"][m][1] += 1
            q["wins"][m] += s
            if j.judge in contestant_judges:
                cell = q["judge"][(j.judge, m)]
                cell[0] += s
                cell[1] += 1
        q["games"][j.pair] += 1
    return stats


def bootstrap(
    judgments: Sequence[Judgment],
    models: Sequence[str],
    n: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict:
    """95% (by default) percentile intervals for PRD score, win rate and Bradley-Terry.

    Questions are resampled with replacement within each dataset, so every sample
    keeps the dataset mix. Also returns ``rank_ub``: 1 + the number of models whose
    PRD interval lies entirely above this model's (the "Rank (UB)" used by Arena).
    """
    keep = set(models)
    stats = _question_stats(judgments, keep)
    by_ds = defaultdict(list)
    for q in stats:
        by_ds[q[0]].append(q)
    rng = random.Random(seed)
    samples = {"prd": defaultdict(list), "win_rate": defaultdict(list), "bradley_terry": defaultdict(list)}
    for _ in range(n):
        counts = Counter()
        for qs in by_ds.values():
            counts.update(rng.choices(qs, k=len(qs)))
        judge_tot = defaultdict(lambda: [0.0, 0])
        model_tot = defaultdict(lambda: [0.0, 0])
        wins, games = defaultdict(float), defaultdict(int)
        for q, c in counts.items():
            st = stats[q]
            for k, (s, cnt) in st["judge"].items():
                judge_tot[k][0] += c * s
                judge_tot[k][1] += c * cnt
            for m, (s, cnt) in st["model"].items():
                model_tot[m][0] += c * s
                model_tot[m][1] += c * cnt
            for m, w in st["wins"].items():
                wins[m] += c * w
            for p, g in st["games"].items():
                games[p] += c * g
        per_judge = defaultdict(dict)
        for (judge, m), (s, cnt) in judge_tot.items():
            if cnt:
                per_judge[judge][m] = s / cnt
        prd, _, _ = metrics.peer_rank_from(per_judge, models)
        bt = metrics.bradley_terry_from(wins, games, models)
        for m in models:
            samples["prd"][m].append(prd[m])
            s, cnt = model_tot[m]
            samples["win_rate"][m].append(s / cnt if cnt else float("nan"))
            samples["bradley_terry"][m].append(bt[m])

    def interval(values: list[float]) -> list[float]:
        v = sorted(x for x in values if x == x)
        if not v:
            return [float("nan"), float("nan")]
        lo_i = int(round((1 - level) / 2 * (len(v) - 1)))
        hi_i = int(round((1 + level) / 2 * (len(v) - 1)))
        return [v[lo_i], v[hi_i]]

    ci = {metric: {m: interval(vals[m]) for m in models} for metric, vals in samples.items()}
    prd_ci = ci["prd"]
    rank_ub = {
        m: 1 + sum(1 for k in models if k != m and prd_ci[k][0] > prd_ci[m][1]) for m in models
    }
    return {"ci": ci, "rank_ub": rank_ub, "n": n, "seed": seed, "level": level}
