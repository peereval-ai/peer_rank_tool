"""Build a leaderboard from pairwise judgments."""
from __future__ import annotations

import csv
import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from . import metrics
from .data import Judgment, load_judgments, normalize


@dataclass
class Result:
    """Scores for one slice of the data (one dataset, or everything)."""

    name: str
    models: list[str]
    questions: int
    judgments: int
    prd: dict[str, float]
    judge_weights: dict[str, float]
    win_rate: dict[str, float]
    bradley_terry: dict[str, float]
    self_bias: dict[str, dict[str, float]]
    pairwise: dict[str, dict[str, float]]
    position_bias: dict[str, dict[str, float]]
    prd_iterations: int = 0

    def ranking(self) -> list[str]:
        """Models best first: by PRD score, or by win rate when PRD is unavailable."""
        key = self.win_rate if any(math.isnan(v) for v in self.prd.values()) else self.prd
        return sorted(self.models, key=lambda m: key[m], reverse=True)

    def rows(self) -> list[dict]:
        return [
            {
                "rank": i,
                "model": m,
                "prd_score": self.prd[m],
                "judge_weight": self.judge_weights.get(m, float("nan")),
                "win_rate": self.win_rate[m],
                "bradley_terry": self.bradley_terry[m],
                "self_bias_pp": self.self_bias[m]["bias_pp"],
            }
            for i, m in enumerate(self.ranking(), 1)
        ]

    def to_markdown(self) -> str:
        head = "| Rank | Model | PRD score | Judge weight | Win rate | Bradley-Terry | Self-bias (pp) |"
        sep = "|---:|---|---:|---:|---:|---:|---:|"
        lines = [head, sep]
        def f(v: float, spec: str) -> str:
            return "–" if v is None or math.isnan(v) else format(v, spec)

        for r in self.rows():
            lines.append(
                f"| {r['rank']} | {r['model']} | {f(r['prd_score'], '.3f')} | {f(r['judge_weight'], '.3f')} | "
                f"{f(r['win_rate'], '.3f')} | {f(r['bradley_terry'], '+.3f')} | {f(r['self_bias_pp'], '+.1f')} |"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()} | {"ranking": self.ranking()}


@dataclass
class Leaderboard:
    overall: Result
    by_dataset: dict[str, Result] = field(default_factory=dict)

    def to_markdown(self) -> str:
        parts = [f"## Overall ({self.overall.questions} questions, {self.overall.judgments} judgments)\n",
                 self.overall.to_markdown()]
        for name, r in self.by_dataset.items():
            parts += [f"\n## {name} ({r.questions} questions, {r.judgments} judgments)\n", r.to_markdown()]
        return "\n".join(parts)

    def to_json(self, path: str | Path) -> None:
        data = {"overall": self.overall.to_dict(), "by_dataset": {k: v.to_dict() for k, v in self.by_dataset.items()}}
        Path(path).write_text(json.dumps(data, indent=2))

    def to_csv(self, path: str | Path) -> None:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["slice", "rank", "model", "prd_score", "judge_weight",
                                              "win_rate", "bradley_terry", "self_bias_pp"])
            w.writeheader()
            for name, r in [("overall", self.overall), *self.by_dataset.items()]:
                for row in r.rows():
                    w.writerow({"slice": name, **row})


def complete_questions(judgments: Sequence[Judgment], models: Sequence[str], judges: Sequence[str]) -> set[tuple[str, str]]:
    """Questions that have every (pair, judge, display order) verdict."""
    need = len(models) * (len(models) - 1) * len(judges)
    seen: dict[tuple[str, str], set] = {}
    for j in judgments:
        seen.setdefault((j.dataset, j.question_id), set()).add((j.model_a, j.model_b, j.judge))
    return {q for q, cells in seen.items() if len(cells) == need}


def score_slice(name: str, judgments: Sequence[Judgment], models: Sequence[str]) -> Result:
    keep = set(models)
    prd, weights, iters = metrics.peer_rank([j for j in judgments if j.judge in keep], models)
    return Result(
        name=name,
        models=list(models),
        questions=len({(j.dataset, j.question_id) for j in judgments}),
        judgments=len(judgments),
        prd=prd,
        judge_weights=weights,
        win_rate=metrics.win_rates(judgments, models),
        bradley_terry=metrics.bradley_terry(judgments, models),
        self_bias=metrics.self_bias(judgments, models),
        pairwise=metrics.pairwise_matrix(judgments, models),
        position_bias=metrics.position_bias(judgments),
        prd_iterations=iters,
    )


def score(
    judgments: Sequence[Judgment] | str | Path | Iterable[str | Path],
    models: Sequence[str] | None = None,
    by_dataset: bool = True,
    complete_only: bool = False,
) -> Leaderboard:
    """Score pairwise judgments and return a :class:`Leaderboard`.

    Args:
        judgments: raw judgment dicts, :class:`Judgment` objects, or path(s) to judgment JSONL files.
        models: contestants to keep (default: every model that appears in a pair).
            Comparisons involving other models are dropped. Judges that are not
            contestants still count for win rate, Bradley-Terry and the pairwise
            matrix, but get no PRD weight (PRD only uses contestant judges).
        by_dataset: also score each dataset separately.
        complete_only: keep only questions that have every verdict
            (all pairs x all judges x both display orders).
    """
    if isinstance(judgments, list) and judgments and isinstance(judgments[0], dict):
        judgments = normalize(judgments)
    elif not (isinstance(judgments, list) and (not judgments or isinstance(judgments[0], Judgment))):
        judgments = load_judgments(judgments)  # type: ignore[arg-type]
    models = list(models) if models else metrics.models_in(judgments)
    keep = set(models)
    js = [j for j in judgments if j.model_a in keep and j.model_b in keep]
    if not any(j.judge in keep for j in js):
        warnings.warn("peereval: no judge is also a contestant, so PRD scores and judge weights are "
                      "unavailable; models are ranked by win rate", stacklevel=2)
    if complete_only:
        ok = complete_questions(js, models, sorted({j.judge for j in js}))
        js = [j for j in js if (j.dataset, j.question_id) in ok]
    if not js:
        raise ValueError("no judgments left to score")
    board = Leaderboard(overall=score_slice("overall", js, models))
    if by_dataset:
        for ds in sorted({j.dataset for j in js}):
            board.by_dataset[ds] = score_slice(ds, [j for j in js if j.dataset == ds], models)
    return board
