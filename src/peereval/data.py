"""Loading and normalizing pairwise judgment records."""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

WINNERS = {"A", "B", "tie"}


@dataclass(frozen=True)
class Judgment:
    """One anonymous pairwise verdict.

    ``model_a`` / ``model_b`` are the models in the order the judge saw them, and
    ``winner`` ("A", "B" or "tie") refers to that displayed order.
    """

    dataset: str
    question_id: str
    judge: str
    model_a: str
    model_b: str
    winner: str

    @property
    def pair(self) -> tuple[str, str]:
        return tuple(sorted((self.model_a, self.model_b)))  # type: ignore[return-value]

    def score(self, model: str) -> float:
        """1 if ``model`` won, 0.5 for a tie, 0 if it lost."""
        if model not in (self.model_a, self.model_b):
            raise ValueError(f"{model} is not in this comparison")
        if self.winner == "tie":
            return 0.5
        won = self.model_a if self.winner == "A" else self.model_b
        return 1.0 if won == model else 0.0


def read_jsonl(paths: str | Path | Iterable[str | Path]) -> list[dict]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    rows: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _displayed_order(row: dict) -> tuple[str, str] | None:
    left, right = row.get("solver_left"), row.get("solver_right")
    a, b = row.get("response_a_model"), row.get("response_b_model")
    if a and b:
        return a, b
    position = row.get("position")
    if left and right and position in ("AB", "BA"):
        return (left, right) if position == "AB" else (right, left)
    return None


def normalize(rows: Iterable[dict]) -> list[Judgment]:
    """Turn raw judgment rows into :class:`Judgment` objects.

    * Rows with an error or without a valid winner are dropped.
    * The displayed order comes from ``response_a_model`` / ``response_b_model``.
      Rows that only carry ``solver_left`` / ``solver_right`` + ``position`` are
      also accepted. When both are present and disagree, the displayed models are
      trusted, because they are what the judge actually saw.
    * Exact duplicates (same question, judge, and displayed order) are kept once.
    """
    out: dict[tuple, Judgment] = {}
    dropped = conflicts = 0
    for row in rows:
        winner = (row.get("judgment") or {}).get("winner", row.get("winner"))
        order = _displayed_order(row)
        if row.get("error") or winner not in WINNERS or order is None or order[0] == order[1]:
            dropped += 1
            continue
        left, right, position = row.get("solver_left"), row.get("solver_right"), row.get("position")
        if left and right and position in ("AB", "BA"):
            expected = (left, right) if position == "AB" else (right, left)
            if expected != order:
                conflicts += 1
        j = Judgment(
            dataset=str(row.get("dataset", "default")),
            question_id=str(row["question_id"]),
            judge=str(row["judge_model"]),
            model_a=order[0],
            model_b=order[1],
            winner=winner,
        )
        out[(j.dataset, j.question_id, j.judge, j.model_a, j.model_b)] = j
    if dropped:
        warnings.warn(f"peereval: skipped {dropped} rows without a valid verdict", stacklevel=2)
    if conflicts:
        warnings.warn(
            f"peereval: {conflicts} rows had a `position` that disagrees with "
            "response_a_model/response_b_model; scored by the displayed models",
            stacklevel=2,
        )
    return list(out.values())


def load_judgments(paths: str | Path | Iterable[str | Path]) -> list[Judgment]:
    """Read one or more judgment JSONL files and normalize them."""
    return normalize(read_jsonl(paths))
