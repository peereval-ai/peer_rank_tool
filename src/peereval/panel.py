"""Judge weights and weighted panel verdicts.

Fit PRD judge weights once, on peer judgments where the judges also answered the
questions, then reuse them to combine the same judges' verdicts on new pairs of
answers into one panel verdict.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from . import metrics
from .data import Judgment, load_judgments, normalize


def _as_judgments(judgments) -> list[Judgment]:
    if isinstance(judgments, list) and judgments and isinstance(judgments[0], dict):
        return normalize(judgments)
    if isinstance(judgments, list) and (not judgments or isinstance(judgments[0], Judgment)):
        return judgments
    return load_judgments(judgments)


def fit_weights(
    judgments: Sequence[Judgment] | str | Path | Iterable[str | Path],
    models: Sequence[str] | None = None,
) -> dict[str, float]:
    """PRD judge weights from peer judgments.

    ``judgments`` must come from a peer evaluation: the judges are also contestants,
    i.e. they answered the questions and judged each other's answers. Weights sum to 1;
    the lowest-scoring judge gets 0. Same weights as ``score(...).overall.judge_weights``.
    """
    js = _as_judgments(judgments)
    models = list(models) if models else metrics.models_in(js)
    keep = set(models)
    js = [j for j in js if j.model_a in keep and j.model_b in keep]
    _, weights, _ = metrics.peer_rank(js, models)
    if not weights:
        raise ValueError("no judge is also a contestant, so PRD weights cannot be computed")
    return weights


def save_weights(weights: dict[str, float], path: str | Path) -> None:
    Path(path).write_text(json.dumps(weights, indent=2) + "\n", encoding="utf-8")


def load_weights(path: str | Path) -> dict[str, float]:
    """Read ``{judge: weight}`` JSON, or the judge weights from ``peereval score --json`` output."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "overall" in data:
        data = data["overall"]["judge_weights"]
    return {str(k): float(v) for k, v in data.items()}


def panel_verdicts(
    judgments: Sequence[Judgment] | str | Path | Iterable[str | Path],
    weights: dict[str, float] | None = None,
) -> list[dict]:
    """Combine several judges' verdicts on each pair of answers into one panel verdict.

    For every (dataset, question, pair), each judge's preference for ``model_a`` is
    averaged over the display orders it saw (win 1, tie 0.5, loss 0). The panel
    preference is the weighted mean over judges; judges missing from ``weights`` get
    weight 0. With ``weights=None`` every judge counts equally.

    Returns one dict per pair with ``model_a``/``model_b`` (sorted), ``preference``
    (panel preference for ``model_a``, or ``None`` if every judge of the pair has
    weight 0), ``winner`` (``model_a``, ``model_b``, ``"tie"`` or ``None``) and the
    per-judge preferences.
    """
    js = _as_judgments(judgments)
    per = defaultdict(lambda: defaultdict(list))
    for j in js:
        a, _ = j.pair
        per[(j.dataset, j.question_id, j.pair)][j.judge].append(j.score(a))
    out = []
    for (dataset, qid, (a, b)), by_judge in per.items():
        prefs = {judge: sum(v) / len(v) for judge, v in sorted(by_judge.items())}
        w = {judge: (1.0 if weights is None else weights.get(judge, 0.0)) for judge in prefs}
        total = sum(w.values())
        pref = sum(w[judge] * p for judge, p in prefs.items()) / total if total > 0 else None
        winner = None if pref is None else a if pref > 0.5 else b if pref < 0.5 else "tie"
        out.append({"dataset": dataset, "question_id": qid, "model_a": a, "model_b": b,
                    "preference": pref, "winner": winner, "judges": prefs})
    return out
