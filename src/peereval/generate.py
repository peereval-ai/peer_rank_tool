"""Generate solver answers through OpenAI-compatible endpoints.

Only the final answer is stored: ``<think>`` blocks are removed, and answers that
hit the token limit, come back empty, or are cut off by a content filter are
recorded with an ``error`` and retried on the next run.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from .data import read_jsonl
from .judge import ModelConfig, strip_reasoning

INCOMPLETE = {"length", "max_tokens", "content_filter"}


def _question_text(row: dict) -> str:
    for key in ("question", "prompt", "instruction"):
        if isinstance(row.get(key), str) and row[key].strip():
            return row[key]
    raise KeyError(f"question {row.get('question_id')} has no question/prompt/instruction field")


def run_generation(
    questions: Sequence[dict] | str | Path | Iterable[str | Path],
    models: Sequence[ModelConfig],
    output: str | Path,
    concurrency: int = 4,
    strip_think: bool = True,
    dry_run: bool = False,
) -> int | dict:
    """Ask every model every question and append answers to ``output``.

    ``questions`` rows need ``question_id`` and one of ``question`` / ``prompt`` /
    ``instruction`` (``dataset`` is optional). Already-valid answers in ``output``
    are skipped, so the run is resumable. Returns the number of rows written, or with
    ``dry_run=True`` a dict with the number of pending calls (nothing is called).
    """
    if not (isinstance(questions, list) and questions and isinstance(questions[0], dict)):
        questions = read_jsonl(questions)  # type: ignore[arg-type]
    output = Path(output)
    done = set()
    if output.exists():
        for r in read_jsonl(output):
            if not r.get("error") and str(r.get("response", "")).strip():
                done.add((str(r.get("dataset", "default")), str(r["question_id"]), r["model"]))
    pending = [
        (q, m.name)
        for q in questions
        for m in models
        if (str(q.get("dataset", "default")), str(q["question_id"]), m.name) not in done
    ]
    if dry_run:
        return {"pending_calls": len(pending), "prompt_chars": sum(len(_question_text(q)) for q, _ in pending)}
    calls = {m.name: m.make_chat() for m in models}

    def work(item: tuple[dict, str]) -> dict:
        q, name = item
        row = {"dataset": str(q.get("dataset", "default")), "question_id": str(q["question_id"]),
               "model": name, "question": _question_text(q)}
        try:
            out = calls[name](row["question"])
            text = strip_reasoning(out["text"]) if strip_think else out["text"].strip()
            error = None
            if out["finish_reason"] in INCOMPLETE:
                error = f"incomplete answer (finish_reason={out['finish_reason']})"
            elif not text:
                error = "empty answer"
            row.update(response=text, finish_reason=out["finish_reason"], usage=out["usage"], error=error)
        except Exception as exc:  # recorded and retried on the next run
            row.update(response="", finish_reason="error", usage=None, error=f"{type(exc).__name__}: {exc}")
        row["timestamp"] = datetime.now(timezone.utc).isoformat()
        return row

    written = 0
    lock = threading.Lock()
    output.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=concurrency) as pool, open(output, "a", encoding="utf-8") as f:
        for fut in as_completed([pool.submit(work, it) for it in pending]):
            row = fut.result()
            with lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
    return written


def valid_answers(rows: Iterable[dict]) -> list[dict]:
    """Latest valid answer per (dataset, question, model)."""
    keep: dict[tuple, dict] = {}
    for r in rows:
        if not r.get("error") and str(r.get("response", "")).strip():
            keep[(str(r.get("dataset", "default")), str(r["question_id"]), r["model"])] = r
    return list(keep.values())
