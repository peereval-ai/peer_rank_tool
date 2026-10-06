"""Generate anonymous pairwise judgments from solver answers.

Judges are called through any OpenAI-compatible chat endpoint (OpenAI, OpenRouter,
vLLM, SGLang, ...). Every pair of answers to a question is judged by every judge in
both display orders. Output rows use the same schema that :func:`peereval.score` reads.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .data import read_jsonl

JUDGE_PROMPT = """You are an impartial evaluator. The two responses are anonymized and their authors are unknown.
Question:
{question}

Response A:
{response_a}

Response B:
{response_b}

Which response is better for this task? Judge correctness, relevance, completeness, clarity, and the task-specific requirements. Do not infer authorship from style. Return JSON only:
{{"winner":"A"|"B"|"tie","confidence":0.0,"reason":"brief explanation"}}"""

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Remove ``<think>...</think>`` blocks, and anything before a lone ``</think>``."""
    text = _THINK.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def parse_verdict(text: str) -> dict[str, Any]:
    """Extract ``{"winner", "confidence", "reason"}`` from a judge's reply."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if obj.get("winner") in ("A", "B", "tie"):
                return {"winner": obj["winner"], "confidence": float(obj.get("confidence", 0.0)),
                        "reason": str(obj.get("reason", ""))}
        except (ValueError, TypeError):
            pass
    m = re.search(r'"winner"\s*:\s*"(A|B|tie)"', text, re.IGNORECASE)
    if not m:
        raise ValueError("judge reply has no valid winner")
    return {"winner": m.group(1), "confidence": 0.0, "reason": "recovered from malformed JSON"}


@dataclass
class JudgeConfig:
    """One judge model behind an OpenAI-compatible endpoint."""

    name: str                       # id used in the output (should match the solver's model name)
    model: str                      # model string sent to the endpoint
    base_url: str | None = None     # e.g. https://openrouter.ai/api/v1 or http://localhost:8000/v1
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 8192
    max_tokens_param: str = "max_tokens"  # OpenAI reasoning models need "max_completion_tokens"
    temperature: float | None = 0.0       # None = don't send (some reasoning models reject it)
    extra_body: dict[str, Any] = field(default_factory=dict)  # e.g. {"chat_template_kwargs": {"enable_thinking": False}}

    def make_call(self) -> Callable[[str], str]:
        from openai import OpenAI  # optional dependency: pip install "peereval[judge]"

        client = OpenAI(base_url=self.base_url, api_key=os.environ.get(self.api_key_env, "EMPTY"))

        def call(prompt: str) -> str:
            kwargs: dict[str, Any] = {self.max_tokens_param: self.max_tokens}
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                extra_body=self.extra_body or None,
                **kwargs,
            )
            return resp.choices[0].message.content or ""

        return call


def load_judge_configs(path: str | Path) -> list[JudgeConfig]:
    return [JudgeConfig(**c) for c in json.loads(Path(path).read_text())]


def _key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:24]


def build_tasks(answers: Sequence[dict], judges: Sequence[str]) -> list[dict]:
    """All (question, pair, judge, display order) cells for questions every model answered."""
    by_q: dict[tuple[str, str], dict[str, dict]] = {}
    for a in answers:
        by_q.setdefault((str(a.get("dataset", "default")), str(a["question_id"])), {})[a["model"]] = a
    models = sorted({a["model"] for a in answers})
    tasks = []
    for (ds, qid), per_model in sorted(by_q.items()):
        if set(per_model) != set(models):
            continue
        question = per_model[models[0]]["question"]
        for left, right in combinations(models, 2):
            for judge in judges:
                for position, (a, b) in (("AB", (left, right)), ("BA", (right, left))):
                    tasks.append({
                        "dataset": ds, "question_id": qid, "question": question,
                        "solver_left": left, "solver_right": right,
                        "response_a_model": a, "response_b_model": b,
                        "judge_model": judge, "position": position,
                        "_a": per_model[a]["response"], "_b": per_model[b]["response"],
                    })
    return tasks


def run_judging(
    answers: Sequence[dict] | str | Path | Iterable[str | Path],
    judges: Sequence[JudgeConfig],
    output: str | Path,
    concurrency: int = 4,
    strip_think: bool = True,
    prompt_template: str = JUDGE_PROMPT,
) -> int:
    """Judge every pair of answers with every judge and append verdicts to ``output``.

    Already-valid verdicts in ``output`` are skipped, so an interrupted run can be
    resumed by calling this again. Failed calls are written with an ``error`` field
    and retried on the next run. Returns the number of new rows written.
    """
    if not (isinstance(answers, list) and answers and isinstance(answers[0], dict)):
        answers = read_jsonl(answers)  # type: ignore[arg-type]
    output = Path(output)
    done = set()
    if output.exists():
        for r in read_jsonl(output):
            if not r.get("error") and (r.get("judgment") or {}).get("winner") in ("A", "B", "tie"):
                done.add(r.get("cache_key"))
    calls = {j.name: j.make_call() for j in judges}
    tasks = build_tasks(answers, [j.name for j in judges])
    if not tasks:
        warnings.warn("peereval: no question was answered by every model, so there is nothing to judge", stacklevel=2)
    pending = []
    for t in tasks:
        ra, rb = (strip_reasoning(t["_a"]), strip_reasoning(t["_b"])) if strip_think else (t["_a"], t["_b"])
        prompt = prompt_template.format(question=t["question"], response_a=ra, response_b=rb)
        t["cache_key"] = _key(t["dataset"], t["question_id"], t["response_a_model"], t["response_b_model"],
                              t["judge_model"], hashlib.sha256(prompt.encode()).hexdigest())
        if t["cache_key"] not in done:
            t["_prompt"] = prompt
            pending.append(t)

    lock = threading.Lock()

    def work(t: dict) -> dict:
        row = {k: v for k, v in t.items() if not k.startswith("_")}
        try:
            raw = calls[t["judge_model"]](t["_prompt"])
            row.update(judgment=parse_verdict(raw), raw_response=raw, error=None)
        except Exception as exc:  # recorded and retried on the next run
            row.update(judgment={"winner": None}, raw_response="", error=f"{type(exc).__name__}: {exc}")
        row["timestamp"] = datetime.now(timezone.utc).isoformat()
        return row

    written = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=concurrency) as pool, open(output, "a", encoding="utf-8") as f:
        for fut in as_completed([pool.submit(work, t) for t in pending]):
            row = fut.result()
            with lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
    return written
