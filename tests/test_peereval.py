import json
import math
import warnings

import pytest

from peereval import normalize, parse_verdict, score, strip_reasoning
from peereval.judge import build_tasks


def row(q, left, right, judge, position, winner, **extra):
    a, b = (left, right) if position == "AB" else (right, left)
    return {"dataset": "d", "question_id": q, "solver_left": left, "solver_right": right,
            "response_a_model": a, "response_b_model": b, "judge_model": judge,
            "position": position, "judgment": {"winner": winner}, **extra}


def full_round(q, winner_of):
    """Every judge in {x, y, z} judges every pair in both orders; `winner_of(pair)` wins."""
    rows = []
    for left, right in [("x", "y"), ("x", "z"), ("y", "z")]:
        w = winner_of(left, right)
        for judge in "xyz":
            for pos in ("AB", "BA"):
                a = left if pos == "AB" else right
                rows.append(row(q, left, right, judge, pos, "A" if a == w else "B"))
    return rows


def test_winner_uses_displayed_order():
    j_ab, j_ba = normalize([row("1", "x", "y", "z", "AB", "A"), row("1", "x", "y", "z", "BA", "A")])
    assert j_ab.score("x") == 1.0 and j_ab.score("y") == 0.0
    assert j_ba.score("y") == 1.0 and j_ba.score("x") == 0.0


def test_conflicting_position_trusts_displayed_models():
    r = row("1", "x", "y", "z", "AB", "A")
    r["response_a_model"], r["response_b_model"] = "y", "x"  # shown as BA, labelled AB
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        (j,) = normalize([r])
    assert j.score("y") == 1.0
    assert any("disagrees" in str(x.message) for x in w)


def test_tie_and_invalid_rows():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        js = normalize([row("1", "x", "y", "z", "AB", "tie"), row("2", "x", "y", "z", "AB", None),
                        row("3", "x", "y", "z", "AB", "A", error="timeout")])
    assert len(js) == 1 and js[0].score("x") == 0.5


def test_clear_order_is_recovered():
    order = ["x", "y", "z"]
    rows = full_round("1", lambda a, b: min(a, b, key=order.index))
    board = score(rows)
    r = board.overall
    assert r.ranking() == order
    assert r.win_rate == {"x": 1.0, "y": 0.5, "z": 0.0}
    assert math.isclose(sum(r.judge_weights.values()), 1.0)
    assert r.judge_weights["z"] == 0.0  # min-max scaling: lowest scorer gets weight 0
    assert r.bradley_terry["x"] > r.bradley_terry["y"] > r.bradley_terry["z"]
    assert r.pairwise["x"]["z"] == 1.0


def test_self_bias():
    # judge x always picks x; other judges always pick the other model
    rows = []
    for left, right in [("x", "y"), ("x", "z"), ("y", "z")]:
        for judge in "xyz":
            for pos in ("AB", "BA"):
                a, b = (left, right) if pos == "AB" else (right, left)
                if judge == "x" and "x" in (a, b):
                    w = "A" if a == "x" else "B"
                else:
                    w = "B" if a == "x" else "A"
                rows.append(row("1", left, right, judge, pos, w))
    sb = score(rows).overall.self_bias["x"]
    assert sb["self"] == 1.0 and sb["others"] == 0.0 and sb["bias_pp"] == 100.0


def test_complete_only_drops_partial_questions():
    order = ["x", "y", "z"]
    rows = full_round("1", lambda a, b: min(a, b, key=order.index)) + full_round("2", lambda a, b: a)[:-1]
    assert score(rows, complete_only=True).overall.questions == 1
    assert score(rows).overall.questions == 2


def test_parse_verdict_and_strip():
    assert parse_verdict('ok {"winner":"B","confidence":0.7,"reason":"r"}')["winner"] == "B"
    assert parse_verdict('{"winner": "tie", "reason": "bad \\q escape"}')["winner"] == "tie"
    with pytest.raises(ValueError):
        parse_verdict("no json here")
    assert strip_reasoning("<think>plan</think>\nAnswer") == "Answer"
    assert strip_reasoning("unclosed plan</think> Answer") == "Answer"


def test_build_tasks_covers_pairs_judges_orders():
    answers = [{"dataset": "d", "question_id": "1", "model": m, "question": "q", "response": m} for m in "xyz"]
    answers.append({"dataset": "d", "question_id": "2", "model": "x", "question": "q", "response": "x"})
    tasks = build_tasks(answers, ["x", "y"])
    assert len(tasks) == 3 * 2 * 2  # 3 pairs x 2 judges x 2 orders; question 2 is incomplete


def test_external_judge_without_prd():
    rows = [row("1", "x", "y", "ext", pos, "A" if (pos == "AB") else "B") for pos in ("AB", "BA")]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = score(rows).overall
    assert r.win_rate == {"x": 1.0, "y": 0.0}
    assert r.ranking() == ["x", "y"] and r.judge_weights == {}
    assert all(math.isnan(v) for v in r.prd.values())
    assert any("no judge is also a contestant" in str(x.message) for x in w)


def test_external_judge_counts_but_gets_no_weight():
    order = ["x", "y", "z"]
    rows = full_round("1", lambda a, b: min(a, b, key=order.index))
    rows += [row("1", "y", "z", "ext", pos, "B" if pos == "AB" else "A") for pos in ("AB", "BA")]  # ext prefers z
    r = score(rows).overall
    assert "ext" not in r.judge_weights and set(r.judge_weights) == {"x", "y", "z"}
    assert r.win_rate["z"] > 0  # ext's votes count in win rate


def test_bootstrap_intervals_and_rank_ub():
    order = ["x", "y", "z"]
    rows = []
    for q in range(20):
        rows += full_round(str(q), lambda a, b: min(a, b, key=order.index))
    r = score(rows, bootstrap=50, seed=1).overall
    for m in order:
        lo, hi = r.ci["prd"][m]
        assert lo <= r.prd[m] <= hi
    assert r.rank_ub == {"x": 1, "y": 2, "z": 3}  # identical questions: no uncertainty
    assert "95% CI" in r.to_markdown()


def test_generate_marks_incomplete_and_resumes(tmp_path, monkeypatch):
    from peereval import ModelConfig, run_generation

    replies = iter([
        {"text": "<think>plan</think>Answer one", "finish_reason": "stop", "usage": None},
        {"text": "cut off mid", "finish_reason": "length", "usage": None},
        {"text": "Answer two", "finish_reason": "stop", "usage": None},
    ])
    monkeypatch.setattr(ModelConfig, "make_chat", lambda self: (lambda prompt: next(replies)))
    qs = [{"dataset": "d", "question_id": "1", "question": "q1"}, {"dataset": "d", "question_id": "2", "question": "q2"}]
    out = tmp_path / "a.jsonl"
    m = [ModelConfig(name="x", model="x")]
    assert run_generation(qs, m, out, concurrency=1) == 2
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    good = [r for r in rows if not r["error"]]
    assert len(good) == 1 and good[0]["response"] == "Answer one"
    assert run_generation(qs, m, out, concurrency=1) == 1  # only the incomplete one is retried


def test_fit_weights_matches_score_and_sums_to_one():
    rows = full_round("q1", lambda l, r: "x" if "x" in (l, r) else "y")
    from peereval import fit_weights
    w = fit_weights(rows)
    assert w == pytest.approx(score(rows).overall.judge_weights)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["z"] == 0.0 and w["x"] > w["y"] > 0


def test_fit_weights_needs_contestant_judges():
    from peereval import fit_weights
    rows = [row("q1", "x", "y", "outsider", "AB", "A")]
    with pytest.raises(ValueError):
        fit_weights(rows)


def test_panel_verdicts_weighted_and_unweighted():
    from peereval import panel_verdicts
    # judge j1 prefers x in both orders; j2 prefers y in both orders; j3 flips with the order
    rows = []
    for pos in ("AB", "BA"):
        a = "x" if pos == "AB" else "y"
        rows.append(row("q9", "x", "y", "j1", pos, "A" if a == "x" else "B"))
        rows.append(row("q9", "x", "y", "j2", pos, "A" if a == "y" else "B"))
        rows.append(row("q9", "x", "y", "j3", pos, "A"))
    [v] = panel_verdicts(rows)
    assert (v["model_a"], v["model_b"]) == ("x", "y")
    assert v["judges"] == {"j1": 1.0, "j2": 0.0, "j3": 0.5}
    assert v["preference"] == pytest.approx(0.5) and v["winner"] == "tie"
    [v] = panel_verdicts(rows, {"j1": 0.6, "j2": 0.2, "j3": 0.2})
    assert v["preference"] == pytest.approx(0.6 * 1 + 0.2 * 0 + 0.2 * 0.5)
    assert v["winner"] == "x"
    [v] = panel_verdicts(rows, {"j2": 1.0})  # judges missing from the weights get 0
    assert v["preference"] == 0.0 and v["winner"] == "y"
    [v] = panel_verdicts(rows, {"other": 1.0})
    assert v["preference"] is None and v["winner"] is None


def test_weights_roundtrip_and_cli(tmp_path):
    from peereval import load_weights, save_weights
    from peereval.cli import main
    peer = full_round("q1", lambda l, r: "x" if "x" in (l, r) else "y")
    peer_file = tmp_path / "peer.jsonl"
    peer_file.write_text("\n".join(json.dumps(r) for r in peer))
    # `peereval weights` writes the same weights `score --json` reports
    assert main(["weights", str(peer_file), "--out", str(tmp_path / "w.json")]) == 0
    w = load_weights(tmp_path / "w.json")
    score(peer).to_json(tmp_path / "board.json")
    assert load_weights(tmp_path / "board.json") == pytest.approx(w)
    save_weights(w, tmp_path / "w2.json")
    assert load_weights(tmp_path / "w2.json") == pytest.approx(w)
    # `peereval panel` applies them to new pairs
    new = [row("n1", "a", "b", "x", "AB", "A"), row("n1", "a", "b", "x", "BA", "B"),
           row("n1", "a", "b", "z", "AB", "B"), row("n1", "a", "b", "z", "BA", "A")]
    new_file = tmp_path / "new.jsonl"
    new_file.write_text("\n".join(json.dumps(r) for r in new))
    out = tmp_path / "panel.jsonl"
    assert main(["panel", str(new_file), "--weights", str(tmp_path / "w.json"), "--out", str(out)]) == 0
    [v] = [json.loads(l) for l in out.read_text().splitlines()]
    assert v["judges"] == {"x": 1.0, "z": 0.0}
    assert v["winner"] == "a"  # z has weight 0, so x decides
