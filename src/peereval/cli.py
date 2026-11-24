"""Command line: ``peereval score``, ``generate``, ``judge``, ``weights`` and ``panel``."""
from __future__ import annotations

import argparse
import json
import sys

from .generate import run_generation
from .judge import load_judge_configs, run_judging
from .leaderboard import score
from .panel import fit_weights, load_weights, panel_verdicts, save_weights


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="peereval", description="Peer evaluation of LLMs with PRD Peer Rank.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="Rank models from pairwise judgment files.")
    s.add_argument("judgments", nargs="+", help="Judgment JSONL file(s).")
    s.add_argument("--models", nargs="+", help="Contestants to keep (default: all).")
    s.add_argument("--no-by-dataset", action="store_true", help="Only score all data together.")
    s.add_argument("--complete-only", action="store_true",
                   help="Keep only questions that have every pair x judge x order verdict.")
    s.add_argument("--bootstrap", type=int, default=0, metavar="N",
                   help="Bootstrap samples for 95%% confidence intervals and Rank (UB), e.g. 1000.")
    s.add_argument("--seed", type=int, default=0, help="Random seed for the bootstrap.")
    s.add_argument("--json", help="Write full results (scores, weights, matrices) to this JSON file.")
    s.add_argument("--csv", help="Write the leaderboard tables to this CSV file.")

    g = sub.add_parser("generate", help="Generate solver answers to questions.")
    g.add_argument("questions", nargs="+", help="Question JSONL file(s): question_id + question/prompt.")
    g.add_argument("--models", required=True, help="JSON list of model configs (same format as --judges).")
    g.add_argument("--out", required=True, help="Answer JSONL to write (appended; resumable).")
    g.add_argument("--concurrency", type=int, default=4)
    g.add_argument("--keep-think", action="store_true", help="Do not strip <think> blocks from answers.")
    g.add_argument("--dry-run", action="store_true", help="Only count the calls that would be made.")

    j = sub.add_parser("judge", help="Generate pairwise judgments from solver answers.")
    j.add_argument("answers", nargs="+", help="Solver answer JSONL file(s).")
    j.add_argument("--judges", required=True, help="JSON list of judge configs.")
    j.add_argument("--out", required=True, help="Judgment JSONL to write (appended; resumable).")
    j.add_argument("--concurrency", type=int, default=4)
    j.add_argument("--keep-think", action="store_true", help="Do not strip <think> blocks from answers.")
    j.add_argument("--resume-by-cell", action="store_true",
                   help="Treat any valid verdict for the same question/pair/order/judge in --out as done.")
    j.add_argument("--dry-run", action="store_true", help="Only count the calls (and prompt size) that would be made.")

    w = sub.add_parser("weights", help="Fit PRD judge weights from peer judgments.")
    w.add_argument("judgments", nargs="+", help="Peer judgment JSONL file(s): the judges also answered the questions.")
    w.add_argument("--models", nargs="+", help="Contestants to keep (default: all).")
    w.add_argument("--out", required=True, help="Write {judge: weight} JSON here.")

    p = sub.add_parser("panel", help="Combine judges' verdicts on each pair into one weighted panel verdict.")
    p.add_argument("judgments", nargs="+", help="Judgment JSONL file(s) on the pairs to decide.")
    p.add_argument("--weights", help="Judge weights from `peereval weights` (or `peereval score --json`). "
                                     "Default: every judge counts equally.")
    p.add_argument("--out", required=True, help="Write one panel verdict per pair to this JSONL file.")

    args = ap.parse_args(argv)
    if args.cmd == "score":
        board = score(args.judgments, models=args.models, by_dataset=not args.no_by_dataset,
                      complete_only=args.complete_only, bootstrap=args.bootstrap, seed=args.seed)
        print(board.to_markdown())
        if args.json:
            board.to_json(args.json)
        if args.csv:
            board.to_csv(args.csv)
    elif args.cmd == "weights":
        weights = fit_weights(args.judgments, models=args.models)
        save_weights(weights, args.out)
        for judge, wt in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"{judge}\t{wt:.3f}")
    elif args.cmd == "panel":
        rows = panel_verdicts(args.judgments, load_weights(args.weights) if args.weights else None)
        with open(args.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(rows)} panel verdicts to {args.out}", file=sys.stderr)
    elif args.cmd == "generate":
        n = run_generation(args.questions, load_judge_configs(args.models), args.out,
                           concurrency=args.concurrency, strip_think=not args.keep_think, dry_run=args.dry_run)
        print(n if args.dry_run else f"wrote {n} answers to {args.out}", file=sys.stderr)
    else:
        n = run_judging(args.answers, load_judge_configs(args.judges), args.out,
                        concurrency=args.concurrency, strip_think=not args.keep_think,
                        resume_by_cell=args.resume_by_cell, dry_run=args.dry_run)
        print(n if args.dry_run else f"wrote {n} judgments to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
