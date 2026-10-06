"""Command line: ``peereval score`` and ``peereval judge``."""
from __future__ import annotations

import argparse
import sys

from .judge import load_judge_configs, run_judging
from .leaderboard import score


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="peereval", description="Peer evaluation of LLMs with PRD Peer Rank.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="Rank models from pairwise judgment files.")
    s.add_argument("judgments", nargs="+", help="Judgment JSONL file(s).")
    s.add_argument("--models", nargs="+", help="Contestants to keep (default: all).")
    s.add_argument("--no-by-dataset", action="store_true", help="Only score all data together.")
    s.add_argument("--complete-only", action="store_true",
                   help="Keep only questions that have every pair x judge x order verdict.")
    s.add_argument("--json", help="Write full results (scores, weights, matrices) to this JSON file.")
    s.add_argument("--csv", help="Write the leaderboard tables to this CSV file.")

    j = sub.add_parser("judge", help="Generate pairwise judgments from solver answers.")
    j.add_argument("answers", nargs="+", help="Solver answer JSONL file(s).")
    j.add_argument("--judges", required=True, help="JSON list of judge configs.")
    j.add_argument("--out", required=True, help="Judgment JSONL to write (appended; resumable).")
    j.add_argument("--concurrency", type=int, default=4)
    j.add_argument("--keep-think", action="store_true", help="Do not strip <think> blocks from answers.")

    args = ap.parse_args(argv)
    if args.cmd == "score":
        board = score(args.judgments, models=args.models, by_dataset=not args.no_by_dataset,
                      complete_only=args.complete_only)
        print(board.to_markdown())
        if args.json:
            board.to_json(args.json)
        if args.csv:
            board.to_csv(args.csv)
    else:
        n = run_judging(args.answers, load_judge_configs(args.judges), args.out,
                        concurrency=args.concurrency, strip_think=not args.keep_think)
        print(f"wrote {n} judgments to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
