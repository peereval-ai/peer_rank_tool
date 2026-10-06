"""PeerEval: rank LLMs by having them judge each other (PRD Peer Rank)."""
from .data import Judgment, load_judgments, normalize, read_jsonl
from .judge import JudgeConfig, parse_verdict, run_judging, strip_reasoning
from .leaderboard import Leaderboard, Result, score

__version__ = "0.1.0"
__all__ = [
    "Judgment", "JudgeConfig", "Leaderboard", "Result",
    "load_judgments", "normalize", "read_jsonl", "parse_verdict", "run_judging", "score", "strip_reasoning",
]
