"""PeerEval: rank LLMs by having them judge each other (PRD Peer Rank)."""
from .data import Judgment, load_judgments, normalize, read_jsonl
from .generate import run_generation
from .judge import JudgeConfig, ModelConfig, parse_verdict, run_judging, strip_reasoning
from .leaderboard import Leaderboard, Result, score

__version__ = "0.1.1"
__all__ = [
    "Judgment", "JudgeConfig", "ModelConfig", "Leaderboard", "Result",
    "load_judgments", "normalize", "read_jsonl", "parse_verdict", "run_generation", "run_judging", "score", "strip_reasoning",
]
