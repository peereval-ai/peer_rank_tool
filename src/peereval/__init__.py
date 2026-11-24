"""PeerEval: rank LLMs by having them judge each other (PRD Peer Rank)."""
from .data import Judgment, load_judgments, normalize, read_jsonl
from .generate import run_generation
from .judge import JudgeConfig, ModelConfig, parse_verdict, run_judging, strip_reasoning
from .leaderboard import Leaderboard, Result, score
from .panel import fit_weights, load_weights, panel_verdicts, save_weights

__version__ = "0.1.2"
__all__ = [
    "Judgment", "JudgeConfig", "ModelConfig", "Leaderboard", "Result",
    "fit_weights", "load_judgments", "load_weights", "normalize", "panel_verdicts", "read_jsonl", "parse_verdict",
    "run_generation", "run_judging", "save_weights", "score", "strip_reasoning",
]
