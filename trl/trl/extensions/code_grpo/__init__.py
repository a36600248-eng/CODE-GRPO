"""Code GRPO extension package."""

from .backends import Backend, HFBackend, VLLMBackend, build_backend
from .error_utils import summarize_error
from .executor import execute
from .matcher import is_match
from .parser import (
    build_canonical_completion,
    build_token_masks,
    parse_exec_prediction_only,
    parse_exec_response,
    parse_generation_output,
    parse_generation_response,
    parse_logic_prediction_only,
    parse_logic_response,
    parse_prediction_only,
)
from .tree import CodeGRPOTreeRunner
from .types import ExecResult, Node, QuestionRollout, TrainSample

__all__ = [
    "Backend",
    "HFBackend",
    "VLLMBackend",
    "CodeGRPOTreeRunner",
    "ExecResult",
    "Node",
    "QuestionRollout",
    "TrainSample",
    "build_backend",
    "build_canonical_completion",
    "build_token_masks",
    "execute",
    "is_match",
    "parse_exec_prediction_only",
    "parse_exec_response",
    "parse_generation_output",
    "parse_generation_response",
    "parse_logic_prediction_only",
    "parse_logic_response",
    "parse_prediction_only",
    "summarize_error",
]
