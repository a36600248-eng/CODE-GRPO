"""Code GRPO extension package."""

from .backends import Backend, HFBackend, VLLMBackend, build_backend
from .error_utils import summarize_error
from .executor import execute
from .matcher import is_match
from .parser import build_generation_completion, build_token_masks, parse_generation_output, parse_generation_response
from .soft_reward import (
    build_diagnostic_inputs,
    compute_soft_reward,
    compute_zero_pass_beta,
    get_oracle_outputs,
    normalize_soft_reward_to_unit_interval,
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
    "build_generation_completion",
    "build_token_masks",
    "build_diagnostic_inputs",
    "compute_soft_reward",
    "compute_zero_pass_beta",
    "execute",
    "get_oracle_outputs",
    "is_match",
    "normalize_soft_reward_to_unit_interval",
    "parse_generation_output",
    "parse_generation_response",
    "summarize_error",
]
