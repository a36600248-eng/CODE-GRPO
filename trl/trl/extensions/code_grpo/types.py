from dataclasses import dataclass, field
from typing import Any, Literal


CodeStatus = Literal["CORRECT", "FAIL", "SYNTAX_ERROR", "RUNTIME_ERROR", "TIMEOUT"]
ExecKind = Literal["OK", "SYNTAX_ERROR", "RUNTIME_ERROR", "TIMEOUT"]


@dataclass
class ExecResult:
    kind: ExecKind
    value: Any | None = None
    error_type: str | None = None
    error_msg: str | None = None


@dataclass
class Node:
    node_id: str
    parent_id: str | None
    round_idx: int
    code: str | None
    pass_rate: float = 0.0
    R_soft: float = 0.0
    R_code: float = 0.0
    status_code: CodeStatus = "FAIL"
    frozen_code: bool = False
    exec_summary: dict[str, Any] = field(default_factory=dict)
    A_code: float = 0.0
    completion_text: str = ""
    code_token_mask: list[int] = field(default_factory=list)
    prompt_text: str = ""


@dataclass
class TrainSample:
    question_id: str
    prompt_text: str
    completion_text: str
    code_token_mask: list[int]
    A_code: float
    R_code: float
    pass_rate: float
    sft_token_mask: list[int] = field(default_factory=list)
    sft_weight: float = 0.0
    old_per_token_logps: list[float] | None = None


@dataclass
class QuestionRollout:
    question_id: str
    rounds: list[dict[str, Any]]
    node_count: int
    resample_count: int
    train_samples: list[TrainSample]
    mean_R_code: float
    mean_pass_rate: float
    std_R_code: float
    eval_metrics: dict[str, float] = field(default_factory=dict)
    repeat_idx: int | None = None

