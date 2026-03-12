from dataclasses import dataclass, field
from typing import Any, Literal


CodeStatus = Literal["CORRECT", "FAIL", "SYNTAX_ERROR", "RUNTIME_ERROR", "TIMEOUT"]
ReasonStatus = Literal["HONEST", "HALLUCINATION", "NOT_RUN"]
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
    reasoning: str | None
    pass_rate: float = 0.0
    R_soft: float = 0.0
    R_code: float = 0.0
    R_reason: float = 0.0
    status_code: CodeStatus = "FAIL"
    status_reason: ReasonStatus = "NOT_RUN"
    frozen_code: bool = False
    frozen_reason: bool = False
    exec_summary: dict[str, Any] = field(default_factory=dict)
    A_code: float = 0.0
    A_reason: float = 0.0
    completion_text: str = ""
    code_token_mask: list[int] = field(default_factory=list)
    reason_token_mask: list[int] = field(default_factory=list)
    prompt_text: str = ""


@dataclass
class TrainSample:
    question_id: str
    prompt_text: str
    completion_text: str
    code_token_mask: list[int]
    reason_token_mask: list[int]
    A_code: float
    A_reason: float
    R_code: float
    pass_rate: float


@dataclass
class QuestionRollout:
    question_id: str
    audit_indices: list[int]
    rounds: list[dict[str, Any]]
    node_count: int
    resample_count: int
    train_samples: list[TrainSample]
    mean_R_code: float
    mean_R_reason: float
    mean_pass_rate: float
    std_R_code: float
    std_R_reason: float
    eval_metrics: dict[str, float] = field(default_factory=dict)
    repeat_idx: int | None = None
