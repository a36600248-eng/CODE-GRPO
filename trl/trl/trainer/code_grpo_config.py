from dataclasses import dataclass, field

from .grpo_config import GRPOConfig


@dataclass
class CodeGRPOConfig(GRPOConfig):
    """Configuration for CodeGRPOTrainer."""

    # Keep sibling size aligned with GRPO grouping.
    num_generations: int | None = field(default=2, metadata={"help": "Number of sibling candidates (K)."})
    num_generations_eval: int | None = field(default=2, metadata={"help": "Eval sibling candidates (defaults to K)."})

    codegrpo_mode: str = field(default="train", metadata={"help": "Pipeline mode: train or test."})
    backend: str = field(default="hf", metadata={"help": "Generation backend: hf or vllm."})

    K: int = field(default=2, metadata={"help": "Sibling group size per parent expansion."})
    K_reason: int = field(
        default=4,
        metadata={"help": "Sibling size for reason-only rollout when code is frozen and reason is not frozen."},
    )
    T_max: int = field(default=3, metadata={"help": "Maximum rounds per question."})
    N_max: int = field(default=16, metadata={"help": "Maximum generated nodes per question."})
    M_audit: int = field(default=3, metadata={"help": "Fixed audit subset size per question."})
    M_retry: int = field(
        default=1,
        metadata={"help": "Maximum retries (N) when a sibling group is all double-zero rewards."},
    )
    context_round_window: int = field(default=2, metadata={"help": "Rounds of feedback to carry to next prompt."})

    lambda_soft: float = field(default=0.2, metadata={"help": "Soft reward coefficient for code reward."})
    code_compile_reward_scale: float = field(
        default=0.1,
        metadata={"help": "Residual reward scale for compile-success signal in main generation reward."},
    )
    code_format_reward_scale: float = field(
        default=0.05,
        metadata={"help": "Residual reward scale for main-generation format compliance signal."},
    )
    logic_format_reward_scale: float = field(
        default=0.1,
        metadata={"help": "Additive scale for logic-audit format signal in reason reward."},
    )
    exec_case_baseline_ema_alpha: float = field(
        default=0.1,
        metadata={"help": "EMA alpha for execution case baseline used in fallback case-level advantage."},
    )
    soft_reward_ineligible_scale: float = field(
        default=0.3,
        metadata={
            "help": (
                "Scale applied to soft reward when node is soft-reward-ineligible "
                "(e.g., syntax error or missing top-level solve). 0 keeps hard gate."
            )
        },
    )
    format_penalty_logic: float = field(
        default=0.3,
        metadata={"help": "Penalty applied when logic response format is invalid."},
    )
    format_bonus_logic: float = field(
        default=0.05,
        metadata={"help": "Bonus added when logic response format is valid."},
    )
    format_penalty_exec: float = field(
        default=0.3,
        metadata={"help": "Penalty applied when execution response format is invalid."},
    )
    format_bonus_exec: float = field(
        default=0.05,
        metadata={"help": "Bonus added when execution response format is valid."},
    )
    require_reason_before_prediction: bool = field(
        default=True,
        metadata={"help": "Whether <REASON> must appear before prediction tags to count as format-valid."},
    )
    reasoning_max_chars: int = field(
        default=400,
        metadata={"help": "Max allowed characters in reasoning tags for logic/exec parsing."},
    )
    prediction_max_chars: int = field(
        default=200,
        metadata={"help": "Max allowed characters in prediction tags for logic/exec parsing."},
    )
    disallow_code_in_reasoning: bool = field(
        default=True,
        metadata={"help": "Mark format invalid when reasoning contains code-like content."},
    )
    format_outside_noise_chars: int = field(
        default=80,
        metadata={
            "help": (
                "Allowed non-whitespace chars outside required reasoning tags when parsing logic/exec responses. "
                "Use 0 for fully strict parsing."
            )
        },
    )
    generation_outside_noise_chars: int = field(
        default=0,
        metadata={
            "help": (
                "Allowed non-whitespace chars outside the required code block when parsing main-generation "
                "responses. Use 0 to enforce code-only output."
            )
        },
    )
    prefill_generation_code_tag: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether the main-generation prompt should prefill the opening <CODE> tag and expect the model "
                "to continue with code body only."
            )
        },
    )
    use_chat_template_for_codegrpo: bool = field(
        default=True,
        metadata={"help": "Whether to render CodeGRPO prompts with tokenizer chat template before generation."},
    )
    terminal_logic_backprop_bonus: float = field(
        default=0.1,
        metadata={
            "help": (
                "Base bonus scale for terminal logic backpropagation. "
                "Applied to matched logic case-level rewards on ancestors when descendants solve the task."
            )
        },
    )
    terminal_logic_backprop_decay: float = field(
        default=0.7,
        metadata={"help": "Per-hop decay for terminal logic backprop bonus along ancestor chain."},
    )
    terminal_logic_backprop_max_depth: int = field(
        default=6,
        metadata={"help": "Maximum ancestor hops for terminal logic backprop bonus."},
    )
    terminal_backprop_code_similarity_threshold: float = field(
        default=0.75,
        metadata={"help": "Minimum parent-child code similarity to continue terminal bonus backpropagation."},
    )
    frozen_reason_one_shot: bool = field(
        default=True,
        metadata={"help": "Run a single reason-only round after code is frozen/passed, then stop tree growth."},
    )
    unify_reason_when_code_frozen: bool = field(
        default=True,
        metadata={"help": "When code is frozen and passed, use a unified reason/execution audit prompt."},
    )
    beta_reason: float = field(default=1.0, metadata={"help": "Reason loss coefficient."})
    gamma_shrink: float = field(default=0.1, metadata={"help": "Advantage shrink factor for fully-correct nodes."})

    error_max_chars: int = field(default=800, metadata={"help": "Maximum error summary character length."})
    error_max_lines: int = field(default=20, metadata={"help": "Maximum error summary line count."})
    code_timeout_seconds: float = field(default=2.0, metadata={"help": "Timeout in seconds for single test-case exec."})

    eval_round_n: int = field(default=1, metadata={"help": "Round index used by pass@k@round=n metrics."})
    eval_k_list: list[int] = field(default_factory=lambda: [1, 3, 5], metadata={"help": "k values for pass@k metrics."})

    debug_trace_dir: str = field(
        default="traces/rollout",
        metadata={"help": "Relative directory under output_dir for per-question JSON traces."},
    )
    debug_trace_sample_size: int = field(
        default=1,
        metadata={"help": "How many rollout traces to dump per train step when dump_train_traces=True (0 means all)."},
    )
    debug_trace_question_ids: list[str] = field(
        default_factory=list,
        metadata={"help": "Optional question_id whitelist for trace dump. Empty means no whitelist."},
    )
    dump_train_traces: bool = field(
        default=False,
        metadata={"help": "Whether to dump per-question rollout traces during train mode as well."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.codegrpo_mode not in {"train", "test"}:
            raise ValueError(f"codegrpo_mode must be one of ['train', 'test'], got: {self.codegrpo_mode}")
        if self.backend not in {"hf", "vllm"}:
            raise ValueError(f"backend must be one of ['hf', 'vllm'], got: {self.backend}")
        if self.K < 2:
            raise ValueError("K must be >= 2 for sibling-group GRPO.")
        if self.K_reason < 1:
            raise ValueError("K_reason must be >= 1.")
        if self.K != self.num_generations:
            raise ValueError(
                f"K ({self.K}) must equal num_generations ({self.num_generations}) to keep sibling grouping consistent."
            )
        if self.M_audit < 0 or self.M_retry < 0:
            raise ValueError("M_audit and M_retry must be non-negative.")
        if not (0.0 <= self.lambda_soft <= 1.0):
            raise ValueError(f"lambda_soft must be in [0, 1], got: {self.lambda_soft}")
        if not (0.0 <= self.code_compile_reward_scale <= 1.0):
            raise ValueError(
                f"code_compile_reward_scale must be in [0, 1], got: {self.code_compile_reward_scale}"
            )
        if not (0.0 <= self.code_format_reward_scale <= 1.0):
            raise ValueError(
                f"code_format_reward_scale must be in [0, 1], got: {self.code_format_reward_scale}"
            )
        if not (0.0 <= self.logic_format_reward_scale <= 1.0):
            raise ValueError(
                f"logic_format_reward_scale must be in [0, 1], got: {self.logic_format_reward_scale}"
            )
        if not (0.0 <= self.exec_case_baseline_ema_alpha <= 1.0):
            raise ValueError(
                f"exec_case_baseline_ema_alpha must be in [0, 1], got: {self.exec_case_baseline_ema_alpha}"
            )
        if not (0.0 <= self.soft_reward_ineligible_scale <= 1.0):
            raise ValueError(
                f"soft_reward_ineligible_scale must be in [0, 1], got: {self.soft_reward_ineligible_scale}"
            )
        if not (0.0 <= self.format_penalty_logic <= 1.0):
            raise ValueError(f"format_penalty_logic must be in [0, 1], got: {self.format_penalty_logic}")
        if not (0.0 <= self.format_bonus_logic <= 1.0):
            raise ValueError(f"format_bonus_logic must be in [0, 1], got: {self.format_bonus_logic}")
        if not (0.0 <= self.format_penalty_exec <= 1.0):
            raise ValueError(f"format_penalty_exec must be in [0, 1], got: {self.format_penalty_exec}")
        if not (0.0 <= self.format_bonus_exec <= 1.0):
            raise ValueError(f"format_bonus_exec must be in [0, 1], got: {self.format_bonus_exec}")
        if self.eval_round_n < 1:
            raise ValueError("eval_round_n must be >= 1.")
        if not self.eval_k_list:
            raise ValueError("eval_k_list must contain at least one k value.")
        if any(k < 1 for k in self.eval_k_list):
            raise ValueError("All values in eval_k_list must be >= 1.")
        if self.reasoning_max_chars <= 0:
            raise ValueError("reasoning_max_chars must be > 0.")
        if self.prediction_max_chars <= 0:
            raise ValueError("prediction_max_chars must be > 0.")
        if self.format_outside_noise_chars < 0:
            raise ValueError("format_outside_noise_chars must be >= 0.")
        if self.generation_outside_noise_chars < 0:
            raise ValueError("generation_outside_noise_chars must be >= 0.")
        if not (0.0 <= self.terminal_logic_backprop_bonus <= 1.0):
            raise ValueError(
                f"terminal_logic_backprop_bonus must be in [0, 1], got: {self.terminal_logic_backprop_bonus}"
            )
        if not (0.0 <= self.terminal_logic_backprop_decay <= 1.0):
            raise ValueError(
                f"terminal_logic_backprop_decay must be in [0, 1], got: {self.terminal_logic_backprop_decay}"
            )
        if self.terminal_logic_backprop_max_depth < 0:
            raise ValueError("terminal_logic_backprop_max_depth must be >= 0.")
        if not (0.0 <= self.terminal_backprop_code_similarity_threshold <= 1.0):
            raise ValueError(
                "terminal_backprop_code_similarity_threshold must be in [0, 1], "
                f"got: {self.terminal_backprop_code_similarity_threshold}"
            )
        if self.debug_trace_sample_size < 0:
            raise ValueError("debug_trace_sample_size must be >= 0.")
