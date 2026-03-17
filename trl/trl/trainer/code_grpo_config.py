from dataclasses import dataclass, field

from .grpo_config import GRPOConfig


@dataclass
class CodeGRPOConfig(GRPOConfig):
    """Configuration for CodeGRPOTrainer."""

    # Keep sibling size aligned with GRPO grouping.
    num_generations: int | None = field(default=2, metadata={"help": "Number of sibling candidates (K)."})
    num_generations_eval: int | None = field(
        default=1,
        metadata={"help": "Eval generations. For code-only single-trajectory eval this should stay at 1."},
    )

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
    undiff_retry_enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable retry for undiff_unsolved sibling groups: strict size==2, not all-pass, "
                "R_code identical, pass_rate identical. Regenerates one sibling to break the tie."
            )
        },
    )
    undiff_retry_max: int = field(
        default=1,
        metadata={"help": "Maximum retries for undiff_unsolved groups (only used when undiff_retry_enabled=True)."},
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
    code_aux_reward_without_format_scale: float = field(
        default=0.25,
        metadata={
            "help": (
                "Scale applied to compile/soft auxiliary code rewards when the main code response format is invalid. "
                "pass_rate is kept unchanged."
            )
        },
    )
    logic_format_reward_scale: float = field(
        default=0.1,
        metadata={"help": "Additive scale for logic-audit format signal in reason reward."},
    )
    logic_match_reward_scale: float = field(
        default=1.0,
        metadata={"help": "Reward scale for logic-audit answer match in reason reward."},
    )
    logic_confirmed_bonus: float = field(
        default=0.25,
        metadata={"help": "Extra bonus for logic-audit answer match when the child code already passes."},
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
    max_completion_length_code: int | None = field(
        default=None,
        metadata={
            "help": (
                "Optional main-generation completion length override. Falls back to max_completion_length when unset."
            )
        },
    )
    max_completion_length_audit: int | None = field(
        default=None,
        metadata={
            "help": (
                "Optional audit/reason completion length override. Falls back to max_completion_length when unset."
            )
        },
    )
    generation_temperature_code: float | None = field(
        default=None,
        metadata={
            "help": (
                "Optional main-generation sampling temperature override. Falls back to temperature when unset."
            )
        },
    )
    eval_generation_temperature_code: float | None = field(
        default=None,
        metadata={
            "help": (
                "Optional code-only eval temperature override. Set to 0 for deterministic greedy eval."
            )
        },
    )
    generation_top_p_code: float | None = field(
        default=None,
        metadata={
            "help": "Optional main-generation top-p override. Falls back to top_p when unset.",
        },
    )
    generation_min_new_tokens_code: int = field(
        default=16,
        metadata={"help": "Minimum number of new tokens to generate for main code generation."},
    )
    generation_empty_retry_count: int = field(
        default=1,
        metadata={"help": "How many times to retry a main-generation sample when the decoded output is empty."},
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

    # --- Variance reduction options (K=2 specific) ---
    advantage_base: str = field(
        default="R_code",
        metadata={
            "help": (
                "Which reward signal to use for computing A_code sibling advantages. "
                "'R_code': full composite reward (default, may saturate at 1 with clamp01). "
                "'pass_rate': raw pass_rate only (avoids clamp01 saturation when soft/compile/format push R_code to 1)."
            )
        },
    )
    advantage_mode: str = field(
        default="zscore",
        metadata={
            "help": (
                "How to compute sibling group advantages. "
                "'zscore': (r - mean) / (std + eps), standard GRPO (default). "
                "'sign': pairwise sign advantage: +1/-1/0 based on reward comparison (reduces K=2 scale noise). "
                "'mean_only': (r - mean), no std normalization (Dr.GRPO style, preserves reward gap magnitude)."
            )
        },
    )
    code_grpo_loss_type: str = field(
        default="seq_mean",
        metadata={
            "help": (
                "Loss aggregation for code_grpo orthogonal loss. "
                "'seq_mean': per-sequence mean then batch mean (current default, known length bias). "
                "'token_mean': sum all token losses / total active tokens (DAPO/Dr.GRPO style, no length bias). "
            )
        },
    )
    async_rollout_prefetch: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable one-batch-ahead asynchronous rollout prefetch for single-process vLLM training. "
                "The next tree rollout is prepared in a background thread while the current batch trains."
            )
        },
    )
    signal_weighted_sampling: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable signal-weighted sampling (dataloader-build-time snapshot). "
                "Questions that historically produced useful gradient signal are sampled more often; "
                "always-solved or long-term-undifferentiated questions are down-weighted. "
                "NOTE: weights are computed once when the train dataloader is built, using the "
                "signal states accumulated up to that point. They are NOT refreshed each epoch. "
                "At training start the states are empty so all weights equal 1.0 (uniform). "
                "This flag is a placeholder for future epoch-level refresh; "
                "enable only after implementing a sampler rebuild hook."
            )
        },
    )

    error_max_chars: int = field(default=800, metadata={"help": "Maximum error summary character length."})
    error_max_lines: int = field(default=20, metadata={"help": "Maximum error summary line count."})
    code_timeout_seconds: float = field(default=2.0, metadata={"help": "Timeout in seconds for single test-case exec."})

    eval_round_n: int = field(default=1, metadata={"help": "Round index used by pass@k@round=n metrics."})
    eval_k_list: list[int] = field(default_factory=lambda: [1, 3, 5], metadata={"help": "k values for pass@k metrics."})
    eval_code_only_single_trajectory: bool = field(
        default=True,
        metadata={
            "help": (
                "Use code-only eval with a single repair trajectory. "
                "No logic/exec audit is run during eval; metrics report cumulative pass@1 within <= round r."
            )
        },
    )
    eval_T_max_override: int | None = field(
        default=None,
        metadata={
            "help": (
                "Optional eval-only maximum code-repair rounds. When set, code-only eval can run for more rounds "
                "than training T_max."
            )
        },
    )
    eval_repeat_count: int = field(
        default=1,
        metadata={
            "help": (
                "Repeat each code-only eval trajectory this many times with different seeds and average metrics "
                "across repeats."
            )
        },
    )
    run_base_model_baseline_eval: bool = field(
        default=False,
        metadata={
            "help": (
                "Before training, run one eval pass with the raw base model (no PEFT adapter) on the eval split "
                "and save metrics as baseline_eval."
            )
        },
    )
    reward_window_bins: int = field(
        default=10,
        metadata={
            "help": (
                "Split total training steps into N bins and log rolling reward means over each bin. "
                "For example, max_steps=200 and reward_window_bins=10 logs one reward window every 20 steps."
            )
        },
    )
    tensorboard_root_dir: str | None = field(
        default=None,
        metadata={
            "help": (
                "Optional external TensorBoard root directory. When set, event files are written to "
                "<tensorboard_root_dir>/<mode>/<run_id> instead of <run_root>/tensorboard."
            )
        },
    )
    vllm_dynamic_lora_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Optional standalone-eval LoRA adapter path for vLLM dynamic LoRA requests. "
                "When set, vLLM keeps the base model loaded and applies this adapter per request."
            )
        },
    )
    vllm_dynamic_lora_name: str = field(
        default="adapter",
        metadata={"help": "Request name used when applying a dynamic LoRA adapter in vLLM."},
    )
    vllm_dynamic_lora_int_id: int = field(
        default=1,
        metadata={"help": "Integer adapter id used for dynamic LoRA requests in vLLM."},
    )

    debug_trace_dir: str = field(
        default="traces/rollout",
        metadata={"help": "Relative directory under output_dir for per-question JSON traces."},
    )
    debug_trace_sample_size: int = field(
        default=1,
        metadata={"help": "How many rollout traces to dump per train step when dump_train_traces=True (0 means all)."},
    )
    dump_train_trace_interval_steps: int = field(
        default=120,
        metadata={"help": "Dump train rollout traces only every N global steps. <=1 means every step."},
    )
    max_train_trace_files: int = field(
        default=8,
        metadata={"help": "Maximum number of train rollout trace files to dump in a single run (0 means no cap)."},
    )
    debug_trace_question_ids: list[str] = field(
        default_factory=list,
        metadata={"help": "Optional question_id whitelist for trace dump. Empty means no whitelist."},
    )
    trace_store_full_text: bool = field(
        default=False,
        metadata={
            "help": "Store full prompt/output text inside the selected rollout traces for exact replay and debugging."
        },
    )
    dump_train_traces: bool = field(
        default=False,
        metadata={"help": "Whether to dump per-question rollout traces during train mode as well."},
    )
    dump_eval_traces: bool = field(
        default=False,
        metadata={"help": "Whether to dump per-question rollout traces during eval mode."},
    )
    log_eval_trajectories: bool = field(
        default=False,
        metadata={"help": "Whether to log per-question eval trajectory summaries and node rewards to the console."},
    )
    log_train_rollout_details: bool = field(
        default=False,
        metadata={"help": "Whether to log per-question train rollout summaries and node rewards to the console."},
    )
    write_trainer_text_log: bool = field(
        default=False,
        metadata={"help": "Whether to write a duplicated plain-text trainer_events log in addition to jsonl."},
    )
    log_reward_losses: bool = field(
        default=False,
        metadata={"help": "Whether to print per-step loss_code/loss_reason messages to the console."},
    )
    log_trace_dump_events: bool = field(
        default=False,
        metadata={"help": "Whether to print trace dump count messages to the console."},
    )
    compact_logging: bool = field(
        default=True,
        metadata={
            "help": "Log only a compact whitelist of high-signal metrics to console/reporters such as TensorBoard."
        },
    )
    log_kl_metrics: bool = field(
        default=True,
        metadata={
            "help": "Compute and log KL divergence as a monitoring metric even when it is not part of the loss."
        },
    )
    review_bundle_trace_sample_size: int = field(
        default=2,
        metadata={"help": "How many trace files to copy into review_bundle."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.codegrpo_mode not in {"train", "test"}:
            raise ValueError(f"codegrpo_mode must be one of ['train', 'test'], got: {self.codegrpo_mode}")
        if self.backend not in {"hf", "vllm"}:
            raise ValueError(f"backend must be one of ['hf', 'vllm'], got: {self.backend}")
        if self.advantage_base not in {"R_code", "pass_rate"}:
            raise ValueError(f"advantage_base must be 'R_code' or 'pass_rate', got: {self.advantage_base}")
        if self.advantage_mode not in {"zscore", "sign", "mean_only"}:
            raise ValueError(f"advantage_mode must be 'zscore', 'sign', or 'mean_only', got: {self.advantage_mode}")
        if self.code_grpo_loss_type not in {"seq_mean", "token_mean"}:
            raise ValueError(
                f"code_grpo_loss_type must be 'seq_mean' or 'token_mean', got: {self.code_grpo_loss_type}"
            )
        if self.K < 2:
            raise ValueError("K must be >= 2 for sibling-group GRPO.")
        if self.K_reason < 1:
            raise ValueError("K_reason must be >= 1.")
        if self.K != self.num_generations:
            raise ValueError(
                f"K ({self.K}) must equal num_generations ({self.num_generations}) to keep sibling grouping consistent."
            )
        if self.eval_repeat_count < 1:
            raise ValueError("eval_repeat_count must be >= 1.")
        if self.dump_train_trace_interval_steps == 0:
            raise ValueError("dump_train_trace_interval_steps must be != 0.")
        if self.max_train_trace_files < 0:
            raise ValueError("max_train_trace_files must be >= 0.")
        if self.review_bundle_trace_sample_size < 0:
            raise ValueError("review_bundle_trace_sample_size must be >= 0.")
        if self.M_audit < 0 or self.M_retry < 0:
            raise ValueError("M_audit and M_retry must be non-negative.")
        if self.undiff_retry_max < 0:
            raise ValueError("undiff_retry_max must be non-negative.")
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
        if not (0.0 <= self.code_aux_reward_without_format_scale <= 1.0):
            raise ValueError(
                "code_aux_reward_without_format_scale must be in [0, 1], "
                f"got: {self.code_aux_reward_without_format_scale}"
            )
        if not (0.0 <= self.logic_format_reward_scale <= 1.0):
            raise ValueError(
                f"logic_format_reward_scale must be in [0, 1], got: {self.logic_format_reward_scale}"
            )
        if not (0.0 <= self.logic_match_reward_scale <= 1.0):
            raise ValueError(
                f"logic_match_reward_scale must be in [0, 1], got: {self.logic_match_reward_scale}"
            )
        if not (0.0 <= self.logic_confirmed_bonus <= 1.0):
            raise ValueError(f"logic_confirmed_bonus must be in [0, 1], got: {self.logic_confirmed_bonus}")
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
        if self.max_completion_length_code is not None and self.max_completion_length_code <= 0:
            raise ValueError("max_completion_length_code must be > 0 when provided.")
        if self.max_completion_length_audit is not None and self.max_completion_length_audit <= 0:
            raise ValueError("max_completion_length_audit must be > 0 when provided.")
        if self.generation_temperature_code is not None and self.generation_temperature_code < 0.0:
            raise ValueError("generation_temperature_code must be >= 0 when provided.")
        if self.eval_generation_temperature_code is not None and self.eval_generation_temperature_code < 0.0:
            raise ValueError("eval_generation_temperature_code must be >= 0 when provided.")
        if self.generation_top_p_code is not None and not (0.0 < self.generation_top_p_code <= 1.0):
            raise ValueError("generation_top_p_code must be in (0, 1] when provided.")
        if self.generation_min_new_tokens_code < 0:
            raise ValueError("generation_min_new_tokens_code must be >= 0.")
        if self.generation_empty_retry_count < 0:
            raise ValueError("generation_empty_retry_count must be >= 0.")
        if self.max_completion_length_code is not None and self.max_completion_length_code > self.max_completion_length:
            raise ValueError(
                "max_completion_length_code must be <= max_completion_length so shared generation backends remain valid."
            )
        if (
            self.max_completion_length_audit is not None
            and self.max_completion_length_audit > self.max_completion_length
        ):
            raise ValueError(
                "max_completion_length_audit must be <= max_completion_length so shared generation backends remain valid."
            )
        if (
            self.max_completion_length_code is not None
            and self.generation_min_new_tokens_code > self.max_completion_length_code
        ):
            raise ValueError("generation_min_new_tokens_code must be <= max_completion_length_code.")
        if self.generation_min_new_tokens_code > self.max_completion_length:
            raise ValueError("generation_min_new_tokens_code must be <= max_completion_length.")
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
