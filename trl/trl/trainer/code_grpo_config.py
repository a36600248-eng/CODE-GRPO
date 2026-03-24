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
    T_max: int = field(default=3, metadata={"help": "Maximum rounds per question."})
    N_max: int = field(default=16, metadata={"help": "Maximum generated nodes per question."})
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

    lambda_soft: float = field(default=0.2, metadata={"help": "Legacy soft-reward coefficient field; the active main path uses bounded beta-per-test scaling instead."})
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
    soft_reward_ineligible_scale: float = field(
        default=0.3,
        metadata={
            "help": (
                "Scale applied to soft reward when node is soft-reward-ineligible "
                "(e.g., syntax error or missing top-level solve). 0 keeps hard gate."
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
    train_generation_truncate_tokens: int = field(
        default=0,
        metadata={
            "help": (
                "Optional train-only post-generation truncation cap in tokens. "
                "If > 0, generated code longer than this cap is truncated before execution/reward/model training. "
                "Eval is unaffected."
            )
        },
    )
    train_generation_total_token_cap: int = field(
        default=0,
        metadata={
            "help": (
                "Optional train-only total token budget for rendered prompt + generated code. "
                "When > 0, training generation dynamically reduces max_new_tokens to stay within this budget."
            )
        },
    )
    train_generation_completion_reserve_tokens: int = field(
        default=0,
        metadata={
            "help": (
                "Preferred train-only completion token reserve when train_generation_total_token_cap is enabled. "
                "The trainer will first try to trim carried history to preserve at least this much completion space."
            )
        },
    )
    zero_pass_soft_reward_enabled: bool = field(
        default=True,
        metadata={"help": "Enable bounded soft reward for code candidates; field name is legacy."},
    )
    zero_pass_soft_reward_diag_count: int = field(
        default=4,
        metadata={"help": "Maximum number of diagnostic IO cases used for bounded soft reward; field name is legacy."},
    )
    zero_pass_soft_reward_clip_low: float = field(
        default=-2.0,
        metadata={"help": "Lower clip bound for raw bounded soft reward before mapping to [0,1]; field name is legacy."},
    )
    zero_pass_soft_reward_clip_high: float = field(
        default=2.0,
        metadata={"help": "Upper clip bound for raw bounded soft reward before mapping to [0,1]; field name is legacy."},
    )
    zero_pass_soft_reward_beta_scale: float = field(
        default=0.5,
        metadata={
            "help": (
                "Soft reward scale numerator for bounded soft reward. Actual beta is min(beta_scale / k, (1-eps)/k), "
                "so beta stays strictly below the reward value of one passed test case."
            )
        },
    )
    code_io_aux_training_enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable auxiliary training on code+input->output (or error type) pairs. "
                "This uses a supervised loss on the auxiliary targets while code remains the main GRPO target."
            )
        },
    )
    code_io_aux_case_count: int = field(
        default=2,
        metadata={"help": "Maximum number of code-IO auxiliary training cases created per candidate code sample."},
    )
    code_io_aux_include_correct: bool = field(
        default=True,
        metadata={"help": "Whether to create code-IO auxiliary samples for candidates whose execution is correct."},
    )
    code_io_aux_include_incorrect: bool = field(
        default=True,
        metadata={"help": "Whether to create code-IO auxiliary samples for candidates whose execution is incorrect."},
    )
    code_io_aux_include_errors: bool = field(
        default=True,
        metadata={"help": "Whether runtime/compiler errors can be used as auxiliary targets via their error type."},
    )
    code_io_aux_sft_weight_correct: float = field(
        default=1.0,
        metadata={"help": "Supervised-loss weight assigned to correct code-IO auxiliary training samples."},
    )
    code_io_aux_sft_weight_incorrect: float = field(
        default=1.0,
        metadata={"help": "Supervised-loss weight assigned to incorrect code-IO auxiliary training samples."},
    )
    question_prior_enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable question-level EMA prior. This keeps a per-question moving estimate of code success and "
                "reason-signal strength, then uses the derived weight to modulate original-problem keep probability "
                "and iterative-node priority."
            )
        },
    )
    question_prior_ema_momentum: float = field(
        default=0.9,
        metadata={"help": "EMA momentum for per-question code/reason prior statistics."},
    )
    question_prior_high_threshold: float = field(
        default=0.7,
        metadata={"help": "High threshold for question-level EMA code/reason signals."},
    )
    question_prior_low_threshold: float = field(
        default=0.3,
        metadata={"help": "Low threshold for question-level EMA code/reason signals."},
    )
    question_prior_gap_threshold: float = field(
        default=0.2,
        metadata={"help": "Gap threshold for distinguishing 'can reason but cannot code' and the reverse."},
    )
    question_prior_weight_high_value: float = field(
        default=1.0,
        metadata={"help": "Question prior weight for high-reason / low-code questions."},
    )
    question_prior_weight_mid_negative_gap: float = field(
        default=0.7,
        metadata={"help": "Question prior weight for low-reason / high-code questions."},
    )
    question_prior_weight_mastered: float = field(
        default=0.4,
        metadata={"help": "Question prior weight when both code and reason EMA are high."},
    )
    question_prior_weight_too_hard: float = field(
        default=0.2,
        metadata={"help": "Question prior weight when both code and reason EMA are low."},
    )
    question_prior_min_seen_before_too_hard: int = field(
        default=3,
        metadata={"help": "Minimum number of observations before a question can be classified as too_hard."},
    )
    question_prior_weight_default: float = field(
        default=0.8,
        metadata={"help": "Default question prior weight for intermediate cases."},
    )
    question_prior_keep_prob_floor: float = field(
        default=0.05,
        metadata={"help": "Minimum keep probability after applying question-prior downweighting."},
    )
    pseudo_multiround_enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable pseudo-multiround training on top of the single-round zero-pass baseline: "
                "alternate between original problems and an in-memory iterative-node pool."
            )
        },
    )
    pseudo_iterative_pool_capacity: int = field(
        default=2048,
        metadata={"help": "Maximum in-memory iterative-node records kept for pseudo-multiround training."},
    )
    pseudo_iterative_select_count: int = field(
        default=3,
        metadata={"help": "How many iterative candidates to keep when a rollout group has no passing sample."},
    )
    pseudo_iterative_soft_priority_bonus_scale: float = field(
        default=0.1,
        metadata={"help": "Positive soft-reward bonus scale added to repair-shaped iterative-node priority."},
    )
    pseudo_original_downweight_after: int = field(
        default=2,
        metadata={"help": "Start downweighting original problems after this many consecutive no-pass attempts."},
    )
    pseudo_original_keep_prob_decay: float = field(
        default=0.5,
        metadata={"help": "Multiplicative keep-prob decay for original problems after the no-pass threshold."},
    )
    pseudo_original_keep_prob_floor: float = field(
        default=0.1,
        metadata={"help": "Minimum keep probability for original-problem sampling when downweighted."},
    )
    pseudo_original_age_bonus_per_step: float = field(
        default=0.0,
        metadata={"help": "Additive keep-prob bonus per step since this question was last rolled out as an original problem."},
    )
    pseudo_original_age_bonus_max: float = field(
        default=0.0,
        metadata={"help": "Maximum additive keep-prob bonus from original-problem age recovery."},
    )
    pseudo_original_cover_once_before_iterative: bool = field(
        default=False,
        metadata={"help": "Before pseudo-multiround mixing begins, force each original question to be rolled out once as an original problem."},
    )
    pseudo_forced_original_fraction: float = field(
        default=0.0,
        metadata={"help": "Fraction of rollout slots forced to use original problems instead of iterative nodes."},
    )
    pseudo_iterative_ttl_steps: int = field(
        default=0,
        metadata={
            "help": "Drop iterative nodes older than this many trainer steps. 0 disables TTL-based cleanup."
        },
    )
    use_chat_template_for_codegrpo: bool = field(
        default=True,
        metadata={"help": "Whether to render CodeGRPO prompts with tokenizer chat template before generation."},
    )
    beta_sft: float = field(default=1.0, metadata={"help": "Auxiliary SFT loss coefficient."})

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
    sampling_refresh_steps: int = field(
        default=0,
        metadata={
            "help": (
                "Pseudo-epoch refresh interval for question sampling weights. "
                "0 = no refresh (weights frozen at dataloader build time). "
                ">0 = re-read weights every N training steps worth of sampler output. "
                "Used by question-prior weighted sampling."
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
                "No logic/exec audit is run during eval; each round generates one repair candidate and metrics report cumulative pass@1 within <= round r."
            )
        },
    )
    eval_T_max_override: int | None = field(
        default=None,
        metadata={
            "help": (
                "Optional eval-only maximum code-repair rounds. When eval_code_only_single_trajectory=True, "
                "this controls how many consecutive single-trajectory repair rounds are actually executed."
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
    log_trace_dump_events: bool = field(
        default=False,
        metadata={"help": "Whether to print trace dump count messages to the console."},
    )
    log_kl_metrics: bool = field(
        default=True,
        metadata={
            "help": "Compute and log KL divergence as a monitoring metric even when it is not part of the loss."
        },
    )
    log_cuda_memory_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to log CUDA allocated/reserved memory at key training boundaries."},
    )
    log_cuda_memory_logprob_min_seq_len: int = field(
        default=1024,
        metadata={"help": "Only emit HF logprob CUDA memory logs when prompt+target length reaches this threshold."},
    )
    review_bundle_trace_sample_size: int = field(
        default=2,
        metadata={"help": "How many trace files to copy into review_bundle."},
    )

    def __post_init__(self):
        restore_num_generations = None
        restore_num_generations_eval = None
        if self.codegrpo_mode != "train" and self.num_generations is not None and self.num_generations < 2:
            restore_num_generations = self.num_generations
            self.num_generations = 2
            if self.num_generations_eval is None:
                restore_num_generations_eval = self.num_generations_eval
                self.num_generations_eval = 2
        try:
            super().__post_init__()
        finally:
            if restore_num_generations is not None:
                self.num_generations = restore_num_generations
            if restore_num_generations_eval is not None:
                self.num_generations_eval = restore_num_generations_eval
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
        if self.codegrpo_mode == "train":
            if self.K < 2:
                raise ValueError("K must be >= 2 for sibling-group GRPO training.")
            if self.K != self.num_generations:
                raise ValueError(
                    f"K ({self.K}) must equal num_generations ({self.num_generations}) to keep sibling grouping consistent."
                )
        else:
            if self.K < 1:
                raise ValueError("K must be >= 1 for CodeGRPO test/eval mode.")
            if self.K != self.num_generations:
                raise ValueError(
                    f"K ({self.K}) must equal num_generations ({self.num_generations}) in test/eval mode as well."
                )
        if self.eval_repeat_count < 1:
            raise ValueError("eval_repeat_count must be >= 1.")
        if self.dump_train_trace_interval_steps == 0:
            raise ValueError("dump_train_trace_interval_steps must be != 0.")
        if self.max_train_trace_files < 0:
            raise ValueError("max_train_trace_files must be >= 0.")
        if self.review_bundle_trace_sample_size < 0:
            raise ValueError("review_bundle_trace_sample_size must be >= 0.")
        if self.log_cuda_memory_logprob_min_seq_len < 0:
            raise ValueError("log_cuda_memory_logprob_min_seq_len must be >= 0.")
        if self.undiff_retry_max < 0:
            raise ValueError("undiff_retry_max must be non-negative.")
        if self.sampling_refresh_steps < 0:
            raise ValueError("sampling_refresh_steps must be non-negative.")
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
        if not (0.0 <= self.soft_reward_ineligible_scale <= 1.0):
            raise ValueError(
                f"soft_reward_ineligible_scale must be in [0, 1], got: {self.soft_reward_ineligible_scale}"
            )
        if self.generation_outside_noise_chars < 0:
            raise ValueError("generation_outside_noise_chars must be >= 0.")
        if self.max_completion_length_code is not None and self.max_completion_length_code <= 0:
            raise ValueError("max_completion_length_code must be > 0 when provided.")
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
        if self.train_generation_truncate_tokens < 0:
            raise ValueError("train_generation_truncate_tokens must be >= 0.")
        if self.train_generation_total_token_cap < 0:
            raise ValueError("train_generation_total_token_cap must be >= 0.")
        if self.train_generation_completion_reserve_tokens < 0:
            raise ValueError("train_generation_completion_reserve_tokens must be >= 0.")
        if (
            self.train_generation_total_token_cap > 0
            and self.train_generation_completion_reserve_tokens > self.train_generation_total_token_cap
        ):
            raise ValueError(
                "train_generation_completion_reserve_tokens must be <= train_generation_total_token_cap."
            )
        if self.zero_pass_soft_reward_diag_count < 0:
            raise ValueError("zero_pass_soft_reward_diag_count must be >= 0.")
        if self.zero_pass_soft_reward_clip_low >= self.zero_pass_soft_reward_clip_high:
            raise ValueError("zero_pass_soft_reward_clip_low must be < zero_pass_soft_reward_clip_high.")
        if self.zero_pass_soft_reward_beta_scale < 0:
            raise ValueError("zero_pass_soft_reward_beta_scale must be >= 0.")
        if self.code_io_aux_case_count < 0:
            raise ValueError("code_io_aux_case_count must be >= 0.")
        if self.code_io_aux_sft_weight_correct < 0:
            raise ValueError("code_io_aux_sft_weight_correct must be >= 0.")
        if self.code_io_aux_sft_weight_incorrect < 0:
            raise ValueError("code_io_aux_sft_weight_incorrect must be >= 0.")
        if not (0.0 <= self.question_prior_ema_momentum < 1.0):
            raise ValueError("question_prior_ema_momentum must be in [0, 1).")
        if not (0.0 <= self.question_prior_low_threshold <= 1.0):
            raise ValueError("question_prior_low_threshold must be in [0, 1].")
        if not (0.0 <= self.question_prior_high_threshold <= 1.0):
            raise ValueError("question_prior_high_threshold must be in [0, 1].")
        if self.question_prior_low_threshold > self.question_prior_high_threshold:
            raise ValueError("question_prior_low_threshold must be <= question_prior_high_threshold.")
        if self.question_prior_gap_threshold < 0:
            raise ValueError("question_prior_gap_threshold must be >= 0.")
        if self.question_prior_weight_high_value < 0:
            raise ValueError("question_prior_weight_high_value must be >= 0.")
        if self.question_prior_weight_mid_negative_gap < 0:
            raise ValueError("question_prior_weight_mid_negative_gap must be >= 0.")
        if self.question_prior_weight_mastered < 0:
            raise ValueError("question_prior_weight_mastered must be >= 0.")
        if self.question_prior_weight_too_hard < 0:
            raise ValueError("question_prior_weight_too_hard must be >= 0.")
        if self.question_prior_min_seen_before_too_hard < 0:
            raise ValueError("question_prior_min_seen_before_too_hard must be >= 0.")
        if self.pseudo_original_age_bonus_per_step < 0:
            raise ValueError("pseudo_original_age_bonus_per_step must be >= 0.")
        if self.pseudo_original_age_bonus_max < 0:
            raise ValueError("pseudo_original_age_bonus_max must be >= 0.")
        if not (0.0 <= self.pseudo_forced_original_fraction <= 1.0):
            raise ValueError("pseudo_forced_original_fraction must be in [0, 1].")
        if self.question_prior_weight_default < 0:
            raise ValueError("question_prior_weight_default must be >= 0.")
        if not (0.0 <= self.question_prior_keep_prob_floor <= 1.0):
            raise ValueError("question_prior_keep_prob_floor must be in [0, 1].")
        if self.pseudo_iterative_pool_capacity < 0:
            raise ValueError("pseudo_iterative_pool_capacity must be >= 0.")
        if self.pseudo_iterative_select_count < 0:
            raise ValueError("pseudo_iterative_select_count must be >= 0.")
        if self.pseudo_iterative_soft_priority_bonus_scale < 0:
            raise ValueError("pseudo_iterative_soft_priority_bonus_scale must be >= 0.")
        if self.pseudo_original_downweight_after < 0:
            raise ValueError("pseudo_original_downweight_after must be >= 0.")
        if not (0.0 <= self.pseudo_original_keep_prob_decay <= 1.0):
            raise ValueError("pseudo_original_keep_prob_decay must be in [0, 1].")
        if not (0.0 <= self.pseudo_original_keep_prob_floor <= 1.0):
            raise ValueError("pseudo_original_keep_prob_floor must be in [0, 1].")
        if self.pseudo_iterative_ttl_steps < 0:
            raise ValueError("pseudo_iterative_ttl_steps must be >= 0.")
        if self.max_completion_length_code is not None and self.max_completion_length_code > self.max_completion_length:
            raise ValueError(
                "max_completion_length_code must be <= max_completion_length so shared generation backends remain valid."
            )
        if (
            self.max_completion_length_code is not None
            and self.generation_min_new_tokens_code > self.max_completion_length_code
        ):
            raise ValueError("generation_min_new_tokens_code must be <= max_completion_length_code.")
        if self.generation_min_new_tokens_code > self.max_completion_length:
            raise ValueError("generation_min_new_tokens_code must be <= max_completion_length.")
        if self.debug_trace_sample_size < 0:
            raise ValueError("debug_trace_sample_size must be >= 0.")
