"""Code GRPO extension package with lazy exports."""

from importlib import import_module

_EXPORTS = {
    "Backend": ".backends",
    "HFBackend": ".backends",
    "VLLMBackend": ".backends",
    "build_backend": ".backends",
    "summarize_error": ".error_utils",
    "execute": ".executor",
    "is_match": ".matcher",
    "build_generation_completion": ".parser",
    "build_token_masks": ".parser",
    "parse_generation_output": ".parser",
    "parse_generation_response": ".parser",
    "build_diagnostic_inputs": ".soft_reward",
    "compute_soft_reward": ".soft_reward",
    "compute_zero_pass_beta": ".soft_reward",
    "get_oracle_outputs": ".soft_reward",
    "normalize_soft_reward_to_unit_interval": ".soft_reward",
    "CodeGRPOTreeRunner": ".tree",
    "ExecResult": ".types",
    "Node": ".types",
    "QuestionRollout": ".types",
    "TrainSample": ".types",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    return getattr(module, name)
