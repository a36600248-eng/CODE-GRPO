from importlib import import_module

__all__ = ["code_grpo"]


def __getattr__(name: str):
    if name == "code_grpo":
        return import_module(".code_grpo", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
