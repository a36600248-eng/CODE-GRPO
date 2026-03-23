from abc import ABC, abstractmethod

import torch

_SUPPORTED_VLLM_OVERRIDES = {
    "max_new_tokens",
    "min_new_tokens",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
}


def _truncate_at_stop_strings(text: str, stop_strings: list[str] | tuple[str, ...] | None) -> str:
    if not text or not stop_strings:
        return text
    best_end = None
    for stop in stop_strings:
        if not stop:
            continue
        idx = text.find(stop)
        if idx < 0:
            continue
        end = idx + len(stop)
        best_end = end if best_end is None else min(best_end, end)
    return text[:best_end] if best_end is not None else text


def _safe_model_max_length(tokenizer) -> int | None:
    value = getattr(tokenizer, "model_max_length", None)
    if not isinstance(value, int):
        return None
    if value <= 0 or value >= 10_000_000:
        return None
    return value


class Backend(ABC):
    @abstractmethod
    def generate(self, prompt: str, **gen_cfg) -> str:
        pass

    @abstractmethod
    def generate_many(self, prompts: list[str], **gen_cfg) -> list[str]:
        pass

    @abstractmethod
    def logprob(self, prompt: str, target_text: str, **cfg) -> float:
        pass


class HFBackend(Backend):
    def __init__(
        self,
        model,
        tokenizer,
        device,
        generation_defaults: dict | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.generation_defaults = generation_defaults or {}

    def generate(self, prompt: str, **gen_cfg) -> str:
        cfg = {**self.generation_defaults, **gen_cfg}
        stop_strings = cfg.pop("stop_strings", None)
        if float(cfg.get("temperature", 1.0) or 0.0) <= 0.0:
            cfg["do_sample"] = False
            cfg.pop("temperature", None)
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            output_ids = self.model.generate(**encoded, **cfg)
        completion_ids = output_ids[:, encoded["input_ids"].shape[1] :]
        decoded = self.tokenizer.decode(completion_ids[0], skip_special_tokens=True)
        return _truncate_at_stop_strings(decoded, stop_strings)

    def generate_many(self, prompts: list[str], **gen_cfg) -> list[str]:
        if not prompts:
            return []
        cfg = {**self.generation_defaults, **gen_cfg}
        stop_strings = cfg.pop("stop_strings", None)
        num_generations = int(cfg.pop("num_generations", 1) or 1)
        if float(cfg.get("temperature", 1.0) or 0.0) <= 0.0:
            cfg["do_sample"] = False
            cfg.pop("temperature", None)
        expanded_prompts: list[str] = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * num_generations)
        encoded = self.tokenizer(expanded_prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        prompt_width = encoded["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = self.model.generate(**encoded, **cfg)
        completion_ids = output_ids[:, prompt_width:]
        return [
            _truncate_at_stop_strings(self.tokenizer.decode(row, skip_special_tokens=True), stop_strings)
            for row in completion_ids
        ]

    def logprob(self, prompt: str, target_text: str, **cfg) -> float:
        del cfg
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
        target_ids = self.tokenizer(target_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        prompt_len = prompt_ids.shape[1]
        target_len = target_ids.shape[1]
        if target_len == 0:
            return 0.0

        max_supported = _safe_model_max_length(self.tokenizer)
        if max_supported is not None and (prompt_len + target_len) > max_supported:
            return float("nan")

        prompt_ids = prompt_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        input_ids = torch.cat([prompt_ids, target_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)
        try:
            with torch.no_grad():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return float("nan")
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)

        positions = torch.arange(prompt_len - 1, prompt_len - 1 + target_len, device=self.device)
        selected_logits = log_probs[:, positions, :]
        target = target_ids.unsqueeze(-1)
        selected = torch.gather(selected_logits, dim=-1, index=target).squeeze(-1)
        return selected.mean().item()


class VLLMBackend(Backend):
    def __init__(self, vllm_generation, tokenizer, hf_fallback: HFBackend):
        self.vllm_generation = vllm_generation
        self.tokenizer = tokenizer
        self.hf_fallback = hf_fallback

    @staticmethod
    def _split_vllm_overrides(gen_cfg: dict) -> tuple[dict, dict]:
        overrides = {}
        passthrough = {}
        for key, value in gen_cfg.items():
            if key in _SUPPORTED_VLLM_OVERRIDES:
                if key == "max_new_tokens":
                    overrides["max_completion_length"] = int(value)
                elif key == "min_new_tokens":
                    overrides["min_tokens"] = int(value)
                else:
                    overrides[key] = value
            else:
                passthrough[key] = value
        return overrides, passthrough

    def generate(self, prompt: str, **gen_cfg) -> str:
        stop_strings = gen_cfg.pop("stop_strings", None)
        generation_overrides, passthrough = self._split_vllm_overrides(gen_cfg)
        if passthrough:
            return self.hf_fallback.generate(prompt, stop_strings=stop_strings, **gen_cfg)
        prompt_ids, completion_ids, _, _, _ = self.vllm_generation.generate(
            [prompt],
            num_generations=1,
            generation_overrides=generation_overrides or None,
        )
        if not completion_ids:
            return ""
        decoded = self.tokenizer.decode(completion_ids[0], skip_special_tokens=True)
        return _truncate_at_stop_strings(decoded, stop_strings)

    def generate_many(self, prompts: list[str], **gen_cfg) -> list[str]:
        if not prompts:
            return []
        stop_strings = gen_cfg.pop("stop_strings", None)
        num_generations = int(gen_cfg.pop("num_generations", 1) or 1)
        generation_overrides, passthrough = self._split_vllm_overrides(gen_cfg)
        if passthrough:
            # vLLM wrapper is configured at initialization time; for per-call overrides, fallback to HF path.
            expanded_prompts: list[str] = []
            for prompt in prompts:
                expanded_prompts.extend([prompt] * num_generations)
            return [self.hf_fallback.generate(prompt, stop_strings=stop_strings, **gen_cfg) for prompt in expanded_prompts]
        _, completion_ids, _, _, _ = self.vllm_generation.generate(
            prompts,
            num_generations=num_generations,
            generation_overrides=generation_overrides or None,
        )
        if not completion_ids:
            return ["" for _ in range(len(prompts) * num_generations)]
        decoded = [
            _truncate_at_stop_strings(self.tokenizer.decode(ids, skip_special_tokens=True), stop_strings)
            for ids in completion_ids
        ]
        expected = len(prompts) * num_generations
        if len(decoded) < expected:
            decoded.extend(["" for _ in range(expected - len(decoded))])
        return decoded[:expected]

    def logprob(self, prompt: str, target_text: str, **cfg) -> float:
        # vLLM generation path does not provide teacher-forcing logprob consistently; use HF model forward path.
        return self.hf_fallback.logprob(prompt, target_text, **cfg)


def build_backend(
    backend_name: str,
    model,
    tokenizer,
    device,
    generation_defaults: dict | None = None,
    vllm_generation=None,
) -> Backend:
    hf_backend = HFBackend(model=model, tokenizer=tokenizer, device=device, generation_defaults=generation_defaults)
    if backend_name == "hf":
        return hf_backend
    if backend_name == "vllm":
        if vllm_generation is None:
            raise ValueError("backend='vllm' requires initialized vLLM generation backend.")
        return VLLMBackend(vllm_generation=vllm_generation, tokenizer=tokenizer, hf_fallback=hf_backend)
    raise ValueError(f"Unknown backend '{backend_name}'. Expected one of: hf, vllm.")
