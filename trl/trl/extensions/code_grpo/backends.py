from abc import ABC, abstractmethod

import torch


class Backend(ABC):
    @abstractmethod
    def generate(self, prompt: str, **gen_cfg) -> str:
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
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            output_ids = self.model.generate(**encoded, **cfg)
        completion_ids = output_ids[:, encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(completion_ids[0], skip_special_tokens=True)

    def logprob(self, prompt: str, target_text: str, **cfg) -> float:
        del cfg
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        target_ids = self.tokenizer(target_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(
            self.device
        )
        input_ids = torch.cat([prompt_ids, target_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)

        prompt_len = prompt_ids.shape[1]
        target_len = target_ids.shape[1]
        if target_len == 0:
            return 0.0

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

    def generate(self, prompt: str, **gen_cfg) -> str:
        del gen_cfg
        prompt_ids, completion_ids, _, _, _ = self.vllm_generation.generate([prompt], num_generations=1)
        if not completion_ids:
            return ""
        return self.tokenizer.decode(completion_ids[0], skip_special_tokens=True)

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

