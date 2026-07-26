"""
DSPy LM adapter for the locally-served, 4-bit quantized Llama model.

DSPy needs an object it can call like an LM to run its optimizer (MIPROv2)
against — this wraps our already-loaded `model`/`tok` (from
envs.agent_harness.load_model) so DSPy drives generation through the same
model we used for the baseline, no separate API needed.
"""

import dspy
import torch


class LocalLlamaLM(dspy.LM):
    """
    Minimal custom DSPy LM. DSPy 2.5+'s LM base class expects __call__ to
    accept either `prompt` (str) or `messages` (list of role/content dicts)
    and return a list of completion strings.
    """

    def __init__(self, model, tokenizer, max_new_tokens: int = 300):
        super().__init__(model="local-llama-3.1-8b-instruct")
        self.hf_model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt=None, messages=None, **kwargs):
        if messages is not None:
            chat_text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            chat_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False,
            )

        inputs = self.tokenizer(chat_text, return_tensors="pt").to(self.hf_model.device)
        out = self.hf_model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_tokens", self.max_new_tokens),
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        decoded = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        del inputs, out
        torch.cuda.empty_cache()
        return [decoded]