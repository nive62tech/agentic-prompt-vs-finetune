"""
Agent harness — the missing link between a served model and grader.py.

Three responsibilities:
  1. Load the model (4-bit quantized, fits a Colab T4).
  2. Run a multi-turn tool-use loop: generate -> parse tool call -> execute
     against the simulator -> feed result back -> repeat until the model
     gives a final non-tool-call answer (or max_turns is hit).
  3. Parse the model's raw text into the {"tool":..., "args":...} format
     grader.py expects.

This file is split into two parts on purpose:
  - PARSING (top half): pure Python, no heavy deps, works anywhere
    (including this sandbox, for self-testing).
  - MODEL / GENERATION (bottom half): needs torch + transformers +
    bitsandbytes, only runs on Colab (GPU). Imports are wrapped in
    try/except so the parsing half stays testable without those deps
    installed.

USAGE (in Colab, after loading model + tokenizer):
    from envs.agent_harness import run_agent
    from envs.tools import TOOL_SCHEMAS, call_tool

    tool_calls, final_text = run_agent(model, tok, task["prompt"], TOOL_SCHEMAS, call_tool)
    result = grade_task(task, tier=1, tool_calls=tool_calls, final_text=final_text)
"""

import json
import re

# ---------------------------------------------------------------------------
# PARSING — no heavy deps, testable anywhere
# ---------------------------------------------------------------------------

def extract_json_objects(text: str) -> list:
    """Find every balanced top-level {...} substring in text and parse it
    as JSON. Skips anything that isn't valid JSON rather than raising."""
    objs = []
    stack = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and start is not None:
                    candidate = text[start:i + 1]
                    try:
                        objs.append(json.loads(candidate))
                    except json.JSONDecodeError:
                        pass
                    start = None
    return objs


def parse_tool_call(text: str):
    """
    Extract a single tool call from raw model output, if present.
    Handles both observed formats:
      - Llama-style:  {"name": "get_weather", "parameters": {"city": "Tokyo"}}
      - Qwen-style:   <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    Returns {"tool": str, "args": dict} or None if no valid tool call found
    (i.e. this is a final natural-language answer).
    """
    cleaned = re.sub(r"</?tool_call>", "", text)
    for obj in extract_json_objects(cleaned):
        if isinstance(obj, dict) and "name" in obj:
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, dict):
                return {"tool": obj["name"], "args": args}
    return None


# ---------------------------------------------------------------------------
# MODEL / GENERATION — needs torch + transformers + bitsandbytes (Colab only)
# ---------------------------------------------------------------------------

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    _HEAVY_DEPS_AVAILABLE = True
except ImportError:
    _HEAVY_DEPS_AVAILABLE = False


def load_model(model_id: str = "meta-llama/Llama-3.1-8B-Instruct"):
    """4-bit quantized load — fits comfortably in a T4's 16GB."""
    if not _HEAVY_DEPS_AVAILABLE:
        raise RuntimeError("torch/transformers/bitsandbytes not installed — run this in Colab.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    return model, tok


def run_agent(model, tok, prompt: str, tool_schemas: list, tool_call_fn,
              max_turns: int = 4, max_new_tokens: int = 300):
    """
    Multi-turn agent loop.

    tool_call_fn: a function(tool_name: str, args: dict) -> dict, i.e.
                  envs.tools.call_tool. Injected as a parameter (not
                  imported directly) so this harness stays decoupled
                  from the specific tool simulators.

    Returns: (tool_calls: list[{"tool", "args"}], final_text: str)
    """
    if not _HEAVY_DEPS_AVAILABLE:
        raise RuntimeError("torch/transformers not installed — run this in Colab.")

    messages = [{"role": "user", "content": prompt}]
    tool_calls_made = []
    last_raw_text = ""

    for _ in range(max_turns):
        chat_text = tok.apply_chat_template(
            messages, tools=tool_schemas, add_generation_prompt=True, tokenize=False
        )
        inputs = tok(chat_text, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        decoded = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        last_raw_text = decoded

        call = parse_tool_call(decoded)
        if call is None:
            return tool_calls_made, decoded  # final natural-language answer

        tool_calls_made.append(call)
        result = tool_call_fn(call["tool"], call["args"])
        messages.append({"role": "assistant", "content": decoded})
        messages.append({"role": "tool", "content": json.dumps(result)})

    # hit max_turns without a final answer — return what we have
    return tool_calls_made, last_raw_text


# ---------------------------------------------------------------------------
# Self-test — parsing logic only, no GPU needed. Run: python envs/agent_harness.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        # Llama-style, clean
        ('{"name": "get_weather", "parameters": {"city": "Tokyo"}}',
         {"tool": "get_weather", "args": {"city": "Tokyo"}}),

        # Qwen-style, wrapped in tags
        ('<tool_call>\n{"name": "get_weather", "arguments": {"city": "London"}}\n</tool_call>',
         {"tool": "get_weather", "args": {"city": "London"}}),

        # Prose + tool call together (Qwen sometimes does this)
        ('I can check that for you.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
         {"tool": "get_weather", "args": {"city": "Paris"}}),

        # Final answer, no tool call at all
        ("Sorry, I couldn't find weather data for that city.", None),

        # Multi-arg call
        ('{"name": "convert_currency", "parameters": {"amount": 100, "base": "USD", "target": "EUR"}}',
         {"tool": "convert_currency", "args": {"amount": 100, "base": "USD", "target": "EUR"}}),

        # Malformed JSON should not crash, should just return None
        ('{"name": "get_weather", "parameters": {city: Tokyo}}', None),
    ]

    print("=== parse_tool_call self-test ===")
    all_pass = True
    for i, (text, expected) in enumerate(test_cases):
        result = parse_tool_call(text)
        ok = result == expected
        all_pass = all_pass and ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] case {i}: got {result}")
        if not ok:
            print(f"         expected {expected}")

    print(f"\n{'All parsing tests passed.' if all_pass else 'SOME TESTS FAILED — check above.'}")
    print(f"Heavy deps (torch/transformers) available: {_HEAVY_DEPS_AVAILABLE}")
