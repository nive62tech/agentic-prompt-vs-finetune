# envs/pilot_test.py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch, json, gc

MODELS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    }
]

TEST_PROMPTS = [
    "What's the weather like in Tokyo right now?",
    "Is it raining in London?",
    "Tell me the temperature in New York.",
    "Check the weather for Mumbai.",
    "How hot is it in Dubai today?",
    "What's the forecast for Paris?",
    "Is Berlin cold right now?",
    "Weather in Sydney please.",
    "Can you check Singapore's weather?",
    "What's the current weather in Cairo?",
]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

def run_model(model_id, prompts, tools):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    results = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
        inputs = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=200)
        decoded = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append({"prompt": prompt, "output": decoded})
        print(f"  done: {prompt[:40]}")

    # free GPU memory before loading the next model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results

if __name__ == "__main__":
    all_results = {}
    for name, model_id in MODELS.items():
        print(f"Running {name}...")
        all_results[name] = run_model(model_id, TEST_PROMPTS, TOOLS)
        with open("results/pilot_results.json", "w") as f:
            json.dump(all_results, f, indent=2)  # save after each model, not just at the end
    print("Done — check results/pilot_results.json")