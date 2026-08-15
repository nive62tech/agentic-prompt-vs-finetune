"""
QLoRA fine-tuning for Tier 3 (error recovery).
Phase 9: --seed argument added, --n now accepts 50 as well as 10
(Tier 3's training pool has ~66 non-overlapping examples, enough for N=50).
Run: !python qlora_finetune_tier3.py --n 10 --seed 42
Saves the adapter to adapters/tier3_n{N}_seed{seed}/.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

from envs.training_data import sample_tier3_training
from envs.tools import call_tool

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def build_chat_example(tokenizer, task: dict) -> str:
    trigger_result = call_tool(task["trigger_tool"], task["trigger_args"])
    target_call = json.dumps({"name": task["trigger_tool"], "parameters": task["trigger_args"]})
    final_answer = f"I'm sorry, I couldn't complete this — {task['recovery_keywords'][0]}. Please check and try again."

    messages = [
        {"role": "user", "content": task["prompt"]},
        {"role": "assistant", "content": target_call},
        {"role": "tool", "content": json.dumps(trigger_result)},
        {"role": "assistant", "content": final_answer},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def build_dataset(tokenizer, training_examples: list, max_length: int = 768) -> Dataset:
    texts = [build_chat_example(tokenizer, ex) for ex in training_examples]

    def tokenize_fn(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=max_length, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    ds = Dataset.from_dict({"text": texts})
    return ds.map(tokenize_fn, batched=True, remove_columns=["text"])


def main(n: int, seed: int, epochs: int = 3, output_dir: str = None):
    output_dir = output_dir or f"adapters/tier3_n{n}_seed{seed}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Sampling {n} Tier 3 training examples (seed={seed})...")
    training_examples = sample_tier3_training(n, seed=seed)
    with open(os.path.join(output_dir, "training_examples.json"), "w") as f:
        json.dump(training_examples, f, indent=2)

    print("Loading base model (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map={"": 0}
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Building dataset...")
    train_ds = build_dataset(tokenizer, training_examples)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Training...")
    trainer.train()

    print(f"Saving adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True, choices=[10, 50], help="Data regime")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data sampling + training")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    main(n=args.n, seed=args.seed, epochs=args.epochs)
