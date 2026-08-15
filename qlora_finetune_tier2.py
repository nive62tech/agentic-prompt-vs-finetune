"""
QLoRA fine-tuning for Tier 2 (multi-tool chains).
Phase 9: --seed argument added (used for training-sample selection and
training reproducibility), and --n now accepts 50 as well as 10.
Run: !python qlora_finetune_tier2.py --n 10 --seed 42
Saves the adapter to adapters/tier2_n{N}_seed{seed}/.
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

from envs.training_data import sample_tier2_training
from grader import resolve_expected_sequence
from envs.tools import call_tool

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def build_chat_example(tokenizer, task: dict) -> str:
    resolved = resolve_expected_sequence(task)
    messages = [{"role": "user", "content": task["prompt"]}]
    for step in resolved:
        result = call_tool(step["tool"], step["args"])
        target = json.dumps({"name": step["tool"], "parameters": step["args"]})
        messages.append({"role": "assistant", "content": target})
        messages.append({"role": "tool", "content": json.dumps(result)})
    return tokenizer.apply_chat_template(messages, tokenize=False)


def build_dataset(tokenizer, training_examples: list, max_length: int = 1024) -> Dataset:
    texts = [build_chat_example(tokenizer, ex) for ex in training_examples]

    def tokenize_fn(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=max_length, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    ds = Dataset.from_dict({"text": texts})
    return ds.map(tokenize_fn, batched=True, remove_columns=["text"])


def main(n: int, seed: int, epochs: int = 3, output_dir: str = None):
    output_dir = output_dir or f"adapters/tier2_n{n}_seed{seed}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Sampling {n} Tier 2 training examples (seed={seed})...")
    training_examples = sample_tier2_training(n, seed=seed)
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
