"""
QLoRA fine-tuning for Tier 1 (single tool call), matching DSPy's scope
for a fair comparison at the same data regimes (N=10, N=50).

Trains a LoRA adapter on top of the 4-bit quantized Llama-3.1-8B-Instruct
using PEFT. Each training example is formatted as a full chat turn
(user request -> correct tool-call JSON), same target format the
baseline harness and DSPy program both use — so all three conditions
are compared on identical output format, not different conventions.

Run this in Colab: `!python qlora_finetune.py --n 10` or `--n 50`.
Saves the adapter to `adapters/tier1_n{N}/`.
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

from envs.training_data import sample_tier1_training

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def build_chat_example(tokenizer, task: dict) -> str:
    """Formats one training example as a full chat turn ending in the
    correct tool-call JSON, matching the format the model produces at
    inference time (see envs/agent_harness.py's parse_tool_call)."""
    target = json.dumps({"name": task["expected_tool"], "parameters": task["expected_args"]})
    messages = [
        {"role": "user", "content": task["prompt"]},
        {"role": "assistant", "content": target},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def build_dataset(tokenizer, training_examples: list, max_length: int = 512) -> Dataset:
    texts = [build_chat_example(tokenizer, ex) for ex in training_examples]

    def tokenize_fn(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=max_length, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    ds = Dataset.from_dict({"text": texts})
    return ds.map(tokenize_fn, batched=True, remove_columns=["text"])


def main(n: int, epochs: int = 3, output_dir: str = None):
    output_dir = output_dir or f"adapters/tier1_n{n}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Sampling {n} training examples...")
    training_examples = sample_tier1_training(n)
    with open(os.path.join(output_dir, "training_examples.json"), "w") as f:
        json.dump(training_examples, f, indent=2)

    print("Loading base model (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto"
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
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
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
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    main(n=args.n, epochs=args.epochs)
