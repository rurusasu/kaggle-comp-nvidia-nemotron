"""
Nemotron-3-Nano-30B LoRA Fine-Tuning Script
============================================
Run this in a Kaggle notebook with GPU enabled.
Requires: transformers, peft, trl, datasets, bitsandbytes
"""

import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "trl", "peft", "bitsandbytes", "datasets", "accelerate"])

import json
import os
import zipfile

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import SFTTrainer

# ============================================================
# Debug: List available input data
# ============================================================
print("=== /kaggle/input contents ===")
for root, dirs, files in os.walk("/kaggle/input"):
    level = root.replace("/kaggle/input", "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    if level < 2:
        subindent = " " * 2 * (level + 1)
        for file in files[:10]:
            print(f"{subindent}{file}")
        if len(files) > 10:
            print(f"{subindent}... and {len(files) - 10} more files")

# ============================================================
# Configuration
# ============================================================
# Auto-detect paths
def find_file(base_dir, filename):
    for root, dirs, files in os.walk(base_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

BASE_MODEL = find_file("/kaggle/input", "config.json")
if BASE_MODEL:
    BASE_MODEL = os.path.dirname(BASE_MODEL)
else:
    BASE_MODEL = "/kaggle/input/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"
print(f"Using base model: {BASE_MODEL}")

TRAIN_CSV = find_file("/kaggle/input", "train.csv")
if not TRAIN_CSV:
    raise FileNotFoundError("train.csv not found in /kaggle/input/")
print(f"Using train CSV: {TRAIN_CSV}")

OUTPUT_DIR = "/kaggle/working/lora_output"
SUBMISSION_ZIP = "/kaggle/working/submission.zip"

LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

NUM_EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = 2e-4
MAX_SEQ_LEN = 2048
SEED = 42

SYSTEM_PROMPT = (
    "You are a precise reasoning assistant. Solve the given puzzle step by step. "
    "Think carefully about the pattern, then provide your final answer inside \\boxed{}."
)


# ============================================================
# Data Preparation
# ============================================================
def format_example(row):
    """Convert a training row to chat format."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": f"Let me analyze this step by step.\n\nAfter careful analysis of the pattern, the answer is:\n\n\\boxed{{{row['answer']}}}"},
    ]
    return {"messages": messages}


def prepare_dataset():
    """Load and prepare the training dataset."""
    df = pd.read_csv(TRAIN_CSV)
    records = [format_example(row) for _, row in df.iterrows()]
    return Dataset.from_list(records)


# ============================================================
# Model Setup
# ============================================================
def setup_model_and_tokenizer():
    """Load base model with quantization and apply LoRA."""
    # Quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ============================================================
# Training
# ============================================================
def train():
    """Run the full training pipeline."""
    print("Preparing dataset...")
    dataset = prepare_dataset()
    print(f"Dataset size: {len(dataset)}")

    print("Loading model...")
    model, tokenizer = setup_model_and_tokenizer()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        seed=SEED,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        gradient_checkpointing=True,
        report_to="none",
        max_grad_norm=1.0,
        weight_decay=0.01,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LEN,
    )

    print("Starting training...")
    trainer.train()

    print("Saving adapter...")
    trainer.save_model(OUTPUT_DIR)

    return OUTPUT_DIR


# ============================================================
# Submission Packaging
# ============================================================
def package_submission(adapter_dir):
    """Package the LoRA adapter into submission.zip."""
    with zipfile.ZipFile(SUBMISSION_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(adapter_dir):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, adapter_dir)
                zf.write(filepath, arcname)
    print(f"Submission saved to {SUBMISSION_ZIP}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    adapter_dir = train()
    package_submission(adapter_dir)
    print("Done!")
