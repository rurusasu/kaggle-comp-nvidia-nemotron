"""Training entrypoint for the Nemotron LoRA fine-tuning competition.

This script:
1. Analyzes the training data
2. Prepares the SFT dataset in chat format
3. Generates the Kaggle training script
4. Saves a sample adapter_config.json for reference

Actual GPU training must be done on Kaggle or a GPU cloud instance.

Usage:
    uv run python scripts/train.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.dataset import load_train, prepare_sft_dataset, classify_task
from src.evaluate import log_experiment
from src.features import analyze_training_data, get_task_summary
from src.model import save_adapter_config, generate_training_script, get_lora_config_dict
from src.utils import Timer, set_seed


def main():
    cfg = Config()
    set_seed(cfg.seed)

    # Step 1: Load and analyze training data
    with Timer("load data"):
        train_df = load_train(cfg)
    print(f"Training data: {len(train_df)} examples")
    print(f"Columns: {train_df.columns.tolist()}")
    print()

    with Timer("analyze data"):
        analyzed = analyze_training_data(train_df, cfg)
        summary = get_task_summary(analyzed)
        print("Task type summary:")
        print(summary)
        print()

    # Step 2: Prepare SFT dataset
    with Timer("prepare SFT dataset"):
        sft_data = prepare_sft_dataset(cfg)
        print(f"SFT dataset: {len(sft_data)} examples")
        print(f"Sample message format:")
        sample = sft_data[0]["messages"]
        for msg in sample:
            print(f"  [{msg['role']}] {msg['content'][:100]}...")
        print()

    # Save SFT data as JSONL for reference
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    sft_path = cfg.processed_dir / "train_sft.jsonl"
    with open(sft_path, "w", encoding="utf-8") as f:
        for example in sft_data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"SFT data saved to {sft_path}")

    # Step 3: Save adapter config
    config_path = save_adapter_config(cfg)
    print(f"Adapter config saved to {config_path}")

    # Step 4: Generate Kaggle training script
    kaggle_script = generate_training_script(cfg)
    kaggle_script_path = cfg.output_dir / "kaggle_training_notebook.py"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    kaggle_script_path.write_text(kaggle_script)
    print(f"Kaggle training script saved to {kaggle_script_path}")

    # Step 5: Log experiment
    lora_config = get_lora_config_dict(cfg)
    log_experiment(
        cfg,
        {
            "experiment": "baseline_sft",
            "num_examples": len(sft_data),
            "task_distribution": summary["count"].to_dict(),
            "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha,
            "learning_rate": cfg.learning_rate,
            "num_epochs": cfg.num_train_epochs,
            "notes": "Baseline SFT with direct answer in boxed format. No chain-of-thought reasoning yet.",
        },
    )

    print()
    print("=" * 60)
    print("BASELINE SETUP COMPLETE")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Upload to Kaggle notebook with GPU access")
    print("2. Run the training script (kaggle_training_notebook.py)")
    print("3. Download the submission.zip and submit")
    print()
    print("Improvement ideas:")
    print("- Add chain-of-thought reasoning in assistant responses")
    print("- Augment training data with more examples per task type")
    print("- Use task-specific system prompts")
    print("- Experiment with different LoRA hyperparameters")
    print("- Add reinforcement learning (GRPO/DPO) after SFT")


if __name__ == "__main__":
    main()
