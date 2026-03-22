"""Feature engineering / data analysis for the reasoning competition.

This module provides analysis tools to understand the training data patterns.
Since this is a LoRA fine-tuning competition, traditional feature engineering
is replaced by data analysis and prompt engineering.
"""

import re

import pandas as pd

from src.config import Config
from src.dataset import classify_task


def analyze_training_data(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Analyze training data and add task type classification."""
    df = df.copy()
    df["task_type"] = df["prompt"].apply(classify_task)
    df["prompt_length"] = df["prompt"].str.len()
    df["answer_length"] = df["answer"].astype(str).str.len()
    df["num_examples"] = df["prompt"].apply(_count_examples)
    return df


def _count_examples(prompt: str) -> int:
    """Count the number of input->output examples in a prompt."""
    # Count arrow patterns like "-> " or "becomes"
    arrows = len(re.findall(r"->|becomes", prompt))
    return arrows


def get_task_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Get summary statistics per task type."""
    if "task_type" not in df.columns:
        df = df.copy()
        df["task_type"] = df["prompt"].apply(classify_task)

    summary = df.groupby("task_type").agg(
        count=("id", "count"),
        avg_prompt_len=("prompt", lambda x: x.str.len().mean()),
        avg_answer_len=("answer", lambda x: x.astype(str).str.len().mean()),
    )
    return summary
