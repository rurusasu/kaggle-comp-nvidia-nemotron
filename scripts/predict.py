"""Local analysis and evaluation script.

Since actual inference requires GPU + the Nemotron model, this script:
1. Tests the answer extraction logic on training data
2. Validates the evaluation metric implementation
3. Reports expected baseline performance characteristics

Usage:
    uv run python scripts/predict.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.dataset import load_train, classify_task
from src.evaluate import extract_boxed_answer, extract_answer, is_correct, compute_accuracy
from src.utils import Timer, set_seed


def main():
    cfg = Config()
    set_seed(cfg.seed)

    with Timer("load data"):
        train_df = load_train(cfg)

    # Test answer extraction logic
    print("Testing answer extraction logic...")
    print()

    test_cases = [
        (r"The answer is \boxed{42}", "42"),
        (r"After analysis, we get \boxed{XXXVIII}", "XXXVIII"),
        (r"Step 1: ... Step 2: ... \boxed{10010111}", "10010111"),
        (r"\boxed{cat imagines book}", "cat imagines book"),
        (r"The result is \boxed{16.65}", "16.65"),
        (r"First \boxed{wrong} then \boxed{correct}", "correct"),
        ("No boxed answer, just 42.5 at the end", "42.5"),
    ]

    all_pass = True
    for text, expected in test_cases:
        result = extract_answer(text)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] extract_answer({text[:50]}...) = {result} (expected: {expected})")

    print()

    # Test is_correct logic
    print("Testing correctness evaluation...")
    correct_cases = [
        ("42", "42", True),
        ("16.65", "16.65", True),
        ("16.64", "16.65", True),  # Within tolerance
        ("XXXVIII", "XXXVIII", True),
        ("wrong", "right", False),
        (None, "42", False),
    ]

    for pred, gt, expected in correct_cases:
        result = is_correct(pred, gt)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] is_correct({pred}, {gt}) = {result} (expected: {expected})")

    print()

    # Analyze answer patterns per task type
    print("Answer patterns by task type:")
    train_df["task_type"] = train_df["prompt"].apply(classify_task)
    for task_type in sorted(train_df["task_type"].unique()):
        subset = train_df[train_df["task_type"] == task_type]
        answers = subset["answer"].astype(str)
        print(f"\n  {task_type} ({len(subset)} examples):")
        print(f"    Answer length: mean={answers.str.len().mean():.1f}, min={answers.str.len().min()}, max={answers.str.len().max()}")
        print(f"    Sample answers: {answers.head(5).tolist()}")

    print()
    if all_pass:
        print("All tests passed!")
    else:
        print("Some tests FAILED - check implementation.")


if __name__ == "__main__":
    main()
