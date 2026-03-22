"""Evaluation utilities matching the competition metric.

The competition evaluates by extracting answers from \\boxed{} format
and comparing with ground truth using exact string match or numerical
tolerance of 1e-2.
"""

import csv
import json
import re
from datetime import UTC, datetime

from src.config import Config


def extract_boxed_answer(text: str) -> str | None:
    """Extract the last \\boxed{...} content from model output.

    Handles nested braces by counting brace depth.
    """
    # Find all \boxed{ occurrences and take the last one
    pattern = r"\\boxed\{"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None

    last_match = matches[-1]
    start = last_match.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth == 0:
        return text[start : i - 1].strip()
    return None


def extract_answer(text: str) -> str | None:
    """Extract answer from model output, trying \\boxed{} first, then fallback heuristics."""
    # Try boxed format first
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed

    # Fallback: look for last number
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]

    return None


def is_correct(prediction: str | None, ground_truth: str, tol: float = 1e-2) -> bool:
    """Check if prediction matches ground truth.

    Uses exact string match first, then tries numerical comparison
    with relative tolerance.
    """
    if prediction is None:
        return False

    # Exact string match
    if prediction.strip() == str(ground_truth).strip():
        return True

    # Numerical comparison
    try:
        pred_val = float(prediction)
        true_val = float(ground_truth)
        if true_val == 0:
            return abs(pred_val) < tol
        return abs(pred_val - true_val) / abs(true_val) < tol
    except (ValueError, ZeroDivisionError):
        pass

    return False


def compute_accuracy(predictions: list[str | None], ground_truths: list[str]) -> float:
    """Compute accuracy score matching competition evaluation."""
    if not predictions:
        return 0.0
    correct = sum(is_correct(p, gt) for p, gt in zip(predictions, ground_truths))
    return correct / len(predictions)


def log_experiment(cfg: Config, result: dict) -> None:
    """Save experiment result as JSON and append to CSV in logs/."""
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    result["timestamp"] = timestamp

    # JSON (detailed, per-experiment)
    json_path = cfg.logs_dir / f"{timestamp}.json"
    json_path.write_text(json.dumps(result, indent=2, default=str))

    # CSV (summary, append-only)
    csv_path = cfg.logs_dir / "experiments.csv"
    flat = {k: str(v) if isinstance(v, list) else v for k, v in result.items()}
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(flat.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(flat)
