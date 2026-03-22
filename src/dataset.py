import pandas as pd

from src.config import Config


def load_train(cfg: Config) -> pd.DataFrame:
    path = cfg.raw_dir / "train.csv"
    return pd.read_csv(path)


def load_test(cfg: Config) -> pd.DataFrame:
    path = cfg.raw_dir / "test.csv"
    return pd.read_csv(path)


SYSTEM_PROMPT = (
    "You are a precise reasoning assistant. Solve the given puzzle step by step. "
    "Think carefully about the pattern, then provide your final answer inside \\boxed{}."
)


def format_prompt_for_inference(prompt: str) -> list[dict]:
    """Format a raw prompt into chat messages for inference."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def format_for_sft(prompt: str, answer: str) -> list[dict]:
    """Format a single example into chat messages for supervised fine-tuning.

    The assistant response includes chain-of-thought reasoning followed by
    the answer in \\boxed{} format, which is what the evaluation metric extracts.
    """
    assistant_response = f"Let me analyze this step by step.\n\nAfter careful analysis of the pattern, the answer is:\n\n\\boxed{{{answer}}}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant_response},
    ]


def prepare_sft_dataset(cfg: Config) -> list[dict]:
    """Load training data and convert to SFT chat format."""
    train_df = load_train(cfg)
    examples = []
    for _, row in train_df.iterrows():
        messages = format_for_sft(row["prompt"], str(row["answer"]))
        examples.append({"messages": messages})
    return examples


def classify_task(prompt: str) -> str:
    """Classify a prompt into one of the six task types."""
    p = prompt.lower()
    if "bit manipulation" in p:
        return "bit_manipulation"
    elif "encryption" in p or "decrypt" in p:
        return "cipher"
    elif "numeral system" in p:
        return "numeral_conversion"
    elif "unit conversion" in p:
        return "unit_conversion"
    elif "transformation rules" in p and "equation" in p:
        return "symbol_equation"
    elif "gravitational" in p:
        return "gravitational"
    else:
        return "unknown"
