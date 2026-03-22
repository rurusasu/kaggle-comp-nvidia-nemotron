"""Submission utilities for the LoRA adapter competition.

The submission format is a zip file containing the LoRA adapter files,
including adapter_config.json and adapter weights.
"""

import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from src.config import Config


def create_submission_zip(
    cfg: Config,
    adapter_dir: Path,
    output_path: Path | None = None,
) -> Path:
    """Package a LoRA adapter directory into submission.zip.

    The adapter directory must contain at minimum:
    - adapter_config.json
    - adapter_model.safetensors (or .bin)
    """
    cfg.submissions_dir.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        output_path = cfg.submissions_dir / f"submission_{timestamp}.zip"

    # Verify adapter_config.json exists
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in adapter_dir.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(adapter_dir)
                zf.write(file, arcname)

    print(f"Submission saved to {output_path}")
    return output_path


def validate_adapter(adapter_dir: Path) -> bool:
    """Validate that a LoRA adapter directory has required files and config."""
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        print("ERROR: adapter_config.json not found")
        return False

    config = json.loads(config_path.read_text())

    # Check LoRA rank <= 32
    rank = config.get("r", 0)
    if rank > 32:
        print(f"ERROR: LoRA rank {rank} exceeds maximum of 32")
        return False

    # Check for weight files
    has_weights = any(adapter_dir.glob("*.safetensors")) or any(adapter_dir.glob("*.bin"))
    if not has_weights:
        print("WARNING: No weight files found (*.safetensors or *.bin)")
        return False

    print(f"Adapter valid: rank={rank}, peft_type={config.get('peft_type')}")
    return True
