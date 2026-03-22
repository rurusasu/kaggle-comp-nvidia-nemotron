# NVIDIA Nemotron Submission Demo - Official Kaggle Notebook

Source: https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo
Author: Ryan Holbrook
Version: 11 of 11
Runtime: 18m 9s, GPU RTX Pro 6000
Public Score: 0.50

## Required Packages and Versions

- `polars` - for CSV reading
- `kagglehub` - for model download
- `mamba_ssm` - imported (SSM model support)
- `torch` - PyTorch with bfloat16 support
- `peft` - LoRA adapter (LoraConfig, get_peft_model, get_peft_model_state_dict, TaskType)
- `transformers` - AutoModelForCausalLM, AutoTokenizer
- Python 3.12

## Full Code

### Cell 1: Load Training Data

```python
import polars as pl

train = pl.read_csv('/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv')
train.head()
```

Output: DataFrame with shape (5, 3), columns: `id` (str), `prompt` (str), `answer` (str)

### Cell 2: Load Model, Configure LoRA, Save Adapter

```python
import os
import kagglehub
import mamba_ssm
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
OUTPUT_DIR = "/kaggle/working"
LORA_RANK = 32  # Can be set to a maximum of 32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16
)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("Model loaded successfully.")

# Initialize LoRA Adapter (Rank 1)
print(f"Initializing LoRA adapter with rank={LORA_RANK}...")
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=16,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# YOUR CODE HERE
# --------------
# model.train()
# --------------

# Save Adapter
print(f"Saving adapter to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
```

Output:
- trainable params: 880,138,240 || all params: 32,458,075,584 || trainable%: 2.7116
- Saving adapter to /kaggle/working...

### Cell 3: Create Submission Zip

```python
import subprocess

subprocess.run("zip -m submission.zip *", shell=True, check=True)
```

Output: Zips adapter_config.json, adapter_model.safetensors, README.md, __notebook__.ipynb

### Cell 4: Done

```python
print('Done.')
```

## Key Patterns

### Data Loading
- Competition data path: `/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv`
- Uses `polars` (not pandas) for CSV reading

### Model Loading
- Model downloaded via `kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")`
- Loaded with `AutoModelForCausalLM.from_pretrained()` with `device_map="auto"`, `trust_remote_code=True`, `dtype=torch.bfloat16`
- Requires `mamba_ssm` to be imported (SSM/hybrid architecture support)

### LoRA Configuration
- Rank: 32 (max allowed is 32)
- Alpha: 16
- Target modules: regex `.*\.(in_proj|out_proj|up_proj|down_proj)$`
- Dropout: 0.05
- Bias: none
- Task type: CAUSAL_LM

### Submission Pattern
- This is a **training notebook** competition - you submit LoRA adapter weights, not inference code
- Save adapter with `model.save_pretrained(OUTPUT_DIR)` to `/kaggle/working`
- Zip all files in working directory: `zip -m submission.zip *`
- The submission.zip contains: adapter_config.json, adapter_model.safetensors

### Environment-Specific Code
- Kaggle input path: `/kaggle/input/nvidia-nemotron-3-reasoning-challenge/`
- Kaggle working directory: `/kaggle/working`
- GPU: RTX Pro 6000
- Model source: Kaggle Models via `kagglehub`
- Utility script loaded from: `/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/`

### Important Notes
- The tokenizer line is commented out in the demo
- The demo does NOT actually train - there's a placeholder `# YOUR CODE HERE` section
- The competition expects you to add your training loop in the marked section
- `mamba_ssm` must be imported before model loading (hybrid Mamba-Transformer architecture)
