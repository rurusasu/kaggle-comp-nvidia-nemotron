# NVIDIA Nemotron Model Reasoning Challenge - Submission Format

## Format

You must submit a LoRA adapter of rank at most 32 for the NVIDIA Nemotron-3-Nano-30B model packaged into a `submission.zip` file.

## Requirements

- The LoRA adapter must include an `adapter_config.json` file
- Maximum LoRA rank: 32
- Compatible with the Nemotron-3-Nano-30B base model
- Packaged as `submission.zip`

## Inference Details

- The model is loaded using the vLLM inference engine
- For each test case, the model is prompted to generate a response
- The model should place its final answer within a `\boxed{}` LaTeX command
- The metric extracts the final answer from the generated text, prioritizing content within the boxed format

## Inference Parameters

| Parameter | Value |
|---|---|
| max_lora_rank | 32 |
| max_tokens | 7680 |
| top_p | 1.0 |
| temperature | 0.0 |
| max_num_seqs | 64 |
| gpu_memory_utilization | 0.85 |
| max_model_len | 8192 |

## Resources

- Submission demo notebook: https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo
- Base model: https://www.kaggle.com/models/metric/nemotron-3-nano-30b-a3b-bf16

## Prize Eligibility

To be eligible for any prize, teams must publish a public Kaggle notebook and solution write-up documenting the methods, datasets, and techniques used to produce the submission. Submissions without qualifying public documentation may be deemed ineligible for prizes.

## Tooling

Participants may use any training framework, tooling, or workflow to produce their LoRA adapter. NVIDIA-provided recipes are optional starting points. Competitors are free to use other ecosystems and libraries (e.g., Hugging Face, Unsloth, Axolotl, TRL, or similar tooling).
