# NVIDIA Nemotron Model Reasoning Challenge - Evaluation

## Evaluation Metric

Submissions are evaluated based on their **Accuracy** in solving the provided tasks.

The NVIDIA Nemotron-3-Nano-30B model (https://www.kaggle.com/models/metric/nemotron-3-nano-30b-a3b-bf16) is loaded with your LoRA adapter (which must include an `adapter_config.json`) using the vLLM inference engine. For each test case, the model is prompted to generate a response and instructed to place its final answer within a `\boxed{}` LaTeX command.

The metric extracts the final answer from the generated text, prioritizing content within the boxed format while falling back to other heuristic patterns or the last numeric value found. A prediction is graded as correct if it matches the ground truth either exactly as a string or within a relative numerical tolerance of 10^-2.

The final score is the proportion of correctly answered questions.

## Metric Implementation

You may view the implementation of the metric here: **NVIDIA Nemotron Metric** (https://www.kaggle.com/code/metric/nvidia-nemotron-metric).

It is being run with the following parameters:

| Parameter | Value |
|---|---|
| max_lora_rank | 32 |
| max_tokens | 7680 |
| top_p | 1.0 |
| temperature | 0.0 |
| max_num_seqs | 64 |
| gpu_memory_utilization | 0.85 |
| max_model_len | 8192 |

## Submitting

You must submit a LoRA adapter of rank at most 32 for the NVIDIA Nemotron-3-Nano-30B model packaged into a `submission.zip` file. You may consider adapting the **NVIDIA Nemotron Submission Demo** (https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo) to produce your submission.
