# NVIDIA Nemotron Model Reasoning Challenge - Overview

**Competition URL:** https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge

**Tagline:** Advance reasoning techniques using NVIDIA Nemotron open models on a novel benchmark

**Type:** Featured Prediction Competition

**Host:** NVIDIA

## Description

Reasoning benchmarks are a useful way to measure progress on structured tasks. When approaches and results are shared openly, the community can compare methods, reproduce improvements, and iterate more effectively.

Today, reasoning improvements are explored across many independent efforts - often using different datasets, prompts, and evaluation setups - making direct comparison difficult. A shared benchmark and common baseline model allow techniques to be tested and compared more consistently.

While language models perform strongly on many tasks, structured reasoning benchmarks remain an active area of research and optimization.

In this competition, participants will work from a shared Nemotron 3 Nano baseline and a novel reasoning benchmark developed by NVIDIA Research. Nemotron provides an open foundation for this challenge, including openly available models, datasets, and training recipes that participants can build on or adapt within their own workflows.

You may experiment with:

- Prompting strategies
- Data filtering and curation
- Synthetic data generation
- Reinforcement learning
- Lightweight fine-tuning
- Or other approaches of your choice

Participants may use any training framework, tooling, or workflow to produce their LoRA adapter. NVIDIA-provided recipes are optional starting points - competitors are free to use other ecosystems and libraries (e.g., Hugging Face, Unsloth, Axolotl, TRL, or similar tooling).

The only requirement is that the final submission produces a compatible LoRA adapter for the Nemotron-3-Nano-30B base model.

Multiple valid solution paths are expected. Clear documentation - including notebooks and write-ups - is encouraged (and required for prize eligibility) to support reproducibility and communal learning.

By bringing models, datasets, and evaluation into an open, shared environment, this challenge creates an opportunity for collaborative iteration - strengthening open reasoning workflows that others can study, reuse, and extend.

## Overview

Develop techniques that improve reasoning accuracy using NVIDIA Nemotron models.

Participants will experiment with prompting, data pipelines, and lightweight fine-tuning while evaluating their approaches on a new reasoning benchmark developed by NVIDIA Research.

## Compute Powered by NVIDIA Blackwell on Google Cloud

We're excited to partner with Google Cloud to offer the G4 VMs powered by NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs as the compute backbone for this challenge. G4 VMs are a strong fit for both fine-tuning and high-throughput inference with Nemotron models so you can iterate quickly on prompts, data pipelines, and tuning strategies while evaluating performance on real benchmarks.

NVIDIA RTX PRO 6000 Blackwell GPUs provide the performance and memory needed to serve open reasoning models efficiently at inference time, enabling responsive evaluation runs and rapid leaderboard iteration as you improve your Nemotron variants. Learn more about the underlying Google Cloud G4 VM and its capabilities here: https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines#g4-series

## Tags

- Reasoning
- Deep Learning
- Pre-Trained Model
- Custom Metric
