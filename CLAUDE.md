# NVIDIA Nemotron Model Reasoning Challenge

## Competition Info

- **URL:** https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
- **Deadline:** 2026-06-15 23:59 UTC
- **Prize:** $106,388
- **Category:** Featured
- **Launched at:** NVIDIA GTC 2026

## Task

NVIDIA Nemotron 3 Nano の推論精度を向上させる。許可されるテクニック:
- プロンプト最適化
- 合成データ生成
- データキュレーション・フィルタリング
- ファインチューニング
- 強化学習

## Evaluation

- **Metric:** pass@1 (majority voting @ 64 generations)
- pass@1: 64世代にわたる平均精度
- maj@64: 64世代にわたる多数決精度

## Submission Format

- Code Competition
- Submission Demo: https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo

## Infrastructure

- Google Cloud G4 VM ベース
- NVIDIA Nemotron モデルを使用（Hugging Face / NVIDIA Developer で公開）

## Data

- `train.csv`, `test.csv`

## Documentation

**IMPORTANT: Before starting any implementation work, you MUST read the relevant docs first.**

- [docs/overview.md](docs/overview.md) — Competition description, goal, background
- [docs/evaluation.md](docs/evaluation.md) — Evaluation metric, scoring methodology
- [docs/submission.md](docs/submission.md) — Submission format, file structure, requirements
- [docs/timeline.md](docs/timeline.md) — Important dates and deadlines
- [docs/rules.md](docs/rules.md) — Full competition rules
- [docs/prizes.md](docs/prizes.md) — Prize structure

### Required Reading Order

1. Before EDA or feature engineering → read `overview.md` and `evaluation.md`
2. Before building submission pipeline → read `submission.md`
3. Before using external data or models → read `rules.md`
4. Before final submission → read `timeline.md` to confirm deadlines

---

# Kaggle Competition Workspace

## Structure

- `src/config.py` — All configuration (paths, params, seed). Change settings HERE, not in other modules.
- `src/dataset.py` — Stateless data I/O. `load_train()` / `load_test()` return raw DataFrames.
- `src/features.py` — Feature engineering. Stateful transforms (fit on train only).
- `src/model.py` — Model train/predict/save/load.
- `src/evaluate.py` — CV splitter, metrics, experiment logging. Owns all writes to `logs/`.
- `src/submit.py` — Generates timestamped submission CSVs.
- `src/utils.py` — `set_seed()`, `Timer`.
- `scripts/train.py` — Training entrypoint. Runs full CV pipeline.
- `scripts/predict.py` — Inference entrypoint. Loads saved models, generates submission.

## Conventions

- Format with ruff (line-length=120, Python 3.14)
- Type hints encouraged
- Config changes go in `src/config.py` only
- Experiment logs go in `logs/` via `src/evaluate.py` only

## Commands

- `task setup` — Install deps + download data
- `task train` — Train models
- `task predict` — Generate predictions
- `task submit` — Submit to Kaggle
- `task lint` — Check code style
- `task test` — Run tests
