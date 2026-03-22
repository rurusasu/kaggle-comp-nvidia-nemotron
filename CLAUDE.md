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

## Current Approach

### Baseline: SFT with LoRA (v1)

- **Method:** Supervised Fine-Tuning (SFT) on train.csv (9,500 examples, 6 task types)
- **Base Model:** Nemotron-3-Nano-30B-A3B-BF16 (4bit quantized)
- **LoRA Config:** rank=32, alpha=16, dropout=0.05, target=in_proj|out_proj|up_proj|down_proj (regex, matches official demo)
- **Training:** 3 epochs, batch=1, grad_accum=8, lr=2e-4, cosine schedule
- **Answer Format:** `\boxed{ANSWER}` extraction

### Workflow

1. **学習は Kaggle Notebook で実行**（ローカル GPU は RTX 4060 8GB で VRAM 不足）
2. Kaggle Notebook URL: https://www.kaggle.com/code/koheimiki/nvidia-nemotron-lora-baseline
3. 学習完了後、`submission.zip`（LoRA adapter）をダウンロード
4. `kaggle competitions submit` で提出

### File Layout

- `kaggle-notebook/notebook.py` — Kaggle 実行用スクリプト
- `kaggle-notebook/kernel-metadata.json` — Kaggle push 用メタデータ
- `outputs/kaggle_training_notebook.py` — 同スクリプトのコピー
- `src/` — ローカル分析・評価用コード
- `data/raw/train.csv` — 6 task types (bit_manipulation, cipher, gravitational, numeral_conversion, symbol_equation, unit_conversion)
- `data/raw/test.csv` — public test (3 examples, real test is hidden)

### Task Types in Training Data

| Type | Count | Description |
|---|---|---|
| bit_manipulation | 1602 | 8-bit binary pattern transformations |
| cipher | 1576 | Substitution cipher decryption |
| gravitational | 1597 | Physics/gravity calculations |
| numeral_conversion | 1576 | Arabic to Roman numeral conversion |
| symbol_equation | 1555 | Symbolic transformation rules |
| unit_conversion | 1594 | Numerical unit conversion |

### Improvement Ideas

- Chain-of-thought reasoning in training responses
- Task-specific prompts per task type
- Data augmentation (generate more examples per task)
- GRPO/RL after SFT
- Larger sequence length for complex reasoning

## Lessons Learned

### Kaggle Notebook 環境の罠

1. **依存パッケージ不足**: Kaggle 環境に `trl`, `peft`, `bitsandbytes`, `accelerate` がプリインストールされていない。Notebook 冒頭で `pip install` が必須。
2. **データマウント**: `kernel-metadata.json` の `competition_sources` を配列形式で指定しないとコンペデータがマウントされない。`"competition": "slug"` 形式は効かなかった。
3. **パス不定**: `/kaggle/input/` 配下のディレクトリ名はコンペやモデルによって異なる。ハードコードせず `os.walk` で自動検出するのが安全。
4. **v1→v6 の試行錯誤**: trl 不足(v1) → pip install 追加(v2) → データ未マウント(v3) → competition_sources 修正(v4) → mamba-ssm 追加(v5, RUNNING成功) → os import 修正+LoRA修正(v6, VRAM不足)
5. **GPU VRAM 不足**: Nemotron-3-Nano-30B は T4 16GB では 4bit 量子化でも載らない。公式 Demo は RTX Pro 6000 (48GB) を使用。**Web UI から GPU タイプを RTX Pro 6000 に変更する必要がある**。

### 対策

- Notebook 冒頭に必ず `pip install` ブロックを入れる
- データパスはハードコードせず自動検出関数を使う
- `kernel-metadata.json` は `competition_sources` (配列) + `model_sources` (配列) で指定
- デバッグ用に `/kaggle/input` の内容を出力するコードを入れる

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
