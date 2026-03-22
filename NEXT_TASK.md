# Next Task: NVIDIA Nemotron

## Status

Kaggle Notebook v7 が RTX Pro 6000 で実行中。結果を確認して次のアクションを決定する。

## Step 1: 結果確認

```bash
uv run kaggle kernels status koheimiki/nvidia-nemotron-lora-baseline
```

### 成功した場合

1. submission.zip をダウンロード:
```bash
uv run kaggle kernels output koheimiki/nvidia-nemotron-lora-baseline -p outputs/kaggle-output
```

2. submission.zip を提出:
```bash
uv run kaggle competitions submit -c nvidia-nemotron-model-reasoning-challenge -f outputs/kaggle-output/submission.zip -m "v1: SFT LoRA baseline, 3 epochs, rank=32"
```

3. CLAUDE.md の Submission History に結果を追記

### エラーの場合

1. ログをダウンロード:
```bash
uv run kaggle kernels output koheimiki/nvidia-nemotron-lora-baseline -p outputs/kaggle-logs-v7
```

2. エラー原因を特定して `kaggle-notebook/notebook.py` を修正
3. 再 push: `cd kaggle-notebook && uv run kaggle kernels push -p .`

## Step 2: モデル改善（v1 提出後）

現在のベースラインは単純な SFT。改善案:

1. **Chain-of-thought in training data**: 現在の training response は `\boxed{answer}` のみ。ステップバイステップの推論を含む training data を生成する。
2. **Task-specific prompting**: 6つのタスクタイプ（bit_manipulation, cipher, gravitational, numeral_conversion, symbol_equation, unit_conversion）ごとに異なるプロンプトを設計。
3. **Data augmentation**: 各タスクタイプについて、パターンを分析して追加の訓練例を生成。
4. **LoRA rank 実験**: rank=16 vs rank=32 の比較。

## Important Notes

- **GPU quota**: RTX Pro 6000 は週30時間、残り約28時間。1回の学習に~30分使う想定。
- **公式 Demo の score は 0.50**（LoRA なしのベースモデル）。最低でもこれを超える必要がある。
- **docs/official-demo.md** に公式 Demo のコード全文あり。
- **docs/evaluation.md** に評価方法の詳細あり。Accuracy ベース、`\boxed{}` から回答抽出。
