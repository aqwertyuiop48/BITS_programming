# Model Engineering Command Reference

This file documents the top-level command usage for the `model-engineering/` project.
Run commands from the `model-engineering/` directory.

## 1. Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## 2. Prepare data

Prepare raw CSV files into the modeling-ready processed format.

```bash
python prepare_data.py --input-csv data/raw/your_raw_data.csv --output-csv data/processed/train.csv --target-col target
```

Prepare Titanic training data:

```bash
python prepare_data.py \
  --input-csv data/raw/titanic_train.csv \
  --output-csv data/processed/titanic_bakeoff_train_1.csv \
  --target-col target \
  --rename-target-from Survived \
  --drop-columns PassengerId,Name,Ticket,Cabin
```

Prepare Titanic inference data:

```bash
python prepare_data.py \
  --input-csv data/raw/titanic_inference.csv \
  --output-csv data/processed/titanic_inference_input.csv \
  --target-col target \
  --drop-columns PassengerId,Name,Ticket,Cabin \
  --allow-missing-target
```

Generic inference data prep:

```bash
python prepare_data.py --input-csv data/raw/your_inference_data.csv --output-csv data/processed/inference_input.csv --target-col target --allow-missing-target
```

## 3. Train a baseline supervised model

Train the default baseline pipeline:

```bash
python train.py --config configs/train/default.yaml
```

Train a Titanic-specific model:

```bash
python train.py --config configs/train/titanic_random_forest.yaml
```

Other Titanic training configs:

```bash
python train.py --config configs/train/titanic_decision_tree.yaml
python train.py --config configs/train/titanic_gradient_boosting.yaml
python train.py --config configs/train/titanic_svm.yaml
python train.py --config configs/train/titanic_all_features.yaml
python train.py --config configs/train/titanic_selected_features.yaml
python train.py --config configs/train/titanic_pca_features.yaml
```

Enable MLflow tracking (recommended - SQLite backend):

```bash
python train.py --config configs/train/titanic_random_forest.yaml --mlflow --mlflow-tracking-uri sqlite:///mlflow.db --mlflow-experiment ml-demo --mlflow-register-model --mlflow-model-name titanic_survival_rf
```

Optional: use the local file-store (legacy) — opt-in by setting `MLFLOW_ALLOW_FILE_STORE=true`:

```bash
MLFLOW_ALLOW_FILE_STORE=true python train.py --config configs/train/titanic_random_forest.yaml --mlflow --mlflow-tracking-uri sqlite:///mlflow.db --mlflow-experiment ml-demo --mlflow-register-model --mlflow-model-name titanic_survival_rf
```

## 4. Train deep learning models

Train deep model demos with a YAML config:

```bash
python train_deep.py --config configs/train/digits_cnn.yaml && python train_deep.py --config configs/train/timeseries_rnn.yaml && python train_deep.py --config configs/train/timeseries_lstm.yaml && python train_deep.py --config configs/train/timeseries_gru.yaml
```

With MLflow (recommended - SQLite backend):

```bash
python train_deep.py --config configs/train/digits_cnn.yaml --mlflow --mlflow-tracking-uri sqlite:///mlflow.db --mlflow-experiment ml-demo-deep --mlflow-register-model --mlflow-model-name digits_cnn_demo
```

Optional: use the local file-store (legacy) — opt-in by setting `MLFLOW_ALLOW_FILE_STORE=true`:

```bash
MLFLOW_ALLOW_FILE_STORE=true python train_deep.py --config configs/train/digits_cnn.yaml --mlflow --mlflow-tracking-uri sqlite:///mlflow.db --mlflow-experiment ml-demo-deep --mlflow-register-model --mlflow-model-name digits_cnn_demo
```

## 5. Train unsupervised models

Train clustering/demo unsupervised pipeline:

```bash
python train_unsupervised.py --config configs/train/titanic_kmeans_unsupervised.yaml
```

With MLflow (recommended - SQLite backend):

```bash
python train_unsupervised.py --config configs/train/titanic_kmeans_unsupervised.yaml --mlflow --mlflow-tracking-uri sqlite:///mlflow.db --mlflow-experiment ml-demo-unsupervised --mlflow-register-model --mlflow-model-name titanic_kmeans_demo
```

Optional: use the local file-store (legacy) — opt-in by setting `MLFLOW_ALLOW_FILE_STORE=true`:

```bash
MLFLOW_ALLOW_FILE_STORE=true python train_unsupervised.py --config configs/train/titanic_kmeans_unsupervised.yaml --mlflow --mlflow-tracking-uri sqlite:///mlflow.db --mlflow-experiment ml-demo-unsupervised --mlflow-register-model --mlflow-model-name titanic_kmeans_demo
```

## 6. Evaluate a saved run

Evaluate a trained run using its `holdout.csv` and serialized model.

```bash
python evaluate.py --run-dir runs/run_001 --target-col target
```

For a Titanic run:

```bash
python evaluate.py --run-dir runs/run_titanic_random_forest --target-col target
```

With MLflow (recommended - SQLite backend):

```bash
python evaluate.py --run-dir runs/run_titanic_random_forest --target-col target --mlflow --mlflow-tracking-uri sqlite:///mlflow.db --mlflow-experiment ml-demo-evaluation
```

Optional: use the local file-store (legacy) — opt-in by setting `MLFLOW_ALLOW_FILE_STORE=true`:

```bash
MLFLOW_ALLOW_FILE_STORE=true python evaluate.py --run-dir runs/run_titanic_random_forest --target-col target --mlflow --mlflow-tracking-uri sqlite:///mlflow.db --mlflow-experiment ml-demo-evaluation
```

## 7. Run inference

Perform batch inference on a saved model run and input CSV:

```bash
python predict.py --run-dir runs/run_001 --input-csv data/processed/inference_input.csv
```

For Titanic inference:

```bash
python predict.py --run-dir runs/run_titanic_random_forest --input-csv data/processed/titanic_inference_input.csv
```

> Note: `runs/run_titanic_random_forest` must contain `model.joblib` before inference will work.

If you want a custom output path:

```bash
python predict.py --run-dir runs/run_titanic_random_forest --input-csv data/processed/titanic_inference_input.csv --output-csv runs/run_titanic_random_forest/predictions_inference.csv
```

## 8. Run smoke test

Perform a quick end-to-end sanity check:

```bash
python smoke_test.py
```

This script will automatically train `runs/run_001` first if needed.

## 9. Print experiment leaderboard

Show the best runs sorted by `roc_auc` by default:

```bash
python leaderboard.py
```

To sort by a different metric:

```bash
python leaderboard.py --sort-by accuracy
```

## 10. LLM demo commands

Run a Gemini LLM demo with config and prompt YAMLs:

```bash
python llm_demo.py --config configs/llm/gemini.yaml --prompt-config configs/prompts/titanic_classification.yaml --query "Male, Age 45, Ticket Class 3"
```

Other examples:

```bash
python llm_demo.py --config configs/llm/gemini.yaml --prompt-config configs/prompts/model_explanation.yaml --query "random forest classifier"
python llm_demo.py --config configs/llm/gemini.yaml --prompt-config configs/prompts/code_review.yaml --query "def train(X, y):\n    model = LogisticRegression()\n    return model.fit(X, y)"
```

Use a fixed budget mode:

```bash
python llm_demo.py --config configs/llm/gemini.yaml --prompt-config configs/prompts/titanic_classification.yaml --query "Female, Age 28, Ticket Class 1" --budget-mode low
```

## 11. RAG demo commands

Run a Gemini Flash RAG demo over repo reports:

```bash
python rag_demo.py --config configs/llm/gemini.yaml --prompt-config configs/prompts/rag_grounded_answer.yaml --query "Explain the Titanic demo workflow"
```

Customize retrieval:

```bash
python rag_demo.py --config configs/llm/gemini.yaml --prompt-config configs/prompts/rag_grounded_answer.yaml --query "What is RAG?" --top-k 5 --chunk-size 1000 --chunk-overlap 100
```

## 12. Token budget report

Summarize LLM demo token usage across runs:

```bash
python token_budget_report.py --task rag_grounded_answer
python token_budget_report.py --task general_assistant --model gemini-2.5-flash
```

## 13. Create Titanic sample data

Generate sample Titanic raw CSV files for training and inference:

```bash
python make_titanic_demo_data.py
```

This writes:

- `data/raw/titanic_train.csv`
- `data/raw/titanic_inference.csv`

## 14. Notes on available run directories

- `runs/run_001`: baseline demo run (often breast-cancer or default bootstrap dataset)
- `runs/run_titanic_random_forest`: Titanic run metadata, but may need training to produce `model.joblib`
- `runs/run_titanic_decision_tree`: similar Titanic run metadata
- `runs/run_titanic_gradient_boosting`: similar Titanic run metadata
- `runs/run_titanic_svm`: similar Titanic run metadata
- `runs/run_titanic_kmeans_unsupervised`: Titanic clustering metadata
- `runs/run_titanic_dbscan_unsupervised`: Titanic clustering metadata

## 15. Troubleshooting

- If inference fails with missing columns, verify that the input CSV has the same feature columns expected by the model.
- If `model.joblib` is missing, train the run first with the appropriate `train.py --config ...` command.
- If `predict.py` cannot find `runs/<run_name>/model.joblib`, check the run directory name and whether the model was saved.
