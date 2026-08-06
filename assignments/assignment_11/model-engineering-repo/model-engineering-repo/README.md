# ML Demo Repo Spine

This repository is a course-ready machine learning project spine focused on reproducibility and traceability.

## Project Structure

```text
ml_demo1/
├── data/
│   ├── raw/
│   ├── processed/
│   └── data_dictionary.md
├── src/
│   ├── features/
│   ├── models/
│   ├── eval/
│   ├── inference/
│   └── utils/
├── notebooks/
├── configs/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── train/
├── runs/
├── reports/
├── prepare_data.py
├── train.py
├── evaluate.py
├── predict.py
├── smoke_test.py
└── requirements.txt
```

## Quickstart

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare raw data into the processed training format:

```bash
python prepare_data.py --input-csv data/raw/your_raw_data.csv --output-csv data/processed/train.csv --target-col target
```

For unlabeled inference data, allow the missing target column:

```bash
python prepare_data.py --input-csv data/raw/your_inference_data.csv --output-csv data/processed/inference_input.csv --target-col target --allow-missing-target
```

3. Train a baseline model:

```bash
python train.py --config configs/train/default.yaml
```

Deep learning demos (config-first, same artifact contract):

```bash
# CNN on digits images
python train_deep.py --config configs/train/digits_cnn.yaml

# RNN / LSTM / GRU on time series sequences
python train_deep.py --config configs/train/timeseries_rnn.yaml
python train_deep.py --config configs/train/timeseries_lstm.yaml
python train_deep.py --config configs/train/timeseries_gru.yaml
```

4. Evaluate a trained run:

```bash
python evaluate.py --run-dir runs/run_001
```

5. Run inference on a CSV:

```bash
python predict.py --run-dir runs/run_001 --input-csv data/processed/inference_input.csv
```

6. Run a pipeline smoke test:

```bash
python smoke_test.py
```

## MLflow Tracking

This repo now supports optional MLflow tracking on training and evaluation scripts.

1. Install dependencies (includes mlflow):

```bash
pip install -r requirements.txt
```

2. Track a supervised training run:

```bash
python train.py --config configs/train/titanic_random_forest.yaml --mlflow
```

3. Track an evaluation run:

```bash
python evaluate.py --run-dir runs/run_titanic_random_forest --target-col target --mlflow
```

4. Track deep or unsupervised runs:

```bash
python train_deep.py --config configs/train/digits_cnn.yaml --mlflow
python train_unsupervised.py --config configs/train/titanic_kmeans_unsupervised.yaml --mlflow
```

By default, runs are written to local file-based tracking URI `file:./mlruns`.

To use a different tracking backend:

```bash
python train.py --config configs/train/default.yaml --mlflow --mlflow-tracking-uri http://localhost:5000 --mlflow-experiment ml-demo
```

To browse local runs with MLflow UI:

```bash
mlflow ui --backend-store-uri ./mlruns
```

### Optional: Auto Register Models In MLflow Registry

Training scripts can automatically register logged models by adding:
- `--mlflow-register-model`
- `--mlflow-model-name <registry_name>`

Example (supervised):

```bash
python train.py --config configs/train/titanic_random_forest.yaml --mlflow --mlflow-register-model --mlflow-model-name titanic_survival_rf
```

Example (deep learning):

```bash
python train_deep.py --config configs/train/digits_cnn.yaml --mlflow --mlflow-register-model --mlflow-model-name digits_cnn_demo
```

Example (unsupervised):

```bash
python train_unsupervised.py --config configs/train/titanic_kmeans_unsupervised.yaml --mlflow --mlflow-register-model --mlflow-model-name titanic_kmeans_demo
```

For local model registry support, prefer a database-backed tracking URI:

```bash
python train.py --config configs/train/titanic_random_forest.yaml --mlflow --mlflow-tracking-uri sqlite:///mlflow.db --mlflow-experiment ml-demo --mlflow-register-model --mlflow-model-name titanic_survival_rf
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Classroom Example: Titanic End to End

Use Titanic passenger survival as a simple real-world classification demo.

Learning goal:
- start with raw CSV files
- create processed training and inference tables
- train a reproducible model
- evaluate holdout performance
- run inference on unlabeled rows
- record the outcome in run artifacts and reports

Suggested raw files:
- `data/raw/titanic_train.csv` with label column `Survived`
- `data/raw/titanic_inference.csv` without the label column

Small Titanic-like sample files are included under `data/raw/` so the demo can run without downloading a dataset.

Prepare labeled training data:

```bash
python prepare_data.py --input-csv data/raw/titanic_train.csv --output-csv data/processed/train.csv --target-col target --rename-target-from Survived --drop-columns PassengerId,Name,Ticket,Cabin
```

Train the baseline pipeline:

```bash
python train.py --config configs/train/default.yaml
```

Evaluate the saved run:

```bash
python evaluate.py --run-dir runs/run_001 --target-col target
```

Prepare unlabeled inference data:

```bash
python prepare_data.py --input-csv data/raw/titanic_inference.csv --output-csv data/processed/inference_input.csv --target-col target --drop-columns PassengerId,Name,Ticket,Cabin --allow-missing-target
```

Run inference:

```bash
python predict.py --run-dir runs/run_001 --input-csv data/processed/inference_input.csv
```

Teaching message:
- `data/raw/` keeps the original source data
- `prepare_data.py` creates the modeling-ready contract in `data/processed/`
- `src/features/preprocess.py` defines reusable transforms
- `train.py` and `predict.py` share the same serialized pipeline
- `runs/` and `reports/` preserve evidence and conclusions

## Workflow Map (Command -> Artifacts)

1. `prepare_data.py`

Command:

```bash
python prepare_data.py --input-csv data/raw/your_raw_data.csv --output-csv data/processed/train.csv --target-col target
```

Reads:
- raw CSV passed via `--input-csv`

Writes:
- `data/processed/train.csv`

Behavior:
- trims column names
- optionally renames a raw target column
- optionally drops selected columns
- can prepare unlabeled inference data with `--allow-missing-target`
- removes duplicate rows

2. `train.py`

Command:

```bash
python train.py --config configs/train/default.yaml
```

Reads:
- `configs/train/default.yaml`
- `configs/data/default.yaml`
- `configs/features/default.yaml`
- `configs/models/baseline_logreg.yaml`
- `data/processed/train.csv` (auto-generated from sklearn demo dataset if missing)

Writes:
- `runs/run_001/model.joblib`
- `runs/run_001/metrics.json`
- `runs/run_001/params.json`
- `runs/run_001/predictions.csv`
- `runs/run_001/holdout.csv`
- `runs/summary.csv`

3. `evaluate.py`

Command:

```bash
python evaluate.py --run-dir runs/run_001 --target-col target
```

Reads:
- `runs/run_001/model.joblib`
- `runs/run_001/holdout.csv`

Writes:
- `runs/run_001/evaluation_metrics.json`

4. `predict.py`

Command:

```bash
python predict.py --run-dir runs/run_001 --input-csv <path_to_input.csv>
```

Reads:
- `runs/run_001/model.joblib`
- input CSV passed via `--input-csv`

Writes:
- `runs/run_001/predictions_inference.csv` (default)
- or custom file path from `--output-csv`

5. `smoke_test.py`

Command:

```bash
python smoke_test.py
```

Behavior:
- Checks whether `runs/run_001/model.joblib` exists.
- If missing, runs `train.py` automatically.
- Loads `runs/run_001/holdout.csv`, predicts on a small sample, and validates expected output columns.
- Prints pass/fail sanity result.

## Deep Workflow Map (YAML -> Artifacts)

Entry command:

```bash
python train_deep.py --config configs/train/digits_cnn.yaml
```

Reads:
- `configs/train/digits_*.yaml`
- `configs/data/digits_*.yaml`
- `configs/models/digits_*.yaml`

Writes under `runs/<run_name>/`:
- `model.pt`
- `metrics.json`
- `params.json`
- `predictions.csv`
- `training_curve.csv`
- `bundle_info.json`

Also appends:
- `runs/summary.csv`

## Token Budget Demo Summary

After running `llm_demo.py` or `rag_demo.py` with different `--budget-mode` values,
use this script to compare token usage and savings:

```bash
python token_budget_report.py --task rag_grounded_answer
```

Optional filters:

```bash
python token_budget_report.py --task general_assistant --model gemini-2.5-flash
```

## Auto Tool Routing (LLM Demo)

`llm_demo.py` can now auto-select tools and token budget from the user query.

Default behavior:
- enable tool routing automatically (`google_search`, `code_execution`, both, or none)
- auto-pick budget mode (`low`, `balanced`, `detailed`) based on detected query intent
- cap max tokens using the chosen budget profile for token optimization

Example (automatic routing + budget):

```bash
python llm_demo.py --config configs/llm/gemini.yaml --prompt-config configs/prompts/general_assistant.yaml --query "What is the latest inflation rate and compute yearly impact for $1200 monthly spend?"
```

Manual overrides:

```bash
# Force fixed budget mode while keeping auto tool routing
python llm_demo.py --config configs/llm/gemini.yaml --prompt-config configs/prompts/general_assistant.yaml --query "Explain random forests" --budget-mode low

# Disable auto routing and use tools from config YAML instead
python llm_demo.py --config configs/llm/gemini_tools_all.yaml --prompt-config configs/prompts/search_plus_compute.yaml --query "Find EUR/USD and compute 5% increase" --disable-auto-route --budget-mode balanced
```

## Auto Budget Routing (RAG Demo)

`rag_demo.py` now supports query-based automatic budget routing for retrieval and output length.

Default behavior:
- `--budget-mode auto` (default) picks `low`, `balanced`, or `detailed` from query complexity
- selected budget controls `top_k`, `chunk_size`, `chunk_overlap`, and max output tokens

Examples:

```bash
# Auto-select retrieval budget from query intent
python rag_demo.py --config configs/llm/gemini.yaml --prompt-config configs/prompts/rag_grounded_answer.yaml --query "Compare RAG design choices in this repo and explain trade-offs"

# Disable auto routing and force fixed budget
python rag_demo.py --config configs/llm/gemini.yaml --prompt-config configs/prompts/rag_grounded_answer.yaml --query "What is RAG?" --disable-auto-route --budget-mode low
```

## Gemini Flash RAG Demo

Use Gemini Flash with retrieval over local markdown reports.

Prerequisites:
- `GEMINI_API_KEY` set in your environment or `.env`
- dependencies installed from `requirements.txt`

Run:

```bash
python rag_demo.py --config configs/llm/gemini.yaml --prompt-config configs/prompts/rag_grounded_answer.yaml --query "How does this repo explain RAG?"
```

Optional retrieval controls:

```bash
python rag_demo.py --config configs/llm/gemini.yaml --prompt-config configs/prompts/rag_grounded_answer.yaml --query "What is model engineering for LLMs?" --docs-dir reports --top-k 5 --chunk-size 1000 --chunk-overlap 100
```

## Gemini Tool-Call Demos

Use these ready-made configs to demo different Gemini tool-calling behaviors.

Search-only demo:

```bash
python llm_demo.py --config gemini_search.yaml --prompt-config search_weather_brief.yaml --query "What is the weather in Bangalore right now?"
```

Code-execution-only demo:

```bash
python llm_demo.py --config gemini_code_execution.yaml --prompt-config code_execution_demo.yaml --query "Compute the CAGR from 120 to 185 over 6 years"
```

Combined tools demo (search + compute):

```bash
python llm_demo.py --config gemini_tools_all.yaml --prompt-config search_plus_compute.yaml --query "Find current Bangalore temperature in Celsius and convert it to Fahrenheit"
```

Note: llm_demo.py supports shorthand names (for example gemini_tools_all.yaml) and resolves from configs/llm and configs/prompts.

New demo files:
- `configs/llm/gemini_search.yaml`
- `configs/llm/gemini_code_execution.yaml`
- `configs/llm/gemini_tools_all.yaml`
- `configs/prompts/search_weather_brief.yaml`
- `configs/prompts/code_execution_demo.yaml`
- `configs/prompts/search_plus_compute.yaml`

Artifacts are saved under `runs/<run_name>/`:
- `response.txt`
- `metadata.json`
- `params.json`
- `bundle_info.json`

## LLM Learning Track

This repo now includes an LLM explainer for classroom teaching:
- `reports/llms_explained.md`

How to position it in class:
- Classical ML in this repo: `train.py` + sklearn pipeline + tabular data.
- Deep learning in this repo: `train_deep.py` + CNN/RNN/LSTM/GRU on Digits.
- LLM concepts: transformer architecture, tokenization, pretraining, fine-tuning, prompting, and RAG.

Suggested teaching flow:
1. Start with reproducibility concepts (`configs/` -> commands -> `runs/` artifacts).
2. Show supervised ML and deep learning runs in this repo.
3. Use `reports/llms_explained.md` to bridge from sequence models (RNN/LSTM/GRU) to transformers and modern LLM systems.

### Gemini API Demo (Free Tier)

Use Google's Gemini API for prompt engineering demonstrations. **Free tier available** at [ai.google.dev](https://ai.google.dev/).

**Setup:**

1. Get free API key at [ai.google.dev](https://ai.google.dev/)
2. Set environment variable:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt  # includes google-genai SDK
   ```

**Usage Examples:**

Classification demo (predict Titanic survival):
```bash
python llm_demo.py --config configs/llm/gemini.yaml \
                   --prompt-config configs/prompts/titanic_classification.yaml \
                   --query "Female, Age 28, Ticket Class 1"
```

Model explanation (teach ML concepts):
```bash
python llm_demo.py --config configs/llm/gemini.yaml \
                   --prompt-config configs/prompts/model_explanation.yaml \
                   --query "random forest classifier"
```

Code review (teaching code best practices):
```bash
python llm_demo.py --config configs/llm/gemini.yaml \
                   --prompt-config configs/prompts/code_review.yaml \
                   --query "def train(X, y):\n    model = LogisticRegression()\n    return model.fit(X, y)"
```

**Artifacts saved to `runs/<run_name>/`:**
- `response.txt` — Full LLM response
- `metadata.json` — Latency, token counts, timestamps
- `params.json` — Configuration parameters used
- `bundle_info.json` — Run metadata
- `runs/summary.csv` — Appended run summary

**Teaching Value:**
- Students see real LLM responses with measurable latency
- Prompt engineering principles (system prompt, few-shot examples, temperature)
- Token counting and generation parameters
- Same artifact reproducibility pattern as traditional ML and deep learning workflows

## Workflow Diagram

![alt text](image-1.png)
```text
notebooks/*  -> exploration only
data/raw/*   -> original source data
prepare_data.py -> raw to processed handoff
src/*        -> reusable production logic
reports/*    -> human-readable findings and decisions
```

## Weekly Course Rhythm

![alt text](image-2.png)

## Reproducibility Checklist

- Keep `data/raw/` unchanged after ingest.
- Create `data/processed/` from `prepare_data.py`, not by hand edits.
- Modify behavior through `configs/` before changing code.
- Use new run names (for example `run_002`) to avoid overwriting prior experiments.
- Record outcomes in `runs/summary.csv` and `reports/weekly/`.

## Notes

- Keep final reusable logic in `src/`, not in notebooks.
- Treat `data/raw/` as read-only.
- Save training-ready tables to `data/processed/`.
- Store per-experiment artifacts under `runs/`.
