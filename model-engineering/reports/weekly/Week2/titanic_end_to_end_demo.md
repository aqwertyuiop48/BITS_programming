# Titanic End-to-End Demo

This note is a ready-to-teach walkthrough for showing the full repo workflow:

raw CSV -> processed CSV -> train -> evaluate -> predict -> report

## Demo Objective

Predict passenger survival from raw Titanic passenger records while teaching:

- config-driven training
- reusable preprocessing
- train/inference consistency
- experiment traceability
- clear separation between exploration and production logic

## Files to Prepare

Place these files in `data/raw/`:

- `titanic_train.csv` with a label column named `Survived`
- `titanic_inference.csv` without the label column

This repo includes small Titanic-like sample CSVs under `data/raw/` so you can run the lesson immediately.

Recommended columns to keep:

- `Pclass`
- `Sex`
- `Age`
- `SibSp`
- `Parch`
- `Fare`
- `Embarked`
- `Survived` for training only

Recommended columns to drop during preparation:

- `PassengerId`
- `Name`
- `Ticket`
- `Cabin`

## Live Demo Script

### 1. Start with raw data

Open the raw CSV and tell students:

"This is real source data. It is useful for exploration, but it is not yet the stable contract our training code expects."

### 2. Create processed training data

Run:

```bash
python prepare_data.py --input-csv data/raw/titanic_train.csv --output-csv data/processed/train.csv --target-col target --rename-target-from Survived --drop-columns PassengerId,Name,Ticket,Cabin
```

Explain:

- the training code expects the label column to be named `target`
- we drop obvious identifier or messy text columns first
- the result is a cleaned training table in `data/processed/`

### 3. Train the model

Run:

```bash
python train.py --config configs/train/default.yaml
```

Explain:

- `train.py` reads config files, not hard-coded paths
- it loads `data/processed/train.csv`
- it splits the data into train and holdout sets
- it calls `src/features/preprocess.py` for imputation, scaling, and encoding
- it fits logistic regression and saves a serialized pipeline

### 4. Evaluate the saved run

Run:

```bash
python evaluate.py --run-dir runs/run_001 --target-col target
```

Explain:

- evaluation is separate from training
- the repo evaluates the saved model artifact on saved holdout data
- metrics are written back into the run folder

### 5. Prepare inference data from raw unlabeled rows

Run:

```bash
python prepare_data.py --input-csv data/raw/titanic_inference.csv --output-csv data/processed/inference_input.csv --target-col target --drop-columns PassengerId,Name,Ticket,Cabin --allow-missing-target
```

Explain:

- inference data often arrives without labels
- the preparation step still standardizes the columns
- the model will receive the same feature names and shape expectations as training

### 6. Run batch inference

Run:

```bash
python predict.py --run-dir runs/run_001 --input-csv data/processed/inference_input.csv
```

Explain:

- `predict.py` loads the saved pipeline from the run folder
- preprocessing is not rewritten manually
- this is how the repo enforces train/inference consistency

### 7. Review artifacts and close the loop

Open these outputs:

- `runs/run_001/metrics.json`
- `runs/run_001/evaluation_metrics.json`
- `runs/run_001/predictions_inference.csv`
- `runs/summary.csv`

Then summarize in:

- `reports/evaluation/baseline_report.md`
- or `reports/model_card_v1.md`

## What to Say at the End

"The important lesson is not just that we trained a classifier. The lesson is that we moved from raw source data to a reproducible, evaluated, inference-ready artifact without hiding critical logic in a notebook."

## Teaching Payoff

- Students see where raw data becomes model-ready data.
- Students see why preprocessing belongs in reusable code.
- Students see that evaluation uses held-out data, not training memory.
- Students see that prediction reuses the same pipeline as training.
- Students see how experiment results become documented decisions.