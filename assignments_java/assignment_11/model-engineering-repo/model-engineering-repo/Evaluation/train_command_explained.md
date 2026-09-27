# What Happened When I Ran Training

Command used:

```bash
python train.py --config configs/train/default.yaml
```

This note explains the command in simple terms for non-ML readers.

## 1. Big Picture

Running this command does a full training cycle:

1. Reads settings (what files to use, where to save outputs).
2. Loads the dataset.
3. Splits data into a learning part and a testing part.
4. Trains a model on the learning part.
5. Tests the model on unseen data.
6. Saves model files and results for reproducibility.

Think of it as teaching a student, giving a test, and storing both the answer sheet and score report.

## 2. Files It Reads

The script starts from `train.py` and loads:

- `configs/train/default.yaml` (main run settings)
- `configs/data/default.yaml` (dataset path, target column, split settings)
- `configs/features/default.yaml` (feature handling options)
- `configs/models/baseline_logreg.yaml` (model type and hyperparameters)
- `data/processed/train.csv` (training dataset)

## 3. What It Does Step By Step

### Step A: Load configuration

From `configs/train/default.yaml`, it learns:

- run name (currently `run_001`)
- where run outputs should be stored (`runs/`)
- artifact filenames (`model.joblib`, `metrics.json`, etc.)

### Step B: Load data

From `configs/data/default.yaml`, it uses:

- dataset path: `data/processed/train.csv`
- target column: `target`
- test size: `0.2` (20%)
- random state: `42` (for reproducible splits)

If the dataset does not exist, the script auto-creates a demo dataset from sklearn breast cancer data so the pipeline still runs.

### Step C: Prepare features and target

- Input columns become **X** (features)
- `target` becomes **y** (the correct answer)

### Step D: Split data

The data is split into:

- **Train set (80%)**: used to learn patterns
- **Test/Holdout set (20%)**: used only for evaluation

In your run:

- total rows: 569
- holdout rows: 114
- predictions rows: 114

### Step E: Build preprocessing + model pipeline

The training pipeline does:

- Numeric missing values -> fill with median
- Categorical missing values -> fill with most frequent value
- Categorical text -> one-hot encoding
- Numeric columns -> scaling
- Model -> Logistic Regression

### Step F: Train and evaluate

The model is trained on train set, then predicts on holdout set.

Metrics are calculated, including:

- accuracy
- precision
- recall
- f1
- roc_auc

## 4. Files It Writes

Inside `runs/run_001/`:

- `model.joblib`: trained model object
- `metrics.json`: evaluation summary
- `params.json`: config references used in this run
- `predictions.csv`: row-level predictions with truth labels and score
- `holdout.csv`: saved holdout dataset used for testing

In `runs/`:

- `summary.csv`: experiment ledger with one row per run

## 5. Your Current Results (Plain English)

From `runs/run_001/metrics.json`:

- accuracy: `0.9824561403508771`
- precision: `0.9861111111111112`
- recall: `0.9861111111111112`
- f1: `0.9861111111111112`
- roc_auc: `0.9953703703703703`

Interpretation:

- Around 98 out of 100 predictions were correct.
- Balance between catching positives and avoiding wrong positive calls is strong.
- Ranking quality is excellent.

## 6. Important Operational Note

Because run name is fixed to `run_001` in config, running the same command again updates files in the same run folder.

If you want separate experiment history, change run name (for example, `run_002`) before retraining.

## 7. Recommended Next Steps

1. Run formal evaluation script:

```bash
python evaluate.py --run-dir runs/run_001
```

2. Keep experiments separate by changing `run_name` before each new run.
3. Inspect error rows in `runs/run_001/predictions.csv` where `y_true != y_pred`.
