# Titanic Feature Reduction Bakeoff Report

## Objective

Compare three pipelines using the same Titanic passenger dataset to show the
practical impact of feature strategy on training, evaluation, and inference.

1. **All features** — Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
2. **Selected features** — Pclass, Sex, Age, Fare, Embarked (SibSp + Parch dropped)
3. **PCA** — all features compressed to 4 principal components

## Dataset

- Raw training: `data/raw/titanic_train.csv` (200 rows, original 12 preserved)
- Raw inference: `data/raw/titanic_inference.csv` (50 rows, unlabeled)
- Processed training: `data/processed/titanic_bakeoff_train.csv`
- Processed inference: `data/processed/titanic_bakeoff_inference.csv`
- Columns after preparation: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, target
- After one-hot encoding Sex and Embarked: ~9 model columns

## Repro Commands

```bash
python make_titanic_demo_data.py

python prepare_data.py --input-csv data/raw/titanic_train.csv \
  --output-csv data/processed/titanic_bakeoff_train.csv \
  --target-col target --rename-target-from Survived \
  --drop-columns PassengerId,Name,Ticket,Cabin

python prepare_data.py --input-csv data/raw/titanic_inference.csv \
  --output-csv data/processed/titanic_bakeoff_inference.csv \
  --target-col target --drop-columns PassengerId,Name,Ticket,Cabin \
  --allow-missing-target

python train.py    --config configs/train/titanic_all_features.yaml
python evaluate.py --run-dir runs/run_titanic_all      --target-col target
python predict.py  --run-dir runs/run_titanic_all      --input-csv data/processed/titanic_bakeoff_inference.csv --output-csv runs/run_titanic_all/predictions_inference.csv

python train.py    --config configs/train/titanic_selected_features.yaml
python evaluate.py --run-dir runs/run_titanic_selected --target-col target
python predict.py  --run-dir runs/run_titanic_selected --input-csv data/processed/titanic_bakeoff_inference.csv --output-csv runs/run_titanic_selected/predictions_inference.csv

python train.py    --config configs/train/titanic_pca_features.yaml
python evaluate.py --run-dir runs/run_titanic_pca      --target-col target
python predict.py  --run-dir runs/run_titanic_pca      --input-csv data/processed/titanic_bakeoff_inference.csv --output-csv runs/run_titanic_pca/predictions_inference.csv
```

## Results

| Run | Strategy | Features In | Accuracy | F1 | ROC-AUC | Artifacts |
|---|---|---:|---:|---:|---:|---|
| run_titanic_all | All features | 7 cols → ~9 model | 0.74 | 0.698 | 0.792 | `runs/run_titanic_all/` |
| run_titanic_selected | Selected (drop SibSp, Parch) | 5 cols → ~7 model | 0.74 | 0.711 | 0.800 | `runs/run_titanic_selected/` |
| run_titanic_pca | PCA compressed | 9 model → 4 PCs | 0.62 | 0.558 | 0.609 | `runs/run_titanic_pca/` |

## What Each Result Teaches

### All features vs Selected features

- Accuracy is identical (0.74).
- F1 and ROC-AUC slightly improve with selection (0.711 vs 0.698; 0.800 vs 0.792).
- SibSp and Parch were dropped. They carry some survival signal in the full Titanic
  dataset but are weak when used independently in a small sample.
- **Lesson:** removing redundant or weak columns can tighten metrics and always
  reduces model complexity, maintenance, and monitoring burden.
- The feature config change (`configs/features/titanic_selected_features.yaml`)
  is the only thing that changed. No model code was touched.

### All features vs PCA

- PCA with 4 components drops accuracy from 0.74 to 0.62 and ROC-AUC from 0.792 to 0.609.
- After one-hot encoding, the model has ~9 input dimensions. Compressing to 4 PCs
  throws away roughly half the variance.
- The categorical features (Sex, Embarked) are binarized before PCA, which makes
  PCA compression on this dataset lossy and harder to reason about.
- **Lesson:** PCA is not always the right reduction tool. On small mixed-type
  datasets, aggressive compression can hurt measurably. Interpretability loss is
  also significant — you can no longer say "Sex was the strongest predictor."

### Prediction impact

All 50 inference rows received predictions and scores in each run.
Open the three prediction files side by side to show:

- `runs/run_titanic_all/predictions_inference.csv`
- `runs/run_titanic_selected/predictions_inference.csv`
- `runs/run_titanic_pca/predictions_inference.csv`

Some rows near the decision boundary flip between runs — good for class discussion
on how feature strategy changes operational decisions, not just aggregate metrics.

## Run Ledger

Check `runs/summary.csv` for all three rows side by side.

## Recommendation

For this dataset and objective, use **selected features** (Pclass, Sex, Age, Fare,
Embarked). It matches the accuracy of all features, improves F1 and ROC-AUC by a
small margin, uses fewer columns, and keeps every prediction explainable in plain
terms: class, gender, age, fare paid, and port of departure.

PCA is not recommended here because the dataset is small, features are mixed-type,
and interpretability is a meaningful requirement for survival prediction.

## Feature Config Reference

| Config | File |
|---|---|
| All features | `configs/features/titanic_all_features.yaml` |
| Selected features | `configs/features/titanic_selected_features.yaml` |
| PCA | `configs/features/titanic_pca_features.yaml` |
