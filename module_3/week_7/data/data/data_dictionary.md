# Data Dictionary

| Column | Type | Meaning | Allowed Values | Notes |
|---|---|---|---|---|
| feature_1 | float | Example numeric feature | Any real number | Check outliers |
| feature_2 | category | Example categorical feature | A, B, C | Verify category drift |
| target | int | Binary label | 0, 1 | Confirm no leakage |

## Leakage and Missingness Warnings

- Do not include post-outcome fields as model features.
- Track missingness patterns before and after preprocessing.

## Titanic Sample Dataset Walkthrough

This repo includes small Titanic-like sample files for end-to-end teaching:

- `data/raw/titanic_train.csv`
- `data/raw/titanic_inference.csv`

These files support a simple binary classification problem:

- predict whether a passenger survived
- training data includes the label column `Survived`
- inference data excludes the label because it represents new unseen rows

### Teaching Objective

Use the Titanic sample to demonstrate:

- raw data vs processed data
- target column handling
- numeric and categorical preprocessing
- train/evaluate/predict workflow
- why some columns should be dropped before modeling

### Titanic Column Guide

| Column | Type | Meaning | Example Values | Modeling Notes |
|---|---|---|---|---|
| PassengerId | int | Passenger row identifier | 1, 2, 101 | Useful for traceability, usually dropped for modeling |
| Survived | int | Training label | 0, 1 | Renamed to `target` during preparation |
| Pclass | int/category | Ticket class | 1, 2, 3 | Can act as a strong socioeconomic proxy |
| Name | text | Passenger name | Anna Harper | Usually dropped in this baseline repo |
| Sex | category | Passenger sex | male, female | Good baseline categorical feature |
| Age | float | Passenger age | 16, 32, 45 | Has missing values; imputation is part of preprocessing |
| SibSp | int | Siblings/spouses aboard | 0, 1 | Small-count numeric feature |
| Parch | int | Parents/children aboard | 0, 1, 2 | Small-count numeric feature |
| Ticket | text | Ticket identifier | PC 17599 | Usually noisy for a simple baseline |
| Fare | float | Fare paid | 7.25, 82.1708 | Useful numeric feature |
| Cabin | text | Cabin identifier | C85, B28 | Often sparse or missing |
| Embarked | category | Port of embarkation | S, C, Q | Good low-cardinality categorical feature |

### Recommended Baseline Modeling Choice

For the simple classroom demo:

- keep `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`
- drop `PassengerId`, `Name`, `Ticket`, `Cabin`
- rename `Survived` to `target`

This choice keeps the lesson focused on structured tabular modeling rather than text handling.

### Raw to Processed Example

Training preparation:

```bash
python prepare_data.py --input-csv data/raw/titanic_train.csv --output-csv data/processed/titanic_bakeoff_train.csv --target-col target --rename-target-from Survived --drop-columns PassengerId,Name,Ticket,Cabin
```

Inference preparation:

```bash
python prepare_data.py --input-csv data/raw/titanic_inference.csv --output-csv data/processed/titanic_bakeoff_inference.csv --target-col target --drop-columns PassengerId,Name,Ticket,Cabin --allow-missing-target
```
