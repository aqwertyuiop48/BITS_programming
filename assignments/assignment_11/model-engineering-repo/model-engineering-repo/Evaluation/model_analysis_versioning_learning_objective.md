# Model Analysis, Versioning, and Model Card v1

## Learning Objective

By the end of this lesson, learners can treat a trained model as a production-like artifact, not just a notebook result.

Learners will be able to:
- Analyze model behavior beyond headline accuracy.
- Identify error slices and interpret bias indicators.
- Recognize drift and training-serving skew risks.
- Explain model registry concepts and versioning policy.
- Package, version, and reproduce a training run.
- Write a complete Model Card v1.

## Repository Mapping (What To Use)

Core scripts:
- train.py
- evaluate.py
- predict.py
- leaderboard.py

Core config files for this lesson:
- configs/train/titanic_random_forest.yaml
- configs/data/titanic_bakeoff.yaml
- configs/features/titanic_all_features.yaml
- configs/models/titanic_random_forest.yaml

Core run artifacts:
- runs/run_titanic_random_forest/model.joblib
- runs/run_titanic_random_forest/metrics.json
- runs/run_titanic_random_forest/evaluation_metrics.json
- runs/run_titanic_random_forest/params.json
- runs/run_titanic_random_forest/bundle_info.json
- runs/run_titanic_random_forest/predictions.csv

## Concept Notes For Teaching

### 1) Error Slicing

Why:
- Aggregate metrics can hide weak segments.

How in this repo:
- Use predictions.csv to compute per-slice metrics (for example by Sex, Pclass, age bucket).

Teaching cue:
- Show that equal overall accuracy does not imply equal subgroup performance.

### 2) Bias Indicators

Why:
- Subgroup disparities are often early warning signals for unfair impact.

How in this repo:
- Compare positive recall and predicted-positive-rate across subgroup slices.
- Flag large deltas for review before model promotion.

### 3) Drift Risks

Why:
- Feature distributions can change between training and serving.

How in this repo:
- Compare train data (data/processed/titanic_bakeoff_train.csv) and inference data (data/processed/titanic_bakeoff_inference.csv).
- Monitor missingness deltas, numeric mean shifts, and categorical distribution changes.

### 4) Model Registry Concepts

Registry stores:
- Immutable versioned model bundles.
- Metadata: run_id, configs, metrics, card, owner, approval state.

Typical lifecycle:
- Candidate -> Staging -> Production -> Archived

Key rule:
- Promote only if reproducibility checks and quality/safety gates pass.

### 5) Artifact Packaging

Model package should contain:
- model.joblib
- bundle_info.json
- metrics.json
- evaluation_metrics.json
- params.json
- model_card_v1.md

Optional additions:
- dependency manifest, checksums, signature, changelog.

## Lab: Production-Like Model Versioning

### Lab Goal

Package and version a trained model artifact, reproduce the run, and produce Model Card v1.

### Step 1: Reproduce Training Run

Run:

```powershell
python train.py --config configs/train/titanic_random_forest.yaml
python evaluate.py --run-dir runs/run_titanic_random_forest --target-col target
```

Verify outputs in runs/run_titanic_random_forest:
- model.joblib
- metrics.json
- evaluation_metrics.json
- params.json
- bundle_info.json
- predictions.csv

### Step 2: Slice Error Analysis

Use predictions.csv and calculate per-slice:
- accuracy
- recall for positive class
- predicted-positive-rate

Required slices:
- Sex
- Pclass
- Age bucket

Discussion questions:
- Which slice has lowest recall?
- Is performance gap large enough to block promotion?

### Step 3: Check Drift Risks

Compare train and inference distributions:
- Missingness deltas per feature
- Numeric shifts (normalized by train std)
- Categorical total-variation distance

Define alert thresholds (example):
- numeric shift absolute value > 0.5
- categorical TV distance > 0.15
- missingness delta > 10 percentage points

### Step 4: Package Artifact

Example package layout:

```text
models/
  titanic_survival/
    v1.0.0/
      model.joblib
      bundle_info.json
      metrics.json
      evaluation_metrics.json
      params.json
      model_card_v1.md
```

Optional Windows zip example:

```powershell
New-Item -ItemType Directory -Force models/titanic_survival/v1.0.0 | Out-Null
Copy-Item runs/run_titanic_random_forest/model.joblib models/titanic_survival/v1.0.0/
Copy-Item runs/run_titanic_random_forest/bundle_info.json models/titanic_survival/v1.0.0/
Copy-Item runs/run_titanic_random_forest/metrics.json models/titanic_survival/v1.0.0/
Copy-Item runs/run_titanic_random_forest/evaluation_metrics.json models/titanic_survival/v1.0.0/
Copy-Item runs/run_titanic_random_forest/params.json models/titanic_survival/v1.0.0/
Copy-Item reports/model_card_v1.md models/titanic_survival/v1.0.0/
Compress-Archive -Path models/titanic_survival/v1.0.0/* -DestinationPath models/titanic_survival_v1.0.0.zip -Force
```

### Step 5: Versioning Decision

Assign semantic version:
- Major: contract-breaking change (feature/schema/model family incompatible)
- Minor: quality improvement, same contract
- Patch: metadata or packaging-only fix

For this lab output:
- Version = v1.0.0
- Status = candidate (promote after checklist and review)

### Step 6: Write and Review Model Card v1

Required sections:
- Intended use
- Dataset description
- Metrics
- Threshold policy
- Known failure modes
- Training-serving skew risks
- Deployment assumptions
- Inference constraints (latency expectations)

Use:
- reports/model_card_v1.md

## Assessment Rubric (Quick)

- Reproducibility complete (commands + artifacts + configs): 30%
- Analysis depth (slices + bias indicators + drift): 30%
- Versioning and packaging quality: 20%
- Model Card v1 completeness and clarity: 20%

## Deliverable

- Full Model Card v1: reports/model_card_v1.md
- Reproducibility checklist: included in reports/model_card_v1.md