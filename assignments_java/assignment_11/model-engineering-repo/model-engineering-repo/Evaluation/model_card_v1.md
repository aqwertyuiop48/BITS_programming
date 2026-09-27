# Model Card v1

## Model Identity

- Model name: Titanic Survival Classifier (Random Forest)
- Version: 1.0.0
- Run ID: run_titanic_random_forest
- Training timestamp (UTC): 2026-04-09T15:01:15.701179+00:00
- Artifact path: runs/run_titanic_random_forest/model.joblib
- Bundle descriptor: runs/run_titanic_random_forest/bundle_info.json

## Intended Use

This model is intended for classroom instruction on reproducible supervised ML workflows, including training, evaluation, artifact packaging, and model documentation.

Approved use:
- Offline batch inference over Titanic-like tabular passenger data.
- Demonstrations of model analysis topics such as thresholding, error slicing, drift checks, and reproducibility.

Not approved use:
- Real-world safety-critical or legal decisions.
- Use on data with materially different schema/population without retraining and revalidation.

## Dataset Description

- Dataset config: configs/data/titanic_bakeoff.yaml
- Training table: data/processed/titanic_bakeoff_train.csv
- Inference demo table: data/processed/titanic_bakeoff_inference.csv
- Target column: target
- Split policy: 75/25 holdout with stratification and random_state=42

Input features expected by the serialized pipeline:
- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

Preprocessing assumptions:
- One-hot encoding for categorical columns.
- Numeric scaling enabled.
- No PCA in this run (pca_enabled: false).

Known data caveats:
- Missing values exist in Age and are handled by the training pipeline.
- Dataset is small enough that metric variance across folds is non-trivial.

## Metrics

Source: runs/run_titanic_random_forest/metrics.json and runs/run_titanic_random_forest/evaluation_metrics.json

Holdout metrics:
- Accuracy: 0.7600
- Precision: 0.8333
- Recall: 0.6250
- F1: 0.7143
- ROC-AUC: 0.8173

Cross-validation summary (5-fold, ROC-AUC):
- Mean: 0.8221
- Std: 0.0620
- Fold values: [0.8973, 0.8889, 0.7467, 0.7644, 0.8133]

Interpretation:
- Precision is materially higher than recall, indicating the model is conservative about predicting positive class.
- A threshold policy should be explicit because operating point strongly changes false negative rate.

## Threshold Policy

Current production-like default:
- Decision threshold: 0.50 on model probability.

Policy guidance:
- If missing positive cases is costly, tune threshold lower (for example 0.40-0.45) to increase recall.
- If false positives are costlier, keep threshold at or above 0.50.

Release gate (minimum acceptable):
- ROC-AUC >= 0.78
- F1 >= 0.68
- Recall >= 0.60

Escalation policy:
- If any gate fails during retrain, mark candidate model as "do not promote" and run error-slice review before registry update.

## Error Slicing and Bias Indicators

Slice metrics from runs/run_titanic_random_forest/predictions.csv:

By Sex:
- female: n=20, accuracy=0.8000, positive recall=0.8750, predicted-positive-rate=0.8000
- male: n=30, accuracy=0.7333, positive recall=0.1250, predicted-positive-rate=0.0667

By Pclass:
- class 1: n=9, accuracy=0.6667, positive recall=0.6667
- class 2: n=15, accuracy=0.8667, positive recall=0.7500
- class 3: n=26, accuracy=0.7308, positive recall=0.5000

By age bucket:
- <=16: n=9, accuracy=0.7778, positive recall=0.6667
- 17-30: n=17, accuracy=0.8235, positive recall=0.8182
- 31-50: n=17, accuracy=0.7647, positive recall=0.5000

Bias indicator interpretation:
- Large difference in positive recall between male and female slices suggests disparate error behavior that must be discussed before promotion.
- This is a teaching dataset and not a real policy model, but the same fairness review process should apply.

## Known Failure Modes

- Under-recall for some subgroups (especially male slice in this run).
- Performance drops on slices with low sample size and sparse signal.
- Sensitivity to data quality issues in Age and Embarked.
- Potential instability when retraining on small datasets due to sampling variance.

## Training-Serving Skew Risks

- Schema mismatch risk: serving data must contain exactly the expected feature names.
- Categorical domain drift: unseen category frequencies for Sex/Embarked can shift one-hot behavior.
- Missingness pattern shift: training Age missingness differs from inference file and can impact imputation behavior.

Skew controls:
- Validate input schema against bundle_info.json features before inference.
- Monitor missingness rates and category frequencies against training baseline.
- Keep preprocessing embedded in serialized pipeline; do not reimplement preprocessing ad hoc in serving code.

## Drift Risks

Simple train-vs-inference drift indicators (demo comparison):
- Age missingness: train 14.5% vs inference 6.0% (delta -8.5pp)
- Categorical distribution total-variation shift:
	- Embarked: 0.0600
	- Sex: 0.0400
- Numeric mean shift in std units:
	- Age: +0.1285
	- Fare: -0.1578
	- Pclass: +0.1014

Drift alert guidance:
- Warn if any numeric |z-shift| > 0.5
- Warn if categorical TV distance > 0.15
- Warn if missingness delta > 10 percentage points

## Deployment Assumptions

- Batch inference context (CSV in, CSV out).
- Same feature definitions and value semantics as training data contract.
- Model file and bundle descriptor are deployed together.
- Python runtime and package versions are pinned from requirements.txt for reproducibility.

## Inference Constraints (Latency Expectations)

Measured on demo environment using 50-row batch, 50 iterations:
- Avg batch latency: 69.188 ms
- Avg per-row latency: 1.3838 ms

Operational expectation:
- Batch jobs should keep average latency below 100 ms per 50-row batch on similar hardware.
- Significant regression (>2x) should block promotion pending investigation.

## Model Registry Concepts and Versioning Guidance

Treat each trained run as a candidate immutable artifact:
- Candidate payload: model.joblib, bundle_info.json, metrics.json, evaluation_metrics.json, params.json.
- Registry metadata should include model name, semantic version, run_id, data config, feature config, model config, and metrics.

Versioning policy (recommended):
- MAJOR: schema/feature contract break or model family change that breaks compatibility.
- MINOR: compatible quality improvement with same IO contract.
- PATCH: metadata/documentation fix or non-behavioral packaging fix.

Promotion states:
- Staging: passes reproducibility and quality gates.
- Production: approved after slice analysis + drift risk review.
- Archived: retained for audit and rollback.

## Reproducibility Checklist

### Training Reproduction

- [x] Fixed seed captured (seed=42 in configs/train/titanic_random_forest.yaml)
- [x] Data config recorded (configs/data/titanic_bakeoff.yaml)
- [x] Feature config recorded (configs/features/titanic_all_features.yaml)
- [x] Model config recorded (configs/models/titanic_random_forest.yaml)
- [x] Train command reproducible:
	- python train.py --config configs/train/titanic_random_forest.yaml
- [x] Evaluation command reproducible:
	- python evaluate.py --run-dir runs/run_titanic_random_forest --target-col target

### Artifact Integrity

- [x] model.joblib present
- [x] bundle_info.json present
- [x] metrics.json present
- [x] evaluation_metrics.json present
- [x] params.json present
- [x] predictions.csv present

### Packaging and Versioning

- [x] Artifacts can be packaged as a versioned bundle (zip/tar)
- [x] Version tag assigned (v1.0.0 for this card)
- [x] Threshold gates documented
- [x] Failure modes documented
- [x] Training-serving skew risks documented
- [x] Deployment assumptions documented
- [x] Inference latency expectations documented

### Suggested Artifact Package Layout

Create package for registry upload:

1. Create a package directory:
	 - models/titanic_survival/v1.0.0/
2. Copy files:
	 - model.joblib
	 - bundle_info.json
	 - metrics.json
	 - evaluation_metrics.json
	 - params.json
	 - model_card_v1.md
3. Add checksum file (recommended) and archive.

This card is v1 and should be updated after each promoted retrain.
