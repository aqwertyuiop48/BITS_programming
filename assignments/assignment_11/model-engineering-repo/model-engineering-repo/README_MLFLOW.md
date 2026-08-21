# MLflow README for Students

This guide explains MLflow from first principles through setup and hands-on usage in this repository.

## 1. What is MLflow?

MLflow is an open source platform for managing the machine learning lifecycle.

At a high level, MLflow helps you:
- Track experiments (parameters, metrics, artifacts)
- Compare runs and pick winners
- Package and register models with versions
- Preserve reproducibility and auditability

Why this matters in class:
- You move from "I trained a model once" to "I can reproduce, compare, and promote models like production artifacts."

## 2. Core MLflow Concepts

### 2.1 Tracking Server / Tracking URI
Where MLflow stores run metadata.

Common options:
- File store: file:./mlruns
- SQLite backend: sqlite:///mlflow.db

### 2.2 Experiment
A logical container for related runs.

Example:
- ml-demo-supervised
- ml-demo-deep
- ml-demo-unsupervised

### 2.3 Run
One execution of a training or evaluation command.

A run contains:
- Params: seed, config paths, model params
- Metrics: accuracy, f1, roc_auc, clustering metrics, etc.
- Artifacts: configs, model files, predictions, bundle metadata
- Tags: task type, run name, model type

### 2.4 Model Registry
A versioned catalog of models.

Example:
- Model name: titanic_survival_rf
- Versions: 1, 2, 3, ...

## 3. What was added in this repo

MLflow was integrated into these scripts:
- train.py
- train_deep.py
- train_unsupervised.py
- evaluate.py

Shared helper layer:
- src/utils/mlflow_utils.py

Dependency:
- requirements.txt includes mlflow>=2.13.0

Docs:
- README.md includes quick MLflow usage snippets

## 4. New CLI Flags

Training scripts now support:
- --mlflow
- --mlflow-tracking-uri
- --mlflow-experiment
- --mlflow-run-name
- --mlflow-register-model
- --mlflow-model-name
- --mlflow-model-artifact-path

Evaluation script supports:
- --mlflow
- --mlflow-tracking-uri
- --mlflow-experiment
- --mlflow-run-name

Safety rule:
- --mlflow-register-model requires --mlflow

## 5. Setup

## 5.1 Create/activate environment (recommended)

PowerShell:

```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& .\.venv\Scripts\Activate.ps1)
```

## 5.2 Install dependencies

```powershell
python -m pip install -r requirements.txt
```

If you are not using venv, be consistent with one interpreter for all installs and runs.

## 6. Quick Start Commands

## 6.1 Supervised training with MLflow

```powershell
python train.py --config configs/train/titanic_random_forest.yaml --mlflow
```

## 6.2 Evaluation with MLflow

```powershell
python evaluate.py --run-dir runs/run_titanic_random_forest --target-col target --mlflow
```

## 6.3 Deep learning training with MLflow

```powershell
python train_deep.py --config configs/train/digits_cnn.yaml --mlflow
```

## 6.4 Unsupervised training with MLflow

```powershell
python train_unsupervised.py --config configs/train/titanic_kmeans_unsupervised.yaml --mlflow
```

## 7. Local Registry Setup (Recommended)

Use SQLite for local demos where registry/versioning is needed.

## 7.1 Train + register supervised model

```powershell
python train.py --config configs/train/titanic_random_forest.yaml --mlflow --mlflow-tracking-uri sqlite:///mlflow.db --mlflow-experiment ml-demo-registry-validation --mlflow-run-name rf-register-check --mlflow-register-model --mlflow-model-name titanic_survival_rf
```

## 7.2 Train + register unsupervised model

```powershell
python train_unsupervised.py --config configs/train/titanic_kmeans_unsupervised.yaml --mlflow --mlflow-tracking-uri sqlite:///mlflow.db --mlflow-register-model --mlflow-model-name titanic_kmeans_demo
```

## 7.3 Start UI

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:
- http://127.0.0.1:5000

## 8. What gets logged in this repo

### 8.1 Params
Examples:
- seed
- data_config, feature_config, model_config
- model hyperparameters
- cv settings

### 8.2 Metrics
Examples:
- Supervised: accuracy, precision, recall, f1, roc_auc, cv_* 
- Deep: accuracy, f1_macro, train loss, best test accuracy
- Unsupervised: silhouette, davies_bouldin, calinski_harabasz, cluster_count, noise_ratio
- Evaluation: eval_accuracy, eval_f1, eval_roc_auc, etc.

### 8.3 Artifacts
Examples:
- config YAML files
- run artifacts folder (model file, metrics.json, params.json, predictions, bundle_info)
- evaluation_metrics.json

### 8.4 Tags
Examples:
- task
- run_name
- model_type
- run_dir (evaluation)

## 9. Suggested Classroom Demo Flow (15-20 min)

1. Explain concepts (experiment, run, params/metrics/artifacts, registry).
2. Run one supervised training command with --mlflow.
3. Open MLflow UI and inspect run details.
4. Run a second model config and compare metrics.
5. Register best run using --mlflow-register-model.
6. Show model versions in registry.
7. Explain how this maps to real promotion workflows.

## 10. Troubleshooting

### 10.1 Module not found (mlflow or langchain packages)
Cause:
- Install and run are using different Python interpreters.

Fix:
- Use the same interpreter for both install and run.
- Check with:

```powershell
python -c "import sys; print(sys.executable)"
python -m pip --version
```

### 10.2 Registration does not appear
Cause:
- File backend used or wrong tracking URI.

Fix:
- Prefer sqlite:///mlflow.db for local registry demos.
- Ensure you passed --mlflow-register-model and --mlflow-model-name.

### 10.3 Warning about file backend deprecation
Meaning:
- file:./mlruns still works, but DB backends are preferred.

Fix:
- Move to sqlite:///mlflow.db or another DB backend.

### 10.4 Invalid API key in RAG demos
This is unrelated to MLflow.
Set a valid key in the expected env var (for Gemini config: GEMINI_API_KEY).

## 11. Best Practices for Students

- Keep run names meaningful and consistent.
- Log config files as artifacts for reproducibility.
- Treat registered model versions as immutable snapshots.
- Use experiment names to separate tasks (supervised vs deep vs unsupervised).
- Always compare runs before registering/promoting.

## 12. One-command cheat sheet

Supervised tracked run:

```powershell
python train.py --config configs/train/titanic_random_forest.yaml --mlflow --mlflow-tracking-uri sqlite:///mlflow.db --mlflow-experiment ml-demo --mlflow-run-name rf_v1
```

Register the model:

```powershell
python train.py --config configs/train/titanic_random_forest.yaml --mlflow --mlflow-tracking-uri sqlite:///mlflow.db --mlflow-experiment ml-demo --mlflow-run-name rf_v1_register --mlflow-register-model --mlflow-model-name titanic_survival_rf
```

Open UI:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

You now have a full experiment log + versioned registry record suitable for production-style teaching demos.
