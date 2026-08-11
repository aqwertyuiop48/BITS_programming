# Supervised and Unsupervised Learning

## Purpose of This Document

This note explains the difference between supervised and unsupervised learning at two levels:

1. Concept level: what each learning style is trying to achieve.
2. Engineering level: how each style is trained and evaluated in this repository.

Use this document as a reference when deciding which workflow to run, how to interpret metrics, and how to explain results in reports.

## Core Idea in One Minute

- Supervised learning uses labeled data and learns to predict a known outcome.
- Unsupervised learning uses unlabeled data and learns patterns or structure.

In notation:

- Supervised: dataset has $(X, y)$ and we learn $f(X) \rightarrow y$.
- Unsupervised: dataset has only $X$ and we learn structure in $X$ (clusters, low-dimensional embeddings, anomalies).

Both methods can be useful in the same project, but they answer different questions.

## Supervised Learning

### What It Solves

Supervised learning answers: "Given what I know about this row, what outcome should I predict?"

In a Titanic setting, that is typically:

- Input features: age, fare, passenger class, family size, and encoded categories.
- Target label: survival status.

### Typical Tasks

- Classification: predict a category, such as survived or not survived.
- Regression: predict a continuous value, such as cost or time.

### Objective Function View

Training optimizes a loss function over labeled examples:

$$
\min_{\theta} \frac{1}{n}\sum_{i=1}^{n} L\left(f_{\theta}(x_i), y_i\right)
$$

where $L$ could be log loss for classification or squared error for regression.

### Common Algorithms

- Logistic regression
- Decision tree
- Random forest
- Gradient boosting
- Support vector machine

This repository includes several of these under the model configs.

### Supervised Workflow in This Repo

1. Prepare data into modeling format.
2. Run supervised training.
3. Evaluate on holdout data.
4. Save artifacts for reproducibility.

Example command:

```bash
python train.py --config configs/train/default.yaml
```

Related configs include:

- `configs/train/default.yaml`
- `configs/train/titanic_decision_tree.yaml`
- `configs/train/titanic_random_forest.yaml`
- `configs/train/titanic_gradient_boosting.yaml`
- `configs/train/titanic_svm.yaml`

Typical supervised artifacts in a run directory:

- `model.joblib`: serialized preprocessing + model pipeline.
- `metrics.json`: quantitative performance.
- `params.json`: captured run parameters.
- `predictions.csv`: model outputs.
- `holdout.csv`: holdout dataset for evaluation evidence.

### How Supervised Models Are Evaluated

Because labels exist, evaluation compares predictions to ground truth.

Common classification metrics:

- Accuracy: fraction of correct predictions.
- Precision: among predicted positives, how many are true positives.
- Recall: among true positives, how many are found.
- F1: harmonic mean of precision and recall.
- ROC-AUC: ranking quality across thresholds.

Confusion matrix terms:

- TP: true positives
- FP: false positives
- TN: true negatives
- FN: false negatives

Example formulas:

$$
	ext{Precision} = \frac{TP}{TP + FP}, \quad
	ext{Recall} = \frac{TP}{TP + FN}, \quad
F1 = 2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}
$$

### Common Supervised Pitfalls

- Data leakage from future or target-derived features.
- Overfitting to train data without holdout validation.
- Poor threshold choice when class distribution is imbalanced.
- Inconsistent preprocessing between train and inference.

## Unsupervised Learning

### What It Solves

Unsupervised learning answers: "What structure exists in this dataset when no label is provided?"

In Titanic-like data, this might mean:

- finding passenger segments with similar profiles,
- identifying dense groups and outliers,
- generating hypotheses for future labeling.

### Typical Tasks

- Clustering (k-means, DBSCAN)
- Dimensionality reduction (PCA, manifold methods)
- Anomaly detection

### Objective View

There is no single universal target. Each algorithm defines its own objective.

- K-means minimizes within-cluster squared distance to centroids.
- DBSCAN groups dense regions and marks sparse points as noise.

### Common Algorithms

- K-means: fast centroid-based clustering, must choose number of clusters.
- DBSCAN: density-based clustering, detects noise points, no fixed cluster count required.

This repository includes both approaches through unsupervised configs.

### Unsupervised Workflow in This Repo

1. Prepare features consistently with supervised pipeline standards.
2. Train clustering model.
3. Compute internal validation metrics.
4. Produce assignments and cluster profiles.
5. Write interpretation narrative and caveats.

Example commands:

```bash
python train_unsupervised.py --config configs/train/titanic_kmeans_unsupervised.yaml
python train_unsupervised.py --config configs/train/titanic_dbscan_unsupervised.yaml
```

The DBSCAN model parameters are configured in:

- `configs/models/titanic_dbscan.yaml`

and currently include:

- `eps: 1.2`
- `min_samples: 5`

Typical unsupervised artifacts in a run directory:

- `model.joblib`
- `metrics.json`
- `params.json`
- `cluster_assignments.csv`
- `cluster_profile.csv`

### How Unsupervised Models Are Evaluated

Without ground-truth labels, evaluation combines internal metrics plus interpretation.

Common internal metrics:

- Silhouette score: separation vs compactness (higher is usually better).
- Davies-Bouldin index: average similarity between clusters (lower is better).
- Calinski-Harabasz index: ratio of between-cluster to within-cluster dispersion (higher is better).

Additional checks:

- Number of clusters produced.
- Percentage of DBSCAN noise points.
- Stability across random seeds and preprocessing variants.
- Whether cluster summaries are interpretable and actionable.

### Unsupervised Pitfalls

- Treating one metric as absolute proof of business value.
- Ignoring feature scaling effects on distance-based clustering.
- Choosing k in k-means without sensitivity analysis.
- Using DBSCAN parameters that collapse to one giant cluster or mostly noise.
- Over-interpreting clusters that are not stable across reruns.

## Side-by-Side Comparison

| Dimension | Supervised | Unsupervised |
|---|---|---|
| Labels required | Yes | No |
| Primary objective | Predict known outcome | Discover latent structure |
| Typical output | Class/value prediction | Cluster id, profile, embedding |
| Evaluation basis | Ground truth comparison | Internal metrics + interpretability |
| Key risk | Leakage/overfitting | False sense of structure |
| Repo training script | `train.py` | `train_unsupervised.py` |
| Example train config | `configs/train/default.yaml` | `configs/train/titanic_dbscan_unsupervised.yaml` |

## Choosing the Right Approach

Use supervised learning when:

- the target is defined and available,
- decisions require measurable prediction quality,
- you need model accountability against known outcomes.

Use unsupervised learning when:

- labels are missing or expensive,
- your goal is exploration, segmentation, or anomaly detection,
- you want to generate candidate features or hypotheses.

Many mature workflows use both in sequence:

1. Unsupervised exploration to understand natural groups and outliers.
2. Feature and labeling refinement based on those insights.
3. Supervised training for stable prediction and controlled deployment.

## End-to-End Repo Mapping

### Supervised Path

1. Data prep to processed CSV.
2. Train using a supervised config.
3. Evaluate with holdout ground truth.
4. Save run artifacts and report findings.

Representative commands:

```bash
python prepare_data.py --input-csv data/raw/titanic_train.csv --output-csv data/processed/train.csv --target-col target --rename-target-from Survived
python train.py --config configs/train/default.yaml
python evaluate.py --run-dir runs/run_001 --target-col target
```

### Unsupervised Path

1. Data prep and feature transformations.
2. Train k-means or DBSCAN.
3. Review clustering metrics and profile tables.
4. Document segment meaning, caveats, and monitoring plan.

Representative commands:

```bash
python train_unsupervised.py --config configs/train/titanic_kmeans_unsupervised.yaml
python train_unsupervised.py --config configs/train/titanic_dbscan_unsupervised.yaml
```

## Reporting Guidance

For supervised reports, include:

- problem statement and prediction target,
- train/validation/holdout setup,
- key metrics and confusion matrix interpretation,
- failure patterns and threshold trade-offs,
- recommendation and known limitations.

For unsupervised reports, include:

- clustering objective and why clustering is appropriate,
- preprocessing and distance assumptions,
- metric values with caveats,
- per-cluster profile narrative,
- deployment risks and monitoring considerations.

## Practical Checklist

Before accepting a supervised model:

- Confirm no obvious leakage features.
- Confirm holdout evaluation exists.
- Confirm metrics align with decision cost.
- Confirm inference pipeline uses same preprocessing.

Before accepting an unsupervised clustering result:

- Confirm features are appropriately scaled.
- Confirm cluster counts and noise rates are reasonable.
- Confirm profiles are interpretable by domain stakeholders.
- Confirm results are stable under small perturbations.

## Final Takeaways

- Supervised learning is best when you need accountable predictions for a known target.
- Unsupervised learning is best when you need structure discovery in unlabeled data.
- In practice, they are complementary: unsupervised methods help you understand data; supervised methods help you operationalize decisions.
