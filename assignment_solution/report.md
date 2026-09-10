# Taking the recommendation model to production

## 1. Containerization

The solution uses FastAPI and Uvicorn for a small typed inference API. The
Dockerfile uses `python:3.11-slim` because it reduces image size and attack
surface relative to a full Python image. Its builder stage installs packages
into `/deps`; the runtime stage copies only those packages, the application,
and the approved model artifact. Pip caches and build tooling never ship. The
runtime changes to the unprivileged `appuser` (UID 10001), and the model should
be mounted read-only. CI scans the completed image for high and critical CVEs.

## 2. Deployment choice

Kubernetes is the deployment target. A 50 MB model and a 200 ms p95 objective
make pure serverless risky: cold starts can consume the budget just as the
evening surge arrives. A managed endpoint is a good lower-operations alternative,
but Kubernetes gives direct portable control over traffic splitting and scaling.
A serverless alternative would need provisioned concurrency or warm pools,
reducing its cost advantage.

Start with two warm pods. HPA scales from two to twenty replicas using request
rate and CPU, with headroom for the 10x evening spike. Cluster Autoscaler adds
nodes if pods cannot schedule. Establish the per-pod concurrent request limit
through load testing before choosing the HPA target. Readiness probes, a Pod
Disruption Budget, and rolling capacity prevent a deployment from removing
healthy capacity.

## 3. Optimization

`optimize_model.py` converts the ONNX candidate using dynamic INT8
quantization. For this CPU-oriented classifier it should reduce artifact size,
memory use, and inference latency, at the cost of a possible small accuracy
change. The optimization script scores FP32 and INT8 on held-out data and fails
the release when the accuracy drop exceeds 1 percentage point. CI records model
size and runs a p95 load test, promoting only if the model remains below 200 ms.

## 4. Monitoring and drift

Operational metrics are throughput, p50/p95/p99 latency, HTTP error rate,
timeouts, CPU/memory, restarts, and replica count. Model-level metrics are
feature drift, missing-feature rate, prediction distribution, and delayed-label
accuracy/precision/recall. Both are necessary: uptime cannot show a quietly
degrading model.

`monitoring/drift_monitor.py` uses a two-sample Kolmogorov-Smirnov test for
each continuous feature, comparing a rolling live-input window with the training
baseline. A p-value below 0.05 indicates feature drift. Alert only when three
important features drift for three consecutive hourly windows; this filters
short-lived noise. Prometheus collects service/Kubernetes metrics, Evidently or
the supplied monitor publishes drift results, and Grafana visualizes them.
Slack receives warnings; PagerDuty pages on-call only for sustained drift, p95
SLO failures, or material label-based quality regressions.

## 5. CI/CD and rollback

The pipeline runs unit tests, then an ML-specific gate that loads the candidate
and rejects it when its held-out accuracy is over 1% below production. It then
builds and scans the image, registers immutable image/model digests, and deploys
a 10% Argo Rollouts canary. Compare candidate and stable latency, error rate,
prediction distribution, and delayed-label performance before increasing traffic
to 25%, 50%, then 100%.

Rollback is immediate and requires no rebuild. The model registry records the
last approved artifact and the stable ReplicaSet remains running during the
canary. If a threshold breaches, abort the rollout and route 100% of traffic
back to the stable ReplicaSet. Quarantine the failed model version, retain its
metrics for incident review, and retrain or investigate before another canary.

## Architecture

`architecture.mmd` shows the path: Git push → tests → ML accuracy gate → build
and scan → registry → canary → Kubernetes/HPA → metrics and drift monitor →
Grafana → Slack/PagerDuty → rollback or retraining.
