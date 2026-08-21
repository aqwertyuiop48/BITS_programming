
# Model Engineering Reference

A comprehensive guide to machine learning model families, their use cases, strengths, and tradeoffs. This reference helps practitioners select the right model family based on data characteristics, problem requirements, computational constraints, and deployment needs.

## Overview

**Classical supervised ML** — You have labeled examples (input + correct answer). The model learns patterns from these examples to predict outcomes on new, unseen data. Think of it like learning from a textbook with worked solutions. Best for straightforward prediction tasks where you have historical data.

**Unsupervised learning** — You have data but no labels or "correct answers." The model finds hidden patterns, groups, or structure on its own. Like sorting a pile of photos by similarity without being told what to look for. Useful for exploration and understanding your data better.

**Dimensionality reduction** — Real data often has many features (columns), which can slow down models and create noise. This family compresses data while keeping the important information, like summarizing a long document into key points. Helps make models faster and sometimes reveals hidden patterns.

**Anomaly detection** — Find the unusual or suspicious cases in your data—fraud, equipment failure, or rare events. Think of a security system flagging odd behavior. Works by learning what "normal" looks like, then flagging cases that deviate significantly from it.

**Recommender systems** — Suggest relevant items: products, movies, articles. Uses patterns from your behavior (and similar users) to predict what you'll like next. Powers Netflix suggestions, Amazon recommendations, and Spotify playlists.

**Deep learning** — Powerful neural networks that automatically discover features and patterns in complex data like images, text, and video. Requires large datasets and computing power but can solve highly nonlinear problems humans can't easily hand-program.

**Foundation / LLM systems** — Large pretrained models (like ChatGPT) that understand language and can perform many tasks without retraining. Use them for text understanding, generation, translation, and multimodal tasks. Accessible but expensive and prone to hallucinations.

**Reinforcement learning** — An agent learns by taking actions, receiving rewards or penalties, and adjusting strategy over time. Like training a dog with treats or teaching a computer to play games. Good for sequential decision-making and adaptive systems.

**Decision layers** — Bridge between model predictions and real-world actions. Convert model scores (e.g., 0.75 probability) into business decisions (approve/reject) with thresholds, confidence rules, or human review. Ensures predictions translate to safe, explainable outcomes.


## Model Families Reference

<details>
<summary><strong>1) Classical supervised ML</strong> — predict known outcomes from labeled data</summary>

- Linear models (Linear Regression, Logistic Regression, Ridge, Lasso, Elastic Net, SGD, GLMs)
- Tree-based models (Decision Tree, Random Forest, Extra Trees, Bagging)
- Boosting models (Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost)
- Margin / kernel methods (Linear SVM, Kernel SVM, SVR)
- Distance / instance-based (k-NN classifier, k-NN regressor)
- Probabilistic models (Naive Bayes, Bayesian linear models, graphical models)
- Time-series forecasting (ARIMA, ETS, Prophet, state-space models)

</details>

<details>
<summary><strong>2) Unsupervised learning</strong> — discover structure without labels</summary>

- Clustering (K-Means, DBSCAN, Hierarchical, GMM, Spectral, Mean Shift)
- Association / pattern discovery (Apriori, FP-Growth)
- Topic / latent structure (LDA, NMF, PLS)

</details>

<details>
<summary><strong>3) Dimensionality reduction</strong> — make data smaller without losing signal</summary>

- Linear reduction (PCA, Truncated SVD, ICA, Factor Analysis)
- Nonlinear / manifold methods (UMAP, t-SNE, Isomap, LLE)
- Learned compression (Autoencoders, VAEs)

</details>

<details>
<summary><strong>4) Anomaly detection</strong> — find what looks abnormal</summary>

- Isolation Forest, One-Class SVM, Local Outlier Factor, Robust covariance, Reconstruction-based methods

</details>

<details>
<summary><strong>5) Recommender systems</strong> — suggest the next best item or action</summary>

- Collaborative filtering (User-user, Item-item, Matrix factorization)
- Content-based recommenders
- Hybrid recommenders
- Ranking / learning-to-rank (LambdaMART, Pairwise, Neural ranking)

</details>

<details>
<summary><strong>6) Deep learning</strong> — learn complex patterns with neural networks</summary>

- Feedforward / tabular nets (MLP, Wide & Deep, DeepFM, TabNet)
- Computer vision (CNNs, Vision Transformers, Detection/Segmentation)
- Sequence / time-series (RNN, LSTM, GRU, Temporal CNNs, Transformers)
- Generative models (GANs, VAEs, Diffusion, Autoregressive)
- Graph learning (GCN, GraphSAGE, GAT, Graph Transformers)

</details>

<details>
<summary><strong>7) Foundation / LLM systems</strong> — general-purpose pretrained models</summary>

- Text embeddings (Sentence-transformers, OpenAI, Gemini)
- Text foundation models (GPT, Claude, Llama, Gemini, Mistral, Cohere, Qwen)
- Multimodal models (Image+text, Audio+text, Video+text, Document-aware)
- Specialized patterns (Prompted, Fine-tuned, RAG, Assistants, Agents)
- Edge / on-device models (Distilled, Quantized, Mobile)

</details>

<details>
<summary><strong>8) Reinforcement learning</strong> — learn by trial, feedback, and reward</summary>

- Bandits (epsilon-greedy, UCB, Thompson sampling)
- Classical RL (Q-learning, SARSA, DQN)
- Policy-gradient / actor-critic (REINFORCE, A2C, PPO, SAC)

</details>

<details>
<summary><strong>9) Decision layers</strong> — turn model scores into usable actions</summary>

- Thresholded classifiers, Calibrated probability models, Abstain/reject-option, Human-in-the-loop, Rules + model hybrids, Multi-stage cascades

</details>

## Comparison Table

| Model Family | Typical Examples | Best Used For | Strengths | Tradeoffs / Cautions |
|---|---|---|---|---|
| **Linear models** | Linear Regression, Logistic Regression, Ridge, Lasso, Elastic Net | Simple prediction problems, strong baselines, interpretable tabular tasks | Fast, easy to explain, easy to deploy | May underfit complex patterns |
| **Tree-based models** | Decision Tree, Random Forest | Tabular classification/regression, mixed feature types | Strong performance, handles nonlinearity well | Can become less interpretable as complexity grows |
| **Boosting models** | Gradient Boosting, XGBoost, LightGBM, CatBoost | High-performing tabular tasks | Often excellent accuracy on structured data | More tuning, complexity, harder to explain |
| **SVMs** | Linear SVM, Kernel SVM | Medium-sized classification tasks, margin-based separation | Can work well on clean datasets | Harder to scale and explain in many settings |
| **Nearest-neighbor models** | k-NN | Small datasets, similarity-based decisions | Simple intuition | Can be slow at inference, sensitive to scaling |
| **Probabilistic models** | Naive Bayes, Gaussian models | Text classification, simple probabilistic baselines | Fast, lightweight, good baseline tools | Strong assumptions may limit performance |
| **Dimensionality reduction** | PCA, Truncated SVD, ICA | Compressing overlapping numeric features, preprocessing | Reduces complexity and redundancy | Reduced interpretability |
| **Clustering models** | K-Means, DBSCAN, Hierarchical, GMM | Segmentation, pattern discovery, unlabeled data | Useful for exploration and grouping | Harder to validate; need business meaning |
| **Neural networks** | MLP, CNN, RNN/LSTM, Transformers | Complex nonlinear tasks, images, text, sequences | Powerful and flexible | More compute, harder debugging, reproducibility burden |
| **LLM systems** | GPT, Claude, Llama, Gemini, Mistral | Language understanding/generation tasks | Strong for unstructured text | Cost, latency, hallucinations, evaluation complexity |
| **RAG systems** | Retrieval + generation pipelines | Question answering, grounded generation | Better grounding than pure prompting | Retrieval quality becomes critical |
| **Agent systems** | Tool-calling, multi-step agents | Multi-step workflows with decisions | Flexible orchestration | Easy to overcomplicate; fragile and costly |
| **Decision layers** | Thresholded classifiers, calibrated outputs | Business decision systems, risk-sensitive workflows | Connects model output to product behavior | Requires business-cost thinking |

## Key Principle

In model engineering, we do not ask "What is the best model?" We ask "Which model family is the right fit for the data, the problem, the cost, the latency, and the deployment constraints?"

## Summary

This reference covers nine major model families:

1. **Classical supervised ML** — Linear, tree, boosting, SVM, k-NN, probabilistic, time-series
2. **Unsupervised learning** — Clustering, association rules, topic modeling
3. **Dimensionality reduction** — PCA, UMAP, autoencoders
4. **Anomaly detection** — Isolation Forest, One-Class SVM, LOF
5. **Recommender systems** — Collaborative filtering, content-based, ranking
6. **Deep learning** — Neural networks for vision, NLP, graphs, generative tasks
7. **Foundation/LLM systems** — Pretrained models for text, multimodal, edge deployment
8. **Reinforcement learning** — Bandits, Q-learning, policy gradients
9. **Decision layers** — Thresholding, calibration, rules, cascades

Use the comparison table to match model families against your data characteristics, problem requirements, computational constraints, and deployment needs.

