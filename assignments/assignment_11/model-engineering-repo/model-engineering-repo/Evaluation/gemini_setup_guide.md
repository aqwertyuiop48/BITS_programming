# Gemini LLM Integration Guide

This guide walks through using Google's Gemini API for LLM demonstrations in your teaching workflow.

## Why Gemini?

- **Free tier**: No credit card required initially
- **Production-ready**: Google's latest models available
- **Teaching-friendly**: Simple API, good for prompt engineering lessons
- **Reproducible**: Save prompts, responses, and metrics to `runs/` just like traditional ML models

## Setup (5 minutes)

### 1. Get Free API Key

Visit [ai.google.dev](https://ai.google.dev/) and click "Get API Key" button. No credit card required for the free tier.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs `google-generativeai>=0.3.0` along with existing dependencies.

### 3. Set API Key

**Option A: Environment Variable (Recommended)**
```bash
# Linux / macOS
export GEMINI_API_KEY="your-key-here"

# Windows PowerShell
$env:GEMINI_API_KEY="your-key-here"

# Windows CMD
set GEMINI_API_KEY=your-key-here
```

**Option B: Verify it's Set**
```bash
# Check if environment variable is set (Linux/macOS)
echo $GEMINI_API_KEY

# Check if environment variable is set (Windows PowerShell)
$env:GEMINI_API_KEY
```

## Basic Usage

### Run a Classification Demo

Ask Gemini to predict Titanic passenger survival:

```bash
python llm_demo.py --config configs/llm/gemini.yaml \
                   --prompt-config configs/prompts/titanic_classification.yaml \
                   --query "Female, Age 28, Ticket Class 1"
```

Expected output:
```
[titanic_classification_20240425_143022] Calling Gemini API...
  Model: gemini-2.0-flash
  Task: titanic_classification
  Temperature: 0.5

======================================================================
Response (245 ms):
======================================================================
1. Explanation: Female passenger in first class during the era of "women and children first" policy.
2. Prediction: Survived
3. Confidence: high
======================================================================

Artifacts saved to: runs/titanic_classification_20240425_143022
  - response.txt
  - metadata.json
  - params.json
  - bundle_info.json
```

### Run a Model Explanation Demo

Ask Gemini to explain a machine learning concept:

```bash
python llm_demo.py --config configs/llm/gemini.yaml \
                   --prompt-config configs/prompts/model_explanation.yaml \
                   --query "random forest classifier"
```

### Run a Code Review Demo

Ask Gemini to review Python code:

```bash
python llm_demo.py --config configs/llm/gemini.yaml \
                   --prompt-config configs/prompts/code_review.yaml \
                   --query "def train(X, y):\n    model = LogisticRegression()\n    return model.fit(X, y)"
```

## Understanding the Artifacts

After each run, a directory is created under `runs/` with these files:

### response.txt
The full LLM response text.

### metadata.json
Runtime metrics:
```json
{
  "model": "gemini-2.0-flash",
  "timestamp": "2024-04-25T14:30:22.123456Z",
  "latency_ms": 245,
  "prompt_tokens": null,
  "response_tokens": null,
  "total_tokens": null,
  "task": "titanic_classification",
  "query": "Female, Age 28, Ticket Class 1"
}
```

**Note:** Token counts may be `null` on free tier (API limitation).

### params.json
Configuration parameters used:
```json
{
  "llm_config": "configs/llm/gemini.yaml",
  "prompt_config": "configs/prompts/titanic_classification.yaml",
  "model_name": "gemini-2.0-flash",
  "temperature": 0.5,
  "max_tokens": 1024
}
```

### bundle_info.json
Run metadata for tracking:
```json
{
  "run_name": "titanic_classification_20240425_143022",
  "task_name": "titanic_classification",
  "model": "gemini-2.0-flash",
  "prompt_config": "configs/prompts/titanic_classification.yaml",
  "llm_config": "configs/llm/gemini.yaml",
  "created_at": "2024-04-25T14:30:22.123456Z",
  "files": {
    "response": "response.txt",
    "metadata": "metadata.json",
    "params": "params.json"
  }
}
```

## Customizing Prompts

Create your own prompt config in `configs/prompts/your_task.yaml`:

```yaml
task_name: "your_task_name"
task_description: "What this demo teaches"

# Instruction to the LLM (sets its role)
system_prompt: |
  You are teaching a machine learning concept.
  Be clear and concise.

# Template for user query with placeholder
user_prompt_template: |
  Explain: {query}
  
  Answer in 3 bullet points.

# Optional: override generation parameters
generation_override:
  temperature: 0.5
  max_tokens: 512
```

Then run:
```bash
python llm_demo.py --config configs/llm/gemini.yaml \
                   --prompt-config configs/prompts/your_task.yaml \
                   --query "your question here"
```

## Customizing LLM Settings

Edit `configs/llm/gemini.yaml` to adjust:

- **model_name**: Change to `gemini-1.5-flash` or another available model
- **temperature**: Lower (0.1) for deterministic, Higher (0.9) for creative
- **max_tokens**: Maximum length of response
- **safety_settings**: Content filtering levels

Example:
```yaml
model_name: "gemini-1.5-flash"
generation_params:
  temperature: 0.3  # More deterministic
  max_tokens: 256   # Shorter responses
```

## Teaching Ideas

### Lesson 1: Prompt Engineering
Show how temperature affects response creativity:
```bash
# Deterministic
python llm_demo.py ... --prompt-config model_explanation.yaml --query "logistic regression" 
# (manually edit config, set temperature: 0.1)

# Creative
python llm_demo.py ... --prompt-config model_explanation.yaml --query "logistic regression"
# (manually edit config, set temperature: 0.9)
```

Then compare responses in `runs/*/response.txt`.

### Lesson 2: Few-Shot Learning
Modify your prompt config to include examples:
```yaml
examples:
  - input: "Example problem"
    output: "Example solution"
  - input: "Another example"
    output: "Another solution"
```

### Lesson 3: Latency Measurement
Compare `metadata.json` across multiple runs to see how:
- Model complexity affects latency
- Response length affects latency
- Time of day affects API latency

### Lesson 4: Comparing Models
Modify `configs/llm/gemini.yaml` to test different models:
- `gemini-2.0-flash` (latest, free)
- `gemini-1.5-flash` (previous generation)
- `gemini-pro` (older)

Compare response quality and latency in runs/summary.csv.

## Troubleshooting

### "GEMINI_API_KEY not found"
```
ERROR: GEMINI_API_KEY not provided and not set in environment.
```

**Solution:** Set the environment variable:
```bash
export GEMINI_API_KEY="your-api-key"
```

### "No module named 'google.generativeai'"
```
ModuleNotFoundError: No module named 'google.generativeai'
```

**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### "The API returned an error"
Free tier has rate limits. Wait a minute and retry.

For production use, upgrade to a paid plan at [ai.google.dev](https://ai.google.dev/).

### Token counts are null
This is expected on free tier. Some API versions don't expose token usage in free tier responses.

## Integration with Course Workflow

This LLM integration fits into your teaching progression:

1. **Classical ML** (Week 1-3): `train.py` + sklearn + tabular data
2. **Deep Learning** (Week 4-6): `train_deep.py` + CNN/RNN/LSTM/GRU
3. **LLM Concepts** (Week 7-8): `llm_demo.py` + Gemini API + `reports/llms_explained.md`

Students can:
- See reproducibility in action (configs → artifacts → metrics)
- Experiment with prompts and measure latency
- Understand how LLMs differ from classical and deep learning models
- Bridge from sequence models (RNN/LSTM) to transformer-based LLMs

## Resources

- Gemini API Docs: https://ai.google.dev/
- Free API Key: https://ai.google.dev/
- Prompt Engineering Tips: https://ai.google.dev/docs/prompt_best_practices
- LLM Concepts in This Repo: `reports/llms_explained.md`
