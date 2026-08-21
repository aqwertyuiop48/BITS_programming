# Class 2 — Instructor Demo Guide
## CNN API Deployment, Drift Detection & Traffic Testing

> **Before class**: Run the one-time model export in class1 (Step 0 below).
> Everything else takes < 2 minutes to start.

---

## Step 0 — One-Time Model Export (from Class 1)

Run the last cell of `class1/Part_2_Overfitting_and_Generalization.ipynb`.
This trains three models and saves the two best to:

```
class2/models/
├── v1/
│   ├── model.pth        ← Baseline ResNet-18 (no regularisation, ~91% test acc)
│   └── metadata.json
└── v2/
    ├── model.pth        ← Dropout ResNet-18 (best model, ~97% test acc)
    └── metadata.json
```

You only need to do this **once**. The files persist for all future class2 sessions.

---

## Step 1 — Install & Start

**Install (do once):**
```bash
cd class2
pip install -r requirements-api.txt
pip install -r requirements-streamlit.txt
```

**Start the simple API — Part 1 demo (Terminal 1):**
```bash
cd class2
uvicorn api_simple:app --reload --port 8000
```

**Start the multi-model API — Part 3 demo (Terminal 1):**
```bash
cd class2
uvicorn api_multimodel:app --reload --port 8000
```

**Start the Streamlit UI (Terminal 2, works with api_multimodel):**
```bash
cd class2
streamlit run streamlit_app.py
```
Opens at `http://localhost:8501`

---

## How to Demo — Part by Part

---

### Part 1 (40 min) — Interface Layer: CNN API Deployment

**Goal**: Students understand how to wrap a model in a REST endpoint with input validation.

**API**: `api_simple.py` — single model, three endpoints.

**Demo script:**

1. Open `http://localhost:8000/docs` — show the auto-generated Swagger UI.
   - Point out `/health`, `/info`, `/predict`
   - "This is what FastAPI gives you for free."

2. Call `/health` in the browser. Show the JSON response.

3. Upload an image via Swagger `/predict` — walk through the response:
   - `prediction`, `confidence`, `class_probabilities`, `model_version`, `latency_ms`

4. Show input validation:
   - Upload a `.txt` file → `400 Bad Request`
   - "Why 400 and not 500? Because the error is the client's fault, not the server's."

5. Open `Part_1_CNN_API_Deployment.ipynb` and run through the test cases table.

**Key question:**
> "What would break in production if we skipped the input validation?"

---

### Part 2 (40 min) — Data Layer: Distribution & Drift Testing

**Goal**: Students see how input distribution shift degrades accuracy.

**Demo script (notebook-based):**

1. Open `Part_2_Data_Drift_Detection.ipynb` and run it.
2. Show the original vs brightness-shifted histograms side by side.
   - "The model was trained on images like the left one. The right one is what a dirty camera sees."
3. Show the KL divergence increasing as shift magnitude increases.
4. Show accuracy dropping under heavy shift.
5. "In production, you'd set an alert threshold: if JSD > X, retrain."

**Key question:**
> "What real-world events cause data drift in an ADAS system?"
> (Answers: rain, night, lens fog, camera angle change, geographic change)

---

### Part 3 (40 min) — Deployment Layer: Versioning & Traffic Testing

**Goal**: Students understand canary deployment, A/B testing, and latency measurement.

**API**: `api_multimodel.py` — two model versions, canary routing, rate limiting.

**Demo script:**

1. Start `uvicorn api_multimodel:app --reload --port 8000`

2. Show `/predict` with canary routing — some requests go to v1, some to v2.

3. Switch to the **A/B Compare** tab in Streamlit.
   - Upload an image → click **Compare Both Models**
   - Show that v1 and v2 sometimes disagree

4. Show `/metrics` — watch v1/v2 request counts and latency.

5. **Rate limiting demo** — click **Predict** 6 times quickly:
   - Requests 1–5: succeed (200 OK)
   - Request 6: rejected — HTTP 429
   - "The server returned 429 Too Many Requests."

6. Open `Part_3_Versioning_Traffic_Testing.ipynb` and run the traffic simulation.

**Key question:**
> "If 100 users hit the API at the same time, what happens without rate limiting?"

---

### Lab 2 (60 min) — Student Hands-On

Students open `Lab_2_System_Testing_Student.ipynb` and build:

| Phase | Deliverable | Reference |
|-------|-------------|-----------|
| A — Interface | CNN API + 8-test suite | Part 1 notebook, `test_part1.py` |
| B — Data | Drift detection with KL divergence | Part 2 notebook |
| C — Deployment | Model comparison + traffic routing | Part 3 notebook, `test_part3.py` |

Release `Lab_2_System_Testing_Solution.ipynb` after submission.

---

## API Endpoints Reference

### api_simple.py (Part 1)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status + device info |
| `/info` | GET | Model metadata |
| `/predict` | POST | Single prediction |
| `/docs` | GET | Swagger UI |

### api_multimodel.py (Part 3)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status + device info |
| `/info` | GET | Model metadata |
| `/predict` | POST | Canary-routed prediction (70% v1, 30% v2) |
| `/predict?model_version=v1` | POST | Force v1 |
| `/predict?model_version=v2` | POST | Force v2 |
| `/predict-both` | POST | Both models, A/B response |
| `/metrics` | GET | Aggregated stats |
| `/stats` | GET | Detailed breakdown |
| `/logs?limit=50` | GET | Raw prediction log |
| `/docs` | GET | Swagger UI |

**Rate limit** (api_multimodel only): 5 requests per 60 seconds per IP on `/predict`.

---

## Test Suites

```bash
# Part 1 tests (api_simple)
cd class2
pytest test_part1.py -v

# Part 3 tests (api_multimodel)
cd class2
pytest test_part3.py -v
```

---

## Troubleshooting

**"Loaded with random weights" in API log:**
The `class2/models/` folder is empty. Run the last cell of
`class1/Part_2_Overfitting_and_Generalization.ipynb` first.

**Port 8000 already in use:**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
# macOS / Linux
lsof -ti:8000 | xargs kill
```

**Streamlit can't reach API:**
Make sure `uvicorn api_multimodel:app --reload --port 8000` is running in a separate terminal.

---

## Learning Checklist

**Part 1 — Interface Layer**
- [ ] `/predict` returns `prediction`, `confidence`, `model_version`, `class_probabilities`
- [ ] Invalid file type → HTTP 400
- [ ] Corrupt image → HTTP 400
- [ ] Swagger UI open and working at `/docs`

**Part 2 — Data Layer**
- [ ] Histogram comparison plotted for brightness and contrast shifts
- [ ] KL divergence increases with shift magnitude
- [ ] Accuracy drops are charted and discussed

**Part 3 — Deployment Layer**
- [ ] Canary split visible in `/metrics` (`v1_requests` vs `v2_requests`)
- [ ] Rate limit triggered — HTTP 429
- [ ] A/B comparison shows when v1 and v2 disagree
- [ ] Students can read `/logs` and explain what each field means

**Lab 2**
- [ ] Student's API returns correct JSON (Phase A)
- [ ] At least 8 tests written and passing (Phase A)
- [ ] Drift detection with KL divergence (Phase B)
- [ ] Model comparison report (Phase C)
