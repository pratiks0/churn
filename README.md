# Customer Churn Prediction – Zero-Cost, Beginner-Friendly (Local Production-Style)

A complete, **zero-cost** project you can run fully on your laptop. It includes:
- Synthetic **customer churn** dataset (so there’s no dependency on paid/cloud data)
- A clean **ML pipeline** (scikit-learn) with preprocessing + Logistic Regression
- A **FastAPI** microservice exposing `POST /predict` for real-time predictions
- Simple, production-like structure with `Dockerfile` (optional, if you use Docker)

> This project is designed for interviews (like NowVertical) to demonstrate **Python + SQL (concepts) + ML + Data Engineering mindset + API deployment** — all without paid cloud infra.

---

## 1) Project Structure

```
churn-zero-cost/
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ churn.csv                 # generated synthetic dataset
│  └─ generate_synthetic_churn.py
├─ src/
│  ├─ train.py                  # trains pipeline and saves artifacts/model.joblib
│  ├─ predict.py                # helper to load model & predict
│  └─ schema.py                 # pydantic models for API I/O
├─ app/
│  ├─ main.py                   # FastAPI app (POST /predict)
│  └─ Dockerfile                # optional containerization
├─ artifacts/
│  ├─ model.joblib              # trained pipeline (after running train.py)
│  └─ metrics.json              # basic metrics
└─ tests/
   └─ test_api.http             # sample HTTP requests for VS Code REST Client
```

---

## 2) Quickstart (No Cloud, No Cost)

### A) Create & activate a virtual environment
```bash
cd churn-zero-cost

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### B) Install dependencies
```bash
pip install -r requirements.txt
```

### C) Generate synthetic dataset
```bash
python data/generate_synthetic_churn.py
# Creates data/churn.csv (5,000 rows)
```

### D) Train the model
```bash
python src/train.py
# Saves artifacts/model.joblib and artifacts/metrics.json
```

### E) Run the API
```bash
uvicorn app.main:app --reload
# Server at http://127.0.0.1:8000
# Docs at  http://127.0.0.1:8000/docs
```

### F) Test the API
- Open browser at `/docs` and try **POST /predict** interactively.
- Or use `curl`:
```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @- <<'JSON'
{
  "records": [{
    "tenure": 5,
    "monthly_charges": 850.0,
    "total_charges": 4250.0,
    "contract_type": "month-to-month",
    "payment_method": "upi",
    "internet_service": "fiber",
    "streaming": true,
    "tech_support": false,
    "support_calls": 4,
    "num_services": 3
  }]
}
JSON
```

---

## 3) What You’ll Learn / Explain in Interviews

- **Problem framing**: churn (binary classification), features that drive churn (tenure, support calls, contract type, etc.).
- **Data engineering mindset**: how data flows from raw → clean → model → API.
- **ML pipeline**: reproducible preprocessing with `ColumnTransformer`, model registry artifact.
- **Deployment basics**: serving predictions over an HTTP endpoint (`/predict`), and validating input with Pydantic.
- **Zero-cost**: everything local; optional Docker for production-style packaging.

---

## 4) Optional: Docker (Local “Production-Style”)

```bash
# (build image)
docker build -t churn-api ./app

# (run container)
docker run -p 8000:8000 -v $(pwd):/app churn-api
# API → http://127.0.0.1:8000/docs
```

> The Docker image expects `artifacts/model.joblib` to exist. Train locally first.

---

## 5) Next Steps (If You Want to Expand)
- Add a simple **dashboard** (e.g., Streamlit) for batch scoring & plotting churn risk distribution.
- Persist predictions in a local SQLite DB.
- Add **threshold tuning** & a confusion matrix report.
- Swap Logistic Regression for **GradientBoostingClassifier** and compare AUC.
- Add a small **data quality check** (e.g., reject negative charges).

---

## 6) FAQ

**Q. Why synthetic data?**  
A. Avoids licensing/cost issues and still mimics real churn patterns for interviews.

**Q. Is this “production”?**  
A. It’s **production-style**: a clean pipeline, saved model, validation, and an HTTP API. You can containerize with Docker. It’s enough to demonstrate job-ready skills in interviews without paid cloud infra.

**Q. How do I talk about this in interviews?**  
- “I built an end-to-end churn prediction system with a scikit-learn pipeline and served it via FastAPI. The API accepts JSON, validates inputs, and returns churn probabilities. I can containerize it for portability.”
