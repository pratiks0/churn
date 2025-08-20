# Customer Churn Prediction 

- A clean **ML pipeline** (scikit-learn) with preprocessing + Logistic Regression
- A **FastAPI** microservice exposing `POST /predict` for real-time predictions
---

## 1) Project Structure

```
churn-zero-cost/
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ churn.csv                 
│  └─ generate_synthetic_churn.py
├─ src/
│  ├─ train.py                 
│  ├─ predict.py               
│  └─ schema.py                
├─ app/
│  ├─ main.py                   # FastAPI app (POST /predict)
│  └─ Dockerfile                # optional containerization
├─ artifacts/
│  ├─ model.joblib             
│  └─ metrics.json            
└─ tests/
   └─ test_api.http             # sample HTTP requests for VS Code REST Client
```

---)

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
