from fastapi import FastAPI, HTTPException
from typing import List
from src.schema import PredictRequest, PredictResponse, PredictResponseItem
from src.predict import predict_proba, load_artifacts   # changed here

app = FastAPI(title="Churn Prediction API", version="1.0.0")

@app.get("/health")
def health():
    try:
        load_artifacts()   # changed here
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        probs = predict_proba([r.model_dump() for r in request.records])
        items: List[PredictResponseItem] = []
        for p in probs:
            items.append(PredictResponseItem(churn_probability=round(float(p), 4),
                                             churn_label=int(p >= 0.5)))
        return PredictResponse(predictions=items)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model artifact not found. Run training first.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
