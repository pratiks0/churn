from pathlib import Path
import joblib
import pandas as pd
from scipy.sparse import hstack

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"
OHE_PATH = ARTIFACTS_DIR / "ohe.joblib"
NUM_IMPUTER_PATH = ARTIFACTS_DIR / "num_imputer.joblib"
CAT_IMPUTER_PATH = ARTIFACTS_DIR / "cat_imputer.joblib"

numeric_features = ["tenure", "monthly_charges", "total_charges", "support_calls", "num_services"]
categorical_features = ["contract_type", "payment_method", "internet_service", "streaming", "tech_support"]

_model = None
_scaler = None
_ohe = None
_num_imputer = None
_cat_imputer = None

def load_artifacts():
    global _model, _scaler, _ohe, _num_imputer, _cat_imputer
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        _ohe = joblib.load(OHE_PATH)
        _num_imputer = joblib.load(NUM_IMPUTER_PATH)
        _cat_imputer = joblib.load(CAT_IMPUTER_PATH)
    return _model, _scaler, _ohe, _num_imputer, _cat_imputer

def predict_proba(records: list[dict]) -> list[float]:
    model, scaler, ohe, num_imputer, cat_imputer = load_artifacts()
    X = pd.DataFrame.from_records(records)

    # process numeric
    X_num = scaler.transform(num_imputer.transform(X[numeric_features]))

    # process categorical
    X_cat = ohe.transform(cat_imputer.transform(X[categorical_features]))

    # combine
    X_proc = hstack([X_num, X_cat])

    probs = model.predict_proba(X_proc)[:, 1]
    return probs.tolist()
