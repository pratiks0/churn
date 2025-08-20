from pathlib import Path
import joblib
import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"

_preprocessor = None
_model = None

def load_artifacts():
    global _preprocessor, _model
    if _preprocessor is None or _model is None:
        _preprocessor = joblib.load(PREPROCESSOR_PATH)
        _model = joblib.load(MODEL_PATH)
    return _preprocessor, _model

def predict_proba(records: list[dict]) -> list[float]:
    preprocessor, model = load_artifacts()
    X = pd.DataFrame.from_records(records)
    X_proc = preprocessor.transform(X)
    probs = model.predict_proba(X_proc)[:, 1]
    return probs.tolist()
