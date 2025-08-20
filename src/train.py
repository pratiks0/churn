import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "churn.csv"
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

def main():
    df = pd.read_csv(DATA_PATH)
    target = "churn"

    numeric_features = ["tenure", "monthly_charges", "total_charges", "support_calls", "num_services"]
    categorical_features = ["contract_type", "payment_method", "internet_service", "streaming", "tech_support"]

    X = df[numeric_features + categorical_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Fit preprocessing separately
    X_train_proc = preprocessor.fit_transform(X_train)

    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(X_train_proc, y_train)

    # Evaluate
    X_test_proc = preprocessor.transform(X_test)
    y_prob = clf.predict_proba(X_test_proc)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }

    # Save preprocessor & model separately
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(clf, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] Saved preprocessor → {PREPROCESSOR_PATH}")
    print(f"[OK] Saved model → {MODEL_PATH}")
    print(f"[OK] Metrics: {json.dumps(metrics, indent=2)}")

if __name__ == "__main__":
    main()
