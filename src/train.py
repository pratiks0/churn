import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.sparse import hstack

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "churn.csv"
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"
OHE_PATH = ARTIFACTS_DIR / "ohe.joblib"
NUM_IMPUTER_PATH = ARTIFACTS_DIR / "num_imputer.joblib"
CAT_IMPUTER_PATH = ARTIFACTS_DIR / "cat_imputer.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

numeric_features = ["tenure", "monthly_charges", "total_charges", "support_calls", "num_services"]
categorical_features = ["contract_type", "payment_method", "internet_service", "streaming", "tech_support"]

def main():
    df = pd.read_csv(DATA_PATH)
    target = "churn"

    X = df[numeric_features + categorical_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # imputers
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    X_num_train = num_imputer.fit_transform(X_train[numeric_features])
    X_cat_train = cat_imputer.fit_transform(X_train[categorical_features])

    # scaling + encoding
    scaler = StandardScaler()
    ohe = OneHotEncoder(handle_unknown="ignore")

    X_num_train = scaler.fit_transform(X_num_train)
    X_cat_train = ohe.fit_transform(X_cat_train)

    X_train_proc = hstack([X_num_train, X_cat_train])

    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(X_train_proc, y_train)

    # evaluate
    X_num_test = scaler.transform(num_imputer.transform(X_test[numeric_features]))
    X_cat_test = ohe.transform(cat_imputer.transform(X_test[categorical_features]))
    X_test_proc = hstack([X_num_test, X_cat_test])

    y_prob = clf.predict_proba(X_test_proc)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    # save everything
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(ohe, OHE_PATH)
    joblib.dump(num_imputer, NUM_IMPUTER_PATH)
    joblib.dump(cat_imputer, CAT_IMPUTER_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("[OK] Saved all artifacts")
    print(metrics)

if __name__ == "__main__":
    main()
