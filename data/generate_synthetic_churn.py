import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

N = 5000

# Categorical distributions
contract_types = np.array(["month-to-month", "one-year", "two-year"])
contract_p = np.array([0.6, 0.25, 0.15])

payment_methods = np.array(["upi", "debit_card", "credit_card", "wallet", "net_banking"])
payment_p = np.array([0.35, 0.2, 0.25, 0.1, 0.1])

internet_services = np.array(["dsl", "fiber", "wireless", "none"])
internet_p = np.array([0.35, 0.4, 0.2, 0.05])

tenure = rng.integers(0, 73, size=N)  # months 0-72
num_services = rng.integers(0, 6, size=N)

# Base monthly charges by internet type and services count
internet_base = {"dsl": 500, "fiber": 800, "wireless": 600, "none": 200}

internet = rng.choice(internet_services, size=N, p=internet_p)
contract = rng.choice(contract_types, size=N, p=contract_p)
payment = rng.choice(payment_methods, size=N, p=payment_p)

streaming = rng.random(N) < np.clip((num_services - 1) / 5.0, 0, 1)
tech_support = rng.random(N) < np.clip((4 - num_services) / 6.0, 0.1, 0.9)  # fewer services → more likely support?

base_charges = np.array([internet_base[i] for i in internet])
monthly_charges = base_charges + num_services * 75 + rng.normal(0, 50, size=N)
monthly_charges = np.clip(monthly_charges, 150, None)

total_charges = tenure * monthly_charges + rng.normal(0, 500, size=N)
total_charges = np.clip(total_charges, 0, None)

# Support calls: higher if no tech support and short tenure
support_calls = rng.poisson(lam=np.clip( ( (~tech_support).astype(int) * 1.5 ) + ( (12 - np.minimum(tenure,12)) / 6.0 ), 0.5, 6), size=N)

# Logistic model for churn probability (not used in training, just to generate labels)
z = -2.0
z += 1.2 * (tenure < 6)      # new users churn more
z += 0.6 * (tenure < 12)
z += 0.9 * (contract == "month-to-month")
z -= 0.5 * (contract == "one-year")
z -= 1.0 * (contract == "two-year")
z += 0.6 * (~tech_support)
z += 0.7 * (support_calls >= 3)
z += 0.5 * (monthly_charges > 900)
z += 0.3 * (internet == "fiber")
z -= 0.3 * (payment == "credit_card")  # autopay proxy
z -= 0.2 * streaming

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

prob = sigmoid(z)
churn = rng.binomial(1, prob)

df = pd.DataFrame({
    "tenure": tenure,
    "monthly_charges": np.round(monthly_charges, 2),
    "total_charges": np.round(total_charges, 2),
    "contract_type": contract,
    "payment_method": payment,
    "internet_service": internet,
    "streaming": streaming.astype(bool),
    "tech_support": tech_support.astype(bool),
    "support_calls": support_calls,
    "num_services": num_services,
    "churn": churn
})

out_path = Path(__file__).parent / "churn.csv"
df.to_csv(out_path, index=False)
print(f"[OK] Wrote {out_path} with shape: {df.shape}")
