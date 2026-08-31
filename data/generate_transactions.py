"""
Synthetic bank transaction generator.

Since we don't have real bank data, we create customers with a "normal"
behavior profile, then deliberately inject a few "anomalous" (suspicious)
transactions. We know exactly where these anomalies are (is_anomaly column)
- this gives us a ground truth to measure how well the model/agent
performs later on.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

COUNTRIES = ["AZ", "TR", "GE", "RU", "DE", "US", "AE"]
MERCHANT_TYPES = ["grocery", "electronics", "travel", "restaurant", "online_retail", "cash_withdrawal", "jewelry"]

def generate_customers(n=30):
    """Creates a 'normal' behavior profile for each customer."""
    customers = []
    for i in range(n):
        customers.append({
            "customer_id": f"CUST{i:04d}",
            "home_country": "AZ",
            "avg_amount": np.random.uniform(20, 300),   # typical transaction amount
            "std_amount": np.random.uniform(10, 50),     # amount variability
            "typical_hour_start": np.random.randint(7, 11),
            "typical_hour_end": np.random.randint(18, 23),
        })
    return pd.DataFrame(customers)


def generate_normal_transactions(customers, n_per_customer=40, start_date=None):
    """Generates normal transactions for each customer, matching their profile."""
    if start_date is None:
        start_date = datetime(2026, 8, 1)

    rows = []
    txn_counter = 0
    for _, cust in customers.iterrows():
        for _ in range(n_per_customer):
            day_offset = np.random.randint(0, 30)
            hour = np.random.randint(cust["typical_hour_start"], cust["typical_hour_end"])
            minute = np.random.randint(0, 60)
            ts = start_date + timedelta(days=int(day_offset), hours=int(hour), minutes=int(minute))

            amount = max(1, np.random.normal(cust["avg_amount"], cust["std_amount"]))

            rows.append({
                "transaction_id": f"TXN{txn_counter:06d}",
                "customer_id": cust["customer_id"],
                "timestamp": ts,
                "amount": round(amount, 2),
                "country": cust["home_country"],
                "merchant_type": random.choice(MERCHANT_TYPES),
                "is_anomaly": 0,
                "anomaly_type": None,
            })
            txn_counter += 1
    return pd.DataFrame(rows), txn_counter


def inject_anomalies(customers, txn_counter, n_anomalies=25, start_date=None):
    """Creates deliberately 'suspicious' transactions - across different anomaly types."""
    if start_date is None:
        start_date = datetime(2026, 8, 1)

    anomaly_types = [
        "large_amount",       # much larger amount than profile
        "odd_hour",            # odd hour, e.g. middle of the night
        "foreign_country",     # a country the customer never transacts in
        "rapid_succession",    # several transactions within a few minutes
    ]

    rows = []
    sample_customers = customers.sample(n_anomalies, replace=True, random_state=1)

    for i, (_, cust) in enumerate(sample_customers.iterrows()):
        anomaly_type = anomaly_types[i % len(anomaly_types)]
        day_offset = np.random.randint(0, 30)
        base_ts = start_date + timedelta(days=int(day_offset))

        if anomaly_type == "large_amount":
            amount = cust["avg_amount"] * np.random.uniform(8, 20)
            ts = base_ts + timedelta(hours=int(np.random.randint(cust["typical_hour_start"], cust["typical_hour_end"])))
            country = cust["home_country"]
            merchant = random.choice(MERCHANT_TYPES)
            rows.append(_make_row(txn_counter, cust, ts, amount, country, merchant, anomaly_type))
            txn_counter += 1

        elif anomaly_type == "odd_hour":
            ts = base_ts + timedelta(hours=random.choice([1, 2, 3, 4]))
            amount = max(1, np.random.normal(cust["avg_amount"], cust["std_amount"]))
            rows.append(_make_row(txn_counter, cust, ts, amount, cust["home_country"], "cash_withdrawal", anomaly_type))
            txn_counter += 1

        elif anomaly_type == "foreign_country":
            foreign = random.choice([c for c in COUNTRIES if c != cust["home_country"]])
            ts = base_ts + timedelta(hours=int(np.random.randint(9, 20)))
            amount = max(1, np.random.normal(cust["avg_amount"] * 2, cust["std_amount"]))
            rows.append(_make_row(txn_counter, cust, ts, amount, foreign, "cash_withdrawal", anomaly_type))
            txn_counter += 1

        elif anomaly_type == "rapid_succession":
            base_hour = int(np.random.randint(9, 20))
            for burst_i in range(4):
                ts = base_ts + timedelta(hours=base_hour, minutes=burst_i * 3)
                amount = max(1, np.random.normal(cust["avg_amount"], cust["std_amount"]))
                rows.append(_make_row(txn_counter, cust, ts, amount, cust["home_country"], "online_retail", anomaly_type))
                txn_counter += 1

    return pd.DataFrame(rows)


def _make_row(txn_id, cust, ts, amount, country, merchant, anomaly_type):
    return {
        "transaction_id": f"TXN{txn_id:06d}",
        "customer_id": cust["customer_id"],
        "timestamp": ts,
        "amount": round(amount, 2),
        "country": country,
        "merchant_type": merchant,
        "is_anomaly": 1,
        "anomaly_type": anomaly_type,
    }


def main():
    customers = generate_customers(n=30)
    normal_txns, counter = generate_normal_transactions(customers, n_per_customer=40)
    anomaly_txns = inject_anomalies(customers, counter, n_anomalies=25)

    all_txns = pd.concat([normal_txns, anomaly_txns], ignore_index=True)
    all_txns = all_txns.sort_values("timestamp").reset_index(drop=True)

    customers.to_csv("data/customers.csv", index=False)
    all_txns.to_csv("data/transactions.csv", index=False)

    print(f"Number of customers: {len(customers)}")
    print(f"Total transactions: {len(all_txns)}")
    print(f"Anomalous transactions: {all_txns['is_anomaly'].sum()} ({all_txns['is_anomaly'].mean()*100:.1f}%)")
    print("\nAnomaly type distribution:")
    print(all_txns[all_txns['is_anomaly']==1]['anomaly_type'].value_counts())


if __name__ == "__main__":
    main()
