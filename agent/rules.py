"""
Rule-based initial filter.

Goal: filter down 1000+ transactions to only the genuinely suspicious ones
(a few dozen) before sending anything to the LLM. This matters for both
speed and cost - LLM calls are expensive, so sending every transaction
makes no sense.

We use 4 simple rules:
1. Is the amount a significant outlier vs. the customer's own average (z-score)
2. Is the hour outside the customer's typical active hours
3. Is the country different from the customer's home country
4. Are there many transactions from the same customer in a short window (15 min)
"""

import pandas as pd
import numpy as np


def flag_large_amount(txn, customer_profile, z_threshold=3.0):
    z = (txn["amount"] - customer_profile["avg_amount"]) / max(customer_profile["std_amount"], 1)
    if z > z_threshold:
        return True, f"Amount ({txn['amount']}) is {z:.1f} standard deviations above the customer's average"
    return False, None


def flag_odd_hour(txn, customer_profile):
    hour = txn["timestamp"].hour
    if hour < customer_profile["typical_hour_start"] or hour > customer_profile["typical_hour_end"]:
        if hour < 6:  # middle of the night is especially suspicious
            return True, f"Transaction occurred at {hour}:00 - outside the customer's typical active hours"
    return False, None


def flag_foreign_country(txn, customer_profile):
    if txn["country"] != customer_profile["home_country"]:
        return True, f"Transaction occurred in {txn['country']}, but the customer's home country is {customer_profile['home_country']}"
    return False, None


def flag_rapid_succession(txn, all_customer_txns, window_minutes=15, min_count=3):
    ts = txn["timestamp"]
    window_start = ts - pd.Timedelta(minutes=window_minutes)
    window_end = ts + pd.Timedelta(minutes=window_minutes)
    nearby = all_customer_txns[
        (all_customer_txns["timestamp"] >= window_start) &
        (all_customer_txns["timestamp"] <= window_end)
    ]
    if len(nearby) >= min_count:
        return True, f"{len(nearby)} transactions occurred within a {window_minutes}-minute window (unusual)"
    return False, None


def apply_rules(transactions_df, customers_df):
    """
    Applies all rules to every transaction and returns the flagged ones.
    Result: for each flagged transaction, which rules fired and why.
    """
    customers_df = customers_df.set_index("customer_id")
    flagged = []

    for cust_id, group in transactions_df.groupby("customer_id"):
        if cust_id not in customers_df.index:
            continue
        profile = customers_df.loc[cust_id]

        for _, txn in group.iterrows():
            reasons = []

            for check_fn in [flag_large_amount, flag_odd_hour, flag_foreign_country]:
                is_flag, reason = check_fn(txn, profile)
                if is_flag:
                    reasons.append(reason)

            is_flag, reason = flag_rapid_succession(txn, group)
            if is_flag:
                reasons.append(reason)

            if reasons:
                flagged.append({
                    "transaction_id": txn["transaction_id"],
                    "customer_id": cust_id,
                    "timestamp": txn["timestamp"],
                    "amount": txn["amount"],
                    "country": txn["country"],
                    "merchant_type": txn["merchant_type"],
                    "rule_reasons": reasons,
                    "true_is_anomaly": txn["is_anomaly"],  # for evaluation only, not sent to the LLM
                })

    return pd.DataFrame(flagged)


def main():
    transactions = pd.read_csv("data/transactions.csv", parse_dates=["timestamp"])
    customers = pd.read_csv("data/customers.csv")

    flagged = apply_rules(transactions, customers)

    total = len(transactions)
    flagged_count = len(flagged)
    true_anomalies = transactions["is_anomaly"].sum()
    caught = flagged["true_is_anomaly"].sum() if len(flagged) else 0

    print(f"Total transactions: {total}")
    print(f"Flagged by rule-based filter: {flagged_count} ({flagged_count/total*100:.1f}%)")
    print(f"True anomalies: {true_anomalies}")
    print(f"Caught (recall): {caught}/{true_anomalies} = {caught/true_anomalies*100:.1f}%")
    print(f"False positives: {flagged_count - caught}")

    flagged.to_csv("data/flagged_transactions.csv", index=False)
    print(f"\nFlagged transactions saved to: data/flagged_transactions.csv")


if __name__ == "__main__":
    main()
