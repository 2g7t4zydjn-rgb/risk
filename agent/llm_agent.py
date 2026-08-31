"""
Risk Reasoning Agent

This is the "brain" of the project. For every transaction that passed the
rule-based filter, it:
  1. Retrieves the policy sections relevant to the flag reasons (RAG)
  2. Passes the transaction + policy sections to the LLM
  3. Gets back a structured (JSON) risk assessment:
     risk_level, reasoning, cited_policy, recommended_action

Uses Groq (free tier, no credit card required) with an open-source model.
To run: requires the GROQ_API_KEY environment variable.
    export GROQ_API_KEY="your-api-key"
    python3 agent/llm_agent.py

Get a free key at https://console.groq.com/keys
"""

import os
import sys
import json
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
from rag.retriever import PolicyRetriever, build_query_from_transaction

SYSTEM_PROMPT = """You are a risk analyst assistant for PasaBank. You will be
given a flagged bank transaction and the relevant sections of the bank's
risk policy. Your job is to:

1. Assess the transaction based only on the given policy sections
2. Only cite the policy sections provided - never invent a rule
3. Give clear, concise reasoning

Respond ONLY in the following JSON format, nothing else:
{
  "risk_level": "Low" | "Medium" | "High" | "Critical",
  "reasoning": "1-2 sentence explanation of why this level",
  "cited_policy": "the cited rule number(s), e.g. '1.1, 4.1'",
  "recommended_action": "the action recommended by the policy"
}"""


class RiskAgent:
    def __init__(self, policy_filepath):
        self.client = Groq()  # reads GROQ_API_KEY from the environment automatically
        self.retriever = PolicyRetriever(policy_filepath)

    def assess(self, flagged_txn_row):
        query = build_query_from_transaction(flagged_txn_row)
        policy_chunks = self.retriever.retrieve(query, top_k=3)
        policy_context = "\n\n".join([c["text"] for c in policy_chunks])

        reasons = flagged_txn_row["rule_reasons"]
        if isinstance(reasons, str):
            reasons = eval(reasons)

        user_message = f"""TRANSACTION:
- ID: {flagged_txn_row['transaction_id']}
- Customer: {flagged_txn_row['customer_id']}
- Amount: {flagged_txn_row['amount']}
- Country: {flagged_txn_row['country']}
- Merchant: {flagged_txn_row['merchant_type']}
- System-flagged reasons: {'; '.join(reasons)}

RELEVANT POLICY SECTIONS:
{policy_context}

Assess this transaction."""

        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            max_tokens=2000,
            reasoning_effort="low",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        raw_text = response.choices[0].message.content
        try:
            result = json.loads(raw_text)
        except json.JSONDecodeError:
            result = {"risk_level": "N/A", "reasoning": raw_text, "cited_policy": "N/A", "recommended_action": "N/A"}

        result["transaction_id"] = flagged_txn_row["transaction_id"]
        return result


def main():
    flagged = pd.read_csv("data/flagged_transactions.csv")
    agent = RiskAgent("policy/risk_policy.md")

    results = []
    for _, row in flagged.iterrows():
        try:
            assessment = agent.assess(row)
        except Exception as e:
            print(f"{row['transaction_id']}: FAILED ({e})")
            assessment = {
                "transaction_id": row["transaction_id"],
                "risk_level": "N/A",
                "reasoning": f"LLM call failed: {e}",
                "cited_policy": "N/A",
                "recommended_action": "N/A",
            }
        results.append(assessment)
        print(f"{assessment['transaction_id']}: {assessment['risk_level']} - {assessment['reasoning']}")

    pd.DataFrame(results).to_csv("data/risk_assessments.csv", index=False)
    print("\nResults saved to: data/risk_assessments.csv")


if __name__ == "__main__":
    main()
