"""
Risk Monitoring Dashboard

To run (from the project root):
    export GROQ_API_KEY="your-api-key"
    streamlit run dashboard/app.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from agent.rules import apply_rules
from agent.llm_agent import RiskAgent

st.set_page_config(page_title="Real-Time Risk Monitor", layout="wide")

st.title("🏦 Real-Time Transaction Risk Monitor")
st.caption("Rule-based filter + RAG + LLM reasoning agent")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
POLICY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "policy", "risk_policy.md")

@st.cache_data
def load_data():
    transactions = pd.read_csv(f"{DATA_DIR}/transactions.csv", parse_dates=["timestamp"])
    flagged = pd.read_csv(f"{DATA_DIR}/flagged_transactions.csv")
    return transactions, flagged

transactions, flagged = load_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total transactions", len(transactions))
col2.metric("Flagged", len(flagged))
col3.metric("True anomalies", int(transactions["is_anomaly"].sum()))
recall = flagged["true_is_anomaly"].sum() / transactions["is_anomaly"].sum() * 100
col4.metric("Recall (initial filter)", f"{recall:.0f}%")

st.divider()

st.subheader("Flagged transactions")

if "assessments" not in st.session_state:
    st.session_state.assessments = {}

if st.button("🔍 Analyze flagged transactions with the LLM", type="primary"):
    if not os.environ.get("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not found. Run: export GROQ_API_KEY='...' in your terminal and restart.")
    else:
        agent = RiskAgent(POLICY_PATH)
        progress = st.progress(0)
        for i, (_, row) in enumerate(flagged.iterrows()):
            if row["transaction_id"] not in st.session_state.assessments:
                try:
                    assessment = agent.assess(row)
                except Exception as e:
                    assessment = {
                        "transaction_id": row["transaction_id"],
                        "risk_level": "N/A",
                        "reasoning": f"LLM call failed: {e}",
                        "cited_policy": "N/A",
                        "recommended_action": "N/A",
                    }
                st.session_state.assessments[row["transaction_id"]] = assessment
            progress.progress((i + 1) / len(flagged))
        st.success("Analysis complete!")

risk_colors = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴", "N/A": "⚪"}

for _, row in flagged.iterrows():
    assessment = st.session_state.assessments.get(row["transaction_id"])
    badge = risk_colors.get(assessment["risk_level"], "⚪") if assessment else "⏳"

    with st.expander(f"{badge} {row['transaction_id']} — {row['customer_id']} — {row['amount']} AZN"):
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Transaction details:**")
            st.write(f"- Country: {row['country']}")
            st.write(f"- Merchant: {row['merchant_type']}")
            st.write(f"- Time: {row['timestamp']}")
            st.write(f"- System flag reasons: {row['rule_reasons']}")
        with c2:
            if assessment:
                st.write("**LLM Risk Assessment:**")
                st.write(f"- Risk level: {badge} {assessment['risk_level']}")
                st.write(f"- Reasoning: {assessment['reasoning']}")
                st.write(f"- Cited policy: {assessment['cited_policy']}")
                st.write(f"- Recommended action: {assessment['recommended_action']}")
            else:
                st.write("Click the button above to run the analysis.")
