# Real-Time Transaction Risk Monitoring Agent

An AI system that analyzes bank transactions in real time, detects
suspicious activity, and explains every decision by citing the bank's own
risk policy.

## Architecture

```
Transaction stream
      │
      ▼
┌─────────────────────┐
│ Rule-based filter     │  ← statistical/threshold checks (fast, cheap)
│ (agent/rules.py)      │
└──────────┬────────────┘
           │ only flagged transactions
           ▼
┌─────────────────────┐
│ RAG Retriever          │  ← finds the policy section matching the flag reason
│ (rag/retriever.py)     │
└──────────┬────────────┘
           ▼
┌─────────────────────┐
│ LLM Reasoning Agent    │  ← risk level + reasoning + recommendation (Claude)
│ (agent/llm_agent.py)   │
└──────────┬────────────┘
           ▼
┌─────────────────────┐
│ Dashboard (Streamlit)  │
└─────────────────────┘
```

## Why two stages (rules + LLM)?

Sending every transaction to the LLM would be both expensive and slow. So
cheap statistical rules first narrow 1000+ transactions down to a few
dozen suspicious candidates, and only those go to the LLM. The LLM's role
isn't detection (the rules handle that) - it's **explaining and adding
context**: which policy section applies, why, and what should happen next.

## Results (on synthetic data)

- 43 out of 1243 transactions (3.5%) were deliberately injected as "anomalous"
- Rule-based filter: **100% recall** (caught all 43), with only 5 false positives
- The filter reduced the volume from 1243 → 48 transactions (96% reduction) - the LLM only sees these

## Running it

```bash
pip install -r requirements.txt

# 1. Generate synthetic data
python3 data/generate_transactions.py

# 2. Apply the rule-based filter
python3 agent/rules.py

# 3. Run the LLM agent (requires a free Groq API key: console.groq.com/keys)
export GROQ_API_KEY="your-api-key"
python3 agent/llm_agent.py

# 4. Open the dashboard
streamlit run dashboard/app.py
```

## Possible next steps

- Add real streaming (Kafka/websocket) to strengthen the "real-time" aspect
- Measure precision/recall on the LLM's final decision, not just the rule filter
- Test against multiple bank policy documents (different banks' rules)
