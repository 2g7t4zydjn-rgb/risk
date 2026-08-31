"""
Simple RAG retriever - TF-IDF based.

Note: neural embedding models (sentence-transformers, etc.) require
downloading a model from the internet on first use. To keep this runnable
in offline/restricted environments, we use TF-IDF instead - since the
source document is small (a single policy file), this works well enough
in practice. In a real production setup you'd swap in OpenAI/Anthropic
embeddings or sentence-transformers instead - the code structure stays
the same, you'd just change the `embed()` step.
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def chunk_policy_document(filepath):
    """Splits the document into chunks based on ## and rule headers."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Extract each rule starting with "**N.N**" as a separate chunk
    pattern = r"(\*\*\d+\.\d+\*\*.*?)(?=\*\*\d+\.\d+\*\*|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)

    chunks = []
    for m in matches:
        clean = m.strip()
        if clean:
            rule_id_match = re.match(r"\*\*(\d+\.\d+)\*\*", clean)
            rule_id = rule_id_match.group(1) if rule_id_match else "N/A"
            chunks.append({"rule_id": rule_id, "text": clean})

    return chunks


class PolicyRetriever:
    def __init__(self, policy_filepath):
        self.chunks = chunk_policy_document(policy_filepath)
        self.texts = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query, top_k=3):
        """Returns the top_k policy chunks most relevant to the query."""
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        top_indices = sims.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in top_indices if sims[i] > 0]


def build_query_from_transaction(txn_row):
    """Builds a search query from a flagged transaction's characteristics."""
    parts = []
    for reason in eval(txn_row["rule_reasons"]) if isinstance(txn_row["rule_reasons"], str) else txn_row["rule_reasons"]:
        parts.append(reason)
    return " ".join(parts)


def main():
    retriever = PolicyRetriever("policy/risk_policy.md")
    print(f"Loaded {len(retriever.chunks)} policy chunks.\n")

    test_query = "Amount is 10 times larger than the customer's average, 4 transactions in a 15-minute window"
    results = retriever.retrieve(test_query, top_k=3)

    print(f"Query: {test_query}\n")
    print("Most relevant policy chunks:")
    for r in results:
        print(f"  [{r['rule_id']}] {r['text'][:100]}...")


if __name__ == "__main__":
    main()
