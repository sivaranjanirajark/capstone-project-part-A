"""Part D, tasks 1-4 - policy RAG assistant.

Run:  python policy_rag.py
In :  data/policies/*.txt, data/raw/policy_eval.csv, mock_llm.py
The refund agent (refund_agent.py) imports retrieve() from this file.
"""
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm

POLICY_DIR = Path("data") / "policies"
CHUNK_WORDS = 40
OVERLAP = 10


# ------------------------------------------------------------------ Task 1: chunking
def chunk_documents(chunk_words=CHUNK_WORDS, overlap=OVERLAP):
    """Split each policy file into word chunks; record the source file with every chunk."""
    chunks = []
    step = chunk_words - overlap
    for path in sorted(POLICY_DIR.glob("*.txt")):
        words = path.read_text(encoding="utf-8").split()
        start = 0
        while True:
            chunks.append({"source": path.name, "text": " ".join(words[start:start + chunk_words])})
            if start + chunk_words >= len(words):      # this chunk reaches/passes the last word
                break
            start += step
    return chunks


# ------------------------------------------------------------------ Task 2: vectorise + retrieve
def build_index(overlap=OVERLAP):
    chunks = chunk_documents(overlap=overlap)
    vec = TfidfVectorizer()
    matrix = vec.fit_transform([c["text"] for c in chunks])   # fitted on chunk texts only
    return {"chunks": chunks, "vec": vec, "matrix": matrix}


INDEX = build_index()


def retrieve(query, k=3, index=None):
    idx = index or INDEX
    q = idx["vec"].transform([query])                         # same fitted vectorizer
    sims = cosine_similarity(q, idx["matrix"])[0]
    top = sims.argsort()[::-1][:k]
    return [{"source": idx["chunks"][i]["source"], "score": float(sims[i]),
             "text": idx["chunks"][i]["text"]} for i in top]


# ------------------------------------------------------------------ Task 3: augment + answer
SYSTEM_PROMPT = (
    "You are a POLICY ASSISTANT for Instamart support. Answer ONLY from the supplied CONTEXT. "
    "If the context does not contain the answer, say that you could not find the information."
)


def answer(question, index=None):
    hits = retrieve(question, k=3, index=index)
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    resp = mock_llm.chat([{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user}])
    return resp.choices[0].message.content, [h["source"] for h in hits]


# ------------------------------------------------------------------ Task 4: evaluation
def evaluate(overlap, label):
    index = build_index(overlap)
    ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    doc_ok = fact_ok = fallback_ok = n_ans = n_unans = 0
    failures = []
    for r in ev.itertuples(index=False):
        text, sources = answer(r.question, index)
        if r.expected_doc == "NONE":
            n_unans += 1
            if text.startswith("I could not find"):
                fallback_ok += 1
            else:
                failures.append((r.question, "should have fallen back"))
        else:
            n_ans += 1
            d = r.expected_doc in sources
            f = r.expected_fact in text                      # case-sensitive substring
            doc_ok += d
            fact_ok += f
            if not (d and f):
                failures.append((r.question, f"doc_in_top3={d}, fact_in_answer={f}"))
    print(f"\n--- Evaluation: {label} (overlap={overlap}, chunks={len(index['chunks'])}) ---")
    print(f"Answerable: expected_doc in top 3 = {doc_ok}/{n_ans} | expected_fact in answer = {fact_ok}/{n_ans}")
    print(f"Unanswerable: fallback answer = {fallback_ok}/{n_unans}")
    for q, why in failures:
        print("  FAILS:", q, "->", why)


if __name__ == "__main__":
    chunks = chunk_documents()
    print("Total chunks:", len(chunks))
    print(pd.Series([c["source"] for c in chunks]).value_counts().sort_index().to_dict())

    top = retrieve("refund policy late delivery")
    print("\nTop 3 for 'refund policy late delivery':", [(h["source"], round(h["score"], 3)) for h in top])
    assert all(h["source"] == "refund_policy.txt" for h in top)

    text, sources = answer("Who must approve a refund above INR 1000?")
    print("\nSample answer:", text, "| sources:", sources)

    evaluate(10, "overlap 10")
    evaluate(0, "zero overlap")