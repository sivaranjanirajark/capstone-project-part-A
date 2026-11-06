"""Part C - Support ticket triage with guardrails.

Pipeline per ticket (file order):
  input guardrails -> PII redaction -> retry-wrapped model call -> output validation
  -> token budget -> routing using outputs/customer_segments.csv (Part B)

Run:  python triage.py
In :  data/raw/tickets.csv, outputs/customer_segments.csv, mock_llm.py (unmodified)
Out:  outputs/triage_budget_demo.csv, outputs/triage_results.csv
"""
import json
import random
import re
import time
from pathlib import Path

import pandas as pd

import mock_llm
from mock_llm import TransientLLMError, est_tokens

# ------------------------------------------------------------------ settings
MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
CATEGORIES = ["billing", "delivery", "technical", "fraud"]
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
random.seed(42)

# ------------------------------------------------------------------ data
tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False, dtype=str)
segments = pd.read_csv(OUT / "customer_segments.csv")
segments["is_premium"] = segments["is_premium"].astype(str).str.lower().eq("true")
seg_lookup = segments.set_index("customer_id")[["segment_name", "is_premium"]].to_dict("index")

# ------------------------------------------------------------------ redaction (order matters)
PATTERNS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("PHONE", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("ORDER_ID", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]


def redact(text, counts=None):
    for name, pat, placeholder in PATTERNS:
        text, n = pat.subn(placeholder, text)
        if counts is not None:
            counts[name] += n
    return text


def contains_pii(text):
    return any(pat.search(text) for _, pat, _ in PATTERNS)


# ------------------------------------------------------------------ prompts
ZERO_SHOT = (
    "You CLASSIFY customer support tickets. "
    "Allowed categories: billing, delivery, technical, fraud. "
    'Reply with JSON only, exactly like {"category": "...", "reason": "..."}.'
)

FEW_SHOT = (
    "You CLASSIFY customer support tickets. "
    "Allowed categories: billing, delivery, technical, fraud. "
    'Reply with JSON only, exactly like {"category": "...", "reason": "..."}.\n'
    "EXAMPLES:\n"
    'Ticket: I was charged twice for [ORDER_ID]. -> {"category": "billing", "reason": "double charge"}\n'
    'Ticket: The rider never delivered [ORDER_ID]. -> {"category": "delivery", "reason": "not received"}\n'
    'Ticket: The app freezes when I open the cart. -> {"category": "technical", "reason": "app problem"}\n'
    'Ticket: Someone hacked my account and used my wallet. -> {"category": "fraud", "reason": "unauthorized use"}'
)

assert "CLASSIFY" in ZERO_SHOT and "CLASSIFY" in FEW_SHOT
assert "EXAMPLES:" not in ZERO_SHOT and "EXAMPLES:" in FEW_SHOT


# ------------------------------------------------------------------ resilient call
def call_with_retry(ticket_id, messages):
    """Call the mock; up to MAX_RETRIES retries with exponential backoff + jitter."""
    n = 0
    while True:
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            if n >= MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * random.uniform(0.9, 1.15)
            print(f"  [retry] ticket={ticket_id} retry={n} wait={wait:.4f}s")
            time.sleep(wait)
            n += 1


# ------------------------------------------------------------------ output validation
def validate(reply):
    """Return (status, category). Never guess."""
    try:
        data = json.loads(reply)
        if isinstance(data, dict) and data.get("category") in CATEGORIES:
            return "classified_json", data["category"]
        return "flagged_invalid_output", ""
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", str(reply).lower())
        found = {c for c in CATEGORIES if c in words}
        if len(found) == 1:
            return "classified_keyword", found.pop()
        return "flagged_invalid_output", ""


# ------------------------------------------------------------------ routing
def route(status, category, customer_id):
    if status in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"):
        return "human_review", "NORMAL"
    if category == "fraud":
        return "security_team", "HIGH"
    if seg_lookup.get(customer_id, {}).get("is_premium", False):
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"


# ------------------------------------------------------------------ one full run
def run_pipeline(system_prompt, token_budget=None, label=""):
    mock_llm.reset()                                   # fresh counters before each full run
    rows, total_tokens, transient_seen = [], 0, 0
    print(f"\n=== Run: {label} (budget={token_budget}) ===")
    for t in tickets.itertuples(index=False):
        status, category, used = None, "", 0
        text = t.text
        if not text.strip():
            status = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        elif token_budget is not None and total_tokens >= token_budget:
            status = "budget_exhausted"
        else:
            clean = redact(text)
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": clean}]
            assert not contains_pii(messages[1]["content"]), "PII reached the prompt"
            assert messages[1]["content"] == clean
            resp = call_with_retry(t.ticket_id, messages)
            used = resp.usage.total_tokens
            total_tokens += used
            status, category = validate(resp.choices[0].message.content)
        r, p = route(status, category, t.customer_id)
        seg = seg_lookup.get(t.customer_id, {}).get("segment_name", "")
        rows.append({"ticket_id": t.ticket_id, "customer_id": t.customer_id, "status": status,
                     "category": category, "segment_name": seg, "priority": p, "route": r,
                     "tokens_used": used, "true_category": t.true_category})
    return pd.DataFrame(rows), total_tokens


def summarise(df, total_tokens, label):
    classified = df[df["status"].isin(["classified_json", "classified_keyword"])]
    correct = (classified["category"] == classified["true_category"]).sum()
    acc = correct / len(classified) * 100
    print(f"\n--- {label} summary ---")
    print(df["status"].value_counts().to_dict())
    print(f"Classified: {len(classified)} | correct: {correct} | accuracy: {acc:.1f}%")
    print(f"Total tokens: {total_tokens} | tokens per classified ticket: "
          f"{total_tokens / len(classified):.2f}")
    return classified


COLS = ["ticket_id", "customer_id", "status", "category", "segment_name",
        "priority", "route", "tokens_used"]

if __name__ == "__main__":
    # ---- Task 1: guardrail counts
    n_empty = (tickets["text"].str.strip() == "").sum()
    n_long = tickets["text"].map(lambda s: est_tokens(s) > MAX_INPUT_TOKENS).sum()
    print("Tickets:", len(tickets), "| empty:", n_empty, "| long:", n_long,
          "| long est_tokens:", sorted({est_tokens(s) for s in tickets["text"]
                                        if est_tokens(s) > MAX_INPUT_TOKENS}))

    # ---- Task 2: redaction counts over ALL 164 tickets
    counts = {"EMAIL": 0, "PHONE": 0, "ORDER_ID": 0}
    redacted_all = [redact(s, counts) for s in tickets["text"]]
    print("Substitutions:", counts, "| total:", sum(counts.values()))
    leftover = sum(len(pat.findall(s)) for s in redacted_all for _, pat, _ in PATTERNS)
    print("Patterns left after redaction:", leftover)
    assert leftover == 0

    # ---- Tasks 4-6: zero-shot and few-shot full runs (no budget)
    zero_df, zero_tok = run_pipeline(ZERO_SHOT, None, "zero-shot")
    zero_cls = summarise(zero_df, zero_tok, "ZERO-SHOT")
    print("\nZero-shot confusion table (rows = true, columns = predicted):")
    print(pd.crosstab(zero_cls["true_category"], zero_cls["category"]))

    few_df, few_tok = run_pipeline(FEW_SHOT, None, "few-shot")
    summarise(few_df, few_tok, "FEW-SHOT")

    # ---- Task 7a: budget demo (zero-shot, TOKEN_BUDGET = 6000)
    DEMO_BUDGET = 6000
    demo_df, demo_tok = run_pipeline(ZERO_SHOT, DEMO_BUDGET, "zero-shot budget demo")
    demo_df[COLS].to_csv(OUT / "triage_budget_demo.csv", index=False)
    print("\nbudget_exhausted tickets in demo:", (demo_df["status"] == "budget_exhausted").sum(),
          "| tokens used:", demo_tok)

    # ---- Task 7b: final results (few-shot, TOKEN_BUDGET = 100000)
    final_df, final_tok = run_pipeline(FEW_SHOT, 100000, "FINAL few-shot")
    final_df[COLS].to_csv(OUT / "triage_results.csv", index=False)
    print("\nFinal rows:", len(final_df))
    print(final_df["route"].value_counts().to_dict())
    print(pd.crosstab(final_df["route"], final_df["priority"]))