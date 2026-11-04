"""Part D, tasks 5-8 - refund agent (tools, ReAct loop, safe stopping, resumable runs).

Run:  python refund_agent.py
In :  data/instamart.db (Part A), data/raw/agent_scenarios.csv, policy_rag.py, mock_llm.py
Out:  outputs/agent_runs.jsonl, outputs/refund_ledger.jsonl, state/<run_id>.json
"""
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

import mock_llm
from policy_rag import retrieve

DB_PATH = Path("data") / "instamart.db"
OUT = Path("outputs")
STATE = Path("state")
LEDGER = OUT / "refund_ledger.jsonl"
RUNS = OUT / "agent_runs.jsonl"
OUT.mkdir(exist_ok=True)
STATE.mkdir(exist_ok=True)

SYSTEM_PROMPT = (
    "You are a refund AGENT for Instamart support. Look up the order, check the policy, "
    "calculate the refund, then issue it or escalate it. Use the tools one step at a time."
)


class SimulatedCrash(Exception):
    """Raised on purpose to demonstrate resuming from a checkpoint."""


# ------------------------------------------------------------------ Task 5: tools
def get_order(order_id):
    con = sqlite3.connect(DB_PATH)
    try:
        row = con.execute(
            "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
            (order_id,)).fetchone()
    finally:
        con.close()
    if row is None:
        return {"error": "order not found"}
    oid, status, amount, minutes = row
    return {"order_id": oid, "status": status, "order_amount": amount,
            "delivery_minutes": None if minutes is None or pd.isna(minutes) else float(minutes)}


def search_policy(query):
    return {"results": retrieve(query)}


def calculate_refund(order_amount, delivery_minutes):
    """> 45 minutes -> 100 % of the order amount; otherwise 0."""
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0.0}


def _ledger(tool, order_id, refund_inr):
    line = {"timestamp": datetime.now().isoformat(timespec="seconds"), "tool": tool,
            "order_id": order_id, "refund_inr": refund_inr}
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


def _spec(name, description, props):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": list(props)}}


TOOLS_SPEC = [
    _spec("get_order", "Look up one order by id in the Instamart database.",
          {"order_id": {"type": "string"}}),
    _spec("search_policy", "Search the policy documents and return the best matching chunks.",
          {"query": {"type": "string"}}),
    _spec("calculate_refund", "Compute the refund in INR from the order amount and delivery minutes.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}),
    _spec("issue_refund", "Issue an approved refund of refund_inr for the order and record it.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human support lead.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy,
            "calculate_refund": calculate_refund, "issue_refund": issue_refund,
            "escalate_to_human": escalate_to_human}


# ------------------------------------------------------------------ Task 8: checkpoints
def save_checkpoint(st):
    (STATE / f"{st['run_id']}.json").write_text(json.dumps(st, default=str), encoding="utf-8")


# ------------------------------------------------------------------ Tasks 6-7: the loop
def _fallback(st, status, reason):
    done = ", ".join(s["tool"] for s in st["steps"]) or "none"
    st["status"] = status
    st["final_answer"] = (f"HUMAN HANDOFF: {reason}. Scenario {st['scenario_id']}; "
                          f"tools completed so far: {done}; tokens used: {st['tokens_used']}.")
    return _result(st)


def _result(st):
    return {"run_id": st["run_id"], "scenario_id": st["scenario_id"], "status": st["status"],
            "final_answer": st["final_answer"], "steps": st["steps"], "llm_calls": st["llm_calls"],
            "tokens_used": st["tokens_used"], "tools_executed": st["tools_executed"]}


def _loop(st, max_iterations, token_limit, time_limit_seconds, crash_after_tools):
    start = time.monotonic()
    while True:
        if time.monotonic() - start >= time_limit_seconds:                     # (c) timeout
            return _fallback(st, "FALLBACK_TIMEOUT", "time limit reached")
        if st["llm_calls"] >= max_iterations:                                  # (a) max iterations
            return _fallback(st, "FALLBACK_MAX_ITERATIONS", "no final answer within the iteration limit")

        resp = mock_llm.chat(st["messages"], tools=TOOLS_SPEC)
        st["llm_calls"] += 1
        st["tokens_used"] += resp.usage.total_tokens
        if st["tokens_used"] > token_limit:                                    # (b) token limit
            return _fallback(st, "FALLBACK_TOKEN_LIMIT", "token limit exceeded")   # tool NOT executed

        msg = resp.choices[0].message
        if not msg.tool_calls:                                                 # final answer
            st["final_answer"] = msg.content
            st["status"] = msg.content.split(":")[0].strip()
            return _result(st)

        tc = msg.tool_calls[0]
        call = {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
        st["messages"].append({"role": "assistant", "content": None, "tool_calls": [call]})
        observation = TOOL_MAP[tc.name](**tc.arguments)                        # look up + call
        st["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                               "content": json.dumps(observation, default=str)})
        st["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": observation})
        st["tools_executed"].append(tc.name)
        st["step"] += 1
        save_checkpoint(st)                                                    # after each observation
        if crash_after_tools and len(st["tools_executed"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool execution #{crash_after_tools}")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, run_id=None, crash_after_tools=None, log=True):
    st = {"run_id": run_id or f"{scenario_id}_{int(time.time() * 1000)}",
          "scenario_id": scenario_id, "status": None, "final_answer": None,
          "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user", "content": user_message}],
          "tokens_used": 0, "llm_calls": 0, "step": 0, "steps": [], "tools_executed": [],
          "limits": {"max_iterations": max_iterations, "token_limit": token_limit,
                     "time_limit_seconds": time_limit_seconds}}
    mock_llm.reset()
    res = _loop(st, max_iterations, token_limit, time_limit_seconds, crash_after_tools)
    if log:
        with open(RUNS, "a", encoding="utf-8") as f:
            f.write(json.dumps({k: res[k] for k in
                                ("scenario_id", "status", "final_answer", "llm_calls",
                                 "tokens_used", "steps")}, default=str) + "\n")
    return res


def resume_agent(run_id):
    """Load state/<run_id>.json and continue; already-executed tools are never repeated."""
    st = json.loads((STATE / f"{run_id}.json").read_text(encoding="utf-8"))
    lim = st["limits"]
    return _loop(st, lim["max_iterations"], lim["token_limit"], lim["time_limit_seconds"], None)


# ------------------------------------------------------------------ demonstrations
if __name__ == "__main__":
    for p in (RUNS, LEDGER):                    # fresh outputs so reruns are reproducible
        p.unlink(missing_ok=True)

    scen = pd.read_csv("data/raw/agent_scenarios.csv", keep_default_na=False)
    rows = []
    for r in scen.itertuples(index=False):
        res = run_agent(r.message, r.scenario_id)
        refund = next((s["observation"]["refund_inr"] for s in res["steps"]
                       if s["tool"] in ("issue_refund", "escalate_to_human")), 0.0)
        ok = (abs(refund - float(r.expected_refund_inr)) <= 0.01
              and res["status"].startswith(r.expected_outcome))
        rows.append({"scenario": r.scenario_id, "status": res["status"], "refund_inr": refund,
                     "expected": r.expected_refund_inr, "llm_calls": res["llm_calls"],
                     "tokens_used": res["tokens_used"], "match": ok})
    print("== 7 scenarios ==")
    print(pd.DataFrame(rows).to_string(index=False))

    s1 = scen.iloc[0]
    d1 = run_agent(s1.message, "S1", token_limit=200, run_id="S1_token_demo", log=False)
    print("\nS1 with token_limit=200 ->", d1["status"], "| tokens:", d1["tokens_used"])
    d2 = run_agent(s1.message, "S1", time_limit_seconds=0, run_id="S1_time_demo", log=False)
    print("S1 with time_limit_seconds=0 ->", d2["status"], "| llm_calls:", d2["llm_calls"])

    print("\n== Crash and resume (S1) ==")
    try:
        run_agent(s1.message, "S1", run_id="S1_crash_demo", crash_after_tools=2, log=False)
    except SimulatedCrash as e:
        print("Crashed:", e)
    final = resume_agent("S1_crash_demo")
    print("Resumed status:", final["status"], "|", final["final_answer"])
    print("Tools executed across crash + resume:", final["tools_executed"])
    assert final["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]