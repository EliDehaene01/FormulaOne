"""
Lightweight cost/latency logging for every Azure OpenAI call this project
makes (chat completions in llm/client.py and llm/summary.py, embeddings in
rag/embeddings.py) — timestamp, deployment, token counts, estimated cost,
and latency, written to a local SQLite table. No new Azure resource: this
is entirely local, queryable with print_summary() below or any SQLite
browser.

Run the summary with:
    python -m backend.observability
    python -m backend.observability --since 2026-08-01
"""

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "storage" / "llm_usage.sqlite"

# Published Azure OpenAI / OpenAI per-token pricing, USD per 1,000,000
# tokens, checked August 2026 (Azure OpenAI tracks OpenAI's direct API
# pricing for these models):
#   https://azure.microsoft.com/en-us/pricing/details/azure-openai/
#   https://developers.openai.com/api/docs/pricing
# This is the ONLY place estimated cost is computed from — update here if
# pricing changes, or if a deployment gets renamed (an unrecognized
# deployment name logs cost as NULL rather than a silently-wrong $0, so a
# stale/missing entry here is visible in the summary instead of hidden).
PRICING_USD_PER_MILLION_TOKENS = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}


def _connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            call_type TEXT NOT NULL,       -- 'chat' or 'embedding'
            deployment TEXT NOT NULL,
            input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            estimated_cost_usd REAL,       -- NULL when `deployment` isn't in PRICING_USD_PER_MILLION_TOKENS
            latency_ms REAL NOT NULL,
            caller TEXT NOT NULL           -- which function made this call, e.g. "llm.client.chat"
        )
        """
    )
    return conn


def _estimate_cost(deployment: str, input_tokens: int, output_tokens: int) -> float | None:
    pricing = PRICING_USD_PER_MILLION_TOKENS.get(deployment)
    if pricing is None:
        return None
    return round((input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000, 8)


def log_call(*, call_type: str, deployment: str, input_tokens: int, output_tokens: int, latency_ms: float, caller: str) -> None:
    """Call this right after an Azure OpenAI request returns — see llm/client.py, llm/summary.py, rag/embeddings.py for the call sites."""
    cost = _estimate_cost(deployment, input_tokens, output_tokens)
    conn = _connection()
    conn.execute(
        "INSERT INTO llm_calls (timestamp, call_type, deployment, input_tokens, output_tokens, estimated_cost_usd, latency_ms, caller) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), call_type, deployment, input_tokens, output_tokens, cost, latency_ms, caller),
    )
    conn.commit()
    conn.close()


def print_summary(since: str | None = None) -> None:
    """Per-call-type totals (calls, tokens, cost, average latency), optionally filtered to timestamp >= `since` (ISO date/datetime)."""
    conn = _connection()
    query = (
        "SELECT call_type, COUNT(*), SUM(input_tokens), SUM(output_tokens), SUM(estimated_cost_usd), AVG(latency_ms) "
        "FROM llm_calls"
    )
    params: list = []
    if since:
        query += " WHERE timestamp >= ?"
        params.append(since)
    query += " GROUP BY call_type ORDER BY call_type"
    rows = conn.execute(query, params).fetchall()
    conn.close()

    if not rows:
        print("No logged LLM calls" + (f" since {since}." if since else "."))
        return

    print(f"{'type':12s} {'calls':>6s} {'in tokens':>10s} {'out tokens':>11s} {'cost ($)':>10s} {'avg ms':>9s}")
    print("-" * 62)
    total_calls = 0
    total_cost = 0.0
    any_unpriced = False
    for call_type, calls, in_tok, out_tok, cost, avg_latency in rows:
        if cost is None:
            any_unpriced = True
        print(f"{call_type:12s} {calls:>6d} {in_tok or 0:>10d} {out_tok or 0:>11d} {(cost or 0.0):>10.4f} {avg_latency:>9.0f}")
        total_calls += calls
        total_cost += cost or 0.0
    print("-" * 62)
    print(f"Total calls: {total_calls}   Total estimated cost: ${total_cost:.4f}")
    if any_unpriced:
        print("Note: at least one deployment name isn't in PRICING_USD_PER_MILLION_TOKENS — its cost is excluded from the total above, not zero.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize logged Azure OpenAI usage (cost + latency).")
    parser.add_argument("--since", type=str, default=None, help="Only include calls at/after this ISO date or datetime, e.g. 2026-08-01")
    args = parser.parse_args()
    print_summary(args.since)


if __name__ == "__main__":
    main()
