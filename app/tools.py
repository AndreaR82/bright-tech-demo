"""The non-AI half of the agent: SQL, product search, loan maths.

Two rules from the spec live in this file:

  * the SQL tool is hard-scoped to John's customer id — the model cannot reach
    another customer's data even if it writes a query that tries;
  * every number in an answer comes from here, never from the model.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "bank.db"
PRODUCTS_PATH = DATA_DIR / "products.json"
CUSTOMER_ID = "CUST-001"

MAX_ROWS = 40
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|replace)\b", re.I
)

SCHEMA_FOR_PROMPT = """\
transactions(txn_id, txn_date TEXT 'YYYY-MM-DD', merchant, description, category, amount REAL, account_id)
    amount is negative for money out, positive for money in.
    category is one of: groceries, dining_out, food_delivery, transport, rent, utilities,
    subscriptions, fitness, entertainment, shopping, health, travel, income,
    savings_transfer, loan_repayment, credit_card_payment, interest
accounts(account_id, account_name, account_type, balance, interest_rate, opened_date)
customers(customer_id, full_name, date_of_birth, suburb, state, occupation, gross_annual_income)

SQLite. Today is date('now'). Use date('now','-12 months') style filters.
Every table is already restricted to this customer — never filter on customer_id."""


def _connect() -> sqlite3.Connection:
    """Read-only connection whose tables are customer-scoped views.

    The model's SQL runs against views, so a query for another customer returns
    nothing at all — the permission check is the database, not the prompt.
    """
    con = sqlite3.connect(":memory:")
    con.execute("ATTACH DATABASE ? AS src", (f"file:{DB_PATH}?mode=ro",))
    # SQLite forbids bound parameters inside a view, so the id is inlined. It is a
    # module constant, never anything the model or a visitor supplied.
    assert re.fullmatch(r"[A-Z0-9\-]+", CUSTOMER_ID), "customer id must be a literal"
    for table in ("transactions", "accounts", "customers"):
        con.execute(
            f"CREATE TEMP VIEW {table} AS SELECT * FROM src.{table}"
            f" WHERE customer_id = '{CUSTOMER_ID}'"
        )
    return con


def run_sql(sql: str) -> dict[str, Any]:
    """Run one read-only SELECT and return columns + rows for display."""
    started = time.monotonic()
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned.lower().startswith(("select", "with")):
        return {"ok": False, "error": "only SELECT queries are allowed", "sql": cleaned, "seconds": 0.0}
    if ";" in cleaned or FORBIDDEN.search(cleaned):
        return {"ok": False, "error": "query rejected by the SQL guard", "sql": cleaned, "seconds": 0.0}

    try:
        con = _connect()
        cur = con.execute(cleaned)
        columns = [d[0] for d in cur.description or []]
        rows = [list(r) for r in cur.fetchmany(MAX_ROWS)]
        con.close()
    except sqlite3.Error as exc:
        return {"ok": False, "error": str(exc), "sql": cleaned, "seconds": time.monotonic() - started}

    return {
        "ok": True,
        "sql": cleaned,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "seconds": time.monotonic() - started,
    }


# ---------------------------------------------------------------- products ---

_catalogue = json.loads(PRODUCTS_PATH.read_text())
PRODUCTS: list[dict[str, Any]] = _catalogue["products"]
RATES_AS_AT: str = _catalogue["rates_as_at"]
BANK_NAME: str = _catalogue["bank_name"]

_STOPWORDS = {
    "the", "a", "an", "for", "of", "to", "and", "or", "is", "are", "what", "which",
    "me", "my", "i", "with", "on", "in", "best", "should", "can", "loan", "loans",
}
_SYNONYMS = {
    "mortgage": "home", "house": "home", "variable": "variable", "fix": "fixed",
    "fixed": "fixed", "saving": "savings", "save": "savings", "card": "credit",
    "deposit": "deposit", "offset": "offset",
}


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for w in words:
        w = _SYNONYMS.get(w, w)
        if w not in _STOPWORDS and len(w) > 2:
            out.append(w)
    return out


def search_products(query: str, k: int = 4, product_type: str | None = None) -> dict[str, Any]:
    """Keyword search over the catalogue.

    Deliberately not a vector database: 14 products, and swapping in
    EmbeddingGemma later only changes the body of this function.
    """
    started = time.monotonic()
    q = _tokens(query)
    scored: list[tuple[float, dict[str, Any]]] = []
    for p in PRODUCTS:
        if product_type and p["type"] != product_type:
            continue
        haystack = _tokens(f"{p['name']} {p['type']} {p['text']}")
        if not haystack:
            continue
        overlap = sum(haystack.count(t) for t in q)
        score = overlap / (len(haystack) ** 0.5)
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda s: s[0], reverse=True)
    hits = [p for _, p in scored[:k]] or PRODUCTS[:k]
    return {
        "ok": True,
        "query": query,
        "rates_as_at": RATES_AS_AT,
        "products": hits,
        "count": len(hits),
        "seconds": time.monotonic() - started,
    }


def get_product(product_id: str) -> dict[str, Any] | None:
    return next((p for p in PRODUCTS if p["id"].lower() == product_id.lower()), None)


# -------------------------------------------------------------- calculator ---


def monthly_repayment(principal: float, annual_rate_pct: float, years: int = 30) -> float:
    r = annual_rate_pct / 100 / 12
    n = years * 12
    if r == 0:
        return principal / n
    return principal * r / (1 - (1 + r) ** -n)


def borrowing_power(
    gross_annual_income: float,
    monthly_living_expenses: float,
    monthly_debt_repayments: float = 0.0,
    assessment_rate_pct: float = 8.94,
    years: int = 30,
) -> dict[str, Any]:
    """Deliberately simple, and labelled as such on screen.

    Assessed at a buffered rate (product rate + 3%), the way a lender would.
    """
    started = time.monotonic()
    net_monthly = gross_annual_income * 0.715 / 12  # rough after-tax
    surplus = net_monthly - monthly_living_expenses - monthly_debt_repayments
    surplus_for_loan = max(surplus * 0.9, 0.0)  # keep a buffer
    r = assessment_rate_pct / 100 / 12
    n = years * 12
    max_loan = surplus_for_loan * (1 - (1 + r) ** -n) / r if r else surplus_for_loan * n
    return {
        "ok": True,
        "net_monthly_income": round(net_monthly, 2),
        "monthly_surplus": round(surplus, 2),
        "assessment_rate_pct": assessment_rate_pct,
        "term_years": years,
        "max_loan": round(max_loan, -3),
        "seconds": time.monotonic() - started,
    }


def loan_repayment(principal: float, annual_rate_pct: float, years: int = 30) -> dict[str, Any]:
    started = time.monotonic()
    monthly = monthly_repayment(principal, annual_rate_pct, years)
    return {
        "ok": True,
        "principal": round(principal, 2),
        "annual_rate_pct": annual_rate_pct,
        "term_years": years,
        "monthly_repayment": round(monthly, 2),
        "total_interest": round(monthly * years * 12 - principal, 2),
        "seconds": time.monotonic() - started,
    }
