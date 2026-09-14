"""The agent: a fixed safety envelope around agentic specialists.

    input check → route → specialist (agentic, ≤4 tool calls) → draft
    → advice check ∥ fact check → ≤1 rewrite → release

Every stage yields trace events, which are what the right-hand panel draws. The
model decides *how* to answer; this file decides *that it gets checked*.
"""

from __future__ import annotations

import concurrent.futures
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

from . import llm, tools

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text())


CFG = load_config()

# ------------------------------------------------------------------ schemas ---

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["safe", "off_topic", "other_customer", "offensive", "injection"]},
        "reason": {"type": "string", "maxLength": 90},
    },
    "required": ["label", "reason"],
    "additionalProperties": False,
}

ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "specialist": {"type": "string", "enum": ["spending_analyst", "product_advisor", "small_talk"]},
        "reason": {"type": "string", "maxLength": 90},
        "follow_up": {"type": "string", "maxLength": 40},
    },
    "required": ["specialist", "reason", "follow_up"],
    "additionalProperties": False,
}

SQL_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["query", "finish"]},
        "sql": {"type": "string", "maxLength": 600},
        "note": {"type": "string", "maxLength": 120},
    },
    "required": ["action", "sql", "note"],
    "additionalProperties": False,
}

PRODUCT_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["search", "repayment", "borrowing_power", "finish"]},
        "query": {"type": "string", "maxLength": 120},
        "principal": {"type": "number"},
        "annual_rate_pct": {"type": "number"},
        "years": {"type": "integer"},
        "note": {"type": "string", "maxLength": 120},
    },
    "required": ["action", "note"],
    "additionalProperties": False,
}

ADVICE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": ["factual_information", "personalised_recommendation"],
        },
        "sentence": {"type": "string", "maxLength": 200},
        "reason": {"type": "string", "maxLength": 120},
    },
    "required": ["label", "sentence", "reason"],
    "additionalProperties": False,
}

# The judge was fine-tuned on this exact contract — see gemma4-groundedness-judge.
JUDGE_SYSTEM = (
    "You are a strict groundedness judge. You are given a Document and a Claim. "
    "Decide whether the Claim is fully supported by the Document.\n"
    "Rules:\n"
    "- Use ONLY the Document. Ignore outside knowledge and your own beliefs.\n"
    "- A Claim is GROUNDED only if every part of it is directly stated or "
    "entailed by the Document.\n"
    "- If any part is contradicted by, or absent from, the Document, the Claim "
    "is NOT GROUNDED.\n"
    "Reason briefly step by step, then end your answer with a final line in "
    "exactly this form:\n"
    "Verdict: GROUNDED\n"
    "or\n"
    "Verdict: NOT GROUNDED"
)
_VERDICT_RE = re.compile(r"verdict\s*[:\-]?\s*\**\s*(not\s+grounded|ungrounded|not\s+supported|unsupported|grounded|supported)\b", re.I)

REFUSALS = {
    "off_topic": "I can only help with {customer}'s banking — spending, accounts and our products.",
    "other_customer": "I can only see {customer}'s own accounts. I can't look up anyone else's banking.",
    "offensive": "Let's keep this about banking. Ask me about {customer}'s spending or our products.",
    "injection": "I'll stick to my job here: {customer}'s accounts and our products.",
}


# ------------------------------------------------------------------ session ---


@dataclass
class Session:
    number: int = 1
    turn: int = 0
    history: list[dict[str, str]] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    def transcript(self, limit: int = 6) -> list[dict[str, str]]:
        return self.history[-limit * 2 :]


@dataclass
class Counters:
    turns: int = 0
    advice_blocked_eager: int = 0
    advice_blocked_careful: int = 0
    fact_failures: int = 0
    input_blocks: int = 0
    tokens: int = 0
    seconds: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.seconds if self.seconds else 0.0


COUNTERS = Counters()


# ------------------------------------------------------------------- helpers ---


def _prompt(name: str, **extra: Any) -> str:
    return CFG["prompts"][name].format(
        bank_name=CFG["bank_name"],
        customer_name=CFG["customer_name"],
        schema=tools.SCHEMA_FOR_PROMPT,
        max_tool_calls=CFG["limits"]["max_tool_calls"],
        answer_words=CFG["limits"]["answer_words"],
        **extra,
    )


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_CHECKABLE_RE = re.compile(r"\d|\bper cent\b|%|\$", re.I)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def _render_facts(facts: list[dict[str, Any]]) -> str:
    """Tool results as a plain-text document — the writer's only source, and the
    fact checker's Document."""
    out = []
    for f in facts:
        if f["kind"] == "sql":
            r = f["result"]
            if r.get("ok"):
                header = " | ".join(str(c) for c in r["columns"])
                rows = "\n".join(" | ".join(str(v) for v in row) for row in r["rows"])
                out.append(f"Query: {f['result']['sql']}\n{header}\n{rows}")
        elif f["kind"] == "products":
            for p in f["result"]["products"]:
                bits = [f"{p['name']} ({p['id']})", f"type: {p['type']}"]
                for key in ("rate", "comparison_rate", "annual_fee", "max_lvr", "purchase_rate", "base_rate"):
                    if key in p:
                        bits.append(f"{key}: {p[key]}")
                bits.append(p["text"])
                out.append(". ".join(bits))
            out.append(f"Rates as at {f['result']['rates_as_at']}.")
        elif f["kind"] == "calculator":
            r = {k: v for k, v in f["result"].items() if k not in ("ok", "seconds")}
            out.append(f"Calculator ({f['name']}): " + ", ".join(f"{k} = {v}" for k, v in r.items()))
    return "\n\n".join(out) if out else "(no data was retrieved)"


# -------------------------------------------------------------------- stages ---


def _step(step: str, model: str | None, kind: str = "ai") -> dict[str, Any]:
    return {
        "type": "step_start",
        "step": step,
        "model": llm.label_for(model) if model else "—",
        "kind": kind,
        "at": time.time(),
    }


def _done(seconds: float, outcome: str, badge: str = "pass", detail: Any = None) -> dict[str, Any]:
    return {
        "type": "step_end",
        "seconds": round(seconds, 2),
        "outcome": outcome,
        "badge": badge,
        "detail": detail,
    }


def _spending_agent(question: str, session: Session) -> Iterator[dict[str, Any]]:
    """Agentic: the model writes SQL, reads the result, decides whether to go again."""
    facts: list[dict[str, Any]] = []
    messages = [{"role": "system", "content": _prompt("spending_analyst")}]
    messages += session.transcript(2)
    messages.append({"role": "user", "content": question})

    for _ in range(CFG["limits"]["max_tool_calls"]):
        yield _step("Spending Analyst", llm.BASE_MODEL)
        # Generous cap: constrained JSON that truncates mid-string is unparseable,
        # and a 4B model occasionally pads. Unused tokens cost nothing.
        c = llm.complete(messages, schema=SQL_ACTION_SCHEMA, max_tokens=600)
        COUNTERS.tokens += c.completion_tokens
        if c.error or not c.data:
            yield _done(c.seconds, "⚠️ model error", "warn", {"error": c.error})
            break
        action = c.data
        if action["action"] == "finish" or not action.get("sql"):
            yield _done(c.seconds, "done", "neutral", {"note": action.get("note", "")})
            break
        yield _done(c.seconds, action.get("note") or "wrote a query", "neutral", {"sql": action["sql"]})

        yield _step("SQL query", None, kind="tool")
        result = tools.run_sql(action["sql"])
        if result.get("ok"):
            yield _done(result["seconds"], f"{result['row_count']} rows", "neutral",
                        {"sql": result["sql"], "columns": result["columns"], "rows": result["rows"][:8]})
            facts.append({"kind": "sql", "result": result})
            messages.append({"role": "assistant", "content": f"Query: {action['sql']}"})
            messages.append({"role": "user", "content": f"Result:\n{_render_facts([facts[-1]])}\n\nAnother query, or finish?"})
        else:
            yield _done(result.get("seconds", 0), "❌ query failed", "warn", {"error": result.get("error")})
            messages.append({"role": "assistant", "content": f"Query: {action['sql']}"})
            messages.append({"role": "user", "content": f"That query failed: {result.get('error')}. Fix it or finish."})

    yield {"type": "facts", "facts": facts}


def _product_agent(question: str, session: Session) -> Iterator[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    messages = [{"role": "system", "content": _prompt("product_advisor")}]
    messages += session.transcript(2)
    messages.append({"role": "user", "content": question})

    for _ in range(CFG["limits"]["max_tool_calls"]):
        yield _step("Product Advisor", llm.BASE_MODEL)
        c = llm.complete(messages, schema=PRODUCT_ACTION_SCHEMA, max_tokens=400)
        COUNTERS.tokens += c.completion_tokens
        if c.error or not c.data:
            yield _done(c.seconds, "⚠️ model error", "warn", {"error": c.error})
            break
        action = c.data
        kind = action["action"]
        if kind == "finish":
            yield _done(c.seconds, "done", "neutral", {"note": action.get("note", "")})
            break
        yield _done(c.seconds, action.get("note") or kind, "neutral", None)

        if kind == "search":
            yield _step("Product search", None, kind="tool")
            result = tools.search_products(action.get("query") or question)
            yield _done(result["seconds"], f"{result['count']} products", "neutral",
                        {"products": [p["name"] for p in result["products"]]})
            facts.append({"kind": "products", "result": result})
        elif kind == "repayment":
            yield _step("Loan calculator", None, kind="tool")
            result = tools.loan_repayment(
                float(action.get("principal") or 500000),
                float(action.get("annual_rate_pct") or 5.94),
                int(action.get("years") or 30),
            )
            yield _done(result["seconds"], f"${result['monthly_repayment']:,.0f}/mo", "neutral", result)
            facts.append({"kind": "calculator", "name": "repayment", "result": result})
        else:  # borrowing_power
            yield _step("Loan calculator", None, kind="tool")
            spend = tools.run_sql(
                "SELECT ROUND(AVG(m),2) FROM (SELECT strftime('%Y-%m',txn_date) k, SUM(-amount) m"
                " FROM transactions WHERE amount<0 AND category NOT IN ('savings_transfer','credit_card_payment')"
                " AND txn_date >= date('now','-6 months') GROUP BY k)"
            )
            monthly_expenses = (spend["rows"][0][0] if spend.get("ok") and spend["rows"] else 3500.0) or 3500.0
            result = tools.borrowing_power(105000.0, float(monthly_expenses), 612.0)
            yield _done(result["seconds"], f"max ${result['max_loan']:,.0f}", "neutral", result)
            facts.append({"kind": "calculator", "name": "borrowing_power", "result": result})

        messages.append({"role": "assistant", "content": f"{kind}: {action.get('note','')}"})
        messages.append({"role": "user", "content": f"Result:\n{_render_facts([facts[-1]])}\n\nAnother tool, or finish?"})

    yield {"type": "facts", "facts": facts}


def _advice_check(draft: str) -> tuple[dict[str, Any], float]:
    c = llm.complete(
        [{"role": "system", "content": _prompt("advice_check")},
         {"role": "user", "content": f"Draft reply:\n{draft}"}],
        model=llm.ADVICE_MODEL, schema=ADVICE_SCHEMA, max_tokens=160,
    )
    COUNTERS.tokens += c.completion_tokens
    return (c.data or {"label": "factual_information", "sentence": "", "reason": "check unavailable"}), c.seconds


def _fact_check(draft: str, document: str) -> tuple[list[dict[str, Any]], float]:
    """Sentence-level, only sentences that actually state a checkable fact."""
    started = time.monotonic()
    claims = [s for s in _sentences(draft) if _CHECKABLE_RE.search(s)]
    if not claims:
        return [], time.monotonic() - started

    def judge(claim: str) -> dict[str, Any]:
        c = llm.complete(
            [{"role": "system", "content": JUDGE_SYSTEM},
             {"role": "user", "content": f"Document:\n{document.strip()}\n\nClaim:\n{claim.strip()}"}],
            model=llm.JUDGE_MODEL, max_tokens=150,
        )
        COUNTERS.tokens += c.completion_tokens
        m = _VERDICT_RE.search(c.text or "")
        grounded = True if not m else m.group(1).lower() in ("grounded", "supported")
        return {"claim": claim, "grounded": grounded}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        verdicts = list(pool.map(judge, claims[:4]))
    return verdicts, time.monotonic() - started


def _draft(
    question: str,
    document: str,
    session: Session,
    mode: str,
    extra: str = "",
    temperature: float = 0.0,
) -> Iterator[Any]:
    """Streams tokens; the last yielded value is the finished text.

    `temperature` stays 0 at the booth (same question → same answer, which matters
    when a visitor retries). Training-data generation raises it for variety.
    """
    system = _prompt("writer_careful" if mode == "careful" else "writer_eager")
    messages = [{"role": "system", "content": system}]
    messages += session.transcript(3)
    messages.append({"role": "user", "content": f"Question: {question}\n\nData you may use:\n{document}{extra}"})
    pieces: list[str] = []
    for piece in llm.stream(messages, max_tokens=200, temperature=temperature):
        pieces.append(piece)
        yield {"type": "draft_token", "text": piece}
    yield {"type": "draft_text", "text": "".join(pieces).strip()}


# ---------------------------------------------------------------- the turn ---


def run_turn(question: str, session: Session, mode: str | None = None) -> Iterator[dict[str, Any]]:
    mode = mode or CFG.get("writer_mode", "eager")
    turn_started = time.monotonic()
    session.turn += 1
    session.last_activity = time.time()
    COUNTERS.turns += 1
    customer = CFG["customer_name"].split()[0]
    yield {"type": "turn_start", "turn": session.turn, "question": question, "mode": mode}

    # 1 + 2. Input check and routing run in parallel: the check is never skipped,
    # and the router's answer is simply thrown away if the question is blocked.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    check_future = pool.submit(
        llm.complete,
        [{"role": "system", "content": _prompt("input_check")}, {"role": "user", "content": question}],
        schema=INPUT_SCHEMA, max_tokens=110,
    )
    route_future = pool.submit(
        llm.complete,
        [{"role": "system", "content": _prompt("router")},
         *session.transcript(2),
         {"role": "user", "content": question}],
        schema=ROUTE_SCHEMA, max_tokens=110,
    )

    yield _step("Input check", llm.BASE_MODEL)
    c = check_future.result()
    COUNTERS.tokens += c.completion_tokens
    verdict = (c.data or {}).get("label", "safe")
    if verdict != "safe":
        COUNTERS.input_blocks += 1
        yield _done(c.seconds, f"🚫 {verdict.replace('_', ' ')}", "block", {"reason": (c.data or {}).get("reason")})
        pool.shutdown(wait=False)
        yield {"type": "message", "role": "assistant", "text": REFUSALS[verdict].format(customer=customer)}
        yield {"type": "turn_end", "seconds": round(time.monotonic() - turn_started, 2)}
        return
    yield _done(c.seconds, "✅ Pass", "pass", {"reason": (c.data or {}).get("reason")})

    yield _step("Route", llm.BASE_MODEL)
    c = route_future.result()
    pool.shutdown(wait=False)
    COUNTERS.tokens += c.completion_tokens
    route = (c.data or {}).get("specialist", "spending_analyst")
    follow_up = (c.data or {}).get("follow_up", "").strip()
    label = {"spending_analyst": "Spending Analyst", "product_advisor": "Product Advisor", "small_talk": "Small talk"}[route]
    yield _done(c.seconds, f"→ {label}" + (f" ↩︎ {follow_up}" if follow_up else ""), "neutral",
                {"reason": (c.data or {}).get("reason")})

    # 3. Specialist — the agentic part.
    facts: list[dict[str, Any]] = []
    if route in ("spending_analyst", "product_advisor"):
        agent = _spending_agent if route == "spending_analyst" else _product_agent
        for event in agent(question, session):
            if event["type"] == "facts":
                facts = event["facts"]
            else:
                yield event
    document = _render_facts(facts)

    # 4. Draft — streams into the panel only. Nothing reaches the chat unchecked.
    yield _step("Draft answer", llm.BASE_MODEL)
    draft_started = time.monotonic()
    draft = ""
    for event in _draft(question, document, session, mode):
        if event["type"] == "draft_text":
            draft = event["text"]
        else:
            yield event
    yield _done(time.monotonic() - draft_started, "✍️ Done", "neutral", None)

    # 5. Advice check ∥ fact check.
    yield _step("Advice check", llm.ADVICE_MODEL)
    checks_started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        advice_future = pool.submit(_advice_check, draft)
        fact_future = pool.submit(_fact_check, draft, document)
        advice, advice_seconds = advice_future.result()
        verdicts, fact_seconds = fact_future.result()

    blocked = advice["label"] == "personalised_recommendation"
    if blocked:
        if mode == "eager":
            COUNTERS.advice_blocked_eager += 1
        else:
            COUNTERS.advice_blocked_careful += 1
        yield _done(advice_seconds, "🚫 Blocked", "block", {"reason": advice.get("reason"), "sentence": advice.get("sentence")})
        yield {"type": "draft_blocked", "text": draft, "reason": advice.get("reason", "")}
    else:
        # The detector is binary: not blocked means not personalised. Whether a
        # general-advice warning rides along is a deterministic call on the route,
        # not a model judgement — every product answer gets one.
        badge = "warn" if route == "product_advisor" else "pass"
        text = "🟡 General info" if badge == "warn" else "✅ Factual"
        yield _done(advice_seconds, text, badge, {"reason": advice.get("reason")})

    ungrounded = [v["claim"] for v in verdicts if not v["grounded"]]
    yield _step("Fact check", llm.JUDGE_MODEL)
    if verdicts:
        ok = len(verdicts) - len(ungrounded)
        yield _done(fact_seconds, f"{'✅' if not ungrounded else '❌'} {ok}/{len(verdicts)} verified",
                    "pass" if not ungrounded else "block", {"claims": verdicts})
    else:
        yield _done(fact_seconds, "no claims to check", "neutral", None)
    if ungrounded:
        COUNTERS.fact_failures += 1

    # 6. One rewrite, shared by both checks.
    if blocked or ungrounded:
        notes = []
        if blocked:
            notes.append(
                "Your draft told the customer what he should do. State only the facts and"
                " the differences between the options. Do not tell him what to choose, and"
                " do not add advice about what to do next."
            )
        if ungrounded:
            notes.append("These statements are not supported by the data — fix or remove them: " + " | ".join(ungrounded))
        yield _step("Rewrite", llm.BASE_MODEL)
        rewrite_started = time.monotonic()
        rewritten = ""
        for event in _draft(question, document, session, "careful", extra="\n\nProblems with your first draft:\n" + "\n".join(notes)):
            if event["type"] == "draft_text":
                rewritten = event["text"]
            else:
                yield event
        yield _done(time.monotonic() - rewrite_started, "✍️ Rewritten", "neutral", None)
        draft = rewritten

        # Re-check facts once; anything still unsupported is dropped, not retried.
        verdicts, fact_seconds = _fact_check(draft, document)
        still_bad = [v["claim"] for v in verdicts if not v["grounded"]]
        if still_bad:
            yield _step("Fact check", llm.JUDGE_MODEL)
            kept = [s for s in _sentences(draft) if s not in still_bad]
            draft = " ".join(kept) or "I couldn't verify those numbers, so I'd rather not guess. Try asking a different way."
            yield _done(fact_seconds, f"⚠️ {len(still_bad)} sentence(s) removed", "warn", {"removed": still_bad})

    # Rewritten drafts still go out, and a rewritten product answer is still general
    # information about products, so the warning does not depend on `blocked`.
    warning = "General information only — not a recommendation." if route == "product_advisor" else None
    session.history.append({"role": "user", "content": question})
    session.history.append({"role": "assistant", "content": draft})
    if follow_up:
        session.topics.append(follow_up)

    seconds = time.monotonic() - turn_started
    COUNTERS.seconds += seconds
    yield {"type": "message", "role": "assistant", "text": draft, "warning": warning}
    yield {"type": "turn_end", "seconds": round(seconds, 2)}
