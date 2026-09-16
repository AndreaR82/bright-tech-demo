"""FastAPI app: serves the booth page and streams trace events over SSE.

Endpoints
    GET  /               the booth page
    POST /api/ask        {question, think_hard} → SSE stream of trace events
    POST /api/reset      end the visitor's session, compact it into booth memory
    POST /api/mode       presenter switch: eager | careful
    POST /api/memory/save    commit the pending customer profile to John's file
    POST /api/memory/forget  wipe John's file (presenter control)
    GET  /api/state      cards, counters, booth + customer memory, health
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import llm, memory, pipeline, tools

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Bright Tech Demo")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

SESSION = pipeline.Session()
MEMORY: list[dict[str, Any]] = []
MODE = pipeline.CFG.get("writer_mode", "eager")

# Booth memory is public on a big screen: strip anything that looks personal
# before it can ever be displayed.
REDACTIONS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[redacted]"),
    (re.compile(r"\b(?:\+?61|0)[\d ]{8,12}\b"), "[redacted]"),
    (re.compile(r"\b\d{6,}\b"), "[redacted]"),
    (re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?"), "[amount]"),
]

COMPACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "maxLength": 160},
        "topics": {"type": "array", "items": {"type": "string", "maxLength": 30}, "maxItems": 4},
    },
    "required": ["summary", "topics"],
    "additionalProperties": False,
}


def redact(text: str) -> str:
    for pattern, replacement in REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


class Ask(BaseModel):
    question: str
    # The Think hard card: the writer runs with Gemma's thinking mode on. Only the
    # writer — every check keeps its JSON-constrained decoding.
    think_hard: bool = False
    # Which vLLM endpoint answers this turn: "bf16" or "fp8". Per-request rather
    # than a server global, so a click mid-turn can never split one turn across two
    # backends. Anything unknown resolves to bf16 (llm.resolve).
    precision: str = "bf16"


class Mode(BaseModel):
    mode: str


def sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/ask")
def ask(body: Ask) -> StreamingResponse:
    question = body.question.strip()[: pipeline.CFG["limits"]["max_question_chars"]]

    def events() -> Iterator[str]:
        if not question:
            yield sse({"type": "turn_end", "seconds": 0})
            return
        try:
            # A pill left on FP8 while that server is down would strand the booth,
            # so fall back rather than fail — and say so on the trace.
            # An unknown name resolves to bf16 anyway (llm.resolve), so only warn
            # about a backend that really exists and really is not answering.
            precision = body.precision if body.precision in llm.BACKENDS else "bf16"
            if precision != "bf16" and not llm.availability()[precision]["ok"]:
                yield sse({"type": "step_start", "step": "Precision", "model": "—", "kind": "tool",
                           "at": time.time()})
                yield sse({"type": "step_end", "seconds": 0, "badge": "warn",
                           "outcome": f"⚠️ {precision} endpoint is down — answering in bf16",
                           "detail": {"base_url": llm.BACKENDS[precision].base_url}})
                precision = "bf16"
            for event in pipeline.run_turn(question, SESSION, mode=MODE,
                                           think_hard=body.think_hard, precision=precision):
                yield sse(event)
        except Exception as exc:  # never leave a visitor looking at a frozen screen
            yield sse({"type": "message", "role": "assistant",
                       "text": "Something broke behind the scenes. Try another question."})
            yield sse({"type": "step_end", "seconds": 0, "outcome": f"⚠️ {type(exc).__name__}", "badge": "warn",
                       "detail": {"error": str(exc)}})
            yield sse({"type": "turn_end", "seconds": 0})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def compact_session(session: pipeline.Session) -> dict[str, Any] | None:
    """One Gemma call turns a finished visitor session into a booth-memory card."""
    if not session.history:
        return None
    transcript = "\n".join(
        f"{'Visitor' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in session.history
    )
    c = llm.complete(
        [{"role": "system", "content": pipeline.CFG["prompts"]["compaction"]},
         {"role": "user", "content": transcript}],
        schema=COMPACTION_SCHEMA, max_tokens=160,
        # Pinned to bf16: booth memory goes on a public screen and must be
        # reproducible, and must still be written when the FP8 box is down.
        backend=llm.resolve("bf16"),
    )
    data = c.data or {"summary": "A conversation about banking.", "topics": []}
    card = {
        "session": session.number,
        "turns": len(session.history) // 2,
        "seconds": round(time.time() - session.started),
        "summary": redact(str(data.get("summary", "")))[:160],
        "topics": [redact(str(t))[:30] for t in data.get("topics", [])][:4],
        "compaction_seconds": round(c.seconds, 2),
    }

    # The summary goes on a public screen, so it passes the input guardrail too.
    check = llm.complete(
        [{"role": "system", "content": pipeline._prompt("input_check")},
         {"role": "user", "content": card["summary"]}],
        backend=llm.resolve("bf16"),
        schema=pipeline.INPUT_SCHEMA, max_tokens=90,
    )
    if (check.data or {}).get("label") in ("offensive", "injection"):
        card["summary"] = "A conversation about banking."
        card["topics"] = []
    return card


@app.post("/api/reset")
def reset() -> dict[str, Any]:
    global SESSION
    card = compact_session(SESSION)
    if card:
        MEMORY.insert(0, card)
    SESSION = pipeline.Session(number=SESSION.number + 1)
    return {"ok": True, "card": card, "memory": MEMORY[:12]}


@app.post("/api/memory/save")
def memory_save() -> dict[str, Any]:
    """Commit the pending profile. Only saved memory is ever read back into a prompt,
    so this is where the guardrail sits — see app/memory.py."""
    return memory.save()


@app.post("/api/memory/forget")
def memory_forget() -> dict[str, Any]:
    return memory.forget()


@app.post("/api/mode")
def set_mode(body: Mode) -> dict[str, Any]:
    global MODE
    MODE = "careful" if body.mode == "careful" else "eager"
    return {"ok": True, "mode": MODE}


@app.get("/api/state")
def state() -> dict[str, Any]:
    counters = pipeline.COUNTERS
    top_topic = ""
    if MEMORY:
        counts: dict[str, int] = {}
        for card in MEMORY:
            for topic in card["topics"]:
                counts[topic] = counts.get(topic, 0) + 1
        if counts:
            top_topic = max(counts, key=counts.get)
    return {
        "bank_name": pipeline.CFG["bank_name"],
        "customer_name": pipeline.CFG["customer_name"],
        "cards": pipeline.CFG["cards"],
        "think_hard_card": pipeline.CFG["think_hard_card"],
        "mode": MODE,
        "session": SESSION.number,
        "idle_reset_seconds": pipeline.CFG["limits"]["idle_reset_seconds"],
        "counters": {
            "conversations": len(MEMORY),
            "turns": counters.turns,
            "advice_blocked_eager": counters.advice_blocked_eager,
            "advice_blocked_careful": counters.advice_blocked_careful,
            "fact_failures": counters.fact_failures,
            "input_blocks": counters.input_blocks,
            "tokens_per_second": round(counters.tokens_per_second, 1),
        },
        "precision": {
            "available": llm.availability(),
            # Per-backend decode throughput, accumulated all day: this is the
            # bf16-vs-FP8 comparison the footer shows. Never reset by New visitor.
            "stats": {
                name: {
                    "tok_s": round(c.gen_tokens_per_second, 1),
                    "tokens": c.tokens,
                    "calls": c.calls,
                    "turns": c.turns,
                    "advice_blocked": c.advice_blocked_eager + c.advice_blocked_careful,
                }
                for name, c in pipeline.PERF.items()
            },
        },
        "memory": MEMORY[:12],
        "customer_memory": {**memory.state(), "labels": pipeline.CFG["customer_memory"]},
        "top_topic": top_topic,
        "rates_as_at": tools.RATES_AS_AT,
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    llm_health = llm.health()
    db = tools.run_sql("SELECT COUNT(*) FROM transactions")
    return {
        "vllm": llm_health,
        "transactions": db["rows"][0][0] if db.get("ok") else None,
        "products": len(tools.PRODUCTS),
        "ok": llm_health["ok"] and db.get("ok", False),
    }
