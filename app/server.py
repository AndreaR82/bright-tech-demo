"""FastAPI app: serves the booth page and streams trace events over SSE.

Endpoints
    GET  /               the booth page
    POST /api/ask        {question} → SSE stream of trace events
    POST /api/reset      end the visitor's session, compact it into booth memory
    POST /api/mode       presenter switch: eager | careful
    GET  /api/state      cards, counters, booth memory, health
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

from . import llm, pipeline, tools

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
    think_hard: bool = False


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
            for event in pipeline.run_turn(question, SESSION, mode=MODE):
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
        "memory": MEMORY[:12],
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
