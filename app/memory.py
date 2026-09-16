"""Customer memory: what the assistant knows about John across visits.

The third tier of memory in the demo, and the only one about the customer:

    session memory   this conversation, wiped by "New visitor"      (pipeline.Session)
    booth memory     anonymous, redacted, never read back           (server.MEMORY)
    customer memory  John's own profile, saved to disk, read back   (here)

It is deliberately not redacted and not anonymous — a profile whose amounts were
stripped out would be no profile at all, and John is a synthetic customer. What
guards it instead is the save step: a turn only updates PENDING, and nothing reaches
a later prompt until someone presses Save and the guardrail passes it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from . import llm

PATH = Path(__file__).resolve().parent.parent / "data" / "customer_memory.json"

FIELDS = ("goals", "circumstances", "preferences", "interests")

# Flat, four short strings — no arrays, so the profile stays one bounded object.
# Decoded with plain json_schema, not the compact regex: with the regex Gemma filled
# every field with ": " filler instead of carrying the old values forward. Booth
# memory's compaction schema makes the same choice for the same reason.
CUSTOMER_MEMORY_SCHEMA = {
    "type": "object",
    "properties": {field: {"type": "string", "maxLength": 120} for field in FIELDS},
    "required": list(FIELDS),
    "additionalProperties": False,
}


def _empty() -> dict[str, str]:
    return {field: "" for field in FIELDS}


def _clean(data: Any) -> dict[str, str]:
    """Whatever the model returned, reduced to the four fields as short strings.

    Gemma sometimes copies the field name back into its own value ("goals": "goals:
    buying a first home"), which would make every field read as changed; strip it.
    """
    if not isinstance(data, dict):
        return _empty()
    out = {}
    for field in FIELDS:
        value = " ".join(str(data.get(field, "")).split())
        if value.lower().startswith(f"{field}:"):
            value = value[len(field) + 1 :].lstrip()
        out[field] = value.lstrip(":").strip()[:120]
    return out


def _load() -> dict[str, str]:
    try:
        return _clean(json.loads(PATH.read_text()))
    except (OSError, json.JSONDecodeError):  # a booth demo never dies on a missing file
        return _empty()


SAVED = _load()
PENDING = dict(SAVED)


def changed() -> list[str]:
    """The fields a save would actually change — what the box accents."""
    return [field for field in FIELDS if PENDING[field] != SAVED[field]]


def render(profile: dict[str, str]) -> str:
    """A profile as plain text, empty fields skipped."""
    return "\n".join(f"{field}: {profile[field]}" for field in FIELDS if profile.get(field))


def for_prompt() -> str:
    """The saved profile as plain text for a prompt, or "" when nothing is saved."""
    return render(SAVED)


def update(session: Any, prompt: str, backend: Any = None) -> tuple[dict[str, str], float, int]:
    """Merge the latest exchange into PENDING. One call, whole profile back.

    A merge rather than an append: the model is shown what is already known and
    returns the updated profile, so the call is idempotent and the profile never
    grows past its four bounded fields. Returns the profile plus the call's seconds
    and completion tokens, so the caller can put them on the trace and the counters.
    """
    global PENDING
    exchange = session.conversation(limit=1)
    if not exchange:
        return dict(PENDING), 0.0, 0
    # As JSON, not as "field: value" lines: shown the prose form, the model copied the
    # separator into its own values and every field then read as changed.
    known = json.dumps(PENDING, ensure_ascii=False)
    c = llm.complete(
        [{"role": "system", "content": prompt},
         {"role": "user", "content": f"The profile so far:\n{known}\n\nLatest exchange:\n{exchange}"}],
        schema=CUSTOMER_MEMORY_SCHEMA, max_tokens=300, backend=backend,
    )
    if c.error or not c.data:
        # Keep what we had; the trace row still shows what the attempt cost.
        return dict(PENDING), c.seconds, c.completion_tokens
    PENDING = _clean(c.data)
    return dict(PENDING), c.seconds, c.completion_tokens


def save() -> dict[str, Any]:
    """Commit PENDING to disk — the only path by which memory becomes readable.

    The profile is about to be fed back into later prompts, so it passes the input
    guardrail first, exactly as a booth-memory summary does before going on screen.
    """
    global SAVED
    from . import pipeline  # lazy: pipeline imports this module

    text = render(PENDING)
    if text:
        check = llm.complete(
            [{"role": "system", "content": pipeline._prompt("input_check")},
             {"role": "user", "content": text}],
            schema=pipeline.INPUT_SCHEMA, max_tokens=90,
            # Pinned to bf16 like the booth-memory check: a save must still work
            # when the FP8 box is down, and must not depend on which pill is lit.
            backend=llm.resolve("bf16"),
        )
        label = (check.data or {}).get("label")
        if label in ("offensive", "injection"):
            return {"ok": False, "reason": (check.data or {}).get("reason") or label,
                    "saved": dict(SAVED), "pending": dict(PENDING), "changed": changed()}

    SAVED = dict(PENDING)
    PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PATH.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({**SAVED, "saved_at": round(time.time())}, indent=2))
    tmp.replace(PATH)
    return {"ok": True, "saved": dict(SAVED), "pending": dict(PENDING), "changed": changed()}


def forget() -> dict[str, Any]:
    global SAVED, PENDING
    SAVED = _empty()
    PENDING = _empty()
    PATH.unlink(missing_ok=True)
    return {"ok": True, "saved": dict(SAVED), "pending": dict(PENDING), "changed": []}


def state() -> dict[str, Any]:
    return {"saved": dict(SAVED), "pending": dict(PENDING), "changed": changed()}
