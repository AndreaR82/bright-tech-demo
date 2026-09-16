"""Thin wrapper over the local vLLM server (OpenAI-compatible).

Every call returns the text plus the timing/token facts the trace panel needs,
because in this demo the timings are part of the product, not debug output.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from openai import OpenAI

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
BASE_MODEL = os.environ.get("VLLM_MODEL", "google/gemma-4-E4B-it")
JUDGE_MODEL = os.environ.get("VLLM_JUDGE_MODEL", BASE_MODEL)
ADVICE_MODEL = os.environ.get("VLLM_ADVICE_MODEL", BASE_MODEL)

# What the Model column shows for each served model.
MODEL_LABELS = {
    BASE_MODEL: "Gemma 4B",
    JUDGE_MODEL: "Gemma 4B + judge-LoRA" if JUDGE_MODEL != BASE_MODEL else "Gemma 4B",
    ADVICE_MODEL: "Gemma 4B + advice-LoRA" if ADVICE_MODEL != BASE_MODEL else "Gemma 4B",
}

_client = OpenAI(base_url=BASE_URL, api_key=os.environ.get("VLLM_API_KEY", "EMPTY"), timeout=60.0)


def label_for(model: str) -> str:
    return MODEL_LABELS.get(model, model)


@dataclass
class Completion:
    text: str
    model: str
    seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    data: Any = None  # parsed JSON, when a schema was used
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        return self.completion_tokens / self.seconds if self.seconds > 0 else 0.0


_JSON_CHAR = r'(?:[^"\\\x00-\x1f]|\\["\\/nt])'
# Longer bounds are slow to compile on the server ({0,600} took over a minute), so
# only short strings get one; long ones are bounded by max_tokens instead.
_MAX_BOUNDED_STRING = 160


def compact_json_regex(schema: dict[str, Any]) -> str:
    """A whitespace-free regex for a flat JSON schema, every property in order.

    With plain json_schema decoding, Gemma sometimes emits newlines where the grammar
    wants a comma and pads until max_tokens (an 18 s call that returns cut-off JSON).
    This vLLM build ignores `disable_any_whitespace`, so the grammar itself has to
    leave no room for whitespace — and, because the model can no longer end a value
    with whitespace, every value needs an end of its own: without one it kept going
    ("loan_term_years": 2540000, or a note that never closed). Short strings get
    their maxLength, numbers get as many digits as their `maximum` allows.
    """
    kind = schema.get("type")
    if kind == "object":
        fields = ",".join(f'"{key}":{compact_json_regex(sub)}' for key, sub in schema["properties"].items())
        return r"\{" + fields + r"\}"
    if "enum" in schema:
        return '"(?:' + "|".join(re.escape(value) for value in schema["enum"]) + ')"'
    if kind == "string":
        limit = schema.get("maxLength")
        if limit and limit <= _MAX_BOUNDED_STRING:
            return f'"{_JSON_CHAR}{{0,{limit}}}"'
        return f'"{_JSON_CHAR}*"'
    if kind in ("integer", "number"):
        digits = len(str(int(schema.get("maximum", 10**8))))
        whole = f"(?:0|[1-9][0-9]{{0,{digits - 1}}})"
        return whole if kind == "integer" else whole + r"(?:\.[0-9]{1,2})?"
    raise ValueError(f"compact_json_regex does not support {schema}")


def complete(
    messages: list[dict[str, str]],
    *,
    model: str = BASE_MODEL,
    max_tokens: int = 300,
    temperature: float = 0.0,
    schema: dict[str, Any] | None = None,
    compact: bool = False,
    thinking: bool = False,
    stop: list[str] | None = None,
) -> Completion:
    """One non-streaming call. With `schema`, the reply is constrained JSON; with
    `compact` too, it is decoded through `compact_json_regex` — every property
    emitted, in schema order, with no whitespace. The input and advice checks keep
    plain json_schema decoding, which is what the advice adapter was trained on."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stop:
        kwargs["stop"] = stop
    extra_body: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": thinking}}
    if schema is not None and compact:
        extra_body["structured_outputs"] = {"regex": compact_json_regex(schema)}
    elif schema is not None:
        # vLLM structured outputs. NOTE: this build silently ignores the older
        # `guided_json` extra-body field — it must go through response_format.
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "reply", "schema": schema},
        }
    kwargs["extra_body"] = extra_body

    started = time.monotonic()
    try:
        resp = _client.chat.completions.create(**kwargs)
    except Exception as exc:  # a booth demo never dies on one bad call
        return Completion(text="", model=model, seconds=time.monotonic() - started, error=str(exc))
    seconds = time.monotonic() - started

    text = (resp.choices[0].message.content or "").strip()
    usage = resp.usage
    out = Completion(
        text=text,
        model=model,
        seconds=seconds,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )
    if schema is not None:
        try:
            out.data = json.loads(text)
        except json.JSONDecodeError as exc:
            out.error = f"bad JSON: {exc}"
    return out


def stream(
    messages: list[dict[str, str]],
    *,
    model: str = BASE_MODEL,
    max_tokens: int = 300,
    temperature: float = 0.0,
    thinking: bool = False,
) -> Iterator[str]:
    """Token-by-token, for the draft that streams into the trace panel."""
    try:
        response = _client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
        )
        for chunk in response:
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content
            if piece:
                yield piece
    except Exception as exc:
        yield f"\n[model error: {exc}]"


# Gemma fences its scratchpad with special tokens: `<|channel>thought` … `<channel|>`,
# which the server strips unless `skip_special_tokens` is off — so thinking mode asks
# for them. The `<think>` forms are here for any build that uses those instead.
_THINK_START_RE = re.compile(r"^\s*(?:<\|channel>\s*thought|<think(?:ing)?>)\s*", re.I)
_THINK_END_RE = re.compile(r"<channel\|>|</think(?:ing)?>\s*", re.I)
_MARKER_TAIL = 12   # longest end marker, held back so a split chunk still matches
_START_HOLD = 40    # enough for the opening marker to arrive before anything is shown


def stream_parts(
    messages: list[dict[str, str]],
    *,
    model: str = BASE_MODEL,
    max_tokens: int = 1200,
    temperature: float = 0.0,
) -> Iterator[tuple[str, str]]:
    """Thinking mode on, streamed as ("think", text) parts then ("answer", text).

    vLLM puts the scratchpad in `reasoning_content` when it runs a reasoning
    parser, and inline in the content, fenced by the markers above, when it does
    not. Both end up split here, so the panel can show the thinking and the chat
    never sees it.

    If neither ever appears, every part comes back as "think" and the caller is
    left to fall back — better than swallowing the answer.
    """
    thinking_done = False
    held = ""
    first_think = True

    def think(text: str) -> tuple[str, str]:
        nonlocal first_think
        if first_think:
            text = _THINK_START_RE.sub("", text, count=1)
            first_think = False
        return "think", text

    try:
        response = _client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": True},
                "skip_special_tokens": False,
            },
        )
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if reasoning := getattr(delta, "reasoning_content", None):
                # A server-side parser is already splitting them: content is answer.
                thinking_done = True
                yield think(reasoning)
            piece = delta.content
            if not piece:
                continue
            if thinking_done:
                yield "answer", piece
                continue
            held += piece
            if m := _THINK_END_RE.search(held):
                if head := held[: m.start()]:
                    yield think(head)
                thinking_done = True
                if rest := held[m.end() :]:
                    yield "answer", rest
                held = ""
            elif len(held) > (_START_HOLD if first_think else _MARKER_TAIL):
                yield think(held[:-_MARKER_TAIL])
                held = held[-_MARKER_TAIL:]
        if held:
            yield ("answer", held) if thinking_done else think(held)
    except Exception as exc:
        yield "answer", f"\n[model error: {exc}]"


def health() -> dict[str, Any]:
    """Used by the preflight script and the on-screen status strip."""
    started = time.monotonic()
    try:
        models = [m.id for m in _client.models.list().data]
        return {"ok": True, "models": models, "seconds": round(time.monotonic() - started, 3)}
    except Exception as exc:
        return {"ok": False, "models": [], "error": str(exc), "seconds": round(time.monotonic() - started, 3)}
