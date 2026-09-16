"""Thin wrapper over the local vLLM server (OpenAI-compatible).

Every call returns the text plus the timing/token facts the trace panel needs,
because in this demo the timings are part of the product, not debug output.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from openai import OpenAI

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
BASE_MODEL = os.environ.get("VLLM_MODEL", "google/gemma-4-E4B-it")
JUDGE_MODEL = os.environ.get("VLLM_JUDGE_MODEL", BASE_MODEL)
ADVICE_MODEL = os.environ.get("VLLM_ADVICE_MODEL", BASE_MODEL)

# The second server: the same weights and the same adapters, quantized to FP8.
# Only the quantization flags differ, so a bf16/fp8 A/B isolates one variable.
FP8_URL = os.environ.get("VLLM_FP8_BASE_URL", "http://localhost:8001/v1")
FP8_MODEL = os.environ.get("VLLM_FP8_MODEL", BASE_MODEL)
FP8_JUDGE_MODEL = os.environ.get(
    "VLLM_FP8_JUDGE_MODEL", FP8_MODEL if JUDGE_MODEL == BASE_MODEL else JUDGE_MODEL
)
FP8_ADVICE_MODEL = os.environ.get(
    "VLLM_FP8_ADVICE_MODEL", FP8_MODEL if ADVICE_MODEL == BASE_MODEL else ADVICE_MODEL
)

DEFAULT_PRECISION = os.environ.get("VLLM_PRECISION", "bf16")
_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")


@dataclass(frozen=True)
class Backend:
    """One vLLM endpoint, with the model names and trace labels that go with it."""

    precision: str          # "bf16" | "fp8"
    base_url: str
    model: str              # the base model id this endpoint serves
    judge_model: str
    advice_model: str
    label: str              # "Gemma 4B bf16" | "Gemma 4B FP8"
    client: OpenAI
    labels: dict[str, str]  # model id -> what the trace Model column shows

    def model_for(self, role: str) -> str:
        return {"base": self.model, "judge": self.judge_model, "advice": self.advice_model}[role]


def _backend(precision: str, label: str, url: str, model: str, judge: str, advice: str) -> Backend:
    # Both endpoints serve the same model id, so the id alone can no longer say
    # which one answered — the labels have to hang off the backend, not a module dict.
    labels = {model: label}
    if judge != model:
        labels[judge] = f"{label} + judge-LoRA"
    if advice != model:
        labels[advice] = f"{label} + advice-LoRA"
    return Backend(
        precision=precision,
        base_url=url,
        model=model,
        judge_model=judge,
        advice_model=advice,
        label=label,
        client=OpenAI(base_url=url, api_key=_API_KEY, timeout=60.0),
        labels=labels,
    )


BACKENDS: dict[str, Backend] = {
    "bf16": _backend("bf16", "Gemma 4B bf16", BASE_URL, BASE_MODEL, JUDGE_MODEL, ADVICE_MODEL),
    "fp8": _backend("fp8", "Gemma 4B FP8", FP8_URL, FP8_MODEL, FP8_JUDGE_MODEL, FP8_ADVICE_MODEL),
}


def resolve(precision: str | None = None) -> Backend:
    """The backend for a precision name. Anything unknown falls back to bf16, so a
    stale browser tab or a typo in an env var can never take the booth down."""
    return BACKENDS.get(precision or DEFAULT_PRECISION) or BACKENDS["bf16"]


def label_for(model: str, backend: Backend | None = None) -> str:
    backend = backend or resolve()
    return backend.labels.get(model, model)


_probes: dict[str, tuple[float, dict[str, Any]]] = {}
_probe_lock = threading.Lock()


def _probe(backend: Backend, max_age: float) -> dict[str, Any]:
    with _probe_lock:
        cached = _probes.get(backend.precision)
        if cached and time.monotonic() - cached[0] < max_age:
            return cached[1]
    try:
        models = [m.id for m in backend.client.with_options(timeout=3.0).models.list().data]
        result = {"ok": True, "models": models}
    except Exception as exc:
        result = {"ok": False, "models": [], "error": str(exc)}
    with _probe_lock:
        _probes[backend.precision] = (time.monotonic(), result)
    return result


def availability(max_age: float = 15.0) -> dict[str, dict[str, Any]]:
    """Which endpoints are up, cached so the 10 s /api/state poll doesn't hammer both.

    A short cache also means a server started *after* the app still becomes
    selectable within one poll — you can bring the FP8 box up mid-event.
    """
    return {name: _probe(backend, max_age) for name, backend in BACKENDS.items()}


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
    model: str | None = None,
    backend: Backend | None = None,
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
    plain json_schema decoding, which is what the advice adapter was trained on.

    `model` defaults to whatever the resolved backend serves — it cannot default to
    a module constant, because the two endpoints have their own model names."""
    backend = backend or resolve()
    model = model or backend.model
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
        resp = backend.client.chat.completions.create(**kwargs)
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
    model: str | None = None,
    backend: Backend | None = None,
    max_tokens: int = 300,
    temperature: float = 0.0,
    thinking: bool = False,
    stats: dict[str, Any] | None = None,
) -> Iterator[str]:
    """Token-by-token, for the draft that streams into the trace panel.

    `stats`, when given, is filled with `completion_tokens` and `seconds` so the
    streamed draft counts towards the per-precision throughput comparison."""
    backend = backend or resolve()
    started = time.monotonic()
    try:
        response = backend.client.chat.completions.create(
            model=model or backend.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
        )
        for chunk in response:
            if not chunk.choices:
                _take_usage(chunk, stats)  # the final usage-only chunk arrives here
                continue
            piece = chunk.choices[0].delta.content
            if piece:
                yield piece
    except Exception as exc:
        yield f"\n[model error: {exc}]"
    finally:
        if stats is not None:
            stats["seconds"] = time.monotonic() - started


# Gemma fences its scratchpad with special tokens: `<|channel>thought` … `<channel|>`,
# which the server strips unless `skip_special_tokens` is off — so thinking mode asks
# for them. The `<think>` forms are here for any build that uses those instead.
_THINK_START_RE = re.compile(r"^\s*(?:<\|channel>\s*thought|<think(?:ing)?>)\s*", re.I)
_THINK_END_RE = re.compile(r"<channel\|>|</think(?:ing)?>\s*", re.I)
_MARKER_TAIL = 12   # longest end marker, held back so a split chunk still matches
_START_HOLD = 40    # enough for the opening marker to arrive before anything is shown


def _take_usage(chunk: Any, stats: dict[str, Any] | None) -> None:
    """vLLM sends one final chunk with no choices and a populated `usage`."""
    if stats is not None and getattr(chunk, "usage", None):
        stats["completion_tokens"] = chunk.usage.completion_tokens


def stream_parts(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    backend: Backend | None = None,
    max_tokens: int = 1200,
    temperature: float = 0.0,
    stats: dict[str, Any] | None = None,
) -> Iterator[tuple[str, str]]:
    """Thinking mode on, streamed as ("think", text) parts then ("answer", text).

    vLLM puts the scratchpad in `reasoning_content` when it runs a reasoning
    parser, and inline in the content, fenced by the markers above, when it does
    not. Both end up split here, so the panel can show the thinking and the chat
    never sees it.

    If neither ever appears, every part comes back as "think" and the caller is
    left to fall back — better than swallowing the answer.
    """
    backend = backend or resolve()
    started = time.monotonic()
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
        response = backend.client.chat.completions.create(
            model=model or backend.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={
                "chat_template_kwargs": {"enable_thinking": True},
                "skip_special_tokens": False,
            },
        )
        for chunk in response:
            if not chunk.choices:
                _take_usage(chunk, stats)
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
    finally:
        if stats is not None:
            stats["seconds"] = time.monotonic() - started


def health(backend: Backend | None = None) -> dict[str, Any]:
    """Used by the preflight script and the on-screen status strip.

    The top-level keys describe the one backend asked about, so callers that
    predate the FP8 server keep working — `backends` is additive, and a missing
    optional FP8 box never turns /api/health red.
    """
    backend = backend or resolve()
    started = time.monotonic()
    try:
        models = [m.id for m in backend.client.models.list().data]
        out: dict[str, Any] = {"ok": True, "models": models}
    except Exception as exc:
        out = {"ok": False, "models": [], "error": str(exc)}
    out["seconds"] = round(time.monotonic() - started, 3)

    backends = {}
    for name, other in BACKENDS.items():
        probe = dict(out) if other is backend else _probe(other, max_age=0.0)
        backends[name] = dict(probe, base_url=other.base_url, label=other.label)
    return dict(out, current=backend.precision, backends=backends)
