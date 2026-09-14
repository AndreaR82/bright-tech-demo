"""Thin wrapper over the local vLLM server (OpenAI-compatible).

Every call returns the text plus the timing/token facts the trace panel needs,
because in this demo the timings are part of the product, not debug output.
"""

from __future__ import annotations

import json
import os
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


def complete(
    messages: list[dict[str, str]],
    *,
    model: str = BASE_MODEL,
    max_tokens: int = 300,
    temperature: float = 0.0,
    schema: dict[str, Any] | None = None,
    thinking: bool = False,
    stop: list[str] | None = None,
) -> Completion:
    """One non-streaming call. With `schema`, the reply is constrained JSON."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stop:
        kwargs["stop"] = stop
    extra_body: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": thinking}}
    if schema is not None:
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


def health() -> dict[str, Any]:
    """Used by the preflight script and the on-screen status strip."""
    started = time.monotonic()
    try:
        models = [m.id for m in _client.models.list().data]
        return {"ok": True, "models": models, "seconds": round(time.monotonic() - started, 3)}
    except Exception as exc:
        return {"ok": False, "models": [], "error": str(exc), "seconds": round(time.monotonic() - started, 3)}
