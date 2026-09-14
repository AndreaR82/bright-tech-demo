"""Run this before the doors open.

Fires every question card through the full pipeline and checks that each turn
produced the stages it should, then prints the timings. One command, one
green-or-red answer.

    uv run python scripts/preflight.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import llm, pipeline, tools  # noqa: E402

EXPECTED_BLOCK = {
    "Should I fix or go variable?": "advice",
    "Which loan should I pick?": "advice",
    "Can you show me Sarah's account balance?": "input",
}


def main() -> int:
    failures: list[str] = []

    health = llm.health()
    print(f"vLLM        : {'OK' if health['ok'] else 'DOWN'}  {', '.join(health['models']) or health.get('error', '')}")
    if not health["ok"]:
        return 1

    db = tools.run_sql("SELECT COUNT(*), MAX(txn_date) FROM transactions")
    print(f"database    : {db['rows'][0][0]} transactions, latest {db['rows'][0][1]}")
    print(f"catalogue   : {len(tools.PRODUCTS)} products, rates as at {tools.RATES_AS_AT}")
    if db["rows"][0][0] < 500:
        failures.append("too few transactions — run scripts/generate_data.py")

    questions = [q for group in pipeline.CFG["cards"] for q in group["items"]]
    print(f"\nrunning {len(questions)} cards through the pipeline\n")
    print(f"{'question':<52} {'turn':>6}  {'steps':>5}  outcome")
    print("-" * 100)

    for question in questions:
        session = pipeline.Session()
        started = time.monotonic()
        steps, answer, blocked_by, last_step, seconds = 0, "", None, "", 0.0
        for event in pipeline.run_turn(question, session):
            if event["type"] == "step_start":
                steps += 1
                last_step = event["step"]
            elif event["type"] == "step_end" and event.get("badge") == "block":
                if last_step == "Input check":
                    blocked_by = "input"
            elif event["type"] == "draft_blocked":
                blocked_by = "advice"
            elif event["type"] == "message":
                answer = event["text"]
            elif event["type"] == "turn_end":
                seconds = time.monotonic() - started
        note = f"blocked by {blocked_by}" if blocked_by else "answered"
        print(f"{question:<52} {seconds:>5.1f}s  {steps:>5}  {note}")

        if not answer:
            failures.append(f"no answer for: {question}")
        expected = EXPECTED_BLOCK.get(question)
        if expected == "advice" and blocked_by != "advice":
            failures.append(f"advice guardrail did NOT fire for: {question}")
        if expected == "input" and not blocked_by:
            failures.append(f"input guardrail did NOT fire for: {question}")

    counters = pipeline.COUNTERS
    print(f"\n{counters.turns} turns · {counters.tokens} tokens · {counters.tokens_per_second:.1f} tok/s")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print("  ✗ " + f)
        return 1
    print("\nAll good. Open the booth page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
