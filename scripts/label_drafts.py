"""Label the generated drafts with Claude, using Andrea's definitions.

Claude reads the same definitions the runtime detector uses (from app/config.yaml)
and labels each draft. The Batch API halves the cost and the whole job is one
submission.

    export ANTHROPIC_API_KEY=sk-ant-...
    uv run --with anthropic python scripts/label_drafts.py            # submit + wait
    uv run --with anthropic python scripts/label_drafts.py --estimate # cost only, no API call

Output: data/labelled.jsonl — the draft rows plus {label, sentence, reason}.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

DRAFTS = ROOT / "data" / "drafts.jsonl"
OUT = ROOT / "data" / "labelled.jsonl"
MODEL = "claude-opus-5"

LABELS = {"FACTUAL": "factual_information",
          "PERSONAL": "personalised_recommendation"}

INSTRUCTION = """\
You are labelling drafts written by a bank's AI assistant, to build a training set
for a small model that will do this job locally.

{definitions}

Label the draft below. Reply with ONLY a JSON object, no other text:
{{"label": "FACTUAL" | "PERSONAL",
  "sentence": "the sentence that decided it, copied exactly, or empty",
  "reason": "under 15 words"}}
"""


def definitions() -> tuple[str, bool]:
    """The advice definitions, and whether they are still the placeholders."""
    cfg = yaml.safe_load((ROOT / "app" / "config.yaml").read_text())
    return cfg["prompts"]["advice_check"], bool(cfg.get("advice_definitions_are_placeholders", False))


def load_drafts() -> list[dict]:
    if not DRAFTS.exists():
        sys.exit(f"no drafts at {DRAFTS} — run scripts/gen_drafts.py first")
    return [json.loads(line) for line in DRAFTS.read_text().splitlines() if line.strip()]


def parse_reply(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    label = LABELS.get(str(data.get("label", "")).strip().upper())
    if not label:
        return None
    return {"label": label,
            "sentence": str(data.get("sentence", ""))[:300],
            "reason": str(data.get("reason", ""))[:200]}


def estimate(rows: list[dict], definitions_text: str) -> None:
    # ~4 chars per token is close enough for a cost sanity check.
    prompt_chars = sum(len(definitions_text) + len(r["draft"]) + 400 for r in rows)
    in_tok, out_tok = prompt_chars / 4, len(rows) * 60
    # Opus 5: $5/MTok in, $25/MTok out; Batch API halves both.
    cost = (in_tok / 1e6 * 5 + out_tok / 1e6 * 25) / 2
    print(f"{len(rows)} drafts · ~{in_tok/1000:.0f}k input tokens · ~{out_tok/1000:.0f}k output tokens")
    print(f"estimated batch cost on {MODEL}: ${cost:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", action="store_true", help="print cost and exit")
    ap.add_argument("--limit", type=int, default=0, help="label only the first N drafts")
    args = ap.parse_args()

    rows = load_drafts()

    # Identical drafts recur (the same question reaches both writer modes). Labelling
    # each one twice would pay twice for the same answer, and build_train_data drops
    # the duplicates anyway.
    seen: set[str] = set()
    deduped = []
    for row in rows:
        key = row["draft"].strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    if len(deduped) < len(rows):
        print(f"{len(rows)} drafts → {len(deduped)} unique (skipping {len(rows) - len(deduped)} duplicates)")
    rows = deduped

    if args.limit:
        rows = rows[: args.limit]
    defs, placeholders = definitions()
    estimate(rows, defs)
    if args.estimate:
        return 0
    if placeholders:
        print("\n!! app/config.yaml still has the PLACEHOLDER advice definitions.")
        print("   Labelling now would teach the adapter my guesses, not your compliance rules.")
        print("   Replace prompts.advice_check, then set advice_definitions_are_placeholders: false")
        return 1

    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic()
    system = [{"type": "text",
               "text": INSTRUCTION.format(definitions=defs),
               "cache_control": {"type": "ephemeral"}}]

    requests = [
        Request(
            custom_id=f"draft-{i}",
            params=MessageCreateParamsNonStreaming(
                model=MODEL,
                max_tokens=300,
                system=system,
                messages=[{"role": "user", "content": f"Question: {r['question']}\n\nDraft reply:\n{r['draft']}"}],
            ),
        )
        for i, r in enumerate(rows)
    ]

    batch = client.messages.batches.create(requests=requests)
    print(f"batch {batch.id} submitted ({len(requests)} drafts) — usually under an hour")

    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        counts = batch.request_counts
        print(f"  {batch.processing_status}: {counts.succeeded} done, {counts.processing} processing")
        time.sleep(30)

    labelled, failed = 0, 0
    by_id: dict[str, dict] = {}
    for result in client.messages.batches.results(batch.id):
        if result.result.type != "succeeded":
            failed += 1
            continue
        text = next((b.text for b in result.result.message.content if b.type == "text"), "")
        parsed = parse_reply(text)
        if parsed:
            by_id[result.custom_id] = parsed
        else:
            failed += 1

    with OUT.open("w") as fh:
        for i, row in enumerate(rows):
            parsed = by_id.get(f"draft-{i}")
            if not parsed:
                continue
            fh.write(json.dumps({**row, **parsed}) + "\n")
            labelled += 1

    print(f"\n{labelled} labelled → {OUT}  ({failed} failed)")
    counts: dict[str, int] = {}
    for row in by_id.values():
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    for label, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<30} {n:>5}  ({n/max(labelled,1)*100:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
