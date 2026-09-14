"""Merge agent-produced labels into data/labelled.jsonl.

The Batch API path (scripts/label_drafts.py) needs API credit. This is the
subscription-side alternative: Claude Code subagents label chunks of drafts
against the same prompts.advice_check definitions, writing one
data/_agent_labels/labels_NNN.jsonl per chunk, and this script stitches them
back into the exact schema build_train_data.py expects.

Chunk ids index the DEDUPLICATED draft list, deduplicated the same way
label_drafts.py does it, so the two paths produce interchangeable output.

    uv run python scripts/merge_agent_labels.py
    uv run python scripts/merge_agent_labels.py --strict   # exit 1 on any problem
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "data" / "drafts.jsonl"
LABEL_DIR = ROOT / "data" / "_agent_labels"
OUT = ROOT / "data" / "labelled.jsonl"

LABELS = {"factual_information", "personalised_recommendation"}

# The detector was three-class until a labelling audit showed the middle class
# split on verb choice rather than on anything a 4B model could learn. General
# information was never blocked, so it collapses into the non-blocking class and
# labels from the earlier passes stay usable.
LEGACY = {"general_information": "factual_information"}


def deduped_drafts() -> list[dict]:
    """Same order and dedup rule as label_drafts.py, so ids line up."""
    rows = [json.loads(line) for line in DRAFTS.read_text().splitlines() if line.strip()]
    seen: set[str] = set()
    out = []
    for row in rows:
        key = row["draft"].strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="exit non-zero if anything is off")
    args = ap.parse_args()

    if not DRAFTS.exists():
        sys.exit(f"no drafts at {DRAFTS}")
    drafts = deduped_drafts()

    files = sorted(LABEL_DIR.glob("labels_*.jsonl"))
    if not files:
        sys.exit(f"no label files in {LABEL_DIR} — run the labelling agents first")

    merged: dict[int, dict] = {}
    problems: list[str] = []
    for path in files:
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append(f"{path.name}:{n} bad JSON ({e})")
                continue
            idx = rec.get("id")
            if not isinstance(idx, int) or not 0 <= idx < len(drafts):
                problems.append(f"{path.name}:{n} id out of range: {idx!r}")
                continue
            label = LEGACY.get(rec.get("label"), rec.get("label"))
            if label not in LABELS:
                problems.append(f"{path.name}:{n} bad label: {rec.get('label')!r}")
                continue
            if idx in merged:
                problems.append(f"{path.name}:{n} duplicate id {idx}")
                continue
            # The deciding sentence is meant to be quoted verbatim; a paraphrase
            # means the labeller drifted, so drop it rather than train on it.
            sentence = str(rec.get("sentence", ""))
            if sentence and sentence not in drafts[idx]["draft"]:
                problems.append(f"{path.name}:{n} id {idx} sentence not verbatim — blanked")
                sentence = ""
            merged[idx] = {
                "label": label,
                "sentence": sentence[:300],
                "reason": str(rec.get("reason", ""))[:200],
            }

    with OUT.open("w") as fh:
        for idx in sorted(merged):
            fh.write(json.dumps({**drafts[idx], **merged[idx]}, ensure_ascii=False) + "\n")

    counts = Counter(r["label"] for r in merged.values())
    total = len(merged)
    print(f"{total} labelled of {len(drafts)} unique drafts -> {OUT}")
    for label, n in counts.most_common():
        print(f"  {label:<30} {n:>5}  ({n / max(total, 1) * 100:.0f}%)")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems[:20]:
            print(f"  {p}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
    if total < 300:
        print("\nnote: under ~300 examples a LoRA may not beat the base model.")
    return 1 if (args.strict and problems) else 0


if __name__ == "__main__":
    raise SystemExit(main())
