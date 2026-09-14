"""Turn labelled drafts into chat-format training data.

The training prompt is byte-identical to what the detector sees at runtime
(`pipeline._advice_check`), so the adapter learns the job it will actually do:

    system   = the advice-check definitions from app/config.yaml
    user     = "Draft reply:\\n{draft}"
    assistant= {"label": ..., "sentence": ..., "reason": ...}   (the runtime JSON)

Splits are grouped by question: every draft for one question — eager, careful,
and any counterfactual rewrite of them — lands on the same side. A per-row split
would put near-identical twins in train and test and flatter the eval. Groups are
stratified by whether they hold a personalised_recommendation. The test split is
never trained on — it is what scripts/eval_advice.py scores.

Counterfactual rows (data/counterfactuals_labelled.jsonl, from
scripts/gen_counterfactuals.py) are included when present.

    uv run python scripts/build_train_data.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

LABELLED = ROOT / "data" / "labelled.jsonl"
COUNTERFACTUALS = ROOT / "data" / "counterfactuals_labelled.jsonl"
OUT_DIR = ROOT / "data" / "processed"


def advice_prompt() -> str:
    cfg = yaml.safe_load((ROOT / "app" / "config.yaml").read_text())
    return cfg["prompts"]["advice_check"]


def to_messages(row: dict, system: str) -> dict:
    target = json.dumps(
        {"label": row["label"], "sentence": row.get("sentence", ""), "reason": row.get("reason", "")},
        ensure_ascii=False,
    )
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Draft reply:\n{row['draft']}"},
            {"role": "assistant", "content": target},
        ]
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not LABELLED.exists():
        sys.exit(f"no labels at {LABELLED} — run scripts/label_drafts.py first")
    rows = [json.loads(line) for line in LABELLED.read_text().splitlines() if line.strip()]
    if COUNTERFACTUALS.exists():
        extra = [json.loads(line) for line in COUNTERFACTUALS.read_text().splitlines() if line.strip()]
        print(f"+ {len(extra)} counterfactual drafts from {COUNTERFACTUALS.name}")
        rows += extra

    # The eager and careful writers often produce the same draft for factual
    # questions. Keeping both would weight those examples double.
    seen: set[str] = set()
    deduped = []
    for row in rows:
        key = row["draft"].strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    if len(deduped) < len(rows):
        print(f"dropped {len(rows) - len(deduped)} duplicate drafts")
    rows = deduped

    system = advice_prompt()

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["question"]].append(row)
    strata: dict[bool, list[list[dict]]] = defaultdict(list)
    for items in groups.values():
        strata[any(r["label"] == "personalised_recommendation" for r in items)].append(items)

    rng = random.Random(args.seed)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for _, stratum in sorted(strata.items()):
        rng.shuffle(stratum)
        n_test = max(1, round(len(stratum) * args.test_frac))
        n_val = max(1, round(len(stratum) * args.val_frac))
        for name, part in (("test", stratum[:n_test]), ("val", stratum[n_test : n_test + n_val]),
                           ("train", stratum[n_test + n_val :])):
            splits[name] += [row for items in part for row in items]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        rng.shuffle(items)
        # train/val are chat-rendered for the trainer; test keeps the raw fields
        # so the eval script can score base vs adapter on identical inputs.
        path = OUT_DIR / f"advice_{name}.jsonl"
        with path.open("w") as fh:
            for row in items:
                payload = row if name == "test" else to_messages(row, system)
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        counts = defaultdict(int)
        for row in items:
            counts[row["label"]] += 1
        summary = " · ".join(f"{k.split('_')[0]}:{v}" for k, v in sorted(counts.items()))
        print(f"{path.name:<24} {len(items):>5} rows   {summary}")

    print(f"\ntotal {len(rows)} labelled drafts")
    if len(rows) < 300:
        print("note: under ~300 examples a LoRA may not beat the base model — generate more drafts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
