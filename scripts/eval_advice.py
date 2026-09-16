"""Score the advice detector: base Gemma vs the LoRA, on the held-out test split.

This is the honesty gate. The adapter goes into the demo only if it clearly beats
the base model — and the metric that matters is recall on
`personalised_recommendation`, because that is the one you cannot afford to miss
at a bank event.

    # base model only (works today)
    uv run python scripts/eval_advice.py

    # both, once the adapter is served:
    #   vllm serve ... --enable-lora --lora-modules advice=outputs/gemma4-e4b-advice
    uv run python scripts/eval_advice.py --adapter advice

    # the same drafts through the FP8 endpoint, to see what quantization costs
    # the detector — the number behind the booth's bf16/FP8 toggle
    uv run python scripts/eval_advice.py --adapter advice --precision fp8
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import llm, pipeline  # noqa: E402

TEST = ROOT / "data" / "processed" / "advice_test.jsonl"
LABELS = ["factual_information", "personalised_recommendation"]


def classify(draft: str, model: str, backend: llm.Backend) -> str:
    c = llm.complete(
        [{"role": "system", "content": pipeline._prompt("advice_check")},
         {"role": "user", "content": f"Draft reply:\n{draft}"}],
        model=model, backend=backend, schema=pipeline.ADVICE_SCHEMA, max_tokens=160,
    )
    return (c.data or {}).get("label", "factual_information")


def score(rows: list[dict], model: str, name: str, backend: llm.Backend) -> dict:
    correct = 0
    per_label_total: dict[str, int] = defaultdict(int)
    per_label_hit: dict[str, int] = defaultdict(int)
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    # original drafts vs each counterfactual type: a template matcher scores well
    # on "original" and falls apart on the rewrites.
    slices: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for i, row in enumerate(rows, 1):
        gold = row["label"]
        pred = classify(row["draft"], model, backend)
        per_label_total[gold] += 1
        confusion[(gold, pred)] += 1
        if pred == gold:
            correct += 1
            per_label_hit[gold] += 1
        if i % 20 == 0:
            print(f"  {name}: {i}/{len(rows)}")

        slices[row.get("cf_type", "original")][0] += pred == gold
        slices[row.get("cf_type", "original")][1] += 1

    recalls = {l: per_label_hit[l] / per_label_total[l] for l in LABELS if per_label_total[l]}
    balanced = sum(recalls.values()) / len(recalls) if recalls else 0.0
    return {
        "name": name,
        "accuracy": correct / len(rows),
        "balanced_accuracy": balanced,
        "recalls": recalls,
        "confusion": confusion,
        "slices": slices,
    }


def report(result: dict) -> None:
    print(f"\n=== {result['name']} ===")
    print(f"accuracy           {result['accuracy']:.3f}")
    print(f"balanced accuracy  {result['balanced_accuracy']:.3f}")
    for label, recall in result["recalls"].items():
        flag = "  ← the one that matters" if label == "personalised_recommendation" else ""
        print(f"  recall {label:<30} {recall:.3f}{flag}")
    for name, (hit, total) in sorted(result["slices"].items()):
        print(f"  accuracy on {name:<24} {hit / total:.3f}  (n={total})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="served LoRA name, e.g. 'advice'")
    ap.add_argument("--precision", default="bf16", choices=sorted(llm.BACKENDS),
                    help="which vLLM endpoint to score against (default bf16)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    backend = llm.resolve(args.precision)
    probe = llm.availability(max_age=0.0)[args.precision]
    if not probe["ok"]:
        # Silently scoring bf16 while the report says fp8 would be worse than failing.
        sys.exit(f"{args.precision} endpoint at {backend.base_url} is not responding: {probe.get('error')}")
    if args.adapter and args.adapter not in probe["models"]:
        sys.exit(f"{backend.base_url} does not serve a LoRA called {args.adapter!r} "
                 f"(it serves {probe['models']})")

    if not TEST.exists():
        sys.exit(f"no test split at {TEST} — run scripts/build_train_data.py first")
    rows = [json.loads(line) for line in TEST.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"scoring {len(rows)} held-out drafts")

    print(f"endpoint: {backend.base_url} ({args.precision})")
    # The precision goes in the result name: two runs' reports are otherwise
    # indistinguishable, and this one exists to be compared against another.
    results = [score(rows, backend.model, f"base Gemma 4 E4B ({args.precision}, zero-shot)", backend)]
    if args.adapter:
        results.append(score(rows, args.adapter,
                             f"Gemma 4 E4B {args.precision} + {args.adapter}-LoRA", backend))

    for result in results:
        report(result)

    if len(results) == 2:
        base, tuned = results
        delta = tuned["balanced_accuracy"] - base["balanced_accuracy"]
        miss_base = 1 - base["recalls"].get("personalised_recommendation", 0)
        miss_tuned = 1 - tuned["recalls"].get("personalised_recommendation", 0)
        print(f"\nΔ balanced accuracy {delta:+.3f}")
        print(f"missed recommendations: base {miss_base:.1%} → LoRA {miss_tuned:.1%}")
        verdict = "ship the adapter" if delta > 0.03 and miss_tuned <= miss_base else "keep the base model"
        print(f"verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
