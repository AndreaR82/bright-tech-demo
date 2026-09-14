"""Counterfactual drafts that break the advice detector's free shortcuts.

On the full labelled set, three surface cues predict the label with no
understanding of advice (see app/fin-adv-defin.md, "The shortcut problem"):
an opening like "Since you are" / "Given your" / "Yes,", and any mention of a
lender or adviser (298 drafts, all factual). A model trained on that learns the
templates and the test split — drawn from the same generator — cannot tell.

Four edits, each aimed at one cue. Gemma (the real writer) does the rewriting so
the style stays in-distribution; Claude then labels every rewrite BLIND, and only
rewrites whose blind label matches the intended one are kept.

    type               from                     intended                  breaks
    neutral_open       personalised drafts      personalised_recommendation  prefix is necessary
    referral_personal  personalised drafts      personalised_recommendation  lender mention = safe
    steer_removed      personalised drafts      factual_information          prefix is sufficient
    prefix_factual     factual product drafts   factual_information          prefix is sufficient

    uv run python scripts/gen_counterfactuals.py generate --workers 6   # Gemma, resumable
    uv run python scripts/gen_counterfactuals.py chunk                  # blind chunks for labelling
    #   ... Claude Code subagents write data/_agent_labels/cf/labels_NNN.jsonl ...
    uv run python scripts/gen_counterfactuals.py merge                  # keep agreements

Every row keeps its source question, so build_train_data.py's question-grouped
split puts a rewrite and its source on the same side of the train/test line.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import re
import sys
import threading
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import llm  # noqa: E402

LABELLED = ROOT / "data" / "labelled.jsonl"
OUT = ROOT / "data" / "counterfactuals.jsonl"
OUT_LABELLED = ROOT / "data" / "counterfactuals_labelled.jsonl"
CHUNK_DIR = ROOT / "data" / "_agent_labels" / "cf"

P, F = "personalised_recommendation", "factual_information"

PREFIX_RE = re.compile(r"\s*(since you|given your|yes\b|as you|based on your|for someone|in your case)", re.I)
LENDER_RE = re.compile(r"\b(lender|banker|adviser|advisor|specialist|broker)s?\b", re.I)
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

COMMON = """\
Keep every number and product name exactly as written and add no new numbers.
Write 2 to 4 short sentences of plain prose, under 80 words, no bullet points.
Output only the rewritten reply, nothing else.

Reply:
{draft}"""

PROMPTS = {
    "neutral_open": """\
Rewrite this reply from a bank's assistant to its customer. It recommends a product
to him. Keep that recommendation: the reply must still tell him which product suits
him, would be best for him, or what he should do.

But do NOT open with anything about him or his question — no "Since you", "Given
your", "Yes", "As you", "Based on your", "For someone", "In your case", "If you".
Open with a plain fact about a product, and put the recommendation in the second or
third sentence, worded differently from the original.
""",
    "referral_personal": """\
Below is a reply from a bank's assistant to its customer. Copy it word for word,
and insert exactly ONE new sentence {where}, suggesting he also speak to {who} —
worded your own way, about this same topic.

Change nothing else. Add no other sentences, no sign-off, no new claims.
""",
    "steer_removed": """\
Rewrite this reply from a bank's assistant to its customer. It currently recommends
a product to him. Remove the recommendation: it must no longer say which product
suits him, is best or better for him, could save him money, or what he should do,
and must not say that he qualifies for anything.

Keep the opening words ("{opening}") and keep the product facts: rates, fees,
features, who each product is for. Do not mention a lender or adviser.
""",
    "prefix_factual": """\
Rewrite this reply from a bank's assistant to its customer. It states facts about
products. Start it with an opening that refers to him or his question, in this
style: "{opening}". Adjust the rest so it reads naturally.

Do not recommend anything: do not say which product suits him, is best or better
for him, or what he should do, and do not say that he qualifies for anything.
Do not mention a lender or adviser.
""",
}
INTENDED = {"neutral_open": P, "referral_personal": P, "steer_removed": F, "prefix_factual": F}
WHO = ["a lender", "one of our bankers", "a specialist", "one of our lenders"]
OPENINGS = ["Since you are asking about ...", "Given your question about ...", "Yes, ..."]


def _nums(text: str) -> set[str]:
    return {n.replace(",", "") for n in NUM_RE.findall(text)}


def valid(cf_type: str, source: str, draft: str) -> str | None:
    """Mechanical checks. Returns the reason a rewrite is rejected, or None."""
    if not draft or draft.strip().lower() == source.strip().lower():
        return "empty or unchanged"
    # Gemma pads rewrites with invented claims ("a great fit for your spending
    # habits"); the edits need at most one sentence, so cap the growth.
    if len(draft.split()) > 95 or len(draft.split()) > len(source.split()) + 25:
        return "too long"
    if not _nums(draft) <= _nums(source):
        return "invented a number"
    if cf_type == "neutral_open" and PREFIX_RE.match(draft):
        return "still opens with a cue"
    if cf_type == "referral_personal" and not LENDER_RE.search(draft):
        return "no referral"
    if cf_type in ("steer_removed", "prefix_factual") and LENDER_RE.search(draft):
        return "added a referral"
    if cf_type == "prefix_factual" and not PREFIX_RE.match(draft):
        return "no opening cue"
    if cf_type == "steer_removed" and PREFIX_RE.match(source) and not PREFIX_RE.match(draft):
        return "dropped the opening"
    return None


def jobs(seed: int, n_factual: int) -> list[dict]:
    rows = [json.loads(line) for line in LABELLED.read_text().splitlines() if line.strip()]
    rng = random.Random(seed)
    out = []
    for i, row in enumerate(r for r in rows if r["label"] == P):
        out.append({**row, "cf_type": "neutral_open"})
        # Real careful drafts put the referral last; left to itself Gemma always puts
        # it second, which would make position a cue of its own. Alternate.
        where = "as the final sentence" if i % 2 == 0 else "straight after the first sentence"
        out.append({**row, "cf_type": "referral_personal", "who": WHO[i % len(WHO)], "where": where})
        opening = PREFIX_RE.match(row["draft"])
        # Without a cue to keep, the pair would not isolate the prefix. Keep the
        # first clause if it has one, otherwise let Gemma keep whatever opens it.
        first = re.split(r"(?<=,)\s", row["draft"], maxsplit=1)[0] if opening else row["draft"].split(" ")[0]
        out.append({**row, "cf_type": "steer_removed", "opening": first})
    factual = [r for r in rows if r["label"] == F and r["route"] == "product_advisor"
               and not PREFIX_RE.match(r["draft"]) and not LENDER_RE.search(r["draft"])]
    for i, row in enumerate(rng.sample(factual, min(n_factual, len(factual)))):
        out.append({**row, "cf_type": "prefix_factual", "opening": OPENINGS[i % len(OPENINGS)]})
    return out


_lock = threading.Lock()


def rewrite(job: dict, fh, tries: int = 3) -> str:
    prompt = PROMPTS[job["cf_type"]].format(who=job.get("who", ""), opening=job.get("opening", ""), where=job.get("where", "")) + "\n" + COMMON.format(draft=job["draft"])
    reason = ""
    for attempt in range(tries):
        c = llm.complete([{"role": "user", "content": prompt}], max_tokens=220, temperature=0.7 + 0.1 * attempt)
        draft = c.text.strip().strip('"').strip()
        reason = c.error or valid(job["cf_type"], job["draft"], draft)
        if reason is None:
            row = {
                "question": job["question"], "route": job["route"], "mode": "counterfactual",
                "facts": job["facts"], "draft": draft, "cf_type": job["cf_type"],
                "intended_label": INTENDED[job["cf_type"]], "source_draft": job["draft"],
            }
            with _lock:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            return "ok"
    return reason or "failed"


def cmd_generate(args) -> int:
    done: set[tuple[str, str]] = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            done.add((r["source_draft"], r["cf_type"]))
    todo = [j for j in jobs(args.seed, args.n_factual) if (j["draft"], j["cf_type"]) not in done]
    print(f"{len(done)} already generated · {len(todo)} to go · {args.workers} workers")

    outcomes: Counter[tuple[str, str]] = Counter()
    with OUT.open("a") as fh, concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for i, (job, result) in enumerate(zip(todo, pool.map(lambda j: rewrite(j, fh), todo)), 1):
            outcomes[(job["cf_type"], result)] += 1
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}")
    for (cf_type, result), n in sorted(outcomes.items()):
        print(f"  {cf_type:<18} {result:<24} {n}")
    return 0


def cmd_chunk(args) -> int:
    rows = [json.loads(line) for line in OUT.read_text().splitlines() if line.strip()]
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    if list(CHUNK_DIR.glob("chunk_*.json")):
        sys.exit(f"{CHUNK_DIR} already has chunks — delete them to re-chunk")
    # Shuffled so no labeller sees a run of one type, and blind: no intended label.
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    for k in range(0, len(order), args.size):
        part = sorted(order[k : k + args.size])
        path = CHUNK_DIR / f"chunk_{k // args.size + 1:03d}.json"
        path.write_text(json.dumps([{"id": i, "question": rows[i]["question"], "draft": rows[i]["draft"]} for i in part],
                                   indent=1, ensure_ascii=False))
        print(f"{path.name}  {len(part)} drafts")
    return 0


def cmd_merge(args) -> int:
    rows = [json.loads(line) for line in OUT.read_text().splitlines() if line.strip()]
    labels: dict[int, dict] = {}
    problems = []
    for path in sorted(CHUNK_DIR.glob("labels_*.jsonl")):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            idx = rec.get("id")
            if not isinstance(idx, int) or not 0 <= idx < len(rows) or idx in labels or rec.get("label") not in (P, F):
                problems.append(f"{path.name}:{n} bad record {rec.get('id')!r}")
                continue
            sentence = str(rec.get("sentence", ""))
            if sentence and sentence not in rows[idx]["draft"]:
                problems.append(f"{path.name}:{n} id {idx} sentence not verbatim — blanked")
                sentence = ""
            labels[idx] = {"label": rec["label"], "sentence": sentence[:300], "reason": str(rec.get("reason", ""))[:200]}

    agree: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    kept = 0
    with OUT_LABELLED.open("w") as fh:
        for idx, row in enumerate(rows):
            if idx not in labels:
                continue
            agree[row["cf_type"]][1] += 1
            if labels[idx]["label"] != row["intended_label"]:
                continue
            agree[row["cf_type"]][0] += 1
            kept += 1
            out = {k: v for k, v in row.items() if k not in ("intended_label", "source_draft")}
            fh.write(json.dumps({**out, **labels[idx]}, ensure_ascii=False) + "\n")

    print(f"{len(labels)} labelled of {len(rows)} generated · {kept} kept -> {OUT_LABELLED}")
    for cf_type, (hit, total) in sorted(agree.items()):
        print(f"  {cf_type:<18} {hit:>4}/{total:<4} blind label matched intent ({hit / max(total, 1):.0%})")
    for p in problems[:20]:
        print(f"  ! {p}")
    return 1 if (args.strict and (problems or len(labels) < len(rows))) else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--workers", type=int, default=6)
    g.add_argument("--n-factual", type=int, default=100, help="factual drafts to give an opening cue")
    g.add_argument("--seed", type=int, default=11)
    c = sub.add_parser("chunk")
    c.add_argument("--size", type=int, default=90)
    c.add_argument("--seed", type=int, default=11)
    m = sub.add_parser("merge")
    m.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    return {"generate": cmd_generate, "chunk": cmd_chunk, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
