"""Generate in-distribution drafts for the advice-detector training set.

The detector's job is judging *this* writer's drafts in *this* pipeline, so the
training data is produced by the real thing: the real specialists fetch real
tool results, and the real writer prompts (eager AND careful) write the draft.

Guardrails and rewrites are skipped — we want the raw drafts, including the ones
the guardrail would catch.

    uv run python scripts/gen_drafts.py --n 400 --workers 6

Resumable: re-running tops the file up, skipping questions already generated.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import threading
import time
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import pipeline  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "drafts.jsonl"

# Question templates. "route" is fixed per template so we skip the router call.
# The mix is deliberate: roughly half are advice-shaped, because that is the
# class the detector must never miss.
# Four pools, sampled with weights. The two *_ADVICE pools are deliberately heavy:
# they produce the personalised_recommendation drafts, which is the class the
# detector cannot afford to miss and the rarest one in a plain question mix.
SPENDING = [
    "How much do I spend on {cat} each month?",
    "What did I spend on {cat} {period}?",
    "Has my {cat} spending gone up {period}?",
    "How much could I save if I halved my {cat}?",
    "What's my biggest spending category {period}?",
    "How much do I put into savings each month?",
    "What's my average monthly spending {period}?",
    "Where is my money going?",
    "How much did I spend at {merchant} {period}?",
    "What did {cat} cost me {period}?",
    "How many times did I buy {cat} {period}?",
    "What's my total on {cat} and {cat2} {period}?",
    "Did I spend more on {cat} or {cat2} {period}?",
    "How much is going out in subscriptions?",
    "What's my income after the bills each month?",
    "How much did I put away {period}?",
    "What's the most I spent in one month on {cat}?",
    "Break down my spending {period}",
]
SPENDING_ADVICE = [
    "Am I spending too much on {cat}?",
    "Should I cut back on {cat}?",
    "Could I afford to save another {small_amount} a month?",
    "What should I cut to save for a house?",
    "Is my spending normal for someone my age?",
    "Where should I cut first?",
    "Am I on track to buy a house next year?",
    "Do you think I can afford a {amount} place?",
    "Should I be worried about my {cat} spending?",
    "What would you do about my {cat} habit?",
    "Is {cat} costing me too much compared to most people?",
    "How should I budget from here?",
    "Am I saving enough?",
    "Should I stop my {merchant} membership?",
]
PRODUCTS = [
    "What's the rate on the {term} fixed loan?",
    "What fees does the {product} have?",
    "How much could I borrow?",
    "What's the difference between fixed and variable?",
    "What's the comparison rate on the {product}?",
    "How much deposit do I need?",
    "What would my repayments be on {amount}?",
    "What's the cheapest loan you have?",
    "What's the rate on the {product}?",
    "Do you have a loan for first home buyers?",
    "What's the maximum I can borrow with a 10% deposit?",
    "How much interest would I pay on {amount} over 30 years?",
    "What savings accounts do you offer?",
    "What's the interest rate on the first home saver account?",
    "Can I make extra repayments on the {term} fixed loan?",
    "What happens when a fixed term ends?",
    "How does an offset account work?",
    "What's the annual fee on your cards?",
]
PRODUCTS_ADVICE = [
    "Is an offset account worth it?",
    "Which home loan should I pick?",
    "Should I fix or go variable?",
    "What's the best savings account for me?",
    "Should I pay off my credit card or keep saving?",
    "Which card should I get?",
    "Is now a good time to fix?",
    "Would the offset loan save me money?",
    "Which loan is best for someone like me?",
    "Should I go with the {term} fixed?",
    "Is the {product} right for me?",
    "Would you recommend fixing for {term}?",
    "What would you do in my position?",
    "Should I use my savings as a bigger deposit?",
    "Is it smarter to wait or buy now?",
    "Which is better for me, the offset or the basic variable?",
    "Should I borrow the full {amount}?",
    "Do you think I should switch cards?",
]
CATEGORIES = ["eating out", "food delivery", "groceries", "transport", "entertainment", "shopping",
              "subscriptions", "fuel", "coffee", "takeaway", "the gym", "nights out"]
MERCHANTS = ["Burger Barn", "SwiftFit Gym", "FreshMart", "QuickBite Delivery", "Bean There Cafe",
             "Pasta Palace", "Streamly", "CityLink Transit"]
TERMS = ["1 year", "2 year", "3 year", "5 year"]
PRODUCT_NAMES = ["Offset Variable Home Loan", "Basic Variable Home Loan", "First Home Starter",
                 "Low Rate Card", "Rewards Platinum Card", "Bright Saver", "12 Month Term Deposit"]
AMOUNTS = ["$450,000", "$500,000", "$600,000", "$680,000", "$750,000", "$820,000"]
SMALL_AMOUNTS = ["$200", "$300", "$500", "$750", "$1,000"]
PERIODS = ["", "last month", "last year", "this year", "in the last 6 months", "compared to last year",
           "over the last 3 months", "since January"]
PREFIXES = ["", "", "", "Hi, ", "Quick question — ", "Hey ", "So, "]
SUFFIXES = ["", "", "", "", " thanks", " please", " — what do you reckon?", " can you help?"]

POOLS = [
    (SPENDING, "spending_analyst", 25),
    (SPENDING_ADVICE, "spending_analyst", 22),
    (PRODUCTS, "product_advisor", 27),
    (PRODUCTS_ADVICE, "product_advisor", 26),
]


def build_questions(n: int, seed: int = 7) -> list[dict[str, str]]:
    """Sample n *distinct* questions across the four pools."""
    rng = random.Random(seed)
    pools = [p for p, _, _ in POOLS]
    weights = [w for _, _, w in POOLS]
    routes = {id(p): r for p, r, _ in POOLS}

    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for _ in range(n * 60):  # bounded: the pools are finite, so stop hunting eventually
        if len(out) >= n:
            break
        pool = rng.choices(pools, weights=weights)[0]
        question = rng.choice(pool).format(
            cat=rng.choice(CATEGORIES),
            cat2=rng.choice(CATEGORIES),
            merchant=rng.choice(MERCHANTS),
            term=rng.choice(TERMS),
            product=rng.choice(PRODUCT_NAMES),
            amount=rng.choice(AMOUNTS),
            small_amount=rng.choice(SMALL_AMOUNTS),
            period=rng.choice(PERIODS),
        )
        question = (rng.choice(PREFIXES) + question.strip() + rng.choice(SUFFIXES)).replace("  ", " ").strip()
        if question in seen:
            continue
        seen.add(question)
        out.append({"question": question, "route": routes[id(pool)]})
    rng.shuffle(out)
    return out


_write_lock = threading.Lock()


def gather_facts(question: str, route: str) -> str:
    """Drive the real specialist and keep only its tool results."""
    session = pipeline.Session()
    agent = pipeline._spending_agent if route == "spending_analyst" else pipeline._product_agent
    facts = []
    for event in agent(question, session):
        if event["type"] == "facts":
            facts = event["facts"]
    return pipeline._render_facts(facts)


def write_draft(question: str, document: str, mode: str, temperature: float = 0.8) -> str:
    """Sampled, not greedy: at temperature 0 the eager and careful writers produce
    identical text for factual questions, which halves the usable dataset."""
    session = pipeline.Session()
    text = ""
    for event in pipeline._draft(question, document, session, mode, temperature=temperature):
        if event["type"] == "draft_text":
            text = event["text"]
    return text


def one(item: dict[str, str], fh) -> int:
    question, route = item["question"], item["route"]
    try:
        document = gather_facts(question, route)
        rows = []
        for mode in ("eager", "careful"):
            draft = write_draft(question, document, mode)
            if draft:
                rows.append({
                    "question": question,
                    "route": route,
                    "mode": mode,
                    "facts": document,
                    "draft": draft,
                })
        with _write_lock:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
            fh.flush()
        return len(rows)
    except Exception as exc:
        print(f"  ! {question[:50]}: {type(exc).__name__}: {exc}")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="questions (each yields 2 drafts)")
    ap.add_argument("--workers", type=int, default=6, help="vLLM serves 8 concurrent sequences")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    done: set[str] = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            try:
                done.add(json.loads(line)["question"])
            except Exception:
                pass
        print(f"resuming: {len(done)} questions already in {args.out.name}")

    questions = [q for q in build_questions(args.n) if q["question"] not in done]
    print(f"generating drafts for {len(questions)} questions with {args.workers} workers")

    started = time.monotonic()
    written = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as fh:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, count in enumerate(pool.map(lambda q: one(q, fh), questions), 1):
                written += count
                if i % 10 == 0:
                    rate = (time.monotonic() - started) / i
                    left = rate * (len(questions) - i)
                    print(f"  {i}/{len(questions)} questions · {written} drafts · {rate:.1f}s each · ~{left/60:.0f} min left")

    print(f"\n{written} drafts → {args.out} in {(time.monotonic() - started)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
