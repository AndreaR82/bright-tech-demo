# Bright Tech Demo — booth spec

An interactive booth demo for an internal bank AI event: a mock retail-banking
assistant for a synthetic customer (**John Citizen**), with the entire agent
pipeline exposed live on screen next to the chat.

**Take-home messages**
1. A whole agent — router, specialists, tools, guardrails, memory — runs on one
   small box with one small model (Gemma 4 E4B, ~20 tok/s, no internet).
2. Here is what is actually behind an "AI chatbot": checks, routing, SQL,
   retrieval, drafts, rewrites, verification.

Audience is mixed: engineers *and* branch staff. Everything on screen must be
readable by both.

---

## Booth setup

- DGX Spark **on the booth table**, visible, **no internet**.
- One station: computer monitor + keyboard, **one visitor at a time**.
- Presenter (you) narrates; discreet presenter controls only (no visitor toggles).

## The screen

```
┌──────────────────────────────┬───────────────────────────────────────────────┐
│  💬 John's Banking Assistant │  ⚙️ Behind the scenes                          │
│  chat (~40%)                 │  trace rows (~60%)                            │
│  question cards + input      │  Step | Model | Time | Outcome                │
├──────────────────────────────┴───────────────────────────────────────────────┤
│ 🧠 Today: N conversations · M advice drafts blocked · top topic …            │
│ #34 dining + fixed/variable · #33 borrowing power   🔌 no internet ☁️ $0.00   │
└──────────────────────────────────────────────────────────────────────────────┘
```

- Trace rows are **persistent**: `Step | Model | Time | Outcome`, coloured
  badges (green pass / red blocked / amber warning / grey non-AI).
- The `Model` column carries the LoRA story (`Gemma 4B`, `Gemma 4B + judge-LoRA`,
  `—` for non-AI steps like SQL and the calculator).
- Current turn expanded at top; older turns collapse to one line. Min 16px type.
- **No simple/engineer toggle.** One view for everyone.

## Pipeline (hybrid: fixed safety envelope, agentic specialists)

```
question
  → 🛡️ Input check        (code, always)      base Gemma, JSON-constrained
  → 🧭 Route              (code, always)      base Gemma, JSON-constrained
  → specialist            (AGENTIC, max 4 tool calls)
        📊 Spending Analyst  → SQL over John's transactions (customer-scoped)
        🏠 Product Advisor   → product search + loan calculator
  → ✍️ Draft answer       (code, always)      base Gemma, 60–80 words
  → 🛡️ Advice check ∥ ✅ Fact check           (parallel)
  → (≤1 rewrite, then drop unsupported sentences)
  → release to chat
```

**Rule: the model never does arithmetic.** Numbers come from SQL or the
calculator; the fact checker verifies claims against those tool results.

**Rule: the model decides *how* to answer, not *whether* it is checked.**

### Guardrails

| Guardrail | Model | Labels / behaviour |
|---|---|---|
| Input check | base Gemma | safe / off-topic / other-customer / offensive → block politely |
| Advice check | base Gemma + written definitions today; **advice-LoRA** once trained (`train/`) | `factual information` → pass · `personalised recommendation` → 🚫 block → rewrite. Binary since a labelling audit found the old middle class split on verb choice, not on anything learnable; the general-advice warning is now a deterministic rule — every `product_advisor` answer carries one |
| Fact check | base Gemma + **judge-LoRA** (already trained, bacc 0.82 vs 0.72 zero-shot) | sentence-level, only sentences containing numbers/rates/product names; fail → 1 rewrite → else drop the sentence |

- **Eager vs Careful writer prompt**: presenter switch. Guardrails stay on in
  both. Default **Eager** (compliance lives in the guardrail layer, not the
  prompt — stated on screen, not hidden). Counter: "advice drafts caught today:
  Eager N / Careful M".
- Nothing reaches the chat before the checks pass. The **draft streams into the
  trace panel**; blocked drafts show struck through in red.

## Data

- **John Citizen**, 32, Parramatta, IT project manager, $105k, saving for a
  first home. Everyday + savings (~$78k) + credit card (−$3.2k) + car loan.
- 18 months, ~1,500 synthetic transactions. Discoverable quirks: Sunday-night
  food delivery spikes, an unused gym membership, one concert+hotel splurge.
- **Fictional bank and fictional products** (~14: home loans, savings, cards),
  realistic market-like rates. No real bank data, no scraping, no approvals.
- Products: structured table (rates, fees, LVR, offset) for the calculator and
  the fact checker + text chunks for retrieval (**keyword search day 1**,
  EmbeddingGemma if time allows).
- SQL tool is **hard-scoped to John's customer id** in the query layer.

## Question cards

💰 spending: eating out per month · where spending rose vs last year · savings if
food delivery halved
🏠 first home: 3-year fixed rate (factual) · how much could I borrow
🛡️ guardrails: fix or variable? (blocked) · which loan should I pick? (blocked) ·
Sarah's balance (blocked at input)
🧠 Think hard: one card runs real Gemma thinking (~30s) with a live timer.

## Memory

- **Session memory**: specialists + writer see the conversation; guardrails judge
  the current turn only. Reset by "New visitor" or 90s idle.
- **Booth memory**: at reset, one compaction call → `{topics, guardrail
  outcomes, 1-line summary}` shown in the bottom strip ticker. Redacted, checked
  before display, **never fed back into the chat** (prompt-injection path).
  Deleted at the end of the event.

## Non-goals / deliberate exclusions

- No "break the bot" challenge (separate demo), no guardrails-OFF switch, no
  visitor phones, no React build step, no request logging to disk beyond booth
  memory summaries, no real bank data or branding.

## Runtime

Python + FastAPI + SSE, vanilla HTML/JS (no build step), SQLite, `uv`.
vLLM serves Gemma 4 E4B + judge LoRA; demo talks OpenAI-compatible API.
systemd `Restart=always`; preflight script; labelled "RECORDING — not live"
fallback replay if vLLM dies mid-event.

## Two-day plan

**Day 1** — vLLM verified with LoRA → skeleton + live trace → John's data →
specialists + tools → three guardrails → 8 cards → end-to-end demo.
**Day 2 am** — booth memory, counters, reset, preflight, recording fallback,
speed pass. **Day 2 pm** — you test with 2–3 colleagues, I fix what they find.
Stretch, in order: EmbeddingGemma, FP8/speculative decoding.

## Advice-detector fine-tune

Distillation: Claude labels, Gemma learns, the box serves it offline.

```
gen_drafts.py       real specialists + real writer (eager AND careful) → data/drafts.jsonl
label_drafts.py     Claude Opus 5, Batch API, Andrea's definitions     → data/labelled.jsonl
build_train_data.py stratified split, runtime-identical prompt format  → data/processed/advice_{train,val,test}.jsonl
train/run.sh        QLoRA in the NGC container (reuses gj-train image) → outputs/gemma4-e4b-advice
eval_advice.py      base vs LoRA on held-out test                      → ship it or don't
```

The gate: the adapter only goes in the demo if balanced accuracy beats base by a
clear margin **and** it misses fewer `personalised_recommendation` cases. Training
data uses the exact runtime prompt, so the adapter learns the job it will do.

**Andrea's homework**: monitor + cables + power at the booth; ~~sign-off on the advice
definitions~~ (approved 14 Sep 2026, see `app/fin-adv-defin.md`); 2–3 colleagues to test on day 2.
