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
- Presenter (you) narrates; discreet presenter controls only, with one exception:
  the ⚡ FP8 pill is visitor-clickable, because the speed/quality trade-off is
  something people want to feel for themselves rather than be told about.

## The screen

```
┌──────────────────────────────┬───────────────────────────────────────────────┐
│  💬 John's Banking Assistant │  ⚙️ Behind the scenes                          │
│  chat (~40%)                 │  trace rows (~60%)                            │
│  question cards + input      │  Step | Model | Time | Outcome                │
│                              ├───────────────────────────────────────────────┤
│                              │ 🧠 What we remember about John                 │
│                              │ Goals · Circumstances · Preferences · Looking │
│                              │ [💾 Save to John's memory] [🗑️ Forget]        │
├──────────────────────────────┴───────────────────────────────────────────────┤
│ 🧠 Today: N conversations · M advice drafts blocked · top topic …            │
│ #34 dining + fixed/variable · #33 borrowing power   🔌 no internet ☁️ $0.00   │
└──────────────────────────────────────────────────────────────────────────────┘
```

- Trace rows are **persistent**: `Step | Model | Time | Outcome`, coloured
  badges (green pass / red blocked / amber warning / blue for neutral and
  non-AI steps).
- The `Model` column carries the precision and LoRA story (`Gemma 4B bf16`,
  `Gemma 4B FP8 + advice-LoRA`, `—` for non-AI steps like SQL and the
  calculator), so a turn's trace says which of the two endpoints answered it.
- Current turn expanded at top; older turns collapse to one line. Chat bubbles
  and the input are 16px; the trace table runs 12.5–14px.
- **No simple/engineer toggle.** One view for everyone.

## Pipeline (hybrid: fixed safety envelope, agentic specialists)

```
question
  → 🛡️ Input check        (code, always)      base Gemma, JSON-constrained
  → 🧭 Route              (code, always)      base Gemma, JSON-constrained
  → specialist            (AGENTIC, max 4 tool calls)
        📊 Spending Analyst  → SQL over John's transactions (customer-scoped)
        🏠 Product Advisor   → product search + loan calculator
  → ✍️ Draft answer       (code, always)      base Gemma, under 80 words
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
| Input check | base Gemma | safe / off_topic / other_customer / offensive / injection → block politely |
| Advice check | base Gemma + written definitions today; the **advice-LoRA** is trained (`train/`, see `reports/advice_learning_curve.md`) but not yet served | `factual information` → pass · `personalised recommendation` → 🚫 block → rewrite. Binary since a labelling audit found the old middle class split on verb choice, not on anything learnable; the general-advice warning is now a deterministic rule — every `product_advisor` answer carries one |
| Fact check | base Gemma today; the **judge-LoRA** is trained (bacc 0.82 vs 0.72 zero-shot) but not yet served | sentence-level, only sentences carrying a digit, `%`, `$` or "per cent", and at most four per draft; fail → 1 rewrite → else drop the sentence |

- **Eager vs Careful writer prompt**: presenter switch. Guardrails stay on in
  both. Default **Eager** (compliance lives in the guardrail layer, not the
  prompt — stated on screen, not hidden). `/api/state` counts eager and careful
  blocks separately; the bottom strip shows the total.
- Nothing reaches the chat before the checks pass. The **draft streams into the
  trace panel**; blocked drafts show struck through in red.

## Data

- **John Citizen**, 32, Parramatta, IT project manager, $105k, saving for a
  first home. Everyday + savings (~$78k) + credit card (−$3.2k) + car loan.
- 18 months, 1,500 synthetic transactions. Discoverable quirks: Sunday-night
  food delivery spikes (worse in "deadline weeks"), a fortnightly gym
  membership, one concert+hotel splurge, dining and delivery drifting up ~25%
  over the last six months.
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
find me a credit card to cover my expenses · Sarah's balance (blocked at input)
🧠 Think hard: a ninth card that runs the writer with Gemma's thinking mode on
(~40s on the booth card, live timer on the row). The scratchpad streams into the
trace panel as its own step and is never part of the draft — the guardrails check
the answer exactly as they do on every other card.

## Memory

- **Session memory**: the router rewrites each follow-up into a standalone question
  ("and the 5 year one?" → "what's the rate on the 5 year fixed loan?"), shown in
  the trace as "understood as". Specialists get the conversation plus the last two
  turns' tool results; the writer and the fact check get those results too, so an
  earlier number stays grounded. The input check screens the raw question (with the
  previous one as context); the advice check judges the draft alone. Questions
  blocked at the door never enter session memory. Reset by the "New visitor" button; idle auto-reset is off unless
  `limits.idle_reset_seconds` is set above 0.
- **Booth memory**: at reset, one compaction call → `{1-line summary, up to 4
  topics}` shown in the bottom strip ticker; the guardrail counters live beside
  it, outside the card. Redacted, checked before display, **never fed back into
  the chat** (prompt-injection path). Deleted at the end of the event.
- **Customer memory**: what the assistant knows about *John*, in the box at the
  bottom right. One extraction call after each released answer merges the turn into
  a four-field profile — goals, circumstances, preferences, what he is looking at —
  and the box shows the changed fields accented. The call runs *after* the answer is
  already in the chat, so it never delays a reply.
  Deliberately the opposite posture to booth memory: **not anonymous and not
  redacted**, because a profile with its amounts stripped out is not a profile, and
  John is synthetic. What guards it instead is the save step. A turn only updates the
  *pending* profile; **nothing is read back until someone presses Save**, and the
  profile passes the input guardrail on the way to disk, so a poisoned field cannot
  reach a later prompt. Saved memory is prepended to the router, the specialists and
  the writer (as background — the tool results stay the only source of numbers), and
  shows as a "Customer memory" row at the top of the turn. It survives **New visitor**
  and a restart: that persistence is the point. `data/customer_memory.json`, wiped by
  the 🗑️ Forget button and at the end of the event.
  Note it does not buy a bypass: with the profile loaded, "Which loan should I pick?"
  is still blocked by the advice check and rewritten. Worth narrating at the booth.

## Non-goals / deliberate exclusions

- No "break the bot" challenge (separate demo), no guardrails-OFF switch, no
  visitor phones, no React build step, no real bank data or branding.
- Nothing is logged to disk beyond booth-memory summaries and John's saved
  customer-memory profile — no transcripts, no per-request logs.

## Runtime

Python + FastAPI + SSE, vanilla HTML/JS (no build step), SQLite, `uv`.
vLLM serves Gemma 4 E4B — one base model today, the two LoRAs alongside it
once `--enable-lora` goes on; demo talks the OpenAI-compatible API.
systemd `Restart=always`; preflight script. A labelled "RECORDING — not live"
fallback replay for a mid-event vLLM death is still to be built.

## Two-day plan

**Day 1** — vLLM verified with LoRA → skeleton + live trace → John's data →
specialists + tools → three guardrails → 9 cards → end-to-end demo.
**Day 2 am** — booth memory, counters, reset, preflight, recording fallback,
speed pass. **Day 2 pm** — you test with 2–3 colleagues, I fix what they find.
Stretch, in order: EmbeddingGemma, speculative decoding.

FP8 is done: a second vLLM server on :8001 holds the same weights and the same
advice adapter quantized to FP8, and the ⚡ pill picks between them per question.
Measured at batch size 1, 128 tokens, after warm-up: **bf16 19.2 tok/s → FP8 34.8
tok/s (1.81x)** on the base model. Quantizable linears are only 49% of the
checkpoint — the per-layer embedding tables are another 36% and do not quantize —
so ~1.8x is the ceiling here, not 2x. See README for the launch flags.

## Advice-detector fine-tune

Distillation: Claude labels, Gemma learns, the box serves it offline.

```
gen_drafts.py       real specialists + real writer (eager AND careful) → data/drafts.jsonl
label_drafts.py     Claude Opus 5, Batch API, Andrea's definitions     → data/labelled.jsonl
                    (or Claude Code subagents + merge_agent_labels.py)
build_train_data.py question-grouped split, runtime-identical prompt format → data/processed/advice_{train,val,test}.jsonl
train/run.sh        QLoRA in the NGC container (reuses gj-train image) → outputs/gemma4-e4b-advice
eval_advice.py      base vs LoRA on held-out test                      → ship it or don't
```

The gate: the adapter only goes in the demo if balanced accuracy beats base by a
clear margin **and** it misses fewer `personalised_recommendation` cases. Training
data uses the exact runtime prompt, so the adapter learns the job it will do.

**Andrea's homework**: monitor + cables + power at the booth; ~~sign-off on the advice
definitions~~ (approved 14 Sep 2026, see `app/fin-adv-defin.md`); 2–3 colleagues to test on day 2.
