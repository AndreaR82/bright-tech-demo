# Bright Tech Demo

A booth demo: a mock bank assistant for a synthetic customer (**John Citizen**),
with its entire agent pipeline shown live on the right-hand side of the screen.
Everything runs on the DGX Spark — one small model, no internet, no cloud.

See [SPEC.md](SPEC.md) for the design and the decisions behind it.

## Run it

```bash
uv sync                                     # once
uv run python scripts/generate_data.py      # once — John's 18 months of transactions
uv run uvicorn app.server:app --host 127.0.0.1 --port 8090
```

Then open <http://localhost:8090>. Port 8090 because 8080 is your existing
`spark-web` app.

vLLM must be serving Gemma on port 8000 (`spark-vllm.service`). Override with
`VLLM_BASE_URL`, `VLLM_MODEL`, `VLLM_JUDGE_MODEL`, `VLLM_ADVICE_MODEL`.

To use the trained advice detector rather than the base model for the advice
check, point the app at the served LoRA:

```bash
VLLM_ADVICE_MODEL=advice uv run uvicorn app.server:app --host 127.0.0.1 --port 8090
```

### The FP8 endpoint (the ⚡ toggle)

A second vLLM server on port 8001 serves the same weights and the same adapter,
quantized to FP8, so the booth can switch precision between questions and show
the speed and quality difference on screen. Only the quantization flags differ —
the A/B isolates one variable:

```bash
spark-vllm-fp8.service   # same as spark-vllm.service, plus:
  -p 8001:8000 --name vllm_fp8
  --gpu-memory-utilization 0.22
  --quantization fp8
  --kv-cache-dtype fp8
```

Override with `VLLM_FP8_BASE_URL`, `VLLM_FP8_MODEL`, `VLLM_FP8_ADVICE_MODEL`,
`VLLM_FP8_JUDGE_MODEL`. `VLLM_PRECISION` sets which endpoint is the default for
anything that does not ask for one (bf16 unless you say otherwise).

The FP8 server is optional. With it down the ⚡ pill greys out, every turn runs
in bf16, and preflight still goes green.

**Warm it up before the doors open.** The first few FP8 requests run at roughly
bf16 speed while Cutlass and Triton autotune; after ~5 calls it settles. Its
torch.compile cache key differs from the bf16 server's, so a cold start compiles
from scratch rather than loading the AOT cache.

## Before the doors open

```bash
uv run python scripts/preflight.py
```

Runs every question card through the real pipeline, checks the guardrails
actually fire, then runs two short follow-up conversations (a number carried
across turns, guardrails on "should I take it?"), prints the timings, exits
non-zero if anything is off.

For the full multi-turn check — eight conversations of follow-ups like "and the 5
year one?" — run `uv run python scripts/eval_multiturn.py`.

## At the booth

| Control | What it does |
|---|---|
| **New visitor** button | compacts the session into booth memory, clears both panels |
| Idle auto-clear | off by default (`limits.idle_reset_seconds: 0`); set it to a number of seconds to re-enable |
| **Ctrl+M** | flips the writer prompt: eager ↔ careful (guardrails stay on either way) |
| ⚡ **FP8** toggle | sends the next question to the FP8 endpoint on :8001 instead of bf16 on :8000. Per-question, so a click never splits a turn. Greyed out when :8001 is down. The footer keeps a running `bf16 … · fp8 …` tok/s comparison that survives **New visitor** |
| 🧠 **Think hard** card | writer runs with Gemma's thinking mode on (~40 s); the scratchpad shows in the trace, never in the chat |
| 💾 **Save to John's memory** | commits the customer-memory box (lower right) to `data/customer_memory.json`. Only saved memory is read back into later conversations, so this button is the moment memory becomes real. Greyed out when nothing has changed since the last save |
| 🗑️ **Forget** | wipes John's file and the box. Do this between demos if a visitor has taken his profile somewhere odd |
| `app/config.yaml` | prompts, cards, limits — edit, then restart the server (it is read once at import) |

## Layout

```
app/
  config.yaml   prompts, question cards, limits          ← tune this at the booth
  llm.py        vLLM client (timings + tokens per call)
  tools.py      SQL (scoped to John), product search, loan maths
  pipeline.py   the agent: input check → route → specialist → draft → checks
  memory.py     customer memory: John's profile, saved to disk and read back
  server.py     FastAPI + SSE + booth memory
static/index.html   the booth page (no build step)
scripts/        generate_data.py, preflight.py, eval_multiturn.py, and the
                fine-tuning pipeline (gen_drafts → label → build → eval)
data/           bank.db (generated), products.json (hand-written catalogue),
                customer_memory.json (John's saved profile, written by the
                booth) — all of data/ is gitignored, so a fresh clone needs
                products.json put back before the server will import
```

## Still to do

- serve the groundedness judge (trained in the `gemma4-groundedness-judge` repo)
  and point `VLLM_JUDGE_MODEL` at it. The fact check still runs on the base
  model; the advice detector is served and wired up (`VLLM_ADVICE_MODEL=advice`)
- recording fallback for the "vLLM died mid-event" case
- EmbeddingGemma for product search; speculative decoding for speed
- FP8 + the advice LoRA is erratic at batch size 1 (14–32 tok/s against a steady
  18 on bf16), while FP8 on the base model is a clean 1.81x. Worth chasing:
  `cudagraph_specialize_lora` and the adapter-swap path between base and LoRA
  requests

The advice definitions in `app/config.yaml` are written up from ASIC's guidance
and verified against primary sources (12 Sep 2026), with the reasoning and the
currency notes in [app/fin-adv-defin.md](app/fin-adv-defin.md) — read them before
labelling, since that wording is what the adapter learns. Approved by Andrea on
14 Sep 2026; the sign-off at the top of that file pins the exact prompt by hash.

## Fine-tuning the advice detector

Claude labels, Gemma learns, the box serves it with no internet.

The detector is **binary**: `factual_information` vs `personalised_recommendation`,
i.e. release or block. See [app/fin-adv-defin.md](app/fin-adv-defin.md) for why the
old middle class was dropped.

There are two ways to label. The Batch API path needs API credit, which a Claude Pro
subscription does **not** include. The agent path uses Claude Code subagents instead
and costs no credits:

```bash
uv run python scripts/gen_drafts.py --n 500 --workers 6   # real drafts from the real pipeline
                                                          # (resumable — re-run to top up)

# path A — Batch API (needs credit on the API account, not a Pro subscription)
export ANTHROPIC_API_KEY=sk-ant-...
uv run --with anthropic python scripts/label_drafts.py --estimate   # cost first
uv run --with anthropic python scripts/label_drafts.py             # Batch API, ~1 hour

# path B — Claude Code subagents: chunk the drafts, label each chunk, then
uv run python scripts/merge_agent_labels.py                        # stitch chunks -> labelled.jsonl

# counterfactuals — break the prefix and lender-referral shortcuts (see fin-adv-defin.md)
uv run python scripts/gen_counterfactuals.py generate              # Gemma rewrites, mechanically checked
uv run python scripts/gen_counterfactuals.py chunk                 # blind chunks, no intended label
#   Claude Code subagents label data/_agent_labels/cf/chunk_NNN.json -> labels_NNN.jsonl
uv run python scripts/gen_counterfactuals.py merge                 # keep rewrites the blind label agrees with

uv run python scripts/build_train_data.py                          # question-grouped split, + counterfactuals
bash train/run.sh python train/train_advice.py --config configs/train_advice.yaml --max-steps 20   # smoke
bash train/run.sh python train/train_advice.py --config configs/train_advice.yaml                  # full
uv run python scripts/eval_advice.py --adapter advice              # base vs LoRA

# learning curve — how many labelled rows does the detector need?
bash train/run.sh python -u scripts/eval_checkpoints.py --watch   # score each checkpoint (val+test) as it lands
uv run python scripts/run_subsets.py                               # retrain on nested 150/300/600/900-row subsets
uv run --with matplotlib python scripts/learning_report.py         # -> reports/advice_learning_curve.md
```

**Generated and labelled.** `data/drafts.jsonl` holds 1,560 rows from 547 distinct
questions — **1,152 unique drafts** after de-duplication, ~39 words each, an even
eager/careful split. All 1,152 are labelled (`data/labelled.jsonl`): **1,069 factual
/ 83 personalised recommendation**. The class that matters is 7% of the set — that is
the thin part, the reason `eval_advice.py` reports its recall separately, and the
reason `gen_counterfactuals.py` adds 308 rewritten rows on top (1,460 in all).
Labelling the full set costs about **$3.14** via the Batch API, or no credits at all
via the subagent path.

The same 50 drafts were labelled three times while the definitions were being fixed.
As a binary decision the three passes agree on **49 of 50 (98%)**; under the older
three-class scheme they agreed on only 84%, and every unstable draft sat on the
middle-class boundary. That is the evidence for dropping it.

Training runs in the NGC container from your `gemma4-groundedness-judge` repo
(`gj-train:latest`, bind-mounted — no rebuild) and reuses that repo's HF cache, so
the Gemma weights aren't downloaded twice. Serve the result alongside the base
model with `--enable-lora --lora-modules advice=outputs/gemma4-e4b-advice`, then
point the demo at it with `VLLM_ADVICE_MODEL=advice`.

Watch the loss: a healthy run starts around 1–3. Exactly 0, with `grad_norm` 0 and
`mean_token_accuracy` 0, means the assistant-turn marker wasn't found and every token
was masked — the trainer will still cheerfully save an adapter that learned nothing.
Near 13–15 means masking broke the other way and it's training on the prompt.

**Result so far** ([reports/advice_learning_curve.md](reports/advice_learning_curve.md)):
the full run takes F1 on `personalised_recommendation` from **0.674 to 0.966** and its
recall from **0.508 to 0.949**, scored on the same 373 val+test drafts by first-token
logit. The first 300 labelled rows buy most of it; 900 → 1,087 moves nothing this eval
can resolve. That report is for understanding, not for choosing — the ship-or-don't
gate is still `eval_advice.py --adapter advice` on the untouched test split, and it
needs the adapter served.
