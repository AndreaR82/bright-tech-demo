# FP8 vs bf16 — speed, and what it costs the guardrail

Gemma 4 E4B served twice on the one Spark: bf16 on :8000, the same weights and the
same `advice` adapter quantized to FP8 on :8001. Only the quantization flags
differ (`--quantization fp8 --kv-cache-dtype fp8`), so the comparison isolates one
variable. Measured 16 Sep 2026.

## Speed

Batch size 1, 128 tokens, `ignore_eos`, six interleaved rounds, **on an otherwise
idle box** after five full-shape warm-up calls.

| | median tok/s | range | vs bf16 |
|---|---|---|---|
| bf16 | 19.1 | 18.3–19.9 | 1.00x |
| FP8 | **35.6** | 34.3–37.6 | **1.86x** |
| bf16 + advice-LoRA | 18.2 | 17.0–18.7 | 0.95x |
| FP8 + advice-LoRA | **33.2** | 31.0–33.9 | **1.74x** |

The adapter costs ~5% at either precision; FP8 with the adapter is 1.82x its own
bf16 baseline. Weights load at **11.19 GiB against 15.26 GiB** bf16.

Two things will make you measure this wrong:

- **Warm-up.** The first FP8 calls run at roughly bf16 speed while Cutlass
  (`CutlassFP8ScaledMMLinearKernel`) and Triton autotune. An unwarmed measurement
  said FP8 was *slower* (16.0 vs 17.9). It settles after ~5 calls.
- **Contention.** Running anything else against either server destroys the
  numbers and does it asymmetrically — under load FP8+LoRA looked bimodal
  (14.4–32.1) and appeared to be adapter-swap thrash. It was not. On an idle box
  LoRA requests back-to-back (32.1) and interleaved with base requests (32.8) are
  indistinguishable.

### Why 1.86x and not 2x

The 14.89 GiB bf16 checkpoint, by tensor group:

| group | size | share |
|---|---|---|
| MLP linears | 6.15 GiB | 41.3% |
| PLE per-layer embeddings | 5.40 GiB | 36.3% |
| token embedding (tied to lm_head) | 1.25 GiB | 8.4% |
| attention linears | 1.20 GiB | 8.0% |
| audio tower | 0.58 GiB | 3.9% |
| vision tower | 0.32 GiB | 2.1% |

Only the linears quantize — 7.35 GiB, 49.3% of the checkpoint. The per-layer
embedding tables are another third and stay bf16. Per decoded token the engine
reads linears + lm_head (~8.6 GiB), which FP8 takes to ~4.9 GiB. ~1.8x is the
ceiling for this architecture, so quote that rather than "twice as fast".

## What it costs the advice detector

`scripts/eval_advice.py --adapter advice --precision {bf16,fp8}`, 224 drafts from
the untouched test split.

| | bf16 base | FP8 base | bf16 + LoRA | FP8 + LoRA |
|---|---|---|---|---|
| accuracy | 0.955 | 0.951 | 0.991 | **0.991** |
| balanced accuracy | 0.876 | 0.862 | 0.984 | **0.984** |
| recall `personalised_recommendation` | 0.757 | 0.730 | 0.973 | **0.973** |
| recall `factual_information` | 0.995 | 0.995 | 0.995 | 0.995 |
| slice: neutral_open (n=11) | 0.727 | 0.727 | 0.909 | 0.909 |
| slice: original (n=177) | 0.977 | 0.972 | 1.000 | 1.000 |
| slice: prefix_factual (n=10) | 1.000 | 1.000 | 1.000 | 1.000 |
| slice: referral_personal (n=12) | 0.833 | 0.833 | 1.000 | 1.000 |
| slice: steer_removed (n=14) | 0.929 | 0.929 | 0.929 | 0.929 |

**The two adapter columns are identical** — `diff` on the two report blocks
returns nothing but the heading. Across 224 drafts, quantization does not flip a
single decision the detector makes.

The base model is not equally lucky: FP8 costs it 2 of 74 personalised drafts
(recall 0.757 → 0.730). So FP8 is not numerically free — the fine-tuned adapter
simply has the margin to absorb it where the zero-shot model, sitting nearer the
boundary, does not. The fine-tune bought robustness as well as accuracy.

Both runs print `verdict: ship the adapter`. bf16 base recall reproduces
`reports/base_vllm_eval.txt` (0.757) exactly, which validates the harness.

## Caveat

FP8 changes the *drafts*, even at temperature 0 — the writer's tokens differ, so a
booth turn is not reproducible across the toggle. One traced turn had FP8 draft
something the detector blocked where bf16's draft passed, triggering an extra
rewrite. That is the writer diverging, not the detector disagreeing: on identical
input, per the table above, it does not.
