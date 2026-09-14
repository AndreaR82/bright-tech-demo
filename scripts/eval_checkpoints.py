"""Learning curve for the advice adapter: score every checkpoint on one fixed set.

The base model is step 0, then each checkpoint-N the trainer saves. Every point
is scored on the same drafts (val + test by default), so the curve shows how
F1 / accuracy move with training, and — in epoch 1, where each step is new rows —
roughly how many labelled rows the detector needs.

Scoring is one forward pass per draft, not generation: the runtime JSON always
starts {"label": "  and the two labels differ at their first token, so comparing
those two logits is the choice greedy schema-constrained decoding makes in vLLM.
The two-way softmax also gives a probability, hence ROC-AUC.

Runs in the training container, alongside the trainer:
    bash train/run.sh python -u scripts/eval_checkpoints.py --watch
    bash train/run.sh python -u scripts/eval_checkpoints.py --config configs/subsets/n300.yaml --no-base

Writes into the run dir (resumable):
    curve.jsonl    one line of aggregate metrics per scored step
    probs.jsonl    per-draft P(personalised) per step, for bootstrap error bars
    eval_set.json  the eval drafts, labels and slices, in the order of probs

The curve includes the test split. That is fine for looking, NOT for choosing:
the adapter that ships is the final one, so the ship-or-don't gate stays honest.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
P = "personalised_recommendation"
PREFIX = '{"label": "'


def load_eval(files: list[str]) -> list[dict]:
    """Val is chat-rendered and test is raw; recover draft, label and rewrite type for both."""
    sources: dict[str, dict] = {}
    for name in ("labelled.jsonl", "counterfactuals_labelled.jsonl"):
        path = ROOT / "data" / name
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    sources.setdefault(row["draft"].strip().lower(), row)
    rows = []
    for f in files:
        split = Path(f).stem.replace("advice_", "")
        for line in Path(f).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "messages" in r:
                draft = r["messages"][1]["content"].split("\n", 1)[1]
                label = json.loads(r["messages"][2]["content"])["label"]
                src = sources.get(draft.strip().lower(), {})
            else:
                draft, label, src = r["draft"], r["label"], r
            rows.append({"draft": draft, "label": label, "split": split, "slice": src.get("cf_type", "original")})
    return rows


def label_token_ids(tok, system: str) -> tuple[int, int]:
    probe = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": "Draft reply:\nx"}],
        tokenize=False, add_generation_prompt=True) + PREFIX
    base = tok.encode(probe, add_special_tokens=False)
    ids = []
    for label in ("factual_information", P):
        full = tok.encode(probe + label, add_special_tokens=False)
        if full[: len(base)] != base:
            raise SystemExit(f"prompt prefix re-tokenizes differently before {label!r}")
        ids.append(full[len(base)])
    if ids[0] == ids[1]:
        raise SystemExit(f"both labels start with token {ids[0]} — first-token scoring cannot separate them")
    return ids[0], ids[1]


def probabilities(model, tok, rows: list[dict], system: str, ids: tuple[int, int], bs: int) -> list[float]:
    """P(personalised_recommendation) per draft, from the first label token."""
    import torch

    tok.padding_side = "left"
    out: list[float] = []
    for i in range(0, len(rows), bs):
        texts = [
            tok.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": f"Draft reply:\n{r['draft']}"}],
                tokenize=False, add_generation_prompt=True) + PREFIX
            for r in rows[i : i + bs]
        ]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            try:  # full-vocab logits for every position would cost GBs; keep the last one
                logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
            except TypeError:
                logits = model(**enc).logits[:, -1, :].float()
        two = torch.stack([logits[:, ids[0]], logits[:, ids[1]]], dim=-1).softmax(-1)
        out += two[:, 1].tolist()
    return out


def auc(scores: list[float], gold: list[bool]) -> float | None:
    pos = [s for s, g in zip(scores, gold) if g]
    neg = [s for s, g in zip(scores, gold) if not g]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def metrics(rows: list[dict], probs: list[float]) -> dict:
    gold = [r["label"] == P for r in rows]
    pred = [p > 0.5 for p in probs]
    tp = sum(g and q for g, q in zip(gold, pred))
    fp = sum(q and not g for g, q in zip(gold, pred))
    fn = sum(g and not q for g, q in zip(gold, pred))
    tn = len(rows) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(rows), "positives": tp + fn,
        "accuracy": (tp + tn) / len(rows),
        "balanced_accuracy": (recall + (tn / (tn + fp) if tn + fp else 0.0)) / 2,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "auc": auc(probs, gold),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def grouped(rows: list[dict], probs: list[float], key: str) -> dict:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r[key]].append(i)
    return {k: {m: v for m, v in metrics([rows[i] for i in idx], [probs[i] for i in idx]).items()
                if m in ("n", "positives", "accuracy", "recall", "f1")}
            for k, idx in sorted(groups.items())}


def trainer_context(ckpt: Path, step: int) -> dict:
    """Learning rate, latest train loss and val loss at this step, from the trainer's own log."""
    state = json.loads((ckpt / "trainer_state.json").read_text())
    ctx: dict = {"epoch": None, "lr": None, "train_loss": None, "eval_loss": None}
    for entry in state.get("log_history", []):
        if entry.get("step", 0) > step:
            break
        if "loss" in entry:
            ctx.update(lr=entry.get("learning_rate"), train_loss=entry["loss"], epoch=entry.get("epoch"))
        if "eval_loss" in entry and entry.get("step") == step:
            ctx["eval_loss"] = entry["eval_loss"]
    return ctx


def ready_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    found = []
    for d in run_dir.glob("checkpoint-*"):
        state = d / "trainer_state.json"
        # trainer_state.json is written last; give the save a moment to finish.
        if (d / "adapter_model.safetensors").exists() and state.exists() and time.time() - state.stat().st_mtime > 20:
            found.append((int(d.name.split("-")[1]), d))
    return sorted(found)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_advice.yaml")
    ap.add_argument("--eval-files", nargs="+",
                    default=["data/processed/advice_val.jsonl", "data/processed/advice_test.jsonl"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--no-base", action="store_true", help="skip step 0 (the base model is the same for every run)")
    ap.add_argument("--watch", action="store_true", help="keep scoring new checkpoints until training finishes")
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_dir = Path(cfg["sft"]["output_dir"])
    rows_per_step = cfg["sft"].get("per_device_train_batch_size", 1) * cfg["sft"].get("gradient_accumulation_steps", 1)
    system = yaml.safe_load((ROOT / "app" / "config.yaml").read_text())["prompts"]["advice_check"]
    rows = load_eval(args.eval_files)
    print(f"[curve] {len(rows)} eval drafts, {sum(r['label'] == P for r in rows)} personalised; "
          f"{rows_per_step} rows per step")

    run_dir.mkdir(parents=True, exist_ok=True)
    eval_path = run_dir / "eval_set.json"
    if eval_path.exists() and json.loads(eval_path.read_text()) != rows:
        raise SystemExit(f"{eval_path} holds a different eval set — probs would not line up; move it aside")
    eval_path.write_text(json.dumps(rows, ensure_ascii=False))

    # A step counts as done only with both its metrics and its per-draft probs:
    # the report's error bars need the probs, so older probs-less lines are rescored.
    curve_path, probs_path = run_dir / "curve.jsonl", run_dir / "probs.jsonl"
    probs_steps = ({json.loads(l)["step"] for l in probs_path.read_text().splitlines() if l.strip()}
                   if probs_path.exists() else set())
    curve_lines = [l for l in curve_path.read_text().splitlines() if l.strip()] if curve_path.exists() else []
    kept = [l for l in curve_lines if json.loads(l)["step"] in probs_steps]
    if len(kept) != len(curve_lines):
        print(f"[curve] dropping {len(curve_lines) - len(kept)} scored step(s) without per-draft probs; rescoring")
        curve_path.write_text("".join(l + "\n" for l in kept))
    done = {json.loads(l)["step"] for l in kept}

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(cfg["model"]["base"])
    ids = label_token_ids(tok, system)
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base"],
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
        # Not "auto": GB10 memory is unified, and next to the trainer and vLLM the GPU's
        # reported free memory excludes reclaimable cache, so "auto" offloads layers to
        # CPU and scoring crawls. Pinning to the GPU is safe on a shared pool.
        torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager",
    )
    base.eval()

    def record(step: int, probs: list[float], ctx: dict, seconds: float) -> None:
        line = {"step": step, "rows_seen": step * rows_per_step, **ctx, **metrics(rows, probs),
                "by_split": grouped(rows, probs, "split"), "by_slice": grouped(rows, probs, "slice"),
                "seconds": round(seconds, 1)}
        # probs first: a crash between the two writes leaves an orphan probs line
        # (harmless — readers take the last one per step), never a curve line without probs.
        with probs_path.open("a") as fh:
            fh.write(json.dumps({"step": step, "probs": [round(p, 6) for p in probs]}) + "\n")
        with curve_path.open("a") as fh:
            fh.write(json.dumps(line) + "\n")
        done.add(step)
        print(f"[curve] step {step:>4}  rows {line['rows_seen']:>5}  f1 {line['f1']:.3f}  "
              f"recall {line['recall']:.3f}  bacc {line['balanced_accuracy']:.3f}  "
              f"auc {line['auc'] or 0:.3f}  eval_loss {ctx.get('eval_loss')}  ({seconds:.0f}s)", flush=True)

    if 0 not in done and not args.no_base:
        started = time.monotonic()
        record(0, probabilities(base, tok, rows, system, ids, args.batch_size),
               {"epoch": 0.0, "lr": 0.0, "train_loss": None, "eval_loss": None}, time.monotonic() - started)

    while True:
        for step, ckpt in ready_checkpoints(run_dir):
            if step in done:
                continue
            started = time.monotonic()
            model = PeftModel.from_pretrained(base, str(ckpt))
            model.eval()
            probs = probabilities(model, tok, rows, system, ids, args.batch_size)
            base = model.unload()  # strip the LoRA layers again, without merging them
            record(step, probs, trainer_context(ckpt, step), time.monotonic() - started)
        finished = (run_dir / "adapter_model.safetensors").exists()
        pending = [s for s, _ in ready_checkpoints(run_dir) if s not in done]
        if not args.watch or (finished and not pending):
            break
        time.sleep(args.poll)
    print(f"[curve] done: {len(done)} points -> {curve_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
