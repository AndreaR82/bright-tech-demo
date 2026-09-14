"""QLoRA SFT: teach Gemma 4 E4B to be the advice detector.

Adapted from the gemma4-groundedness-judge trainer, which is the version proven
on this GB10 box — same HF stack (transformers + peft + trl + bitsandbytes), the
same GB10 gotchas (eager attention, no forked dataloader workers, exclude the
vision/audio towers from LoRA injection).

Runs inside the training container:
    bash train/run.sh python train/train_advice.py --config configs/train_advice.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _build_dataset(path: str, tokenizer):
    """jsonl of {"messages": [...]} → HF Dataset with a rendered "text" field."""
    from datasets import Dataset

    rows = read_jsonl(path)

    def render(batch):
        return {
            "text": [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                for m in batch["messages"]
            ]
        }

    ds = Dataset.from_list(rows)
    return ds.map(render, batched=True, remove_columns=ds.column_names)


def _response_marker(tokenizer) -> list[int]:
    """Token ids that mark the start of the assistant turn.

    Derived from the tokenizer's own chat template rather than hardcoded. This
    Gemma 4 build renders '<|turn>model\\n' (3 tokens); classic Gemma renders
    '<start_of_turn>model\\n', which this tokenizer shreds into literal characters
    that never appear in the rendered text. Get it wrong and the collator masks
    every token, so the run reports loss 0 and trains on nothing — silently.
    Better to stop here than to save an adapter that learned nothing.
    """
    probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    full = tokenizer.encode(probe, add_special_tokens=False)
    for cand in ("<|turn>model\n", "<start_of_turn>model\n", "<|turn>assistant\n"):
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if any(full[i : i + len(ids)] == ids for i in range(len(full) - len(ids) + 1)):
            print(f"[collator] assistant-turn marker {cand!r} -> {ids}")
            return ids
    raise SystemExit(
        "could not find an assistant-turn marker in the chat template.\n"
        f"  rendered probe: {probe!r}\n"
        "  add the right marker to _response_marker() before training."
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="QLoRA SFT for the advice detector")
    ap.add_argument("--config", default="configs/train_advice.yaml")
    ap.add_argument("--max-steps", type=int, default=None, help="cap steps for a smoke run")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg["sft"].get("seed", 42))
    mcfg, lcfg, scfg, ecfg = cfg["model"], cfg["lora"], cfg["sft"], cfg.get("export", {})

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=mcfg.get("load_in_4bit", True),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(mcfg["base"])
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        mcfg["base"],
        quantization_config=bnb_cfg if mcfg.get("load_in_4bit", True) else None,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # GB10 ARM: avoids flash-attention compat issues
    )
    model.config.use_cache = False

    use_gc = scfg.get("gradient_checkpointing", True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gc)

    lora_cfg = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg.get("dropout", 0.0),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lcfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        # The checkpoint resolves to the multimodal model, whose vision/audio towers
        # reuse these projection names but wrap them in a layer peft can't inject.
        exclude_modules=r".*\.(vision_tower|audio_tower)\..*",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = _build_dataset(cfg["data"]["train_file"], tokenizer)
    val_path = cfg["data"].get("val_file")
    val_ds = _build_dataset(val_path, tokenizer) if val_path and Path(val_path).exists() else None

    max_steps = args.max_steps if args.max_steps is not None else scfg.get("max_steps", -1)
    if args.max_steps is not None:
        val_ds = None

    sft_args = SFTConfig(
        output_dir=scfg["output_dir"],
        dataset_text_field="text",
        max_seq_length=mcfg["max_seq_length"],
        num_train_epochs=scfg.get("num_train_epochs", 3),
        max_steps=max_steps,
        per_device_train_batch_size=scfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=scfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(scfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=scfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=scfg.get("warmup_ratio", 0.05),
        optim=scfg.get("optim", "adamw_8bit"),
        weight_decay=scfg.get("weight_decay", 0.01),
        logging_steps=scfg.get("logging_steps", 5),
        save_steps=scfg.get("save_steps", 200),
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=scfg.get("eval_steps", 100),
        bf16=scfg.get("bf16", True),
        gradient_checkpointing=use_gc,
        seed=scfg.get("seed", 42),
        report_to="none",
        dataset_num_proc=scfg.get("dataset_num_proc", 1),
        dataloader_num_workers=scfg.get("dataloader_num_workers", 0),  # GB10: forks deadlock
    )

    data_collator = None
    if scfg.get("train_on_responses_only", True):
        # Loss on the assistant turn only. The marker is looked up rather than
        # assumed — see _response_marker for why a wrong one fails silently.
        response_ids = _response_marker(tokenizer)
        data_collator = DataCollatorForCompletionOnlyLM(response_ids, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_args,
        data_collator=data_collator,
    )
    trainer.train()

    out = scfg["output_dir"]
    if ecfg.get("save_adapter", True):
        model.save_pretrained(out)
        tokenizer.save_pretrained(out)
        print(f"[export] adapter saved to {out}")
        print(f"[serve]  vllm serve ... --enable-lora --lora-modules advice={out}")


if __name__ == "__main__":
    main()
