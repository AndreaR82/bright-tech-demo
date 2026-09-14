"""Collect the advice detector's learning-curve results into one markdown report.

    uv run --with matplotlib python scripts/learning_report.py

Reads whatever exists so far, so it is safe to run while the queue is going:
    outputs/gemma4-e4b-advice/   full run: base model (step 0) + a checkpoint every 25 steps
    outputs/subsets/n*/          nested train subsets (scripts/run_subsets.py), a checkpoint per epoch
each holding curve.jsonl, probs.jsonl and eval_set.json from scripts/eval_checkpoints.py.

Writes reports/advice_learning_curve.md and reports/figures/*.png.

Error bars are 95% bootstrap intervals over the eval drafts: 2,000 resamples, the
SAME resamples for every point, so the change between two points gets a paired
interval too. They show how much a score depends on which drafts happened to be
in the eval set. They do not show run-to-run training noise (one seed per size),
so treat them as a lower bound on the real uncertainty. Val loss has no interval.
Per-slice accuracies use Wilson intervals.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "outputs" / "gemma4-e4b-advice"
SUBSETS = ROOT / "outputs" / "subsets"
REPORT = ROOT / "reports" / "advice_learning_curve.md"
FIGURES = ROOT / "reports" / "figures"
BASE_VLLM = ROOT / "reports" / "base_vllm_eval.txt"
P = "personalised_recommendation"
B = 2000
METRICS = ["f1", "recall", "precision", "balanced_accuracy", "auc", "accuracy"]

# Reference palette (dataviz skill): chart chrome + categorical slots 1-3, in order.
SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


# ---------------------------------------------------------------- loading

def load_run(run_dir: Path) -> dict | None:
    need = [run_dir / f for f in ("curve.jsonl", "probs.jsonl", "eval_set.json")]
    if not all(p.exists() for p in need):
        return None
    curve = {}
    for line in need[0].read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            curve[r["step"]] = r
    probs = {}
    for line in need[1].read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            probs[r["step"]] = np.array(r["probs"], dtype=float)  # last line per step wins
    steps = sorted(s for s in curve if s in probs)
    if not steps:
        return None
    # Finished = trained AND every checkpoint scored; otherwise the last scored
    # step of a run still being scored would pose as its final result.
    ckpts = {int(d.name.split("-")[1]) for d in run_dir.glob("checkpoint-*")}
    return {"dir": run_dir, "curve": curve, "probs": probs, "steps": steps,
            "eval": json.loads(need[2].read_text()),
            "finished": (run_dir / "adapter_model.safetensors").exists() and ckpts <= set(steps)}


# ---------------------------------------------------------------- statistics

def _div(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.full(np.broadcast(a, b).shape, np.nan)
    np.divide(a, b, out=out, where=b > 0)
    return out


def _auc(scores: np.ndarray, gold: np.ndarray) -> float:
    pos, neg = scores[gold], scores[~gold]
    if not len(pos) or not len(neg):
        return np.nan
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def metric_matrix(p: np.ndarray, gold: np.ndarray, idx: np.ndarray) -> dict[str, np.ndarray]:
    """Every metric for every row of resample indices idx (shape: resamples x drafts)."""
    g, q, s = gold[idx], (p > 0.5)[idx], p[idx]
    tp, fp = (g & q).sum(1), (~g & q).sum(1)
    fn, tn = (g & ~q).sum(1), (~g & ~q).sum(1)
    recall, specificity = _div(tp, tp + fn), _div(tn, tn + fp)
    return {
        "f1": _div(2 * tp, 2 * tp + fp + fn),
        "recall": recall,
        "precision": _div(tp, tp + fp),
        "balanced_accuracy": (recall + specificity) / 2,
        "accuracy": (tp + tn) / idx.shape[1],
        "auc": np.array([_auc(s[i], g[i]) for i in range(len(idx))]),
    }


class Scorer:
    def __init__(self, eval_set: list[dict]):
        self.gold = np.array([r["label"] == P for r in eval_set])
        self.slices = np.array([r["slice"] for r in eval_set])
        self.splits = np.array([r["split"] for r in eval_set])
        n = len(eval_set)
        self.identity = np.arange(n)[None, :]
        self.resamples = np.random.default_rng(0).integers(0, n, size=(B, n))
        self._cache: dict[int, dict] = {}

    def summary(self, probs: np.ndarray) -> dict:
        key = id(probs)
        if key not in self._cache:
            point = metric_matrix(probs, self.gold, self.identity)
            boot = metric_matrix(probs, self.gold, self.resamples)
            self._cache[key] = {m: {"value": float(point[m][0]),
                                    "lo": float(np.nanpercentile(boot[m], 2.5)),
                                    "hi": float(np.nanpercentile(boot[m], 97.5)),
                                    "boot": boot[m]} for m in METRICS}
        return self._cache[key]

    def delta(self, a: np.ndarray, b: np.ndarray, metric: str) -> tuple[float, float, float]:
        """b minus a, with a paired bootstrap interval."""
        sa, sb = self.summary(a)[metric], self.summary(b)[metric]
        d = sb["boot"] - sa["boot"]
        return sb["value"] - sa["value"], float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    z = 1.96
    centre = (k + z * z / 2) / (n + z * z)
    half = z * math.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return centre - half, centre + half


# ---------------------------------------------------------------- formatting

def ci(m: dict, digits: int = 3) -> str:
    if math.isnan(m["value"]):
        return "—"
    return f"{m['value']:.{digits}f} ({m['lo']:.2f}–{m['hi']:.2f})"


def num(v, spec: str = ".3f") -> str:
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------- figures

def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="semibold")


def _legend(ax, loc: str) -> None:
    ax.legend(loc=loc, frameon=False, fontsize=9, labelcolor=SECONDARY, ncols=3)


def figure_data_size(points: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    xs = np.array([pt["rows"] for pt in points], dtype=float)
    dodge = max(xs.max(), 1) * 0.008
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8), sharex=True, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [3, 2], "hspace": 0.32})
    panels = [("Personalised recommendation: F1, recall, precision",
               [("f1", "F1"), ("recall", "Recall"), ("precision", "Precision")]),
              ("Balanced accuracy and ROC-AUC", [("balanced_accuracy", "Balanced accuracy"), ("auc", "ROC-AUC")])]
    for ax, (title, series) in zip(axes, panels):
        _style(ax, title)
        lows = []
        for i, (key, label) in enumerate(series):
            v = np.array([pt["m"][key]["value"] for pt in points])
            lo = np.array([pt["m"][key]["lo"] for pt in points])
            hi = np.array([pt["m"][key]["hi"] for pt in points])
            lows.append(np.nanmin(lo) if np.isfinite(lo).any() else 0)
            off = (i - (len(series) - 1) / 2) * dodge
            ax.errorbar(xs + off, v, yerr=[np.nan_to_num(v - lo), np.nan_to_num(hi - v)], color=SERIES[i],
                        linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=2,
                        elinewidth=1.2, capsize=3, label=label, zorder=3)
            if i == 0 and np.isfinite(v[-1]):
                ax.annotate(f"{v[-1]:.2f}", (xs[-1] + off, v[-1]), xytext=(10, 0), textcoords="offset points",
                            color=SECONDARY, fontsize=9, va="center")
        _legend(ax, "lower right")
        ax.set_ylim(max(0.0, min(lows) - 0.05), 1.03)
    axes[0].set_ylim(0, 1.03)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([f"{int(x):,}" for x in xs])
    axes[1].set_xlabel("Labelled training rows (0 = base model, zero-shot) · bars: 95% bootstrap CI",
                       color=SECONDARY, fontsize=9.5)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def figure_training(run: dict, scorer: Scorer, train_rows: int, epochs: int, path: Path) -> None:
    import matplotlib.pyplot as plt

    steps = run["steps"]
    curve = [run["curve"][s] for s in steps]
    xs = np.array([c["rows_seen"] for c in curve], dtype=float)
    sums = [scorer.summary(run["probs"][s]) for s in steps]
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [3, 2, 1.3], "hspace": 0.35})

    ax = axes[0]
    _style(ax, "Personalised recommendation: F1, recall, precision")
    for i, (key, label) in enumerate([("f1", "F1"), ("recall", "Recall"), ("precision", "Precision")]):
        v = np.array([s[key]["value"] for s in sums])
        lo = np.array([s[key]["lo"] for s in sums])
        hi = np.array([s[key]["hi"] for s in sums])
        ax.fill_between(xs, lo, hi, color=SERIES[i], alpha=0.10, linewidth=0, zorder=1)
        ax.plot(xs, v, color=SERIES[i], linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=label, zorder=3)
    ax.set_ylim(0, 1.03)
    _legend(ax, "lower right")

    ax = axes[1]
    _style(ax, "Loss")
    for i, (key, label) in enumerate([("train_loss", "Train loss"), ("eval_loss", "Val loss")]):
        pts = [(c["rows_seen"], c.get(key)) for c in curve if c.get(key) is not None]
        if pts:
            px, py = zip(*pts)
            ax.plot(px, py, color=SERIES[i], linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=label, zorder=3)
    ax.set_ylim(bottom=0)
    _legend(ax, "upper right")

    ax = axes[2]
    _style(ax, "Learning rate")
    ax.plot(xs, [c.get("lr") or 0 for c in curve], color=SERIES[0], linewidth=2, zorder=3)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_color(MUTED)
    ax.set_xlabel("Training rows seen (step × 8) · bands: 95% bootstrap CI", color=SECONDARY, fontsize=9.5)

    for a in axes:
        for e in range(1, epochs):
            a.axvline(e * train_rows, color=AXIS, linewidth=1, zorder=0)
    for e in range(1, epochs):
        axes[0].text(e * train_rows, 1.02, f" epoch {e + 1}", color=MUTED, fontsize=8.5, va="top")
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- report

def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["system-ui", "Segoe UI", "DejaVu Sans", "sans-serif"]

    cfg = yaml.safe_load((ROOT / "configs" / "train_advice.yaml").read_text())
    train_rows = sum(1 for l in (ROOT / cfg["data"]["train_file"]).read_text().splitlines() if l.strip())
    epochs = cfg["sft"]["num_train_epochs"]

    full = load_run(FULL)
    if full is None:
        raise SystemExit(f"no scored points in {FULL} yet")
    subsets = {}
    for d in sorted(SUBSETS.glob("n*"), key=lambda d: int(d.name[1:]) if d.name[1:].isdigit() else 0):
        run = load_run(d)
        if run is not None:
            if run["eval"] != full["eval"]:
                raise SystemExit(f"{d} was scored on a different eval set than the full run")
            subsets[int(d.name[1:])] = run

    scorer = Scorer(full["eval"])
    n_eval, n_pos = len(full["eval"]), int(scorer.gold.sum())
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Data-size points: base, each finished subset's final checkpoint, the finished full run.
    points = []
    if 0 in full["probs"]:
        points.append({"rows": 0, "label": "0 (base)", "step": 0, "run": full, "probs": full["probs"][0]})
    for n, run in subsets.items():
        if run["finished"]:
            s = run["steps"][-1]
            points.append({"rows": n, "label": f"{n:,}", "step": s, "run": run, "probs": run["probs"][s]})
    if full["finished"]:
        s = full["steps"][-1]
        points.append({"rows": train_rows, "label": f"{train_rows:,} (full)", "step": s, "run": full,
                       "probs": full["probs"][s]})
    for pt in points:
        pt["m"] = scorer.summary(pt["probs"])

    md: list[str] = []
    md.append("# Advice detector: learning curve\n")
    status = [f"full run {'finished' if full['finished'] else f'in progress (last scored step {full['steps'][-1]})'}"]
    for n in (150, 300, 600, 900):
        run, d = subsets.get(n), SUBSETS / f"n{n}"
        if run and run["finished"]:
            state = "finished"
        elif (d / "adapter_model.safetensors").exists():
            state = "trained, scoring" + (f" (step {run['steps'][-1]} done)" if run else "")
        elif d.exists():
            state = "training"
        else:
            state = "queued"
        status.append(f"n={n} {state}")
    md.append(f"Generated {time.strftime('%Y-%m-%d %H:%M')} · " + " · ".join(status) + "\n")

    md.append("## What was measured\n")
    split_counts = {s: int((scorer.splits == s).sum()) for s in sorted(set(scorer.splits))}
    slice_counts = {s: (int((scorer.slices == s).sum()), int((scorer.gold & (scorer.slices == s)).sum()))
                    for s in sorted(set(scorer.slices))}
    md.append(
        f"- **Eval set:** the same {n_eval} drafts for every point ({n_pos} `personalised_recommendation`), "
        f"made of " + ", ".join(f"{v} {k}" for k, v in split_counts.items()) + ". By draft type: "
        + ", ".join(f"{k} {v[0]} ({v[1]} personalised)" for k, v in slice_counts.items()) + ".")
    md.append("- **Scoring:** one forward pass per draft. The detector's JSON always starts `{\"label\": \"`, and "
              "the two labels differ at their first token, so the higher of those two logits is the label greedy "
              "schema-constrained decoding would pick in vLLM. The two-way softmax gives a probability, used for "
              "ROC-AUC. Threshold 0.5 for everything else. F1, recall and precision are for "
              "`personalised_recommendation`, the class the guardrail must not miss.")
    md.append(f"- **Error bars:** 95% bootstrap intervals over the eval drafts ({B:,} resamples, the same resamples "
              "for every point, so changes between points get paired intervals). They show how much a score "
              "depends on which drafts are in the eval set. They do **not** include training noise: each size is "
              "one run with one seed, so the real uncertainty is larger. Per-type accuracies use Wilson intervals. "
              "Val loss has no interval.")
    md.append(f"- **Training:** QLoRA r=16 on Gemma 4 E4B, {epochs} epochs, cosine schedule, peak learning rate "
              f"{cfg['sft']['learning_rate']}, 8 rows per step. Subsets are stratified by label and nested "
              f"(150 ⊂ 300 ⊂ 600 ⊂ 900 ⊂ {train_rows:,}), with the same hyperparameters, so steps scale with rows.")
    md.append("- **Test split caveat:** the eval set includes the test split, so this report is for understanding, "
              "not for choosing. The adapter that ships is the full run's final checkpoint, and the ship-or-don't "
              "gate stays `scripts/eval_advice.py` on test.\n")

    md.append("## How many labelled rows?\n")
    if len(points) >= 2:
        figure_data_size(points, FIGURES / "data_size.png")
        md.append("![Scores against training rows](figures/data_size.png)\n")
        rows = []
        for i, pt in enumerate(points):
            m, c = pt["m"], pt["run"]["curve"][pt["step"]]
            if i == 0:
                dstr = "—"
            else:
                d, lo, hi = scorer.delta(points[i - 1]["probs"], pt["probs"], "f1")
                dstr = f"{d:+.3f} ({lo:+.2f} to {hi:+.2f})"
            rows.append([pt["label"], str(pt["step"]), ci(m["f1"]), ci(m["recall"]), ci(m["precision"]),
                         ci(m["balanced_accuracy"]), ci(m["auc"]), num(c.get("eval_loss")), dstr])
        md.append(table(["Training rows", "Final step", "F1", "Recall", "Precision", "Balanced acc.", "ROC-AUC",
                         "Val loss", "ΔF1 vs row above"], rows) + "\n")
        last, prev = points[-1], points[-2]
        d, lo, hi = scorer.delta(prev["probs"], last["probs"], "f1")
        verdict = ("the interval includes zero, so this eval cannot tell the two apart"
                   if lo <= 0 <= hi else "the interval excludes zero")
        md.append(f"Reading: F1 goes from {points[0]['m']['f1']['value']:.3f} at {points[0]['label']} rows to "
                  f"{last['m']['f1']['value']:.3f} at {last['label']}. The last step, {prev['label']} → "
                  f"{last['label']} rows, changes F1 by {d:+.3f} (95% CI {lo:+.3f} to {hi:+.3f}); {verdict}. "
                  "Remember the intervals leave out training noise.\n")
    else:
        md.append("_Not enough finished points yet: this section fills in as subset runs finish._\n")

    md.append("## Full run: scores and loss during training\n")
    figure_training(full, scorer, train_rows, epochs, FIGURES / "training_curve.png")
    md.append("![Full-run training curve](figures/training_curve.png)\n")
    rows = []
    for s in full["steps"]:
        c, m = full["curve"][s], scorer.summary(full["probs"][s])
        rows.append([str(s), f"{c['rows_seen']:,}", num(c.get("epoch"), ".2f"), num(c.get("lr"), ".2e"),
                     num(c.get("train_loss")), num(c.get("eval_loss")), ci(m["f1"]), ci(m["recall"]),
                     ci(m["precision"]), ci(m["auc"])])
    md.append(table(["Step", "Rows seen", "Epoch", "LR", "Train loss", "Val loss", "F1", "Recall", "Precision",
                     "ROC-AUC"], rows) + "\n")
    evals = [(s, full["curve"][s]["eval_loss"]) for s in full["steps"] if full["curve"][s].get("eval_loss") is not None]
    if evals:
        best_s, best_l = min(evals, key=lambda t: t[1])
        last_s, last_l = evals[-1]
        best_f1 = max(full["steps"][1:] or full["steps"], key=lambda s: scorer.summary(full["probs"][s])["f1"]["value"])
        md.append(f"Reading: val loss is lowest at step {best_s} ({best_l:.3f}); latest is {last_l:.3f} at step "
                  f"{last_s}. F1 is highest at step {best_f1} "
                  f"({scorer.summary(full['probs'][best_f1])['f1']['value']:.3f}), but look at the intervals before "
                  "reading a peak into it. Beyond the first epoch boundary the model is re-reading rows it has "
                  "already seen.\n")

    if subsets:
        md.append("## Subset runs, per epoch\n")
        rows = []
        for n, run in subsets.items():
            for s in run["steps"]:
                c, m = run["curve"][s], scorer.summary(run["probs"][s])
                rows.append([f"{n:,}", num(c.get("epoch"), ".1f"), str(s), num(c.get("train_loss")),
                             num(c.get("eval_loss")), ci(m["f1"]), ci(m["recall"]), ci(m["precision"]), ci(m["auc"])])
        md.append(table(["Rows", "Epoch", "Step", "Train loss", "Val loss", "F1", "Recall", "Precision", "ROC-AUC"],
                        rows) + "\n")

    if points:
        md.append("## Accuracy by draft type (final checkpoints)\n")
        md.append("Accuracy with a 95% Wilson interval. The counterfactual types are rewrites built to break "
                  "surface shortcuts; a model that learned templates scores well on `original` and badly on them. "
                  "Groups are small, so read direction, not decimals.\n")
        rows = []
        groups = [("slice", s) for s in sorted(set(scorer.slices))] + [("split", s) for s in sorted(set(scorer.splits))]
        for kind, name in groups:
            mask = (scorer.slices if kind == "slice" else scorer.splits) == name
            row = [f"{name}" if kind == "slice" else f"_{name} split_", str(int(mask.sum()))]
            for pt in points:
                correct = ((pt["probs"] > 0.5) == scorer.gold)[mask]
                lo, hi = wilson(int(correct.sum()), int(mask.sum()))
                row.append(f"{correct.mean():.2f} ({lo:.2f}–{hi:.2f})")
            rows.append(row)
        md.append(table(["Draft type", "n"] + [pt["label"] for pt in points], rows) + "\n")

    if BASE_VLLM.exists():
        md.append("## Reference: base model through vLLM\n")
        md.append("The earlier zero-shot run of `scripts/eval_advice.py`: full generation through vLLM, test split "
                  "only. It is not directly comparable with the first-token scoring above.\n")
        md.append("```\n" + BASE_VLLM.read_text().strip() + "\n```\n")

    md.append("## Caveats\n")
    md.append("- One training run per size. The bootstrap intervals leave out seed-to-seed variance, which is "
              "largest for the small subsets.")
    md.append("- Every size trains for 3 epochs, so small subsets get far fewer optimizer steps. That is part of "
              "what \"fewer rows\" means here, not a separate effect.")
    md.append("- Subsets are nested prefixes of one stratified shuffle. A different shuffle would move the small "
              "points more than the large ones.")
    md.append("- Val and test both come from the question-grouped split, and the counterfactual rewrites stay "
              "with their source question, so no eval draft's question appears in training.\n")

    md.append("## Files\n")
    md.append("- Full run: `outputs/gemma4-e4b-advice/` (`curve.jsonl`, `probs.jsonl`, `eval_set.json`, checkpoints)")
    md.append("- Subsets: `outputs/subsets/n*/`, configs in `configs/subsets/`, data in `data/processed/subsets/`")
    md.append("- Scripts: `scripts/eval_checkpoints.py` (scoring), `scripts/run_subsets.py` (queue), "
              "`scripts/learning_report.py` (this report)")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(md) + "\n")
    print(f"report -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
