"""Plot the advice adapter's learning curve from <run-dir>/curve.jsonl.

    uv run --with matplotlib python scripts/plot_curve.py

Three panels on one shared x-axis (training rows seen = step x effective batch),
never a second y-scale on one plot:
  1. precision / recall / F1 on personalised_recommendation, on the fixed eval set
  2. train loss vs val loss
  3. learning rate
Hairlines mark epoch boundaries: left of the first one every row is new, so that
stretch reads as "how many rows"; past it the model is re-reading rows it has seen.
The same numbers print as a table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

# Reference palette (dataviz skill): chart chrome + categorical slots 1-3, in order.
SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "train_advice.yaml"))
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out", default=None, help="PNG path (default <run-dir>/curve.png)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_dir = Path(args.run_dir or ROOT / cfg["sft"]["output_dir"])
    points = sorted((json.loads(l) for l in (run_dir / "curve.jsonl").read_text().splitlines() if l.strip()),
                    key=lambda p: p["step"])
    if not points:
        raise SystemExit("curve.jsonl is empty — nothing scored yet")
    train_rows = sum(1 for l in (ROOT / cfg["data"]["train_file"]).read_text().splitlines() if l.strip())
    epochs = cfg["sft"].get("num_train_epochs", 1)

    fmt = lambda v, spec=".3f": "—" if v is None else format(v, spec)
    print(f"{'step':>5} {'rows':>6} {'epoch':>5} {'lr':>9} {'train':>6} {'val':>6} "
          f"{'f1':>6} {'recall':>6} {'prec':>6} {'bacc':>6} {'auc':>6}")
    for p in points:
        print(f"{p['step']:>5} {p['rows_seen']:>6} {fmt(p.get('epoch'), '.2f'):>5} {fmt(p.get('lr'), '.2e'):>9} "
              f"{fmt(p.get('train_loss')):>6} {fmt(p.get('eval_loss')):>6} {fmt(p['f1']):>6} "
              f"{fmt(p['recall']):>6} {fmt(p['precision']):>6} {fmt(p['balanced_accuracy']):>6} {fmt(p.get('auc')):>6}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["system-ui", "Segoe UI", "DejaVu Sans", "sans-serif"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 10.5), sharex=True, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [3, 2, 1.3], "hspace": 0.35})
    x = [p["rows_seen"] for p in points]
    last = points[-1]
    fig.suptitle("Advice detector learning curve", x=0.08, ha="left", color=INK, fontsize=14, fontweight="semibold")
    fig.text(0.08, 0.935, f"Eval: {last['n']} fixed drafts ({last['positives']} personalised) · "
                          f"0 rows = base model, zero-shot · {train_rows} training rows per epoch",
             color=SECONDARY, fontsize=9.5)

    def style(ax, title: str) -> None:
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
        for e in range(1, epochs):
            ax.axvline(e * train_rows, color=AXIS, linewidth=1, zorder=0)

    def line(ax, key: str, color: str, label: str, end_label: bool = False) -> None:
        pts = [(p["rows_seen"], p.get(key)) for p in points if p.get(key) is not None]
        if not pts:
            return
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, linewidth=2, solid_joinstyle="round", solid_capstyle="round",
                marker="o", markersize=6, markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=2,
                label=label, zorder=3)
        if end_label:
            ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                        color=SECONDARY, fontsize=9, va="center")

    ax = axes[0]
    style(ax, "Personalised recommendation: F1, recall, precision")
    line(ax, "f1", SERIES[0], "F1", end_label=True)
    line(ax, "recall", SERIES[1], "Recall")
    line(ax, "precision", SERIES[2], "Precision")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=SECONDARY, ncols=3)
    for e in range(1, epochs):
        ax.text(e * train_rows, 1.04, f" epoch {e + 1}", color=MUTED, fontsize=8.5, va="top")

    ax = axes[1]
    style(ax, "Loss")
    line(ax, "train_loss", SERIES[0], "Train loss")
    line(ax, "eval_loss", SERIES[1], "Val loss", end_label=True)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=SECONDARY, ncols=2)

    ax = axes[2]
    style(ax, "Learning rate")
    line(ax, "lr", SERIES[0], "Learning rate")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_color(MUTED)
    ax.set_xlabel("Training rows seen (step × effective batch)", color=SECONDARY, fontsize=9.5)
    ax.set_xlim(left=-train_rows * 0.02, right=max(x) * 1.06 if max(x) else 1)

    out = Path(args.out) if args.out else run_dir / "curve.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"\nplot -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
