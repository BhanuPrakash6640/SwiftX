"""
Collect research results into one table + charts.

    python -m research.plots          -> experiments/results/research/{summary.json, *.png}
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from benchmarks.plots import GRID, INK, MUTED, SURFACE  # noqa: E402  (shared style)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results")
OUT = os.path.join(RES, "research")
COL = {"sift": "#2a78d6", "sift_big": "#86b6ef", "transformer": "#eb6834", "hybrid": "#1baf7a", "siftx": "#6250d6", "siftx_cf": "#a3479b"}
NAMES = {"sift": "Sift (Mamba-1)", "sift_big": "Sift, 5 layers", "transformer": "Transformer",
         "hybrid": "Simple hybrid", "siftx": "SiftX",
         "sift_mt": "Sift + timescales", "siftx_no_mt": "SiftX − timescales", "siftx_surprise": "SiftX, surprise write",
         "siftx_recent": "SiftX, recent-only write", "siftx_random": "SiftX, random write", "siftx_always": "SiftX, gate always on",
         "siftx_cf": "SiftX, counterfactual gate"}


def summary(name):
    p = os.path.join(RES, name, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def metrics(name):
    p = os.path.join(RES, name, "metrics.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None


def style(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=160)
    plt.close(fig)
    print("wrote", name)


def main():
    # Final extras (streaming + Triton benchmarks, counterfactual analysis, website memory viewer).
    # Hooked here so the already-running scripts/run_everything.bat picks them up at its last step.
    try:
        from research.extras import main as extras
        extras()
    except Exception as e:                       # never let extras block the plots
        print("[extras] skipped:", e)
    table = {"mqar": {}, "needle": {}, "lm": {}}
    # ---- MQAR ----
    rows = []
    for key in ["sift", "sift_big", "transformer", "hybrid", "siftx", "siftx_cf", "sift_mt", "siftx_no_mt",
                "siftx_surprise", "siftx_recent", "siftx_random", "siftx_always"]:
        seeds = [s for s in (summary(f"mqar_{key}_s{i}") for i in (0, 1)) if s]
        if not seeds:
            continue
        accs = [s["final_val_metric"] for s in seeds]
        rate = [s.get("final_retrieval_rate") for s in seeds if s.get("final_retrieval_rate") is not None]
        table["mqar"][key] = {"accuracy_mean": sum(accs) / len(accs), "accuracy_seeds": accs,
                              "params": seeds[0]["params"], "retrieval_rate": (sum(rate) / len(rate)) if rate else None,
                              "train_time_s": seeds[0]["train_time_s"]}
        rows.append((key, accs))
    if rows:
        fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.2))
        for i, (k, accs) in enumerate(rows[::-1]):
            m = sum(accs) / len(accs)
            ax.barh(i, m * 100, color=COL.get(k, "#9aa7b2"), height=0.6)
            for a in accs:
                ax.plot(a * 100, i, "o", color=INK, markersize=3)
            ax.annotate(f"{m * 100:.1f}%", (m * 100, i), xytext=(5, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
        ax.set_yticks(range(len(rows)), [NAMES[k] for k, _ in rows[::-1]], color=INK)
        ax.set_xlim(0, 112)
        style(ax, "Associative recall (MQAR, 32 pairs in 256 tokens)", "query accuracy (%), dots = seeds")
        ax.grid(axis="y", visible=False)
        save(fig, "mqar_accuracy.png")

    # ---- Needle ----
    an = os.path.join(OUT, "analysis.json")
    analysis = json.load(open(an)) if os.path.exists(an) else {}
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    found = False
    for k in ("sift", "transformer", "hybrid", "siftx"):
        res = analysis.get(f"needle_{k}", {}).get("needle_accuracy_by_length")
        if not res:
            continue
        pts = [(int(L), v) for L, v in res.items() if v is not None and v == v]
        table["needle"][k] = res
        if pts:
            found = True
            xs, ys = zip(*pts)
            ax.plot(xs, [y * 100 for y in ys], marker="o", color=COL[k], label=NAMES[k], linewidth=2)
    if found:
        ax.axvline(512, color=MUTED, linestyle=":", linewidth=1)
        ax.annotate("training length", (512, 4), xytext=(4, 0), textcoords="offset points", color=MUTED, fontsize=8)
        ax.set_xscale("log", base=2)
        ax.set_ylim(-3, 105)
        style(ax, "Needle retrieval as the haystack grows", "sequence length (tokens)", "accuracy (%)")
        ax.legend(frameon=False)
        save(fig, "needle_by_length.png")
    else:
        plt.close(fig)

    # ---- LM ----
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    found = False
    for k in ("sift", "transformer", "hybrid", "siftx", "siftx_cf"):
        rows = metrics(f"lm_{k}")
        s = summary(f"lm_{k}")
        if not rows or not s:
            continue
        found = True
        table["lm"][k] = {"val_loss": s["final_val_loss"], "perplexity": s.get("val_perplexity"),
                          "params": s["params"], "train_time_s": s["train_time_s"],
                          "retrieval_rate": s.get("final_retrieval_rate"),
                          "analysis": analysis.get(f"lm_{k}")}
        ax.plot([r["step"] for r in rows], [r["val_loss"] for r in rows], color=COL[k], label=NAMES[k], linewidth=2)
    if found:
        style(ax, "Tiny Shakespeare: validation loss", "training step", "loss (nats/char)")
        ax.legend(frameon=False)
        save(fig, "lm_val_loss.png")
    else:
        plt.close(fig)

    # ---- efficiency ----
    for dev in ("cuda", "cpu"):
        p = os.path.join(OUT, f"bench_{dev}.json")
        if not os.path.exists(p):
            continue
        b = json.load(open(p))
        table[f"bench_{dev}"] = b
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for k, d in b["forward_ms_batch2"].items():
            pts = [(int(L), v) for L, v in d.items() if v == v]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", color=COL[k], label=NAMES[k], linewidth=2)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        style(ax, f"Forward pass time vs length ({dev}, batch 2)", "sequence length", "ms")
        ax.legend(frameon=False)
        save(fig, f"efficiency_{dev}.png")

    table["analysis"] = analysis
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(table, f, indent=2)
    print(json.dumps({k: v for k, v in table.items() if k in ("mqar", "lm")}, indent=2))


if __name__ == "__main__":
    main()
