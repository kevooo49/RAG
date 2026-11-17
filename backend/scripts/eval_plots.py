"""Plots for the evaluation runs. Called automatically by eval_common.write_run(); can also redraw the plots
of a finished run without rerunning it:

    python eval_plots.py search_eva_stats/run_20261116_232902
    python eval_plots.py embed_eva_stats/run_...  --format pdf      # vector output for the thesis

Reads only results.json and per_query.csv of the run, writes PNG (or PDF/SVG) files into the same folder:

    overall.png         MRR@10 and CtxHit@5 per system, dot + 95% CI (cluster bootstrap)
    diff.png            difference vs the baseline with 95% CI; filled marker = significant after Holm
    hit_at_k.png        Hit@1/3/5/10 per system (how fast the right file appears)
    by_type.png         MRR@10 per question type (lexical / semantic / colloquial)
    by_doc_type.png     MRR@10 per document type (strona / dokument / sylabus)
    by_position.png     MRR@10 by where the answer sits in the document (start / middle / end)
    tradeoff.png        quality vs cost: search -> MRR vs latency (retrieval + expansion);
                        embeddings -> MRR vs query latency and vs index build time
    chunk_size.png      chunking -> MRR@10 and CtxHit@B per strategy, rows sorted by chunk length

Colors: validated categorical palette (fixed slot order, never cycled), every colored mark also has a distinct
marker shape and/or a direct label, so the plots stay readable in grayscale print.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference palette (light mode), validated: first 3 slots pass all-pairs, all 8 pass adjacent pairs.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ACCENT, MUTED = SERIES[0], "#a3a29c"
K_VALUES = (1, 3, 5, 10)
GROUP_ORDER = {
    "type": ["lexical", "semantic", "colloquial"],
    "doc_type": ["strona", "dokument", "sylabus"],
    "position": ["start", "middle", "end"],
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
    "text.color": INK, "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


# ==========================================
# data loading
# ==========================================
def load_run(run: Path):
    res = json.loads((run / "results.json").read_text(encoding="utf-8"))
    rows = []
    with (run / "per_query.csv").open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            r["rank"] = int(r["rank"])
            r["ctx_hit"] = int(r["ctx_hit"])
            if r.get("position") not in (None, ""):
                p = float(r["position"])
                r["position"] = ("start", "middle", "end")[min(2, int(p * 3))]
            else:
                r.pop("position", None)
            rows.append(r)
    return res, rows


def system_order(res: dict) -> List[str]:
    """Rows sorted by MRR@10 (best on top). Colors are assigned separately, by first appearance in the run."""
    return sorted(res["systems"], key=lambda s: -res["systems"][s]["MRR@10"]["mean"])


def label(res: dict, s: str) -> str:
    return res.get("names", {}).get(s, s)


def save(fig, run: Path, name: str, fmt: str):
    path = run / f"{name}.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    return path


# ==========================================
# plots
# ==========================================
def plot_overall(res, run, fmt):
    order = system_order(res)
    base = res.get("baseline")
    metrics = [("MRR@10", "MRR@10 (file level)"), ("CtxHit@5", "CtxHit@5 (answer chunk in the 5 given to the LLM)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(order) + 1.4), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, (m, title) in zip(axes, metrics):
        for yi, s in zip(y, order):
            e = res["systems"][s][m]
            col = INK if s == base else ACCENT
            ax.plot([e["lo"], e["hi"]], [yi, yi], color=col, lw=2, solid_capstyle="round")
            ax.plot(e["mean"], yi, "o", ms=8, color=col, mec=SURFACE, mew=2)
            ax.annotate(f"{e['mean']:.3f}", (e["hi"], yi), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK2)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        lo = min(res["systems"][s][m]["lo"] for s in order)
        hi = max(res["systems"][s][m]["hi"] for s in order)
        pad = (hi - lo) * 0.12 + 0.01
        ax.set_xlim(max(0, lo - pad), min(1.0, hi + pad * 2.2))
    axes[0].set_yticks(y, [label(res, s) + ("  (baseline)" if s == base else "") for s in order])
    fig.suptitle(f"{res['n_questions']} questions, {res['n_docs']} documents - mean with 95% CI",
                 x=0.01, ha="left", fontsize=9, color=INK2, y=0.995)
    return save(fig, run, "overall", fmt)


def plot_diff(res, run, fmt):
    base = res.get("baseline")
    order = [s for s in system_order(res) if s != base and f"d_MRR@10" in res["systems"][s]]
    if not order:
        return None
    metrics = [("d_MRR@10", "Δ MRR@10"), ("d_CtxHit@5", "Δ CtxHit@5")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(order) + 1.6), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, (m, title) in zip(axes, metrics):
        ax.axvline(0, color=INK2, lw=1)
        for yi, s in zip(y, order):
            d = res["systems"][s].get(m)
            if not d:
                continue
            sig = d["p_holm"] < 0.05
            ax.plot([d["lo"], d["hi"]], [yi, yi], color=ACCENT, lw=2, solid_capstyle="round")
            ax.plot(d["diff"], yi, "o", ms=8, color=ACCENT if sig else SURFACE, mec=ACCENT, mew=2)
            ax.annotate(f"{d['diff']:+.3f}{' *' if sig else ''}", (max(d["hi"], d["diff"]), yi), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=8.5, color=INK2)
        ax.set_title(f"{title} vs {label(res, base)}")
        ax.grid(axis="y", visible=False)
        span = max(abs(res["systems"][s][m][k]) for s in order for k in ("lo", "hi") if res["systems"][s].get(m))
        ax.set_xlim(-span * 1.15 - 0.005, span * 1.45 + 0.01)
    axes[0].set_yticks(y, [label(res, s) for s in order])
    fig.suptitle("difference with 95% CI (paired bootstrap) - filled = significant after Holm correction (p < 0.05)",
                 x=0.01, ha="left", fontsize=9, color=INK2, y=0.995)
    return save(fig, run, "diff", fmt)


def colors_for(res, shown: Sequence[str]) -> Dict[str, int]:
    """Slot per shown system in the order the run evaluated them (follows the entity, never its rank).
    At most len(SERIES) systems are ever shown, so no color repeats."""
    return {s: i for i, s in enumerate(x for x in res["systems"] if x in shown)}


def pick_series(res, max_n: int = 8) -> List[str]:
    order = system_order(res)
    base = res.get("baseline")
    keep = order[:max_n]
    if base in res["systems"] and base not in keep:
        keep = keep[:-1] + [base]
    return keep


def plot_hit_at_k(res, rows, run, fmt):
    ranks = defaultdict(list)
    for r in rows:
        ranks[r["system"]].append(r["rank"])
    keep = pick_series(res)
    slot = colors_for(res, keep)
    direct = len(keep) <= 4                           # direct end labels only for few series; legend always
    fig, ax = plt.subplots(figsize=(8.5 if direct else 10, 4.6))
    ends = []
    for s in keep:
        rk = np.array(ranks[s])
        ys = [float(np.mean((rk > 0) & (rk <= k))) for k in K_VALUES]
        i = slot[s] % len(SERIES)
        ax.plot(K_VALUES, ys, color=SERIES[i], lw=2, marker=MARKERS[i], ms=8, mec=SURFACE, mew=1.5,
                label=label(res, s))
        ends.append([ys[-1], label(res, s), SERIES[i]])
    ends.sort(key=lambda e: e[0])
    gap = 0.018                                       # spread end labels so they never collide
    for j in range(1, len(ends)):
        ends[j][0] = max(ends[j][0], ends[j - 1][0] + gap)
    for yv, txt, col in (ends if direct else []):
        ax.text(10.35, yv, txt, va="center", fontsize=8.5, color=INK)
    ax.set_xticks(K_VALUES, [f"Hit@{k}" for k in K_VALUES])
    ax.set_xlim(0.6, 10.2)
    ax.set_ylabel("share of questions with a relevant file in the top k")
    ax.set_title("How fast the right document appears")
    if direct:
        ax.legend(loc="lower right", fontsize=8)
    else:
        handles, labels_ = ax.get_legend_handles_labels()
        best = {txt: yv for yv, txt, _ in ends}
        pairs = sorted(zip(handles, labels_), key=lambda hl: -best.get(hl[1], 0))   # legend in end-value order
        ax.legend([h for h, _ in pairs], [t for _, t in pairs], loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  fontsize=8.5)
    if len(res["systems"]) > len(keep):
        ax.text(0, -0.13, f"showing the {len(keep)} best systems + baseline (all in report.txt)",
                transform=ax.transAxes, fontsize=8, color=INK2)
    return save(fig, run, "hit_at_k", fmt)


def plot_groups(res, rows, run, fmt, field: str, title: str):
    if not rows or field not in rows[0]:
        return None
    rr = defaultdict(list)
    counts = defaultdict(set)
    for r in rows:
        rr[(r["system"], r[field])].append(1.0 / r["rank"] if r["rank"] else 0.0)
        counts[r[field]].add(r["qid"])
    groups = [g for g in GROUP_ORDER.get(field, []) if g in counts] + \
             sorted(g for g in counts if g not in GROUP_ORDER.get(field, []))
    if len(groups) > 3:                               # all-pairs palette is validated for 3 slots only
        groups = sorted(groups, key=lambda g: -len(counts[g]))[:3]
    order = system_order(res)
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(order) + 1.5))
    y = np.arange(len(order))[::-1]
    for gi, g in enumerate(groups):
        xs = [float(np.mean(rr[(s, g)])) if rr[(s, g)] else np.nan for s in order]
        ax.plot(xs, y, MARKERS[gi], ms=8, color=SERIES[gi], mec=SURFACE, mew=1.5, ls="none",
                label=f"{g} (n={len(counts[g])})")
    for yi, s in zip(y, order):                       # thin connector: spread between groups per system
        xs = [np.mean(rr[(s, g)]) for g in groups if rr[(s, g)]]
        if xs:
            ax.plot([min(xs), max(xs)], [yi, yi], color=GRID, lw=2, zorder=0)
    ax.set_yticks(y, [label(res, s) for s in order])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("MRR@10")
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5)
    return save(fig, run, f"by_{field}", fmt)


def _scatter(ax, pts, xlabel, logx=False):
    """pts: (label, x, mean, lo, hi). One hue; identity by direct label."""
    ys = [m for _, _, m, _, _ in pts]
    yr = (max(ys) - min(ys)) or 1.0
    placed: List[tuple] = []                          # (x, y, stack level) of labels already drawn
    for name, x, m, lo, hi in sorted(pts, key=lambda p: -p[2]):
        ax.plot([x, x], [lo, hi], color=ACCENT, lw=2, alpha=0.6, solid_capstyle="round")
        ax.plot(x, m, "o", ms=8, color=ACCENT, mec=SURFACE, mew=2)
        near = [lv for px, py, lv in placed
                if abs(py - m) < 0.06 * yr and (px / x if logx else px - x + 1) < 1.35
                and (x / px if logx else x - px + 1) < 1.35]
        level = max(near) + 1 if near else 0          # neighbours: put this label one line lower
        placed.append((x, m, level))
        ax.annotate(name, (x, m), xytext=(7, 4 - 12 * level), textcoords="offset points", fontsize=8.5,
                    color=INK)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MRR@10 (95% CI)")


def plot_chunk_sizes(res, run, fmt):
    """evaluate_chunking: one row per strategy (sorted by mean chunk length, length in the label),
    MRR@10 and CtxHit@B with 95% CI, one marker per retrieval mode."""
    S, st = res["systems"], res["strategies"]
    modes = list(dict.fromkeys(k.split(":", 1)[0] for k in S))
    strat = sorted({k.split(":", 1)[1] for k in S}, key=lambda sk: st[sk]["mean_chars"])
    panels = [(m, t) for m, t in [("MRR@10", "MRR@10"), ("CtxHit@B", "CtxHit@B (same amount of text for the LLM)")]
              if all(m in S[k] for k in S)]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.8 * len(panels), 0.5 * len(strat) + 1.6), sharey=True,
                             squeeze=False)
    y = np.arange(len(strat))[::-1]
    off = np.linspace(-0.16, 0.16, len(modes)) if len(modes) > 1 else [0.0]
    for ax, (m, title) in zip(axes[0], panels):
        for mi, mode in enumerate(modes):
            for yi, sk in zip(y, strat):
                e = S.get(f"{mode}:{sk}", {}).get(m)
                if not e:
                    continue
                yy = yi + off[mi]
                ax.plot([e["lo"], e["hi"]], [yy, yy], color=SERIES[mi], lw=2, alpha=0.55, solid_capstyle="round")
                ax.plot(e["mean"], yy, MARKERS[mi], ms=7.5, color=SERIES[mi], mec=SURFACE, mew=1.5,
                        label=mode if yi == y[0] else None)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0][0].set_yticks(y, [f"{st[sk]['label']}  ·  {st[sk]['mean_chars']:.0f} ch" for sk in strat])
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5)
    fig.suptitle("rows sorted by mean chunk length (top = smallest) - mean with 95% CI",
                 x=0.01, ha="left", fontsize=9, color=INK2, y=0.995)
    return save(fig, run, "chunk_size", fmt)


def plot_tradeoff(res, run, fmt):
    S = res["systems"]
    if res.get("no_tradeoff"):                      # all variants cost the same (e.g. metadata headers)
        return None
    if "strategies" in res:
        return plot_chunk_sizes(res, run, fmt)
    if "models" in res:                               # evaluate_embeddings
        info = res["models"]
        base_keys = [k for k in S if k in info]
        if not base_keys:
            return None
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
        pts_q = [(label(res, k), max(info[k]["query_ms_median"], 0.1), S[k]["MRR@10"]["mean"],
                  S[k]["MRR@10"]["lo"], S[k]["MRR@10"]["hi"]) for k in base_keys]
        _scatter(axes[0], pts_q, "query embedding latency, median ms (log)", logx=True)
        axes[0].set_title("Quality vs query latency")
        pts_b = [(label(res, k), info[k]["build_s"], S[k]["MRR@10"]["mean"], S[k]["MRR@10"]["lo"],
                  S[k]["MRR@10"]["hi"]) for k in base_keys if info[k].get("build_s")]
        if pts_b:
            _scatter(axes[1], pts_b, "time to embed the whole sample, s (log)", logx=True)
        axes[1].set_title("Quality vs indexing time")
        return save(fig, run, "tradeoff", fmt)
    pts = [(label(res, s), max(e["retrieval_ms_median"] + e.get("llm_ms_mean", 0), 0.1), e["MRR@10"]["mean"],
            e["MRR@10"]["lo"], e["MRR@10"]["hi"]) for s, e in S.items()]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    _scatter(ax, pts, "latency before generation, ms (retrieval median + expansion mean, log)", logx=True)
    ax.set_title("Quality vs latency")
    return save(fig, run, "tradeoff", fmt)


def plot_run(run: Path, fmt: str = "png") -> List[Path]:
    run = Path(run)
    res, rows = load_run(run)
    made = [plot_overall(res, run, fmt), plot_diff(res, run, fmt), plot_hit_at_k(res, rows, run, fmt),
            plot_groups(res, rows, run, fmt, "type", "MRR@10 by question type"),
            plot_groups(res, rows, run, fmt, "doc_type", "MRR@10 by document type"),
            plot_groups(res, rows, run, fmt, "position", "MRR@10 by where the answer is in the document"),
            plot_tradeoff(res, run, fmt)]
    return [p for p in made if p]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="run folder with results.json and per_query.csv")
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    a = ap.parse_args()
    if not (a.run / "results.json").exists():
        sys.exit(f"No results.json in {a.run}")
    for p in plot_run(a.run, a.format):
        print(f"   {p}")


if __name__ == "__main__":
    main()