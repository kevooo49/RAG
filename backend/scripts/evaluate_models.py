"""
Evaluation of LLM models (Generation and Inference) in the RAG architecture.

What this script does:
-------------------
* Measures models: GPT-4o-mini (baseline), GPT-4o (premium), and Bielik (local open-source).
* Saves results (model responses + judge evaluations) in SQLite (LLM Cache),
  which allows interrupted tests to resume freely and instantly.
* Dynamically injects noise from another part of the database into 50% of the queries (variable isolation).
* Uses GPT-5.4 as the judge evaluating on a 1-5 scale:
  - Faithfulness
  - Relevance
  - Language Quality
* Calculates 95% CI (Bootstrap), p-values relative to the baseline (GPT-4o-mini)
  and generates visually consistent charts (Matplotlib) and CSV/JSON exports.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, "..", ".env"))

# =====================================================================
# 1. CONFIGURATION
# =====================================================================

INPUT_FILE = os.path.join(SCRIPT_DIR, "llm_eva_dataset.json")
CACHE_DIR = os.path.join(SCRIPT_DIR, ".eval_cache")
OUT_ROOT = os.path.join(SCRIPT_DIR, "models_eva_stats")

NOISE_PROBABILITY = 0.5
N_BOOTSTRAP = 2000
SEED = 42

C_BLUE, C_ORANGE, C_AQUA, C_PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8e44ad"
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"

random.seed(SEED)
np.random.seed(SEED)


# =====================================================================
# 2. DATA STRUCTURES AND CACHE
# =====================================================================

@dataclass
class GoldenQuery:
    query: str
    qtype: str
    expected_file: str
    context: str
    golden_answer: str

@dataclass
class LLMResult:
    model: str
    query: str
    qtype: str
    has_noise: bool
    latency: float
    response: str
    faithfulness: int
    relevance: int
    language: int
    justification: str

class EvaluationScore(BaseModel):
    faithfulness: int = Field(ge=1, le=5, description="Czy odpowiedź opiera się TYLKO na kontekście? (1-zmyśla, 5-idealnie wierna)")
    relevance: int = Field(ge=1, le=5, description="Czy odpowiedź bezpośrednio trafia w pytanie? (1-nie na temat, 5-w punkt)")
    language_quality: int = Field(ge=1, le=5, description="Jakość języka polskiego, styl i poprawność. (1-słaba/maszynowa, 5-naturalna/płynna)")
    justification: str = Field(description="Krótkie uzasadnienie ocen (1-2 zdania).")

class LLMCache:
    """Persistent cache for generation and evaluation."""
    def __init__(self, cache_path: str):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        self.db = sqlite3.connect(cache_path)
        self.db.execute("CREATE TABLE IF NOT EXISTS gen (k TEXT PRIMARY KEY, response TEXT, latency REAL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS eval (k TEXT PRIMARY KEY, f INT, r INT, l INT, just TEXT)")
        self.db.commit()

    def get_gen(self, key: str) -> Optional[Tuple[str, float]]:
        cur = self.db.execute("SELECT response, latency FROM gen WHERE k = ?", (key,))
        row = cur.fetchone()
        return row if row else None

    def put_gen(self, key: str, response: str, latency: float):
        self.db.execute("INSERT OR REPLACE INTO gen VALUES (?, ?, ?)", (key, response, latency))
        self.db.commit()

    def get_eval(self, key: str) -> Optional[EvaluationScore]:
        cur = self.db.execute("SELECT f, r, l, just FROM eval WHERE k = ?", (key,))
        row = cur.fetchone()
        return EvaluationScore(faithfulness=row[0], relevance=row[1], language_quality=row[2], justification=row[3]) if row else None

    def put_eval(self, key: str, s: EvaluationScore):
        self.db.execute("INSERT OR REPLACE INTO eval VALUES (?, ?, ?, ?, ?)", 
                        (key, s.faithfulness, s.relevance, s.language_quality, s.justification))
        self.db.commit()

def dict_hash(d: str) -> str:
    return hashlib.sha256(d.encode("utf-8")).hexdigest()


# =====================================================================
# 3. STATISTICS AND BOOTSTRAP
# =====================================================================

def make_bootstrap_samples(n: int, n_boot: int) -> List[np.ndarray]:
    rng = np.random.default_rng(SEED)
    return [rng.integers(0, n, n) for _ in range(n_boot)]

def mean_ci(values: np.ndarray, samples: List[np.ndarray]) -> Dict[str, float]:
    means = np.array([values[s].mean() for s in samples])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(values.mean()), "ci_low": float(lo), "ci_high": float(hi)}

def paired_diff(a: np.ndarray, b: np.ndarray, samples: List[np.ndarray]) -> Dict[str, float]:
    diffs = np.array([(a[s] - b[s]).mean() for s in samples])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {"diff": float((a - b).mean()), "ci_low": float(lo), "ci_high": float(hi), "p": float(p)}


# =====================================================================
# 4. MAIN LOGIC
# =====================================================================

def inject_noise(all_contexts: List[str], current_context: str) -> Tuple[str, bool]:
    if random.random() > NOISE_PROBABILITY:
        return current_context, False
    noise = random.choice(all_contexts)
    while noise == current_context:
        noise = random.choice(all_contexts)
    return f"{current_context}\n\n[DODATKOWE INFORMACJE Z INNEGO DOKUMENTU]:\n{noise[:400]}...", True


def run_evaluation(baseline_model: str):
    if not os.path.exists(INPUT_FILE):
        sys.exit(f"Missing dataset: {INPUT_FILE}. Run the sample generator first.")
        
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    queries = [
        GoldenQuery(
            query=d["query"],
            qtype=d.get("type", "unknown"),
            expected_file=d["expected_file"],
            context=d["context"],
            golden_answer=d["golden_answer"]
        ) for d in raw_data
    ]
    all_contexts = [q.context for q in queries]
    cache = LLMCache(os.path.join(CACHE_DIR, "llm_eval.sqlite"))
    
    models = {
        "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini", temperature=0, max_retries=2),
        "gpt-4o": ChatOpenAI(model="gpt-4o", temperature=0, max_retries=2),
        "bielik-11b": ChatOllama(model="SpeakLeash/bielik-11b-v3.0-instruct:Q4_K_M", temperature=0)
    }
    judge_llm = ChatOpenAI(model="gpt-5.4", temperature=0, max_retries=3).with_structured_output(EvaluationScore)

    print(f"\nSTARTING LLM MODELS EVALUATION | Queries: {len(queries)}")
    results_by_model = {name: [] for name in models.keys()}

    for i, q in enumerate(queries, 1):
        print(f"[{i:02d}/{len(queries)}] {q.qtype.upper():<10} | {q.query[:50]}...")
        test_context, has_noise = inject_noise(all_contexts, q.context)
        
        for m_name, llm in models.items():
            prompt = f"Odpowiedz krótko i bezpośrednio na pytanie bazując WYŁĄCZNIE na poniższym tekście.\n\nTEKST:\n{test_context}\n\nPYTANIE:\n{q.query}"
            
            gen_key = dict_hash(f"{m_name}_{prompt}")
            gen_cached = cache.get_gen(gen_key)
            if gen_cached:
                response, latency = gen_cached
            else:
                t0 = time.perf_counter()
                try:
                    res = llm.invoke([HumanMessage(content=prompt)])
                    response, latency = res.content.strip(), time.perf_counter() - t0
                except Exception as e:
                    response, latency = f"ERROR: {e}", 0.0
                cache.put_gen(gen_key, response, latency)

            eval_key = dict_hash(f"{q.query}_{q.golden_answer}_{response}")
            score = cache.get_eval(eval_key)
            if not score:
                eval_prompt = f"Oceń odpowiedź w skali 1-5.\nPYTANIE: {q.query}\nWZORZEC: {q.golden_answer}\nODPOWIEDŹ: {response}"
                try:
                    score = judge_llm.invoke([SystemMessage(content="Jesteś sędzią."), HumanMessage(content=eval_prompt)])
                except Exception:
                    score = EvaluationScore(faithfulness=1, relevance=1, language_quality=1, justification="Błąd")
                cache.put_eval(eval_key, score)

            results_by_model[m_name].append(LLMResult(
                m_name, q.query, q.qtype, has_noise, latency, response,
                score.faithfulness, score.relevance, score.language_quality, score.justification
            ))

    samples = make_bootstrap_samples(len(queries), N_BOOTSTRAP)
    summary = {}
    
    for m_name, res_list in results_by_model.items():
        faith_arr = np.array([r.faithfulness for r in res_list])
        relev_arr = np.array([r.relevance for r in res_list])
        lang_arr = np.array([r.language for r in res_list])
        lat_arr = np.array([r.latency for r in res_list])
        
        noise_idx = np.array([i for i, r in enumerate(res_list) if r.has_noise])
        clean_idx = np.array([i for i, r in enumerate(res_list) if not r.has_noise])
        
        summary[m_name] = {
            "Faithfulness": mean_ci(faith_arr, samples),
            "Relevance": mean_ci(relev_arr, samples),
            "Language": mean_ci(lang_arr, samples),
            "Latency": {"mean": float(lat_arr.mean()), "median": float(np.median(lat_arr))},
            "arrays": {"f": faith_arr, "r": relev_arr, "l": lang_arr},
            "noise_drop": {
                "clean_faith": float(faith_arr[clean_idx].mean()),
                "noise_faith": float(faith_arr[noise_idx].mean())
            }
        }

    paired = {}
    if baseline_model in summary:
        base_arrs = summary[baseline_model]["arrays"]
        for m_name, s in summary.items():
            if m_name != baseline_model:
                paired[m_name] = {
                    "Faith": paired_diff(s["arrays"]["f"], base_arrs["f"], samples),
                    "Relev": paired_diff(s["arrays"]["r"], base_arrs["r"], samples),
                }

    _generate_report_and_plots(summary, paired, results_by_model, baseline_model, len(queries))

# =====================================================================
# 5. REPORTING AND PLOTTING
# =====================================================================

def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    for side in ("left", "bottom"): ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)

def plot_metrics(summary, path):
    names = list(summary.keys())
    metrics = [("Faithfulness", C_BLUE), ("Relevance", C_ORANGE), ("Language", C_AQUA)]
    x = np.arange(len(names))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor=SURFACE)
    _style_axes(ax)
    
    for j, (m, color) in enumerate(metrics):
        means = np.array([summary[n][m]["mean"] for n in names])
        lo = np.array([summary[n][m]["ci_low"] for n in names])
        hi = np.array([summary[n][m]["ci_high"] for n in names])
        ax.bar(x + (j - 1) * width, means, width - 0.03, label=m, color=color,
               yerr=[np.clip(means - lo, 0, None), np.clip(hi - means, 0, None)],
               capsize=2, error_kw={"ecolor": INK_2, "elinewidth": 1})
        for xi, mean_v in zip(x + (j - 1) * width, means):
            ax.text(xi, 0.2, f"{mean_v:.2f}", ha="center", va="bottom", fontsize=8, color="white", weight="bold")
            
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=0)
    ax.set_ylim(1, 5.1)
    ax.set_ylabel("Ocena Sędziego (1-5, słupki: 95% CI)")
    ax.set_title("Jakość generacji LLM w systemie RAG", loc="left")
    ax.legend(loc="lower right", frameon=False, ncol=3, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(fig)

def plot_latency(summary, path):
    names = list(summary.keys())
    y = np.array([summary[n]["Faithfulness"]["mean"] for n in names])
    x = np.array([summary[n]["Latency"]["mean"] for n in names])
    
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    _style_axes(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    
    colors = [C_ORANGE if "4o" in n else C_PURPLE for n in names]
    ax.scatter(x, y, s=100, color=colors, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    
    for i, n in enumerate(names):
        ax.annotate(n, (x[i], y[i]), textcoords="offset points", xytext=(8, -5), fontsize=9, color=INK_2)
        
    ax.set_xlabel("Średni czas generacji (sekundy)")
    ax.set_ylabel("Wierność faktom (Faithfulness)")
    ax.set_title("Szybkość a Wierność", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(fig)

def _generate_report_and_plots(summary, paired, raw_results, baseline, n_queries):
    names = list(summary.keys())
    w = max(len(n) for n in names) + 2
    bar = "=" * 120
    
    L = ["", bar, f"EWALUACJA MODELI LLM | {n_queries} zapytań | Sędzia: GPT-4o (1-5)", bar]
    
    L.append(f"{'Model':<{w}}| {'Faithfulness':<15}| {'Relevance':<15}| {'Language':<15}| {'Latency (s)':<12}")
    L.append("-" * 120)
    for n in names:
        s = summary[n]
        L.append(f"{n:<{w}}| {s['Faithfulness']['mean']:<15.2f}| {s['Relevance']['mean']:<15.2f}| "
                 f"{s['Language']['mean']:<15.2f}| {s['Latency']['mean']:<12.2f}")

    L += ["", "ODPORNOŚĆ NA SZUM (Spadek wierności przy błędnym kontekście)", "-" * 120]
    L.append(f"{'Model':<{w}}| {'Czysty kontekst':<18}| {'Zaszumiony':<15}| {'Spadek':<10}")
    for n in names:
        nd = summary[n]["noise_drop"]
        L.append(f"{n:<{w}}| {nd['clean_faith']:<18.2f}| {nd['noise_faith']:<15.2f}| {nd['clean_faith']-nd['noise_faith']:<10.2f}")

    if paired:
        L += ["", f"ISTOTNOŚĆ STATYSTYCZNA vs {baseline.upper()} (dodatnie = lepsze od bazy)", "-" * 120]
        L.append(f"{'Model':<{w}}| {'dFaithfulness [95% CI]':<28}| {'p-val':<8}| {'dRelevance [95% CI]':<28}| {'p-val':<8}")
        for n, p_data in paired.items():
            f, r = p_data["Faith"], p_data["Relev"]
            L.append(f"{n:<{w}}| {f['diff']:+.2f} [{f['ci_low']:+.2f},{f['ci_high']:+.2f}]".ljust(30) + 
                     f"| {f['p']:<8.3f}| {r['diff']:+.2f} [{r['ci_low']:+.2f},{r['ci_high']:+.2f}]".ljust(30) + f"| {r['p']:<8.3f}")

    L.append(bar)
    report_text = "\n".join(L)
    print(report_text)

    out_dir = os.path.join(OUT_ROOT, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)
        
    with open(os.path.join(out_dir, "per_query.csv"), "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "query", "type", "has_noise", "latency", "faithfulness", "relevance", "language", "response", "judge_note"])
        for m_name, res_list in raw_results.items():
            for r in res_list:
                wr.writerow([r.model, r.query, r.qtype, r.has_noise, round(r.latency, 2), r.faithfulness, r.relevance, r.language, r.response, r.justification])

    plot_metrics(summary, os.path.join(out_dir, "wykres_jakosc_llm.png"))
    plot_latency(summary, os.path.join(out_dir, "wykres_latency_vs_faithfulness.png"))
    print(f"\nSaved report, CSV, and plots in folder: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="gpt-4o-mini", help="Reference model for statistical comparisons.")
    args = parser.parse_args()
    run_evaluation(args.baseline)