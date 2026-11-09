"""
Evaluation of Embeddings Models for the RAG system.
The script isolates vector search and measures:
- Index building time (cost of embedding the entire corpus).
- Inference time (latency of querying the model + searching in ChromaDB).
- Precision (Hit@K, MRR@10) with confidence intervals.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

script_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
load_dotenv(os.path.join(script_dir, '..', '.env'))

import chromadb
from chromadb.utils import embedding_functions
from app.core.config import settings

from unittest.mock import MagicMock
sys.modules['elasticsearch'] = MagicMock()
sys.modules['elasticsearch.helpers'] = MagicMock()

import build_database

# =====================================================================
# 1. CONFIGURATION
# =====================================================================

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
EVAL_DATA_DIR = r"C:\Users\micro\Desktop\Bachelors\RAG\data\Ewaluacja_Probka"
from sites import load_or_exit  # noqa: E402 # type: ignore
EVAL_DATA_DIR = str(load_or_exit().eval_dir)
DATASET_PATH = os.path.join(SCRIPT_DIR := os.path.dirname(__file__), "test_dataset_balanced.json")
CACHE_DIR = os.path.join(SCRIPT_DIR, ".eval_cache")
OUT_ROOT = os.path.join(SCRIPT_DIR, "embed_eva_stats")

MODELS = {
    "OAI-Small": {"provider": "openai", "name": "text-embedding-3-small"},
    "OAI-Large": {"provider": "openai", "name": "text-embedding-3-large"},
    "BGE-m3": {"provider": "local", "name": "BAAI/bge-m3"},
    "PL-Retriever": {"provider": "local", "name": "ipipan/silver-retriever-base-v1"}
}

K_VALUES = (1, 3, 5, 10)
K_MAX = max(K_VALUES)
FETCH_POOL = 40  
N_BOOTSTRAP = 2000
SEED = 42

# =====================================================================
# 2. DATA STRUCTURES
# =====================================================================

@dataclass
class Query:
    text: str
    expected_file: str
    qtype: str

@dataclass
class QueryResult:
    model_name: str
    query: str
    qtype: str
    expected_file: str
    rank: int
    top_files: List[str]
    query_latency_ms: float

def load_dataset() -> List[Query]:
    if not os.path.exists(DATASET_PATH):
        sys.exit(f"Dataset missing: {DATASET_PATH}. Run the question generator first.")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    files = {f for f in os.listdir(EVAL_DATA_DIR) if f.endswith(".md")}
    missing = sorted({r["expected_file"] for r in raw if r["expected_file"] not in files})
    if missing:
        print("Missing reference files (check for typos in file names):")
        for m in missing:
            print("   -", m)
        sys.exit(1)

    return [Query(r["query"], r["expected_file"], r.get("type", "unknown")) for r in raw]

# =====================================================================
# 3. PREPARING THE DATABASE AND MEASURING BUILD TIME
# =====================================================================

def chunk_advanced_production(filepath: str) -> List[Dict[str, Any]]:
    cfg_mock = {
        "sciezka": filepath,
        "zrodlo": "Zbior_Ewaluacyjny",
        "typ": "dokument",
        "naglowki_prawne": True,
        "pogrubienia_jako_naglowki": True
    }
    doc = build_database.prepare_document(filepath, cfg_mock)
    if not doc:
        return []
    return [{"text": c.text_emb, "metadata": md} for _cid, c, md in build_database.chunk_file(filepath, cfg_mock, doc)]

def corpus_fingerprint(model_name: str) -> str:
    h = hashlib.sha256()
    for fn in sorted(os.listdir(EVAL_DATA_DIR)):
        path = os.path.join(EVAL_DATA_DIR, fn)
        if fn.endswith(".md") and os.path.isfile(path):
            h.update(fn.encode("utf-8"))
            with open(path, "rb") as f:
                h.update(f.read())
    with open(build_database.__file__, "rb") as f:
        h.update(f.read())
    h.update(f"model:{model_name}|metric:cosine".encode("utf-8"))
    return h.hexdigest()

def get_embedding_function(model_info: Dict[str, str]):
    if model_info["provider"] == "openai":
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.OPENAI_API_KEY, 
            model_name=model_info["name"]
        )
    elif model_info["provider"] == "local":
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_info["name"]
        )
    raise ValueError("Unknown model provider")

def load_or_build_collection(chroma_client, model_key: str, model_info: Dict[str, str], force: bool) -> Tuple[Any, float]:
    fp_path = os.path.join(CACHE_DIR, f"fp_{model_key.lower()}.txt")
    fp = corpus_fingerprint(model_info["name"])
    
    times_file = os.path.join(CACHE_DIR, "build_times.json")
    build_times = {}
    if os.path.exists(times_file):
        with open(times_file, "r") as f:
            build_times = json.load(f)

    cached_fp = None
    if os.path.exists(fp_path):
        with open(fp_path, "r", encoding="utf-8") as f:
            cached_fp = f.read().strip()

    collection_name = f"eval_emb_{model_key.lower().replace('-', '_')}"
    embedding_fn = get_embedding_function(model_info)

    if not force and cached_fp == fp and model_key in build_times:
        try:
            collection = chroma_client.get_collection(collection_name, embedding_function=embedding_fn)
            if collection.count() > 0:
                print(f"   [=] Loaded from cache for {model_key} (Build Time: {build_times[model_key]:.2f}s).")
                return collection, build_times[model_key]
        except Exception:
            pass

    print(f"   [+] Building index from scratch for {model_key} (measuring time)...")
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass
        
    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    doc_ids, documents, metadatas = [], [], []
    counter = 0
    for filename in sorted(os.listdir(EVAL_DATA_DIR)):
        filepath = os.path.join(EVAL_DATA_DIR, filename)
        if not os.path.isfile(filepath) or not filename.endswith(".md"):
            continue
        for c in chunk_advanced_production(filepath):
            doc_ids.append(f"doc_{counter}")
            documents.append(c["text"])
            c["metadata"]["source_file"] = filename
            metadatas.append(c["metadata"])
            counter += 1

    t0 = time.perf_counter()
    batch_size = 16
    for i in range(0, len(documents), batch_size):
        collection.add(
            ids=doc_ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )
        print(f"      Vectorization progress: {min(i+batch_size, len(documents))}/{len(documents)}", end="\r")
    print()
    build_time = time.perf_counter() - t0

    build_times[model_key] = build_time
    with open(times_file, "w") as f:
        json.dump(build_times, f)
    with open(fp_path, "w", encoding="utf-8") as f:
        f.write(fp)
        
    return collection, build_time

# =====================================================================
# 4. EVALUATION ENGINE
# =====================================================================

def evaluate_model(model_key: str, queries: List[Query], collection: Any) -> List[QueryResult]:
    results = []
    
    # Warm-up (we do not measure the first shot)
    collection.query(query_texts=[queries[0].text], n_results=1)
    
    for i, q in enumerate(queries, start=1):
        t0 = time.perf_counter()
        res = collection.query(query_texts=[q.text], n_results=min(FETCH_POOL, collection.count()))
        latency_ms = (time.perf_counter() - t0) * 1000

        retrieved_metas = res["metadatas"][0] if res["metadatas"] else []
        
        seen, files = set(), []
        for m in retrieved_metas:
            sf = m.get("source_file", "")
            if sf and sf not in seen:
                seen.add(sf)
                files.append(sf)
                if len(files) == K_MAX:
                    break
        
        rank = files.index(q.expected_file) + 1 if q.expected_file in files else 0
        results.append(QueryResult(model_key, q.text, q.qtype, q.expected_file, rank, files[:3], latency_ms))
        
        if i % 10 == 0 or i == len(queries):
            print(f"      {i}/{len(queries)}", end="\r")
    print()
    return results

# =====================================================================
# 5. METRICS (BOOTSTRAP AND SUMMARY)
# =====================================================================

def per_query_scores(results: List[QueryResult]) -> Dict[str, np.ndarray]:
    ranks = np.array([r.rank for r in results])
    safe = np.maximum(ranks, 1)
    found = ranks > 0
    out = {f"Hit@{k}": (found & (ranks <= k)).astype(float) for k in K_VALUES}
    out["MRR@10"] = np.where(found, 1.0 / safe, 0.0)
    return out

def make_bootstrap_samples(queries: List[Query], n_boot: int, seed: int) -> List[np.ndarray]:
    by_file: Dict[str, List[int]] = defaultdict(list)
    for i, q in enumerate(queries):
        by_file[q.expected_file].append(i)
    groups = [np.array(v) for v in by_file.values()]
    rng = np.random.default_rng(seed)
    return [np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
            for _ in range(n_boot)]

def mean_ci(values: np.ndarray, samples: List[np.ndarray]) -> Dict[str, float]:
    means = np.array([values[s].mean() for s in samples])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(values.mean()), "ci_low": float(lo), "ci_high": float(hi)}

def summarize(results_by_mod: Dict[str, List[QueryResult]], build_times: Dict[str, float], queries: List[Query], samples: List[np.ndarray]):
    types = sorted({q.qtype for q in queries})
    scores = {name: per_query_scores(res) for name, res in results_by_mod.items()}
    summary: Dict[str, Any] = {}

    for name, res in results_by_mod.items():
        entry: Dict[str, Any] = {"n_queries": len(res), "build_time_s": build_times[name]}
        for metric, vals in scores[name].items():
            entry[metric] = mean_ci(vals, samples)

        latencies = np.array([r.query_latency_ms for r in res])
        entry["latency_ms"] = {
            "mean": float(latencies.mean()), 
            "median": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95))
        }

        by_type = {}
        for t in types:
            idx = np.array([i for i, r in enumerate(res) if r.qtype == t])
            if len(idx) > 0:
                by_type[t] = {"n": int(len(idx)),
                              "Hit@1": float(scores[name]["Hit@1"][idx].mean()),
                              "Hit@5": float(scores[name]["Hit@5"][idx].mean()),
                              "MRR@10": float(scores[name]["MRR@10"][idx].mean())}
        entry["by_type"] = by_type
        summary[name] = entry

    return summary, scores, types

def fmt_ci(m: Dict[str, float]) -> str:
    return f"{m['mean']:.3f} [{m['ci_low']:.2f}-{m['ci_high']:.2f}]"

# =====================================================================
# 6. GENERATING REPORTS AND PLOTS
# =====================================================================

def build_report(summary, types, n_queries, n_files) -> str:
    names = list(summary.keys())
    w = max(len(n) for n in names) + 2
    L = []
    bar = "=" * 125

    L += ["", bar, f"EWALUACJA MODELI EMBEDDINGOWYCH | {n_queries} zapytań, {n_files} dokumentów | K={K_MAX}", bar]
    L.append(f"{'Model':<{w}}| {'Hit@1':<7}| {'Hit@3':<7}| {'Hit@5':<7}| {'MRR@10':<8}| {'Mediana ms':<12}| {'Czas Budowy (s)':<15}")
    L.append("-" * 125)
    for n in names:
        s = summary[n]
        L.append(f"{n:<{w}}| {s['Hit@1']['mean']:<7.3f}| {s['Hit@3']['mean']:<7.3f}| "
                 f"{s['Hit@5']['mean']:<7.3f}| {s['MRR@10']['mean']:<8.4f}| "
                 f"{s['latency_ms']['median']:<12.1f}| {s['build_time_s']:<15.2f}")

    L += ["", "95% PRZEDZIAŁY UFNOŚCI (Cluster Bootstrap)", "-" * 125]
    L.append(f"{'Model':<{w}}| {'Hit@1':<20}| {'Hit@5':<20}| {'MRR@10':<20}")
    for n in names:
        s = summary[n]
        L.append(f"{n:<{w}}| {fmt_ci(s['Hit@1']):<20}| {fmt_ci(s['Hit@5']):<20}| {fmt_ci(s['MRR@10']):<20}")

    L += ["", "ROZBICIE NA TYPY ZAPYTAŃ (Hit@1 / MRR@10)", "-" * 125]
    header = f"{'Model':<{w}}" + "".join(
        f"| {t + ' (n=' + str(summary[names[0]]['by_type'][t]['n']) + ')':<24}" for t in types if t in summary[names[0]]["by_type"])
    L.append(header)
    for n in names:
        row = f"{n:<{w}}"
        for t in types:
            if t in summary[n]["by_type"]:
                bt = summary[n]["by_type"][t]
                row += f"| {bt['Hit@1']:.2f} / {bt['MRR@10']:.3f}".ljust(26)
        L.append(row)

    L.append(bar)
    return "\n".join(L)

def plot_metrics(summary, path):
    names = list(summary.keys())
    metrics = ["Hit@1", "Hit@3", "MRR@10"]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for j, m in enumerate(metrics):
        means = np.array([summary[n][m]["mean"] for n in names])
        lo = np.array([summary[n][m]["ci_low"] for n in names])
        hi = np.array([summary[n][m]["ci_high"] for n in names])
        bars = ax.bar(x + (j - 1) * width, means, width, label=m,
                      yerr=[np.clip(means - lo, 0, None), np.clip(hi - means, 0, None)], capsize=3)
        ax.bar_label(bars, fmt="%.2f", padding=10, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Wartość metryki")
    ax.set_title("Precyzja Modeli Osadzeń (Vector Search)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def plot_tradeoffs(summary, path):
    names = list(summary.keys())
    latencies = [summary[n]["latency_ms"]["median"] for n in names]
    build_times = [summary[n]["build_time_s"] for n in names]
    
    x = np.arange(len(names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    rects1 = ax1.bar(x - width/2, latencies, width, label='Czas zapytania (ms)', color='#d62728')
    rects2 = ax2.bar(x + width/2, build_times, width, label='Czas indeksowania bazy (s)', color='#9467bd')

    ax1.set_ylabel('Czas pojedynczego zapytania [ms]', color='#d62728')
    ax2.set_ylabel('Czas wektoryzacji korpusu (s)', color='#9467bd')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_title('Koszty operacyjne i czasowe Modeli Embeddings')
    
    ax1.bar_label(rects1, padding=3, fmt='%.1f ms')
    ax2.bar_label(rects2, padding=3, fmt='%.1f s')

    fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

# =====================================================================
# 7. MAIN EXECUTION BLOCK
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="Forces fetching vectors via API again.")
    args = ap.parse_args()

    if not os.path.isdir(EVAL_DATA_DIR):
        sys.exit(f"Data folder missing: {EVAL_DATA_DIR}")
    os.makedirs(CACHE_DIR, exist_ok=True)

    queries = load_dataset()
    n_files = len({q.expected_file for q in queries})

    print("Starting ChromaDB (Persistent)...")
    chroma_client = chromadb.PersistentClient(path=os.path.join(CACHE_DIR, "chroma_embeds"))

    results_by_mod, build_times = {}, {}

    print("\nSTARTING EMBEDDINGS MODELS EVALUATION\n")
    for key, info in MODELS.items():
        print(f"-> Loading model: {key} ({info['name']})")
        collection, b_time = load_or_build_collection(chroma_client, key, info, args.rebuild)
        build_times[key] = b_time
        results_by_mod[key] = evaluate_model(key, queries, collection)

    samples = make_bootstrap_samples(queries, N_BOOTSTRAP, SEED)
    summary, scores, types = summarize(results_by_mod, build_times, queries, samples)

    report = build_report(summary, types, len(queries), n_files)
    print(report)

    out_dir = os.path.join(OUT_ROOT, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    with open(os.path.join(out_dir, "per_query.csv"), "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "type", "query", "expected_file", "rank", "latency_ms"])
        for res_list in results_by_mod.values():
            for r in res_list:
                wr.writerow([r.model_name, r.qtype, r.query, r.expected_file, r.rank, f"{r.query_latency_ms:.1f}"])

    plot_metrics(summary, os.path.join(out_dir, "wykres_precyzji.png"))
    plot_tradeoffs(summary, os.path.join(out_dir, "wykres_kosztow.png"))
    print(f"\nSaved report and plots in folder: {out_dir}")

if __name__ == "__main__":
    main()