"""
Profesjonalny skrypt ewaluacyjny dla strategii chunkowania.
- Ocena na poziomie plików (collapse_to_files), a nie pojedynczych chunków.
- Przedziały ufności 95% (Cluster Bootstrap).
- Statystyki rozmiaru bazy (liczba chunków, średnia długość).
- Rozbicie na typy zapytań (Lexical, Semantic, Colloquial).
- Testowanie statystycznej istotności różnic względem kodu produkcyjnego.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

katalog_skryptu = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(katalog_skryptu, '..')))
load_dotenv(os.path.join(katalog_skryptu, '..', '.env'))

import chromadb
from chromadb.utils import embedding_functions
from app.core.config import settings

# Mockowanie Elasticsearch (dla chunkowania nie potrzebujemy ES, ale build_database go wczytuje)
from unittest.mock import MagicMock
sys.modules['elasticsearch'] = MagicMock()
sys.modules['elasticsearch.helpers'] = MagicMock()

import build_database

# =====================================================================
# 1. KONFIGURACJA
# =====================================================================

EVAL_DATA_DIR = r"C:\Users\micro\Desktop\Bachelors\RAG\data\Ewaluacja_Probka"
DATASET_PATH = os.path.join(SCRIPT_DIR := os.path.dirname(__file__), "test_dataset_balanced.json")
OUT_ROOT = os.path.join(SCRIPT_DIR, "chunk_eva_stats")

EMBED_MODEL = "text-embedding-3-small"
K_VALUES = (1, 3, 5, 10)
K_MAX = max(K_VALUES)
FETCH_POOL = 40  # Pobieramy 40 wektorów, żeby po redukcji duplikatów zostało min 10 plików
N_BOOTSTRAP = 2000
SEED = 42

# =====================================================================
# 2. STRUKTURY DANYCH
# =====================================================================

@dataclass
class Query:
    text: str
    expected_file: str
    qtype: str

@dataclass
class QueryResult:
    strategy: str
    query: str
    qtype: str
    expected_file: str
    rank: int
    top_files: List[str]

@dataclass
class ChunkingStats:
    total_chunks: int
    avg_chunk_length: float

def load_dataset() -> List[Query]:
    if not os.path.exists(DATASET_PATH):
        sys.exit(f"Brak zbioru danych: {DATASET_PATH}. Uruchom najpierw generator pytań.")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    files = {f for f in os.listdir(EVAL_DATA_DIR) if f.endswith(".md")}
    missing = sorted({r["expected_file"] for r in raw if r["expected_file"] not in files})
    if missing:
        print("Brak plików referencyjnych w folderze ewaluacyjnym:")
        for m in missing:
            print("   -", m)
        sys.exit(1)

    return [Query(r["query"], r["expected_file"], r.get("type", "unknown")) for r in raw]

# =====================================================================
# 3. STRATEGIE CHUNKOWANIA
# =====================================================================

def chunk_fixed_size(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def chunk_by_paragraphs(text: str, max_chunk_size: int = 2500) -> List[str]:
    raw_paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    for p in raw_paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(current_chunk) + len(p) <= max_chunk_size:
            current_chunk += "\n\n" + p if current_chunk else p
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = p
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def chunk_advanced_production(filepath: str) -> List[Dict[str, Any]]:
    cfg_mock = {
        "sciezka": filepath,
        "zrodlo": "Zbior_Ewaluacyjny",
        "typ": "dokument",
        "naglowki_prawne": True,
        "pogrubienia_jako_naglowki": True
    }
    dok = build_database.przygotuj_dokument(filepath, cfg_mock)
    if not dok:
        return []

    chunki_produkcyjne = build_database.chunkuj_plik(filepath, cfg_mock, dok)
    
    processed_chunks = []
    for cid, c, md in chunki_produkcyjne:
        processed_chunks.append({
            "text": c.tekst_emb,
            "metadata": md
        })
    return processed_chunks

# =====================================================================
# 4. SILNIK EWALUACYJNY
# =====================================================================

def build_and_evaluate_strategy(
    strategy_name: str,
    queries: List[Query],
    chroma_client: chromadb.ClientAPI,
    embedding_fn: Any
) -> Tuple[List[QueryResult], ChunkingStats]:

    collection_name = f"eval_{strategy_name.lower().replace(' ', '_')}"
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

    print(f"   [+] Przetwarzanie plików dla: {strategy_name}...")
    for filename in os.listdir(EVAL_DATA_DIR):
        filepath = os.path.join(EVAL_DATA_DIR, filename)
        if not os.path.isfile(filepath) or not filename.endswith(".md"):
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        base_meta = {"source_file": filename}

        if strategy_name == "Fixed_Size":
            for c in chunk_fixed_size(content):
                doc_ids.append(f"doc_{counter}")
                documents.append(c)
                metadatas.append(base_meta)
                counter += 1
        elif strategy_name == "Paragraphs":
            for c in chunk_by_paragraphs(content):
                doc_ids.append(f"doc_{counter}")
                documents.append(c)
                metadatas.append(base_meta)
                counter += 1
        elif strategy_name == "Production_Advanced":
            for c in chunk_advanced_production(filepath):
                doc_ids.append(f"doc_{counter}")
                documents.append(c["text"])
                c["metadata"]["source_file"] = filename 
                metadatas.append(c["metadata"])
                counter += 1

    total_len = sum(len(d) for d in documents)
    stats = ChunkingStats(
        total_chunks=len(documents),
        avg_chunk_length=total_len / len(documents) if documents else 0
    )

    print(f"   [+] Ładowanie {len(documents)} wektorów do bazy...")
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        collection.add(
            ids=doc_ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )

    print("   [+] Sprawdzanie zapytań (Vector Search)...")
    results = []
    
    for q in queries:
        res = collection.query(query_texts=[q.text], n_results=min(FETCH_POOL, len(documents)))
        retrieved_metas = res["metadatas"][0] if res["metadatas"] else []
        
        # Redukcja chunków do unikalnych plików (jak w RAGu podrzucającym linki)
        seen, files = set(), []
        for m in retrieved_metas:
            sf = m.get("source_file", "")
            if sf and sf not in seen:
                seen.add(sf)
                files.append(sf)
                if len(files) == K_MAX:
                    break
        
        rank = files.index(q.expected_file) + 1 if q.expected_file in files else 0
        results.append(QueryResult(strategy_name, q.text, q.qtype, q.expected_file, rank, files[:3]))

    return results, stats

# =====================================================================
# 5. METRYKI I STATYSTYKI (Zapożyczone z evaluate_search.py)
# =====================================================================

def per_query_scores(results: List[QueryResult]) -> Dict[str, np.ndarray]:
    ranks = np.array([r.rank for r in results])
    safe = np.maximum(ranks, 1)
    found = ranks > 0
    out = {f"Hit@{k}": (found & (ranks <= k)).astype(float) for k in K_VALUES}
    out["MRR@10"] = np.where(found, 1.0 / safe, 0.0)
    out["nDCG@10"] = np.where(found, 1.0 / np.log2(safe + 1), 0.0)
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

def paired_diff(a: np.ndarray, b: np.ndarray, samples: List[np.ndarray]) -> Dict[str, float]:
    diffs = np.array([(a[s] - b[s]).mean() for s in samples])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float((a - b).mean()), "ci_low": float(lo), "ci_high": float(hi)}

def summarize(results_by_strat: Dict[str, List[QueryResult]], stats_by_strat: Dict[str, ChunkingStats], 
              queries: List[Query], samples: List[np.ndarray]):
    types = sorted({q.qtype for q in queries})
    scores = {name: per_query_scores(res) for name, res in results_by_strat.items()}
    summary: Dict[str, Any] = {}

    for name, res in results_by_strat.items():
        entry: Dict[str, Any] = {"n_queries": len(res)}
        for metric, vals in scores[name].items():
            entry[metric] = mean_ci(vals, samples)

        entry["chunk_stats"] = {
            "total_chunks": stats_by_strat[name].total_chunks,
            "avg_len": stats_by_strat[name].avg_chunk_length
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

# =====================================================================
# 6. RAPORTOWANIE
# =====================================================================

def fmt_ci(m: Dict[str, float]) -> str:
    return f"{m['mean']:.3f} [{m['ci_low']:.2f}-{m['ci_high']:.2f}]"

def build_report(summary, types, paired, baseline, n_queries, n_files) -> str:
    names = list(summary.keys())
    w = max(len(n) for n in names) + 2
    L = []
    bar = "=" * 120

    L += ["", bar, f"EWALUACJA STRATEGII CHUNKOWANIA | {n_queries} zapytań, {n_files} dokumentów | K={K_MAX}", bar]
    L.append(f"{'Strategia':<{w}}| {'Hit@1':<7}| {'Hit@3':<7}| {'Hit@5':<7}| {'MRR@10':<8}| {'nDCG@10':<8}| {'Chunks':<8}| {'Avg_Len':<8}")
    L.append("-" * 120)
    for n in names:
        s = summary[n]
        L.append(f"{n:<{w}}| {s['Hit@1']['mean']:<7.3f}| {s['Hit@3']['mean']:<7.3f}| "
                 f"{s['Hit@5']['mean']:<7.3f}| {s['MRR@10']['mean']:<8.4f}| "
                 f"{s['nDCG@10']['mean']:<8.4f}| {s['chunk_stats']['total_chunks']:<8}| {s['chunk_stats']['avg_len']:<8.0f}")

    L += ["", "95% PRZEDZIAŁY UFNOŚCI (Cluster Bootstrap po plikach)", "-" * 120]
    L.append(f"{'Strategia':<{w}}| {'Hit@1':<20}| {'Hit@5':<20}| {'MRR@10':<20}")
    for n in names:
        s = summary[n]
        L.append(f"{n:<{w}}| {fmt_ci(s['Hit@1']):<20}| {fmt_ci(s['Hit@5']):<20}| {fmt_ci(s['MRR@10']):<20}")

    if paired:
        L += ["", f"RÓŻNICE WZGLĘDEM BAZY '{baseline}' (Wartość dodatnia = poprawa)", "-" * 120]
        L.append(f"{'Strategia':<{w}}| {'dHit@1 [95% CI]':<26}| {'dMRR@10 [95% CI]':<26}")
        for n, d in paired.items():
            a, b = d["Hit@1"], d["MRR@10"]
            L.append(f"{n:<{w}}| {a['diff']:+.3f} [{a['ci_low']:+.2f},{a['ci_high']:+.2f}]".ljust(w + 28)
                     + f"| {b['diff']:+.3f} [{b['ci_low']:+.2f},{b['ci_high']:+.2f}]")

    L += ["", "ROZBICIE NA TYPY ZAPYTAŃ (Hit@1 / MRR@10)", "-" * 120]
    header = f"{'Strategia':<{w}}" + "".join(
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
    ax.set_ylabel("Wartość metryki (Błąd: 95% CI)")
    ax.set_title("Ewaluacja Strategii Chunkowania (Dokumenty)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

# =====================================================================
# 7. GŁÓWNY BLOK URUCHOMIENIOWY
# =====================================================================

def main():
    queries = load_dataset()
    n_files = len({q.expected_file for q in queries})

    chroma_client = chromadb.Client() 
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=settings.OPENAI_API_KEY,
        model_name=EMBED_MODEL
    )

    strategies = ["Fixed_Size", "Paragraphs", "Production_Advanced"]
    results_by_strat, stats_by_strat = {}, {}

    print("\n🔍 ROZPOCZYNAM EWALUACJĘ STRATEGII CHUNKOWANIA\n")
    for strat in strategies:
        print(f"-> Strategia: {strat}")
        res, stats = build_and_evaluate_strategy(strat, queries, chroma_client, openai_ef)
        results_by_strat[strat] = res
        stats_by_strat[strat] = stats

    samples = make_bootstrap_samples(queries, N_BOOTSTRAP, SEED)
    summary, scores, types = summarize(results_by_strat, stats_by_strat, queries, samples)

    baseline = "Production_Advanced"
    paired = {}
    for n in results_by_strat:
        if n == baseline: continue
        paired[n] = {m: paired_diff(scores[n][m], scores[baseline][m], samples) for m in ("Hit@1", "MRR@10")}

    report = build_report(summary, types, paired, baseline, len(queries), n_files)
    print(report)

    out_dir = os.path.join(OUT_ROOT, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    with open(os.path.join(out_dir, "per_query.csv"), "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["strategy", "type", "query", "expected_file", "rank", "top3_files"])
        for res_list in results_by_strat.values():
            for r in res_list:
                wr.writerow([r.strategy, r.qtype, r.query, r.expected_file, r.rank, " | ".join(r.top_files)])

    plot_metrics(summary, os.path.join(out_dir, "wykres_chunkowania.png"))
    print(f"\n✅ Zapisano raport, wykres i pliki CSV w folderze: {out_dir}")

if __name__ == "__main__":
    main()