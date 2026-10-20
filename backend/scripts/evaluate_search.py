"""
Ostateczny skrypt ewaluacyjny systemu RAG dla AGH.
Zawiera: 
- ewaluację na poziomie plików (nie chunków),
- przedziały ufności (cluster bootstrap),
- wymuszony Cosine Similarity w ChromaDB,
- rozdzielone i rzetelne pomiary czasu (Retrieval vs LLM),
- solidny cache uwzględniający parametry bazy i modeli.
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
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
from openai import OpenAI
from sentence_transformers import CrossEncoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..")))
load_dotenv(os.path.join(SCRIPT_DIR, "..", ".env"))

import chromadb
from chromadb.utils import embedding_functions
from app.core.config import settings
import build_database

# =====================================================================
# 1. KONFIGURACJA
# =====================================================================

EVAL_DATA_DIR = r"C:\Users\micro\Desktop\Bachelors\RAG\data\Ewaluacja_Probka"
DATASET_PATH = os.path.join(SCRIPT_DIR, "test_dataset_balanced.json")
CACHE_DIR = os.path.join(SCRIPT_DIR, ".eval_cache")
OUT_ROOT = os.path.join(SCRIPT_DIR, "search_eva_stats")

EMBED_MODEL = "text-embedding-3-small"
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
REWRITE_MODEL = "gpt-4o-mini"
COLLECTION_NAME = "eval_search_modality"
ES_INDEX = "eval_search_index"

K_VALUES = (1, 3, 5, 10)
K_MAX = max(K_VALUES)
CHUNK_POOL = 30      
RERANK_POOL = 30     
RRF_K = 60
N_BOOTSTRAP = 2000
SEED = 42

REWRITE_PROMPT = """Twoim zadaniem jest optymalizacja zapytania do systemu wyszukiwania dokumentów na uczelni AGH (Elasticsearch + Vector DB).
Zastosuj technikę Query Expansion. BEZWZGLĘDNIE zachowaj oryginalny tekst zapytania!
Dołącz oryginalne zapytanie, a po spacji dopisz 3-5 oficjalnych, urzędowych słów kluczowych lub synonimów, które z nim korespondują (np. jeśli padnie "stypa", dopisz "stypendium, fundusz, pomoc materialna", jeśli "wuef" dopisz "wychowanie fizyczne, SWFiS, zaliczenie").

Oryginalne zapytanie: {query}

Zwróć TYLKO jeden ciąg znaków w formacie:
[oryginalne zapytanie] [słowo_kluczowe_1] [słowo_kluczowe_2] [słowo_kluczowe_3]
"""

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
    modality: str
    query: str
    qtype: str
    expected_file: str
    rank: int                 
    top_files: List[str]
    retrieval_ms: float
    rewrite_ms: float
    pool_hit: Optional[bool]  

@dataclass
class Ctx:
    collection: Any
    es: Elasticsearch
    doc_store: Dict[str, Dict[str, str]]
    reranker: Optional[CrossEncoder]

def load_dataset() -> List[Query]:
    if not os.path.exists(DATASET_PATH):
        sys.exit(f"Brak zbioru danych: {DATASET_PATH}. Uruchom najpierw generator.")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    files = {f for f in os.listdir(EVAL_DATA_DIR) if f.endswith(".md")}
    missing = sorted({r["expected_file"] for r in raw if r["expected_file"] not in files})
    if missing:
        print("Brak plików referencyjnych (sprawdź literówki w nazwach):")
        for m in missing:
            print("   -", m)
        sys.exit(1)

    return [Query(r["query"], r["expected_file"], r.get("type", "unknown")) for r in raw]


# =====================================================================
# 3. INDEKSOWANIE (Z CACHE)
# =====================================================================

def corpus_fingerprint() -> str:
    h = hashlib.sha256()
    for fn in sorted(os.listdir(EVAL_DATA_DIR)):
        path = os.path.join(EVAL_DATA_DIR, fn)
        if fn.endswith(".md") and os.path.isfile(path):
            h.update(fn.encode("utf-8"))
            with open(path, "rb") as f:
                h.update(f.read())
    with open(build_database.__file__, "rb") as f:
        h.update(f.read())
    
    # PEŁNY FINGERPRINT Z USTAWIENIAMI (odporność na zmiany konfiguracji)
    config_str = f"embed:{EMBED_MODEL}|metric:cosine|reranker:{RERANKER_MODEL}|rrf:{RRF_K}"
    h.update(config_str.encode("utf-8"))
    return h.hexdigest()

def chunk_file(filepath: str) -> List[Dict[str, Any]]:
    cfg = {
        "sciezka": filepath,
        "zrodlo": "Zbior_Ewaluacyjny",
        "typ": "dokument",
        "naglowki_prawne": True,
        "pogrubienia_jako_naglowki": True,
    }
    dok = build_database.przygotuj_dokument(filepath, cfg)
    if not dok:
        return []
    return [{"text": c.tekst_emb, "metadata": md}
            for _cid, c, md in build_database.chunkuj_plik(filepath, cfg, dok)]

def build_or_load_index(chroma_client, embedding_fn, es: Elasticsearch, force: bool):
    fp_path = os.path.join(CACHE_DIR, "fingerprint.txt")
    fp = corpus_fingerprint()
    cached_fp = None
    if os.path.exists(fp_path):
        with open(fp_path, "r", encoding="utf-8") as f:
            cached_fp = f.read().strip()

    if not force and cached_fp == fp and es.indices.exists(index=ES_INDEX):
        try:
            collection = chroma_client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)
            if collection.count() > 0:
                data = collection.get(include=["documents", "metadatas"])
                doc_store = {i: {"text": d, "source_file": m["source_file"]}
                             for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])}
                print(f"   [=] Wczytano z cache ({len(doc_store)} chunków). Użyj --rebuild aby wymusić ponowne liczenie.")
                return collection, doc_store
        except Exception:
            pass

    print("   [+] Chunkowanie plików ewaluacyjnych...")
    doc_ids, documents, metadatas, doc_store = [], [], [], {}
    for filename in sorted(os.listdir(EVAL_DATA_DIR)):
        filepath = os.path.join(EVAL_DATA_DIR, filename)
        if not os.path.isfile(filepath) or not filename.endswith(".md"):
            continue
        for c in chunk_file(filepath):
            d_id = f"doc_{len(doc_ids)}"
            c["metadata"]["source_file"] = filename
            doc_ids.append(d_id)
            documents.append(c["text"])
            metadatas.append(c["metadata"])
            doc_store[d_id] = {"text": c["text"], "source_file": filename}
    if not documents:
        sys.exit("Chunker nie wyprodukował danych. Sprawdź pliki.")

    print(f"   [+] Wrzucanie {len(documents)} chunków do ChromaDB i Elasticsearch...")
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
        
    # Wymuszenie metryki Cosine w bazie wektorowej
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME, 
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    for i in range(0, len(documents), 100):
        collection.add(ids=doc_ids[i:i + 100],
                       documents=documents[i:i + 100],
                       metadatas=metadatas[i:i + 100])

    if es.indices.exists(index=ES_INDEX):
        es.indices.delete(index=ES_INDEX)
    es.indices.create(index=ES_INDEX, body={
        "mappings": {"properties": {
            "text": {"type": "text", "analyzer": "polish"},
            "source_file": {"type": "keyword"},
        }}
    })
    helpers.bulk(es, [
        {"_index": ES_INDEX, "_id": d_id,
         "_source": {"text": doc_store[d_id]["text"], "source_file": doc_store[d_id]["source_file"]}}
        for d_id in doc_ids
    ])
    es.indices.refresh(index=ES_INDEX)

    with open(fp_path, "w", encoding="utf-8") as f:
        f.write(fp)
    return collection, doc_store


# =====================================================================
# 4. QUERY EXPANSION (CACHE)
# =====================================================================

class QueryRewriter:
    def __init__(self, client: OpenAI):
        self.client = client
        self.path = os.path.join(CACHE_DIR, "rewrites.json")
        self.salt = hashlib.sha256((REWRITE_MODEL + REWRITE_PROMPT).encode()).hexdigest()[:12]
        self.cache: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.cache = json.load(f)

    def get(self, query: str) -> Tuple[str, float]:
        key = f"{self.salt}|{query}"
        if key not in self.cache:
            t0 = time.perf_counter()
            resp = self.client.chat.completions.create(
                model=REWRITE_MODEL,
                messages=[{"role": "user", "content": REWRITE_PROMPT.replace("{query}", query)}],
                temperature=0,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            text = resp.choices[0].message.content.strip()
            if not text.lower().startswith(query.lower()):
                text = f"{query} {text}"
            self.cache[key] = {"text": text, "latency_ms": latency_ms}
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=1)
        entry = self.cache[key]
        return entry["text"], entry["latency_ms"]


# =====================================================================
# 5. STRATEGIE WYSZUKIWANIA
# =====================================================================

def vector_search(ctx: Ctx, query: str, k: int) -> List[Dict]:
    res = ctx.collection.query(query_texts=[query], n_results=k)
    return [{"id": i, "source_file": (m or {}).get("source_file", "")}
            for i, m in zip(res["ids"][0], res["metadatas"][0])]

def keyword_search(ctx: Ctx, query: str, k: int) -> List[Dict]:
    resp = ctx.es.search(index=ES_INDEX, body={"query": {"match": {"text": query}}, "size": k})
    return [{"id": h["_id"], "source_file": h["_source"].get("source_file", "")}
            for h in resp["hits"]["hits"]]

def rrf_fuse(rankings: List[List[Dict]], k: int, rrf_k: int = RRF_K) -> List[Dict]:
    scores: Dict[str, float] = defaultdict(float)
    meta: Dict[str, str] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item["id"]] += 1.0 / (rrf_k + rank)
            meta[item["id"]] = item["source_file"]
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [{"id": i, "source_file": meta[i]} for i, _ in ordered]

def hybrid_search(ctx: Ctx, query: str, k: int) -> List[Dict]:
    return rrf_fuse([vector_search(ctx, query, CHUNK_POOL),
                     keyword_search(ctx, query, CHUNK_POOL)], k)

def rerank(ctx: Ctx, query: str, candidates: List[Dict]) -> List[Dict]:
    pairs = [[query, ctx.doc_store[c["id"]]["text"]] for c in candidates]
    scores = ctx.reranker.predict(pairs, batch_size=16, show_progress_bar=False)
    scored = [{**c, "rerank_score": float(s)} for c, s in zip(candidates, scores)]
    return sorted(scored, key=lambda x: x["rerank_score"], reverse=True)

def collapse_to_files(hits: List[Dict], k: int) -> List[str]:
    seen, out = set(), []
    for h in hits:
        f = h["source_file"]
        if f not in seen:
            seen.add(f)
            out.append(f)
            if len(out) == k:
                break
    return out

def m_vector(ctx, q, sq):
    return vector_search(ctx, sq, CHUNK_POOL), {}

def m_keyword(ctx, q, sq):
    return keyword_search(ctx, sq, CHUNK_POOL), {}

def m_hybrid(ctx, q, sq):
    return hybrid_search(ctx, sq, CHUNK_POOL), {}

def m_rerank(ctx, q, sq):
    cands = hybrid_search(ctx, sq, RERANK_POOL)
    return rerank(ctx, q, cands), {"pool_files": [c["source_file"] for c in cands]}

@dataclass
class Modality:
    key: str
    name: str
    fn: Callable
    use_rewrite: bool = False
    needs_reranker: bool = False

MODALITIES = [
    Modality("vector", "Vector (ChromaDB)", m_vector),
    Modality("keyword", "Keyword (Elasticsearch)", m_keyword),
    Modality("hybrid", "Hybrid (RRF)", m_hybrid),
    Modality("hybrid_exp", "Hybrid + Expansion", m_hybrid, use_rewrite=True),
    Modality("hybrid_rerank", "Hybrid + Re-Rank", m_rerank, needs_reranker=True),
    Modality("full", "Hybrid + Exp + Re-Rank", m_rerank, use_rewrite=True, needs_reranker=True),
]

# =====================================================================
# 6. EWALUACJA
# =====================================================================

def run_modality(mod: Modality, queries: List[Query], ctx: Ctx, rewriter: QueryRewriter) -> List[QueryResult]:
    q0 = queries[0].text
    mod.fn(ctx, q0, rewriter.get(q0)[0] if mod.use_rewrite else q0)

    results = []
    for i, q in enumerate(queries, start=1):
        rewrite_ms, search_q = 0.0, q.text
        if mod.use_rewrite:
            search_q, rewrite_ms = rewriter.get(q.text)

        t0 = time.perf_counter()
        hits, info = mod.fn(ctx, q.text, search_q)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        files = collapse_to_files(hits, K_MAX)
        rank = files.index(q.expected_file) + 1 if q.expected_file in files else 0
        pool_hit = (q.expected_file in info["pool_files"]) if "pool_files" in info else None

        results.append(QueryResult(mod.name, q.text, q.qtype, q.expected_file, rank,
                                   files[:3], retrieval_ms, rewrite_ms, pool_hit))
        if i % 10 == 0 or i == len(queries):
            print(f"      {i}/{len(queries)}", end="\r")
    print()
    return results


# =====================================================================
# 7. METRYKI I STATYSTYKA
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

def summarize(results_by_mod: Dict[str, List[QueryResult]], queries: List[Query],
              samples: List[np.ndarray]):
    types = sorted({q.qtype for q in queries})
    scores = {name: per_query_scores(res) for name, res in results_by_mod.items()}
    summary: Dict[str, Any] = {}

    for name, res in results_by_mod.items():
        entry: Dict[str, Any] = {"n_queries": len(res)}
        for metric, vals in scores[name].items():
            entry[metric] = mean_ci(vals, samples)

        retrieval = np.array([r.retrieval_ms for r in res])
        rewrite = np.array([r.rewrite_ms for r in res])
        
        entry["latency_ms"] = {
            "retrieval_mean": float(retrieval.mean()), 
            "retrieval_median": float(np.median(retrieval)),
            "retrieval_p95": float(np.percentile(retrieval, 95)), 
            "rewrite_mean": float(rewrite.mean()),
        }

        pool = [r.pool_hit for r in res if r.pool_hit is not None]
        if pool:
            entry["pool_recall"] = float(np.mean(pool))

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

def find_hard_queries(queries: List[Query], results_by_mod: Dict[str, List[QueryResult]], best: str):
    hard = []
    for i, q in enumerate(queries):
        if all(not (0 < res[i].rank <= 5) for res in results_by_mod.values()):
            hard.append((q, results_by_mod[best][i].top_files))
    return hard

# =====================================================================
# 8. RAPORTY I WYKRESY
# =====================================================================

def fmt_ci(m: Dict[str, float]) -> str:
    return f"{m['mean']:.3f} [{m['ci_low']:.2f}-{m['ci_high']:.2f}]"

def build_report(summary, types, paired, baseline, hard, best, n_queries, n_files) -> str:
    names = list(summary.keys())
    w = max(len(n) for n in names) + 2
    L: List[str] = []
    bar = "=" * 125

    L += ["", bar, f"EWALUACJA RAG  |  {n_queries} zapytań, {n_files} dokumentów | K={K_MAX}", bar]
    L.append(f"{'Modality':<{w}}| {'Hit@1':<7}| {'Hit@3':<7}| {'Hit@5':<7}| {'Hit@10':<7}| "
             f"{'MRR@10':<8}| {'nDCG@10':<8}| {'Retr_med':<8}| {'Retr_p95':<8}| {'LLM_mean':<8}")
    L.append("-" * 125)
    for n in names:
        s = summary[n]
        L.append(f"{n:<{w}}| {s['Hit@1']['mean']:<7.3f}| {s['Hit@3']['mean']:<7.3f}| "
                 f"{s['Hit@5']['mean']:<7.3f}| {s['Hit@10']['mean']:<7.3f}| {s['MRR@10']['mean']:<8.4f}| "
                 f"{s['nDCG@10']['mean']:<8.4f}| {s['latency_ms']['retrieval_median']:<8.0f}| "
                 f"{s['latency_ms']['retrieval_p95']:<8.0f}| {s['latency_ms']['rewrite_mean']:<8.0f}")
    L.append("(Retr_* = czasy wyszukiwania i rerankingu. LLM_mean = szacowany koszt przepisywania zapytań)")

    L += ["", "95% PRZEDZIAŁY UFNOŚCI (Cluster Bootstrap po plikach)", "-" * 125]
    L.append(f"{'Modality':<{w}}| {'Hit@1':<20}| {'Hit@5':<20}| {'MRR@10':<20}")
    for n in names:
        s = summary[n]
        L.append(f"{n:<{w}}| {fmt_ci(s['Hit@1']):<20}| {fmt_ci(s['Hit@5']):<20}| {fmt_ci(s['MRR@10']):<20}")

    if paired:
        L += ["", f"RÓŻNICE WZGLĘDEM BAZY '{baseline}' (Wartość dodatnia = poprawa)", "-" * 125]
        L.append(f"{'Modality':<{w}}| {'dHit@1 [95% CI]':<26}| {'dMRR@10 [95% CI]':<26}")
        for n, d in paired.items():
            a, b = d["Hit@1"], d["MRR@10"]
            L.append(f"{n:<{w}}| {a['diff']:+.3f} [{a['ci_low']:+.2f},{a['ci_high']:+.2f}]".ljust(w + 28)
                     + f"| {b['diff']:+.3f} [{b['ci_low']:+.2f},{b['ci_high']:+.2f}]")

    L += ["", "ROZBICIE NA TYPY ZAPYTAŃ (Hit@1 / MRR@10)", "-" * 125]
    header = f"{'Modality':<{w}}" + "".join(
        f"| {t + ' (n=' + str(summary[names[0]]['by_type'][t]['n']) + ')':<24}" for t in types if t in summary[names[0]]["by_type"])
    L.append(header)
    for n in names:
        row = f"{n:<{w}}"
        for t in types:
            if t in summary[n]["by_type"]:
                bt = summary[n]["by_type"][t]
                row += f"| {bt['Hit@1']:.2f} / {bt['MRR@10']:.3f}".ljust(26)
        L.append(row)

    pools = {n: s["pool_recall"] for n, s in summary.items() if "pool_recall" in s}
    if pools:
        L += ["", f"PULA KANDYDATÓW RERANKERA (Top {RERANK_POOL} z Hybrydy) - Górny limit skuteczności re-rankingu", "-" * 125]
        for n, v in pools.items():
            L.append(f"{n:<{w}}| pool recall = {v:.3f}  vs  Hit@10 po re-rankingu = {summary[n]['Hit@10']['mean']:.3f}")

    L += ["", f"ZAPYTANIA POMINIĘTE (poza Top 5) PRZEZ WSZYSTKIE METODY: {len(hard)}", "-" * 125]
    for q, top in hard[:30]:
        L.append(f"[{q.qtype}] {q.text[:90]}")
        L.append(f"     oczekiwane: {q.expected_file}")
        L.append(f"     najlepsze Top-3 z '{best}': {' | '.join(top)}")
    L.append(bar)
    return "\n".join(L)

def plot_overall(summary, path):
    names = list(summary.keys())
    metrics = ["Hit@1", "Hit@5", "MRR@10"]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 7))
    for j, m in enumerate(metrics):
        means = np.array([summary[n][m]["mean"] for n in names])
        lo = np.array([summary[n][m]["ci_low"] for n in names])
        hi = np.array([summary[n][m]["ci_high"] for n in names])
        bars = ax.bar(x + (j - 1) * width, means, width, label=m,
                      yerr=[np.clip(means - lo, 0, None), np.clip(hi - means, 0, None)], capsize=3)
        ax.bar_label(bars, fmt="%.2f", padding=14, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Wartość metryki (Błąd: 95% CI)")
    ax.set_title("Skuteczność Wyszukiwania Systemu RAG (Dokumenty)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def plot_pareto(summary, path):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for n, s in summary.items():
        ax.scatter(s["latency_ms"]["retrieval_median"], s["MRR@10"]["mean"], s=70)
        ax.annotate(n, (s["latency_ms"]["retrieval_median"], s["MRR@10"]["mean"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Mediana czasu zapytania z bazy (ms, log-scale, bez narzutu LLM)")
    ax.set_ylabel("MRR@10")
    ax.set_title("Kompromis: Precyzja vs Opóźnienie")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

# =====================================================================
# 9. GŁÓWNY BLOK URUCHOMIENIOWY
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="Wymusza re-indeksację baz z plików.")
    ap.add_argument("--only", type=str, default="", help="Filtrowanie używanych modalności")
    ap.add_argument("--baseline", type=str, default="hybrid", help="Modalność referencyjna do testów")
    args = ap.parse_args()

    if not os.path.isdir(EVAL_DATA_DIR):
        sys.exit(f"Brak folderu danych: {EVAL_DATA_DIR}")
    os.makedirs(CACHE_DIR, exist_ok=True)

    queries = load_dataset()
    n_files = len({q.expected_file for q in queries})
    type_counts = defaultdict(int)
    for q in queries:
        type_counts[q.qtype] += 1
    print(f"Dataset: {len(queries)} zapytań, {n_files} plików, typy: {dict(type_counts)}")

    selected = [m for m in MODALITIES if not args.only or m.key in args.only.split(",")]
    
    print("🔌 Łączenie z bazami danych...")
    chroma_client = chromadb.PersistentClient(path=os.path.join(CACHE_DIR, "chroma"))
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=settings.OPENAI_API_KEY, model_name=EMBED_MODEL)
    llm_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    es = Elasticsearch(getattr(settings, "ELASTICSEARCH_URL", "http://localhost:9200"))

    reranker = None
    if any(m.needs_reranker for m in selected):
        print(f"🧠 Ładowanie wielojęzycznego modelu Cross-Encoder {RERANKER_MODEL}...")
        reranker = CrossEncoder(RERANKER_MODEL, max_length=512)

    collection, doc_store = build_or_load_index(chroma_client, embedding_fn, es, args.rebuild)
    ctx = Ctx(collection, es, doc_store, reranker)
    rewriter = QueryRewriter(llm_client)

    results_by_mod: Dict[str, List[QueryResult]] = {}
    print("\n🚀 ROZPOCZĘCIE EWALUACJI METOD WYSZUKIWANIA\n")
    for mod in selected:
        print(f"-> Testuję: {mod.name}")
        results_by_mod[mod.name] = run_modality(mod, queries, ctx, rewriter)

    samples = make_bootstrap_samples(queries, N_BOOTSTRAP, SEED)
    summary, scores, types = summarize(results_by_mod, queries, samples)

    base_name = next((m.name for m in selected if m.key == args.baseline), selected[0].name)
    paired = {}
    for n in results_by_mod:
        if n == base_name:
            continue
        paired[n] = {m: paired_diff(scores[n][m], scores[base_name][m], samples)
                     for m in ("Hit@1", "MRR@10")}

    best = max(summary, key=lambda n: summary[n]["MRR@10"]["mean"])
    hard = find_hard_queries(queries, results_by_mod, best)
    report = build_report(summary, types, paired, base_name, hard, best, len(queries), n_files)
    print(report)

    out_dir = os.path.join(OUT_ROOT, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(out_dir, "per_query.csv"), "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["modality", "type", "query", "expected_file", "rank", "top3_files",
                     "retrieval_ms", "rewrite_ms", "pool_hit"])
        for res in results_by_mod.values():
            for r in res:
                wr.writerow([r.modality, r.qtype, r.query, r.expected_file, r.rank,
                             " | ".join(r.top_files), f"{r.retrieval_ms:.1f}", f"{r.rewrite_ms:.1f}",
                             "" if r.pool_hit is None else int(r.pool_hit)])

    plot_overall(summary, os.path.join(out_dir, "plot_effectiveness.png"))
    plot_pareto(summary, os.path.join(out_dir, "plot_accuracy_vs_latency.png"))
    print(f"\n✅ Zapisano raport, wykresy i pliki .csv w folderze: {out_dir}")

if __name__ == "__main__":
    main()