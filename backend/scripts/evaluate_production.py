"""
Final Evaluation of the RAG System (Production Stack - Asymmetric Context).
Tests the integrated final pipeline:
- Chunking: custom (Production_Asymmetric) preserving table and list consistency.
- Metadata (Asymmetric): ChromaDB gets a clean 'breadcrumb', Elasticsearch gets the full header.
- Embeddings: text-embedding-3-small (Cost/Benefit optimization).
- Retrieval: Hybrid Search (ChromaDB + Elasticsearch) fused via RRF.
- LLM: Query Expansion (gpt-4o-mini) for user queries.
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
from elasticsearch import Elasticsearch, helpers
from openai import OpenAI

script_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
load_dotenv(os.path.join(script_dir, '..', '.env'))

import chromadb
from chromadb.utils import embedding_functions
from app.core.config import settings
import build_database

# =====================================================================
# 1. PRODUCTION CONFIGURATION
# =====================================================================

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
EVAL_DATA_DIR = r"C:\Users\micro\Desktop\Bachelors\RAG\data\Ewaluacja_Probka"
from sites import load_or_exit  # noqa: E402 # type: ignore
EVAL_DATA_DIR = str(load_or_exit().eval_dir)
DATASET_PATH = os.path.join(script_dir, "test_dataset_balanced.json")
CACHE_DIR = os.path.join(script_dir, ".eval_cache_production")
OUT_ROOT = os.path.join(script_dir, "production_eva_stats")

EMBED_MODEL = "text-embedding-3-small"
REWRITE_MODEL = "gpt-4o-mini"
COLLECTION_NAME = "eval_production"
ES_INDEX = "eval_production_idx"

K_VALUES = (1, 3, 5, 10)
K_MAX = max(K_VALUES)
CHUNK_POOL = 40      
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
# 2. DATA STRUCTURES
# =====================================================================

@dataclass
class Query:
    text: str
    expected_file: str
    qtype: str

@dataclass
class QueryResult:
    query: str
    qtype: str
    expected_file: str
    rank: int                 
    top_files: List[str]
    retrieval_ms: float
    rewrite_ms: float

def load_dataset() -> List[Query]:
    if not os.path.exists(DATASET_PATH):
        sys.exit(f"Missing dataset: {DATASET_PATH}.")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Query(r["query"], r["expected_file"], r.get("type", "unknown")) for r in raw]

# =====================================================================
# 3. INDEXING (WITH CACHE) AND LLM REWRITER
# =====================================================================

def corpus_fingerprint() -> str:
    h = hashlib.sha256()
    for fn in sorted(os.listdir(EVAL_DATA_DIR)):
        path = os.path.join(EVAL_DATA_DIR, fn)
        if fn.endswith(".md") and os.path.isfile(path):
            h.update(fn.encode("utf-8"))
            with open(path, "rb") as f: h.update(f.read())
    with open(build_database.__file__, "rb") as f: h.update(f.read())
    h.update(f"embed:{EMBED_MODEL}|metric:cosine|rrf:{RRF_K}|asymmetric_v1".encode("utf-8"))
    return h.hexdigest()

def chunk_file_asymmetric(filepath: str) -> List[Dict[str, Any]]:
    """Splits a file and returns two versions of text: clean breadcrumb (for vector) and full header (for ES/LLM)."""
    cfg = {"sciezka": filepath, "zrodlo": "Zbior_Ewaluacyjny", "typ": "dokument", "naglowki_prawne": True, "pogrubienia_jako_naglowki": True}
    doc = build_database.prepare_document(filepath, cfg)
    if not doc: return []
    
    chunks = []
    for _cid, c, md in build_database.chunk_file(filepath, cfg, doc):
        title = str(md.get("tytul") or "")
        section = str(md.get("sekcja") or "")
        
        bc_parts = [x for x in (title, section) if x]
        breadcrumb = " › ".join(bc_parts)
        
        parts = c.text_emb.split("\n\n", 1)
        content = parts[1] if len(parts) > 1 else c.text_emb
        
        text_vector = f"{breadcrumb}\n\n{content}" if breadcrumb else content
        text_es = c.text_emb
        
        chunks.append({
            "text_vector": text_vector,
            "text_es": text_es,
            "metadata": md
        })
    return chunks

def build_or_load_index(chroma_client, embedding_fn, es: Elasticsearch, force: bool):
    fp_path, fp = os.path.join(CACHE_DIR, "fp_prod_final.txt"), corpus_fingerprint()
    if not force and os.path.exists(fp_path) and open(fp_path).read().strip() == fp and es.indices.exists(index=ES_INDEX):
        try:
            return chroma_client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)
        except Exception: pass

    print("   [+] Initializing Asymmetric databases (ChromaDB + Elasticsearch)...")
    doc_ids, vector_docs, es_docs, metadatas = [], [], [], []
    for fn in sorted(os.listdir(EVAL_DATA_DIR)):
        filepath = os.path.join(EVAL_DATA_DIR, fn)
        if not fn.endswith(".md"): continue
        for c in chunk_file_asymmetric(filepath):
            d_id = f"doc_{len(doc_ids)}"
            c["metadata"]["source_file"] = fn
            doc_ids.append(d_id)
            vector_docs.append(c["text_vector"])
            es_docs.append(c["text_es"])
            metadatas.append(c["metadata"])

    try: chroma_client.delete_collection(COLLECTION_NAME)
    except Exception: pass
    collection = chroma_client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn, metadata={"hnsw:space": "cosine"})
    
    for i in range(0, len(vector_docs), 100):
        collection.add(ids=doc_ids[i:i+100], documents=vector_docs[i:i+100], metadatas=metadatas[i:i+100])

    if es.indices.exists(index=ES_INDEX): es.indices.delete(index=ES_INDEX)
    es.indices.create(index=ES_INDEX, body={"mappings": {"properties": {"text": {"type": "text", "analyzer": "polish"}, "source_file": {"type": "keyword"}}}})
    helpers.bulk(es, [{"_index": ES_INDEX, "_id": d_id, "_source": {"text": doc, "source_file": md["source_file"]}} for d_id, doc, md in zip(doc_ids, es_docs, metadatas)])
    es.indices.refresh(index=ES_INDEX)

    with open(fp_path, "w") as f: f.write(fp)
    return collection

class QueryRewriter:
    def __init__(self, client: OpenAI):
        self.client, self.path = client, os.path.join(CACHE_DIR, "prod_rewrites_final.json")
        self.salt = hashlib.sha256((REWRITE_MODEL + REWRITE_PROMPT).encode()).hexdigest()[:12]
        self.cache = json.load(open(self.path)) if os.path.exists(self.path) else {}

    def get(self, query: str) -> Tuple[str, float]:
        key = f"{self.salt}|{query}"
        if key not in self.cache:
            t0 = time.perf_counter()
            resp = self.client.chat.completions.create(model=REWRITE_MODEL, messages=[{"role": "user", "content": REWRITE_PROMPT.replace("{query}", query)}], temperature=0)
            text = resp.choices[0].message.content.strip()
            self.cache[key] = {"text": f"{query} {text}" if not text.lower().startswith(query.lower()) else text, "latency_ms": (time.perf_counter() - t0) * 1000}
            with open(self.path, "w") as f: json.dump(self.cache, f, ensure_ascii=False, indent=1)
        return self.cache[key]["text"], self.cache[key]["latency_ms"]

# =====================================================================
# 4. SEARCH AND EVALUATION ENGINE
# =====================================================================

def evaluate_production_stack(queries: List[Query], collection, es, rewriter) -> List[QueryResult]:
    results = []
    rewriter.get(queries[0].text)
    collection.query(query_texts=[queries[0].text], n_results=1)

    for i, q in enumerate(queries, 1):
        search_q, rewrite_ms = rewriter.get(q.text)
        t0 = time.perf_counter()
        
        v_res = collection.query(query_texts=[search_q], n_results=CHUNK_POOL)
        v_hits = [{"id": j, "source_file": m["source_file"]} for j, m in zip(v_res["ids"][0], v_res["metadatas"][0])]
        
        k_resp = es.search(index=ES_INDEX, body={"query": {"match": {"text": search_q}}, "size": CHUNK_POOL})
        k_hits = [{"id": h["_id"], "source_file": h["_source"].get("source_file", "")} for h in k_resp["hits"]["hits"]]

        scores, meta = defaultdict(float), {}
        for ranking in (v_hits, k_hits):
            for rank, item in enumerate(ranking, 1):
                scores[item["id"]] += 1.0 / (RRF_K + rank)
                meta[item["id"]] = item["source_file"]
        
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:K_MAX]
        hits = [{"id": j, "source_file": meta[j]} for j, _ in ordered]
        
        retrieval_ms = (time.perf_counter() - t0) * 1000

        seen, files = set(), []
        for h in hits:
            if h["source_file"] not in seen:
                seen.add(h["source_file"]); files.append(h["source_file"])
                if len(files) == K_MAX: break
                
        rank = files.index(q.expected_file) + 1 if q.expected_file in files else 0
        results.append(QueryResult(q.text, q.qtype, q.expected_file, rank, files[:3], retrieval_ms, rewrite_ms))
        if i % 10 == 0 or i == len(queries): print(f"      Processed: {i}/{len(queries)}", end="\r")
    print()
    return results

# =====================================================================
# 5. METRICS, BOOTSTRAP, REPORT
# =====================================================================

def make_bootstrap_samples(queries: List[Query], n_boot: int, seed: int) -> List[np.ndarray]:
    by_file: Dict[str, List[int]] = defaultdict(list)
    for i, q in enumerate(queries): by_file[q.expected_file].append(i)
    groups = [np.array(v) for v in by_file.values()]
    rng = np.random.default_rng(seed)
    return [np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))]) for _ in range(n_boot)]

def main():
    queries = load_dataset()
    n_files = len({q.expected_file for q in queries})
    
    print("\n STARTING EVALUATION OF THE FINAL RAG PRODUCTION STACK (Asymmetric)\n")
    
    chroma_client = chromadb.PersistentClient(path=os.path.join(CACHE_DIR, "chroma_prod_final"))
    llm_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(api_key=settings.OPENAI_API_KEY, model_name=EMBED_MODEL)
    es = Elasticsearch(getattr(settings, "ELASTICSEARCH_URL", "http://localhost:9200"))
    rewriter = QueryRewriter(llm_client)

    collection = build_or_load_index(chroma_client, embedding_fn, es, force=False)
    results = evaluate_production_stack(queries, collection, es, rewriter)

    ranks = np.array([r.rank for r in results])
    found = ranks > 0
    safe = np.maximum(ranks, 1)
    
    metrics = {f"Hit@{k}": (found & (ranks <= k)).astype(float) for k in K_VALUES}
    metrics["MRR@10"] = np.where(found, 1.0 / safe, 0.0)
    
    samples = make_bootstrap_samples(queries, N_BOOTSTRAP, SEED)
    summary = {m: {"mean": float(vals.mean()), "lo": float(np.percentile([vals[s].mean() for s in samples], 2.5)), "hi": float(np.percentile([vals[s].mean() for s in samples], 97.5))} for m, vals in metrics.items()}
    
    latencies_retr = np.array([r.retrieval_ms for r in results])
    latencies_rewr = np.array([r.rewrite_ms for r in results])

    by_type = {}
    for t in sorted({q.qtype for q in queries}):
        idx = np.array([i for i, r in enumerate(results) if r.qtype == t])
        by_type[t] = {"n": len(idx), "Hit@1": float(metrics["Hit@1"][idx].mean()), "MRR@10": float(metrics["MRR@10"][idx].mean())}

    hard_queries = [r for r in results if r.rank == 0 or r.rank > 5][:5]

    L = ["", "=" * 90, f" WYNIKI OSTATECZNEGO STOSU PRODUKCYJNEGO RAG | {len(queries)} zapytań, {n_files} dokumentów", "=" * 90]
    L.append(f"Architektura: {EMBED_MODEL} + Asymmetric Context + Hybrid Search (RRF) + LLM Query Expansion")
    L.append("-" * 90)
    
    L.append("GŁÓWNE METRYKI SKUTECZNOŚCI (95% CI):")
    for m in ["Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR@10"]:
        L.append(f"  • {m:<8}: {summary[m]['mean']:.3f}  [{summary[m]['lo']:.3f} - {summary[m]['hi']:.3f}]")
    
    L.append("\nWYDAJNOŚĆ (Opóźnienia):")
    L.append(f"  • Wyszukiwanie (Bazy + RRF): Mediana = {np.median(latencies_retr):.1f} ms | p95 = {np.percentile(latencies_retr, 95):.1f} ms")
    L.append(f"  • Szacowany koszt LLM (Rewriter): Średnio = {latencies_rewr.mean():.1f} ms")

    L.append("\nROZBICIE NA TYPY ZAPYTAŃ (Hit@1 / MRR@10):")
    for t, d in by_type.items():
        L.append(f"  • {t.capitalize():<12} (n={d['n']}): {d['Hit@1']:.2f} / {d['MRR@10']:.3f}")

    if hard_queries:
        L.append("\n PRZYKŁADY ZAPYTAŃ POZA TOP 5 (Błędy docelowego systemu):")
        for r in hard_queries:
            L.append(f"  [{r.qtype}] {r.query[:80]}...")
            L.append(f"      Oczekiwano: {r.expected_file} | Zwrócono: {' | '.join(r.top_files)}")

    L.append("=" * 90)
    report_text = "\n".join(L)
    print(report_text)

    out_dir = os.path.join(OUT_ROOT, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report_production_final.txt"), "w", encoding="utf-8") as f: f.write(report_text)
    
    with open(os.path.join(out_dir, "per_query.csv"), "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["qtype", "query", "expected_file", "rank", "top3_files", "retrieval_ms"])
        for r in results: wr.writerow([r.qtype, r.query, r.expected_file, r.rank, " | ".join(r.top_files), f"{r.retrieval_ms:.1f}"])
        
    print(f"\n Saved final report in folder: {out_dir}")

if __name__ == "__main__":
    main()