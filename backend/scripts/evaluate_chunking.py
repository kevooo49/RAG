"""
Evaluation of chunking strategies (vector retrieval, file-level + chunk-level).

What this script does differently than the first version
----------------------------------------------
* "Brute-force" numpy search (exact cosine) across ALL chunks -
  no FETCH_POOL limit, which after file deduplication favored strategies
  with a small number of chunks per file (Hit@10 was underestimated for fine-grained chunking).
* Disk-based embedding cache (SQLite): subsequent runs are free and fast,
  and changing one strategy doesn't force recomputation of the others.
* Controlled comparison. The old comparison mixed THREE things at once:
  chunk boundaries, chunk sizes, and preprocessing/context (text_emb).
  The added variants separate these effects:
    - Fixed_500 / Fixed_1000 / Fixed_2000        -> effect of size
    - Fixed_Matched / Paragraphs_Matched         -> size = production mean
    - *_Matched+Title                            -> baseline + document title
      (is the production advantage just the "added title/header"?)
* Metrics at the file level (Hit@k, MRR@10, nDCG@10) AND at the chunk level
  (CHit@5, CPrec@5): the LLM receives chunks, not files, so a strategy that
  clogs the top-5 with duplicates from one file is penalized here.
* Baseline fixes: no duplicated tail in Fixed_Size, no empty chunks
  (empty inputs crash the OpenAI API), excessively long paragraphs are split,
  separators are counted towards the size limit.
* p-value in paired comparisons, length statistics (median/p95/max),
  number of chunks per file, verification of reference files without chunks.

Usage
------
    python evaluate_chunking.py
    python evaluate_chunking.py --only Fixed_1000,Paragraphs_2500,Production_Advanced
    python evaluate_chunking.py --baseline Fixed_Matched
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..")))

# =====================================================================
# 1. CONFIGURATION
# =====================================================================

DEFAULT_DATA_DIR = r"C:\Users\micro\Desktop\Bachelors\RAG\data\Ewaluacja_Probka"
DEFAULT_DATASET = os.path.join(SCRIPT_DIR, "test_dataset_balanced.json")
CACHE_DIR = os.path.join(SCRIPT_DIR, ".eval_cache")
OUT_ROOT = os.path.join(SCRIPT_DIR, "chunk_eva_stats")

EMBED_MODEL = "text-embedding-3-small"
K_VALUES = (1, 3, 5, 10)
K_MAX = max(K_VALUES)
CHUNK_K = 5
N_BOOTSTRAP = 2000
SEED = 42

MAX_EMBED_CHARS = 16000
EMBED_BATCH_ITEMS = 64
EMBED_BATCH_CHARS = 150000

CH_HIT = f"CHit@{CHUNK_K}"
CH_PREC = f"CPrec@{CHUNK_K}"

C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


# =====================================================================
# 2. DATA STRUCTURES
# =====================================================================

@dataclass
class Query:
    text: str
    expected_file: str
    qtype: str


@dataclass
class Chunk:
    text: str
    source_file: str


@dataclass
class QueryResult:
    strategy: str
    query: str
    qtype: str
    expected_file: str
    rank: int
    chunk_rank: int
    chunk_prec: float
    top_files: List[str]


@dataclass
class ChunkingStats:
    total_chunks: int
    mean_len: float
    median_len: float
    p95_len: float
    max_len: int
    chunks_per_file: float
    expected_files_without_chunks: int
    truncated_chunks: int


def load_dataset(data_dir: str, dataset_path: str) -> List[Query]:
    if not os.path.exists(dataset_path):
        sys.exit(f"Dataset missing: {dataset_path}. Run the question generator first.")
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    files = {f for f in os.listdir(data_dir) if f.endswith(".md")}
    missing = sorted({r["expected_file"] for r in raw if r["expected_file"] not in files})
    if missing:
        print("Missing reference files in the evaluation folder:")
        for m in missing:
            print("   -", m)
        sys.exit(1)
    return [Query(r["query"], r["expected_file"], r.get("type", "unknown")) for r in raw]


def read_corpus(data_dir: str) -> List[Tuple[str, str, str]]:
    out = []
    for fn in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, fn)
        if fn.endswith(".md") and os.path.isfile(path):
            with open(path, "r", encoding="utf-8-sig") as f:
                out.append((fn, path, f.read()))
    return out


# =====================================================================
# 3. EMBEDDINGS (SQLite cache) AND SEARCH
# =====================================================================

class OpenAIEmbedder:
    """Embeddings with persistent cache: key = sha256(model + text)."""

    def __init__(self, api_key: str, model: str, cache_path: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, max_retries=6)
        self.model = model
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        self.db = sqlite3.connect(cache_path)
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")
        self.api_texts = 0
        self.cached_texts = 0

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}\x00{text}".encode("utf-8")).hexdigest()

    def embed(self, texts: List[str]) -> np.ndarray:
        texts = [(t[:MAX_EMBED_CHARS] if t.strip() else " ") for t in texts]
        keys = [self._key(t) for t in texts]

        found: Dict[str, np.ndarray] = {}
        for i in range(0, len(keys), 500):
            part = keys[i:i + 500]
            marks = ",".join("?" * len(part))
            for k, v in self.db.execute(f"SELECT k, v FROM emb WHERE k IN ({marks})", part):
                found[k] = np.frombuffer(v, dtype=np.float32)

        todo: Dict[str, str] = {}
        for k, t in zip(keys, texts):
            if k not in found:
                todo.setdefault(k, t)
        self.cached_texts += len(keys) - sum(1 for k in keys if k in todo)

        items = list(todo.items())
        batch: List[Tuple[str, str]] = []
        size = 0

        def flush():
            nonlocal batch, size
            if not batch:
                return
            resp = self.client.embeddings.create(model=self.model, input=[t for _, t in batch])
            rows = []
            for (k, _), d in zip(batch, sorted(resp.data, key=lambda x: x.index)):
                vec = np.asarray(d.embedding, dtype=np.float32)
                found[k] = vec
                rows.append((k, vec.tobytes()))
            self.db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", rows)
            self.db.commit()
            self.api_texts += len(batch)
            batch, size = [], 0

        for k, t in items:
            if batch and (len(batch) >= EMBED_BATCH_ITEMS or size + len(t) > EMBED_BATCH_CHARS):
                flush()
            batch.append((k, t))
            size += len(t)
        flush()

        return np.vstack([found[k] for k in keys])


def normalize(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)


# =====================================================================
# 4. CHUNKING STRATEGIES
# =====================================================================

def chunk_fixed(text: str, size: int, overlap_ratio: float = 0.1) -> List[str]:
    """Fixed-length window; stops after a window that reaches the end of the text
    (the old version also added a tail that was entirely a repetition of the overlap)."""
    overlap = int(size * overlap_ratio)
    step = max(1, size - overlap)
    out, start = [], 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            out.append(piece)
        if start + size >= len(text):
            break
        start += step
    return out


def chunk_paragraphs(text: str, max_size: int) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    cur = ""
    for p in paras:
        if len(p) > max_size:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.extend(chunk_fixed(p, max_size))
            continue
        if cur and len(cur) + 2 + len(p) > max_size:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def title_from_filename(fn: str) -> str:
    stem = os.path.splitext(fn)[0]
    stem = re.sub(r"_[0-9a-f]{8}$", "", stem)
    return re.sub(r"[-_]+", " ", stem).strip()


@dataclass
class Strategy:
    name: str
    split: Optional[Callable[[str], List[str]]] = None
    title_prefix: bool = False
    production: bool = False
    note: str = ""


def load_production():
    """build_database imports Elasticsearch, which we don't need here."""
    from unittest.mock import MagicMock
    sys.modules["elasticsearch"] = MagicMock()
    sys.modules["elasticsearch.helpers"] = MagicMock()
    import build_database
    return build_database


def chunk_production(bd, filepath: str) -> List[str]:
    cfg = {
        "sciezka": filepath,
        "zrodlo": "Zbior_Ewaluacyjny",
        "typ": "dokument",
        "naglowki_prawne": True,
        "pogrubienia_jako_naglowki": True,
    }
    doc = bd.przygotuj_dokument(filepath, cfg)
    if not doc:
        return []
    return [c.tekst_emb for _cid, c, _md in bd.chunkuj_plik(filepath, cfg, doc)]

def chunk_production_asymmetric(bd, filepath: str) -> List[str]:
    """Hybrid version: preserves structure from the custom parser but uses a cleaner header."""
    cfg = {
        "sciezka": filepath, "zrodlo": "Zbior_Ewaluacyjny", "typ": "dokument",
        "naglowki_prawne": True, "pogrubienia_jako_naglowki": True,
    }
    doc = bd.przygotuj_dokument(filepath, cfg)
    if not doc: return []
    
    out = []
    for _cid, c, md in bd.chunkuj_plik(filepath, cfg, doc):
        title = md.get("tytul", "")
        section = md.get("sekcja", "")

        parts = c.tekst_emb.split("\n\n", 1)
        raw_content = parts[1] if len(parts) > 1 else c.tekst_emb
        
        prefix = section.split(" › ") if section else []
        new_header = f"{title}" + (f" - {prefix[-1]}" if prefix else "")
        
        out.append(f"{new_header}\n\n{raw_content}")
    return out


def build_strategies(prod_mean: Optional[int]) -> List[Strategy]:
    s = [
        Strategy("Fixed_500", lambda t: chunk_fixed(t, 500)),
        Strategy("Fixed_1000", lambda t: chunk_fixed(t, 1000)),
        Strategy("Fixed_2000", lambda t: chunk_fixed(t, 2000)),
        Strategy("Paragraphs_2500", lambda t: chunk_paragraphs(t, 2500)),
    ]
    if prod_mean:
        m = max(200, prod_mean)
        # Notatki ("note") pozostawiam po polsku, gdyż trafiają one wprost do polskiego raportu.
        s += [
            Strategy("Fixed_Matched", lambda t: chunk_fixed(t, m), note=f"rozmiar={m}"),
            Strategy("Paragraphs_Matched", lambda t: chunk_paragraphs(t, m), note=f"max={m}"),
            Strategy("Fixed_Matched+Title", lambda t: chunk_fixed(t, m), True, note=f"rozmiar={m}"),
            Strategy("Paragraphs_Matched+Title", lambda t: chunk_paragraphs(t, m), True, note=f"max={m}"),
        ]
        s.append(Strategy("Production_Advanced", production=True))
        s.append(Strategy("Production_Asymmetric", production=True, note="Hybryda - lekki naglowek"))
    return s

def build_chunks(strat: Strategy, corpus, bd, prod_cache: Dict[str, List[str]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for fn, path, text in corpus:
        if strat.production:
            cache_key = f"{strat.name}_{fn}"
            if cache_key not in prod_cache:
                if strat.name == "Production_Asymmetric":
                    prod_cache[cache_key] = chunk_production_asymmetric(bd, path)
                else:
                    prod_cache[cache_key] = chunk_production(bd, path)
            pieces = prod_cache[cache_key]
        else:
            pieces = strat.split(text)
            if strat.title_prefix:
                title = title_from_filename(fn)
                pieces = [f"{title}\n\n{p}" for p in pieces]
        chunks.extend(Chunk(p, fn) for p in pieces if p and p.strip())
    return chunks


# =====================================================================
# 5. EVALUATION OF A SINGLE STRATEGY
# =====================================================================

def evaluate_strategy(name: str, chunks: List[Chunk], queries: List[Query],
                      chunk_vecs: np.ndarray, query_vecs: np.ndarray) -> List[QueryResult]:
    files = np.array([c.source_file for c in chunks])
    sims = query_vecs @ chunk_vecs.T
    results = []
    for qi, q in enumerate(queries):
        order = np.argsort(-sims[qi], kind="stable")
        ranked = files[order]

        uniq = list(dict.fromkeys(ranked.tolist()))[:K_MAX]
        rank = uniq.index(q.expected_file) + 1 if q.expected_file in uniq else 0

        hits = np.flatnonzero(ranked == q.expected_file)
        chunk_rank = int(hits[0]) + 1 if len(hits) else 0
        prec = float(np.mean(ranked[:CHUNK_K] == q.expected_file))

        results.append(QueryResult(name, q.text, q.qtype, q.expected_file,
                                   rank, chunk_rank, prec, uniq[:3]))
    return results


def compute_chunk_stats(chunks: List[Chunk], corpus, queries: List[Query]) -> ChunkingStats:
    lens = np.array([len(c.text) for c in chunks]) if chunks else np.array([0])
    with_chunks = {c.source_file for c in chunks}
    no_chunks = len({q.expected_file for q in queries} - with_chunks)
    return ChunkingStats(
        total_chunks=len(chunks),
        mean_len=float(lens.mean()),
        median_len=float(np.median(lens)),
        p95_len=float(np.percentile(lens, 95)),
        max_len=int(lens.max()),
        chunks_per_file=len(chunks) / max(1, len(corpus)),
        expected_files_without_chunks=no_chunks,
        truncated_chunks=int((lens > MAX_EMBED_CHARS).sum()),
    )


# =====================================================================
# 6. METRICS AND STATISTICS
# =====================================================================

def per_query_scores(results: List[QueryResult]) -> Dict[str, np.ndarray]:
    ranks = np.array([r.rank for r in results])
    safe = np.maximum(ranks, 1)
    found = ranks > 0
    out = {f"Hit@{k}": (found & (ranks <= k)).astype(float) for k in K_VALUES}
    out["MRR@10"] = np.where(found, 1.0 / safe, 0.0)
    out["nDCG@10"] = np.where(found, 1.0 / np.log2(safe + 1), 0.0)
    cr = np.array([r.chunk_rank for r in results])
    out[CH_HIT] = ((cr > 0) & (cr <= CHUNK_K)).astype(float)
    out[CH_PREC] = np.array([r.chunk_prec for r in results])
    return out


def make_bootstrap_samples(queries: List[Query], n_boot: int, seed: int) -> List[np.ndarray]:
    """Cluster bootstrap: we sample FILES (each has 3 correlated queries)."""
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
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {"diff": float((a - b).mean()), "ci_low": float(lo), "ci_high": float(hi), "p": float(p)}


def summarize(results_by_strat, stats_by_strat, queries, samples):
    types = sorted({q.qtype for q in queries})
    scores = {n: per_query_scores(r) for n, r in results_by_strat.items()}
    summary: Dict[str, Any] = {}
    for name, res in results_by_strat.items():
        entry: Dict[str, Any] = {"n_queries": len(res)}
        for metric, vals in scores[name].items():
            entry[metric] = mean_ci(vals, samples)
        entry["chunk_stats"] = stats_by_strat[name].__dict__
        by_type = {}
        for t in types:
            idx = np.array([i for i, r in enumerate(res) if r.qtype == t])
            if len(idx):
                by_type[t] = {"n": int(len(idx)),
                              "Hit@1": float(scores[name]["Hit@1"][idx].mean()),
                              "Hit@5": float(scores[name]["Hit@5"][idx].mean()),
                              "MRR@10": float(scores[name]["MRR@10"][idx].mean())}
        entry["by_type"] = by_type
        summary[name] = entry
    return summary, scores, types


# =====================================================================
# 7. REPORT AND PLOTS
# =====================================================================

def fmt_ci(m: Dict[str, float]) -> str:
    return f"{m['mean']:.3f} [{m['ci_low']:.2f}-{m['ci_high']:.2f}]"


def build_report(summary, types, paired, baseline, notes, hard, n_queries, n_files) -> str:
    names = list(summary.keys())
    w = max(len(n) for n in names) + 2
    bar = "=" * 138
    L: List[str] = ["", bar,
                    f"EWALUACJA STRATEGII CHUNKOWANIA | {n_queries} zapytań, {n_files} plików referencyjnych | "
                    f"poziom plików K={K_MAX}, poziom chunków K={CHUNK_K}", bar]

    L.append(f"{'Strategia':<{w}}| {'Hit@1':<7}| {'Hit@3':<7}| {'Hit@5':<7}| {'Hit@10':<7}| {'MRR@10':<8}| "
             f"{'nDCG@10':<8}| {CH_HIT:<8}| {CH_PREC:<8}")
    L.append("-" * 138)
    for n in names:
        s = summary[n]
        L.append(f"{n:<{w}}| {s['Hit@1']['mean']:<7.3f}| {s['Hit@3']['mean']:<7.3f}| {s['Hit@5']['mean']:<7.3f}| "
                 f"{s['Hit@10']['mean']:<7.3f}| {s['MRR@10']['mean']:<8.4f}| {s['nDCG@10']['mean']:<8.4f}| "
                 f"{s[CH_HIT]['mean']:<8.3f}| {s[CH_PREC]['mean']:<8.3f}")
    L.append(f"({CH_HIT}: czy któryś z top-{CHUNK_K} chunków pochodzi z właściwego pliku; "
             f"{CH_PREC}: jaki ułamek top-{CHUNK_K} chunków pochodzi z właściwego pliku)")

    L += ["", "STATYSTYKI BAZY CHUNKÓW", "-" * 138]
    L.append(f"{'Strategia':<{w}}| {'Chunks':<8}| {'Ch/plik':<8}| {'Śr.dł.':<8}| {'Med.dł.':<8}| {'P95 dł.':<8}| "
             f"{'Max dł.':<8}| {'Brak ch.':<9}| {'Ucięte':<7}| Uwagi")
    for n in names:
        c = summary[n]["chunk_stats"]
        L.append(f"{n:<{w}}| {c['total_chunks']:<8}| {c['chunks_per_file']:<8.1f}| {c['mean_len']:<8.0f}| "
                 f"{c['median_len']:<8.0f}| {c['p95_len']:<8.0f}| {c['max_len']:<8}| "
                 f"{c['expected_files_without_chunks']:<9}| {c['truncated_chunks']:<7}| {notes.get(n, '')}")
    L.append("(Brak ch. = pliki referencyjne bez ani jednego chunku; Ucięte = chunki > "
             f"{MAX_EMBED_CHARS} znaków, obcięte przed embeddingiem)")

    L += ["", "95% PRZEDZIAŁY UFNOŚCI (cluster bootstrap po plikach)", "-" * 138]
    L.append(f"{'Strategia':<{w}}| {'Hit@1':<20}| {'Hit@5':<20}| {'MRR@10':<20}| {CH_PREC:<20}")
    for n in names:
        s = summary[n]
        L.append(f"{n:<{w}}| {fmt_ci(s['Hit@1']):<20}| {fmt_ci(s['Hit@5']):<20}| "
                 f"{fmt_ci(s['MRR@10']):<20}| {fmt_ci(s[CH_PREC]):<20}")

    if paired:
        L += ["", f"RÓŻNICE WZGLĘDEM '{baseline}' (dodatnie = lepsze od bazy)", "-" * 138]
        L.append(f"{'Strategia':<{w}}| {'dHit@1 [95% CI]':<26}| {'p':<7}| {'dMRR@10 [95% CI]':<26}| {'p':<7}")
        for n, d in paired.items():
            a, b = d["Hit@1"], d["MRR@10"]
            c1 = f"{a['diff']:+.3f} [{a['ci_low']:+.2f},{a['ci_high']:+.2f}]"
            c2 = f"{b['diff']:+.3f} [{b['ci_low']:+.2f},{b['ci_high']:+.2f}]"
            L.append(f"{n:<{w}}| {c1:<26}| {a['p']:<7.3f}| {c2:<26}| {b['p']:<7.3f}")
        L.append("(p = dwustronna p-wartość z bootstrapu; ~9 porównań naraz, więc pojedyncze p<0.05 "
                 "traktuj ostrożnie - patrz przedziały ufności)")

    L += ["", "ROZBICIE NA TYPY ZAPYTAŃ (Hit@1 / MRR@10)", "-" * 138]
    first = summary[names[0]]["by_type"]
    ts = [t for t in types if t in first]
    L.append(f"{'Strategia':<{w}}" + "".join(f"| {t + ' (n=' + str(first[t]['n']) + ')':<24}" for t in ts))
    for n in names:
        row = f"{n:<{w}}"
        for t in ts:
            bt = summary[n]["by_type"].get(t)
            row += (f"| {bt['Hit@1']:.2f} / {bt['MRR@10']:.3f}" if bt else "| -").ljust(26)
        L.append(row)

    L += ["", f"ZAPYTANIA NIETRAFIONE (poza top-5 plików) PRZEZ KAŻDĄ STRATEGIĘ: {len(hard)} "
              "- najpierw sprawdź etykiety / parsowanie pliku", "-" * 138]
    for q, expected in hard[:25]:
        L.append(f"[{q.qtype}] {q.text[:100]}")
        L.append(f"     oczekiwany: {expected}")
    if len(hard) > 25:
        L.append(f"... i {len(hard) - 25} więcej (patrz per_query.csv)")
    L.append(bar)
    return "\n".join(L)


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def plot_metrics(summary, path):
    names = list(summary.keys())
    metrics = [("Hit@1", C_BLUE), ("Hit@5", C_ORANGE), ("MRR@10", C_AQUA)]
    x = np.arange(len(names))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(10, 1.5 * len(names)), 6.5), facecolor=SURFACE)
    _style_axes(ax)
    for j, (m, color) in enumerate(metrics):
        means = np.array([summary[n][m]["mean"] for n in names])
        lo = np.array([summary[n][m]["ci_low"] for n in names])
        hi = np.array([summary[n][m]["ci_high"] for n in names])
        ax.bar(x + (j - 1) * width, means, width - 0.03, label=m, color=color,
               yerr=[np.clip(means - lo, 0, None), np.clip(hi - means, 0, None)],
               capsize=2, error_kw={"ecolor": INK_2, "elinewidth": 1})
        for xi, mean_v, hi_v in zip(x + (j - 1) * width, means, hi):   # label above CI end
            ax.text(xi, max(hi_v, mean_v) + 0.012, f"{mean_v:.2f}", ha="center", va="bottom",
                    fontsize=7, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Wartość metryki (słupki błędu: 95% CI)")
    ax.set_title("Strategie chunkowania - skuteczność wyszukiwania (poziom plików)", loc="left")
    ax.legend(loc="upper left", frameon=False, ncol=3, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(fig)


def plot_size_vs_quality(summary, path):
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    _style_axes(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    offsets = [(7, 6), (7, -14), (-7, 14), (-7, -18)]
    for i, (n, s) in enumerate(summary.items()):
        x, y = s["chunk_stats"]["mean_len"], s["MRR@10"]["mean"]
        prod = n.startswith("Production")
        ax.scatter(x, y, s=70, color=C_ORANGE if prod else C_BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=8, color=INK_2,
                    ha="left" if dx > 0 else "right")
    ax.margins(x=0.12, y=0.15)
    ax.set_xlabel("Średnia długość chunku (znaki)")
    ax.set_ylabel("MRR@10")
    ax.set_title("Rozmiar chunku a jakość (pomarańczowy = produkcja)", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(fig)


# =====================================================================
# 8. EXECUTION
# =====================================================================

def run(data_dir: str, dataset_path: str, only: List[str], baseline: str,
        embedder, out_root: str = OUT_ROOT) -> str:
    queries = load_dataset(data_dir, dataset_path)
    corpus = read_corpus(data_dir)
    n_files = len({q.expected_file for q in queries})
    print(f"Dataset: {len(queries)} queries, {n_files} reference files, {len(corpus)} files in the database")

    bd, prod_cache, prod_mean = None, {}, None
    try:
        bd = load_production()
        probe = Strategy("Production_Advanced", production=True)
        prod_chunks = build_chunks(probe, corpus, bd, prod_cache)
        if prod_chunks:
            prod_mean = int(np.mean([len(c.text) for c in prod_chunks]))
            print(f"Production chunker: {len(prod_chunks)} chunks, avg. length {prod_mean} characters")
    except Exception as e:
        print(f"WARNING: failed to use build_database ({e!r}) - skipping production and *_Matched strategies")

    strategies = build_strategies(prod_mean)
    if only:
        unknown = set(only) - {s.name for s in strategies}
        if unknown:
            sys.exit(f"Unknown strategies: {sorted(unknown)}. Available: {[s.name for s in strategies]}")
        strategies = [s for s in strategies if s.name in only]
    if not strategies:
        sys.exit("No strategies to run.")

    query_vecs = normalize(embedder.embed([q.text for q in queries]))

    results_by_strat: Dict[str, List[QueryResult]] = {}
    stats_by_strat: Dict[str, ChunkingStats] = {}
    notes: Dict[str, str] = {}
    print("\nSTARTING CHUNKING STRATEGY EVALUATION\n")
    for strat in strategies:
        print(f"-> {strat.name}")
        chunks = build_chunks(strat, corpus, bd, prod_cache)
        if not chunks:
            print("   [!] 0 chunks - skipping")
            continue
        stats = compute_chunk_stats(chunks, corpus, queries)
        if stats.expected_files_without_chunks:
            print(f"   [!] {stats.expected_files_without_chunks} reference files without chunks")
        vecs = normalize(embedder.embed([c.text for c in chunks]))
        results_by_strat[strat.name] = evaluate_strategy(strat.name, chunks, queries, vecs, query_vecs)
        stats_by_strat[strat.name] = stats
        notes[strat.name] = strat.note
        print(f"   {len(chunks)} chunks, avg. length {stats.mean_len:.0f}")
    if hasattr(embedder, "api_texts"):
        print(f"\nEmbeddings: {embedder.api_texts} from API, {embedder.cached_texts} from cache")

    samples = make_bootstrap_samples(queries, N_BOOTSTRAP, SEED)
    summary, scores, types = summarize(results_by_strat, stats_by_strat, queries, samples)

    base = baseline if baseline in results_by_strat else next(iter(results_by_strat))
    paired = {n: {m: paired_diff(scores[n][m], scores[base][m], samples) for m in ("Hit@1", "MRR@10")}
              for n in results_by_strat if n != base}

    hard = []
    for i, q in enumerate(queries):
        if all(not (0 < res[i].rank <= 5) for res in results_by_strat.values()):
            hard.append((q, q.expected_file))

    report = build_report(summary, types, paired, base, notes, hard, len(queries), n_files)
    print(report)

    out_dir = os.path.join(out_root, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"config": {"embed_model": EMBED_MODEL, "k_max": K_MAX, "chunk_k": CHUNK_K,
                              "n_bootstrap": N_BOOTSTRAP, "seed": SEED, "n_queries": len(queries),
                              "production_mean_len": prod_mean},
                   "summary": summary, "paired_vs_baseline": {"baseline": base, "results": paired}},
                  f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "per_query.csv"), "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["strategy", "type", "query", "expected_file", "file_rank", "chunk_rank",
                     f"chunk_precision@{CHUNK_K}", "top3_files"])
        for res_list in results_by_strat.values():
            for r in res_list:
                wr.writerow([r.strategy, r.qtype, r.query, r.expected_file, r.rank, r.chunk_rank,
                             f"{r.chunk_prec:.2f}", " | ".join(r.top_files)])

    plot_metrics(summary, os.path.join(out_dir, "wykres_chunkowania.png"))
    plot_size_vs_quality(summary, os.path.join(out_dir, "wykres_rozmiar_vs_jakosc.png"))
    print(f"\nSaved report, CSV, JSON, and plots in folder: {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--only", default="", help="comma-separated list of strategies")
    ap.add_argument("--baseline", default="Production_Advanced")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(os.path.join(SCRIPT_DIR, "..", ".env"))
    try:
        from app.core.config import settings
        api_key = settings.OPENAI_API_KEY
    except ImportError:
        api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        sys.exit("Missing OPENAI_API_KEY.")
    if not os.path.isdir(args.data_dir):
        sys.exit(f"Missing evaluation folder: {args.data_dir}")

    embedder = OpenAIEmbedder(api_key, EMBED_MODEL, os.path.join(CACHE_DIR, "embeddings.sqlite"))
    run(args.data_dir, args.dataset, [s for s in args.only.split(",") if s], args.baseline, embedder)


if __name__ == "__main__":
    main()