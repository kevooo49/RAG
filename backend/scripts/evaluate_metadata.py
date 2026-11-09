#!/usr/bin/env python3
"""
Evaluation of CHUNK METADATA ENRICHMENT before embedding (breadcrumbs / context).

Chunking, search, and embedding models are evaluated separately – here we change ONLY what
text goes into the vector (and into BM25). Chunk splitting is identical for all variants.

    python eval_metadata_enrichment.py --list              # list of variants
    python eval_metadata_enrichment.py --dry-run           # chunk count, tokens, cost – without API
    python eval_metadata_enrichment.py                     # deterministic variants (cheap)
    python eval_metadata_enrichment.py --llm               # + variants with generated context (LLM)
    python eval_metadata_enrichment.py --variants raw breadcrumb prod_header --retrievers dense

What we test (each variant is compared to 'raw' via paired test + Holm correction):
  WHAT we inject:      title | section | breadcrumb | prod_header | breadcrumb+url
  HOW we format it:    breadcrumb | labelled | sentence
  WHERE:               breadcrumb (before content) vs suffix (after content)
  CONTEXT FROM LLM:    llm_context | breadcrumb+llm_context | breadcrumb+summary | doc2query   (--llm)

Why these results are more reliable than standard "raw vs enriched":
  * statistics: bootstrap by FILES (3 queries per file are not independent), CI, p-value, Holm correction,
  * "title in query" analysis: queries containing the document title inflate the gain of injecting the title
    (real users rarely type it) – results are reported separately for both groups,
  * exact search (brute-force), no ANN noise; dense + BM25 + hybrid (RRF),
  * disk cache (SQLite): the same text/query/LLM response computed once – reruns cost nothing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence
from urllib.parse import urlsplit

import numpy as np

# =====================================================================
# 1. CONFIGURATION
# =====================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from sites import load_or_exit  # noqa: E402 # type: ignore
EVAL_DATA_DIR = os.getenv("EVAL_DATA_DIR") or str(load_or_exit().eval_dir)
DATASET_PATH = SCRIPT_DIR / "test_dataset_balanced.json"
OUT_ROOT = SCRIPT_DIR / "meta_eva_stats_claude"
CACHE_PATH = SCRIPT_DIR / "meta_eva_cache.sqlite"

EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
PRICE_PER_1M_EMBED_USD = 0.02
K_VALUES = (1, 3, 5, 10)
POOL = 40
N_BOOTSTRAP = 5000
SEED = 42
BASELINE = "raw"
RETRIEVERS = ("dense", "bm25", "hybrid")
BM25_PREFIX = 5
RET_NAMES = {"dense": "Dense (embeddingi, cosine, brute-force)", "bm25": f"BM25 (lokalny, stemming {BM25_PREFIX}-znakowy)",
             "hybrid": "Hybrid (RRF dense + BM25)"}
QUERY_TYPES = ("lexical", "semantic", "colloquial")
MAX_DOC_CHARS = 20_000

SOURCE_BY_HOST = {"bg.agh.edu.pl": "Biblioteka Główna AGH", 
                  "css.agh.edu.pl": "Centrum Spraw Studenckich AGH",
                  "rekrutacja.agh.edu.pl": "Rekrutacja AGH",
                  "miasteczko.agh.edu.pl": "Miasteczko AGH",
                  "ck.agh.edu.pl": "Centrum Karier AGH",
                  "csm.agh.edu.pl": "Centrum Spraw Międzynarodowych AGH",
                  "swfis.agh.edu.pl": "SWFiS AGH",
                  "sylabusy.agh.edu.pl": "Sylabusy AGH",
                  "cyfronet.agh.edu.pl": "Cyfronet AGH",
                  "sjo.agh.edu.pl": "SJO AGH",
                }

STOP = set("jaki jaka jakie jaką który która które kiedy gdzie czego czemu jest są czy oraz tego tych dla przez "
           "jak kto kogo nich tej tym się było będzie można mogę proszę prosze".split())


# =====================================================================
# 2. DATA: TEST SET, CHUNKS
# =====================================================================
def norm_name(s: str) -> str:
    return unicodedata.normalize("NFC", Path(str(s)).name).strip().lower()


def stems(text: str, n: int = 4) -> set:
    """Coarse stemming (prefix of n characters) – sufficient for heuristics like 'does query contain title'."""
    return {w[:n] for w in re.findall(r"\w{4,}", text.lower()) if w not in STOP}


@dataclass
class Item:
    query: str
    qtype: str
    expected: List[str]
    expected_disp: List[str]


def load_dataset(path: Path) -> List[Item]:
    if not Path(path).exists():
        sys.exit(f"Test dataset missing: {path}")
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = []
    for r in raw:
        exp = r.get("expected_files") or [r.get("expected_file")]
        exp = [e for e in exp if e]
        items.append(Item(r["query"].strip(), r.get("type", "unknown"), [norm_name(e) for e in exp],
                          [Path(e).name for e in exp]))
    return items


@dataclass
class DocRec:
    key: str
    disp: str
    title: str
    body: str


@dataclass
class ChunkRec:
    idx: int
    file: str
    title: str
    section: str
    source: str
    url_words: str
    body: str
    llm_context: str = ""
    doc_summary: str = ""
    questions: str = ""


def import_chunker():
    """build_database (chunking code). The module might import elasticsearch on startup, and here ES is not
    needed – so (as in the previous version of the script) we substitute a mock."""
    from unittest.mock import MagicMock
    for p in (SCRIPT_DIR, SCRIPT_DIR.parent):
        if str(p) not in sys.path:
            sys.path.append(str(p))
    sys.modules.setdefault("elasticsearch", MagicMock())
    sys.modules.setdefault("elasticsearch.helpers", MagicMock())
    import build_database
    return build_database


def cfg_for_file(path: Path, meta: dict, data_dir: Path) -> dict:
    """Chunking config close to production (document type and source label from frontmatter)."""
    t = str(meta.get("typ_dokumentu") or meta.get("typ") or "").lower()
    kind = "sylabus" if "sylab" in t else "dokument" if t in ("pdf", "docx", "doc", "odt", "dokument") else "strona"
    host = urlsplit(str(meta.get("url") or "")).netloc.lower()
    source = SOURCE_BY_HOST.get(host) or {"sylabus": "Sylabusy AGH", "dokument": "Dokumenty AGH"}.get(kind, "AGH")
    return {"sciezka": str(data_dir), "zrodlo": source, "typ": kind, "naglowki_prawne": kind == "dokument",
            "pogrubienia_jako_naglowki": True}


def split_header(text_emb: str, md: dict) -> str:
    """Returns chunk content without the contextual header. Accurately first (reconstructed header), then fallback."""
    title, section, source = (str(md.get(k) or "") for k in ("tytul", "sekcja", "zrodlo"))
    header = f"[{source}] {title}" + (f" › {section}" if section else "")
    if text_emb.startswith(header + "\n\n"):
        return text_emb[len(header) + 2:]
    parts = text_emb.split("\n\n", 1)
    return parts[1] if len(parts) > 1 else text_emb


def url_words(md: dict) -> str:
    segs = [s for s in urlsplit(str(md.get("url") or "")).path.split("/") if s]
    return " ".join(re.sub(r"[-_]+", " ", s) for s in segs)


def load_chunks(data_dir: Path, bd) -> tuple:
    chunks: List[ChunkRec] = []
    docs: Dict[str, DocRec] = {}
    for p in sorted(Path(data_dir).rglob("*.md")):
        doc = bd.prepare_document(p, {})
        if not doc:
            continue
        meta, body, title = doc
        cfg = cfg_for_file(p, meta, data_dir)
        key = norm_name(p.name)
        docs[key] = DocRec(key, p.name, title, body)
        for _, c, md in bd.chunk_file(p, cfg, doc):
            chunks.append(ChunkRec(len(chunks), key, str(md.get("tytul") or title), str(md.get("sekcja") or ""),
                                   str(md.get("zrodlo") or cfg["zrodlo"]), url_words(md),
                                   split_header(c.text_emb, md)))
    return chunks, docs


def validate(items: List[Item], docs: Dict[str, DocRec]) -> None:
    missing = sorted({d for it in items for d, e in zip(it.expected_disp, it.expected) if e not in docs})
    if missing:
        sys.exit("Files from the test dataset that are missing in the corpus (any query for them would yield 0):\n  "
                 + "\n  ".join(missing))


# =====================================================================
# 3. ENRICHMENT VARIANTS
# =====================================================================
@dataclass(frozen=True)
class Variant:
    name: str
    desc: str
    build: Callable[[ChunkRec], str]
    needs: str = ""


def _bc(c: ChunkRec) -> str:
    return " › ".join(x for x in (c.title, c.section) if x)


VARIANTS: List[Variant] = [
    Variant("raw", "chunk text only – no metadata (baseline)", lambda c: c.body),
    Variant("title", "document title only", lambda c: f"{c.title}\n\n{c.body}"),
    Variant("section", "section path only (without title)", lambda c: (f"{c.section}\n\n" if c.section else "") + c.body),
    Variant("breadcrumb", "Title › Section › Subsection", lambda c: f"{_bc(c)}\n\n{c.body}"),
    Variant("prod_header", "[Source] Title › Section (build_database format)", lambda c: f"[{c.source}] {_bc(c)}\n\n{c.body}"),
    Variant("breadcrumb+url", "breadcrumb + words from URL path",
            lambda c: f"{_bc(c)}" + (f"\n{c.url_words}" if c.url_words else "") + f"\n\n{c.body}"),
    Variant("labelled", "labelled fields (Dokument: / Sekcja:)",
            lambda c: f"Dokument: {c.title}\n" + (f"Sekcja: {c.section}\n" if c.section else "") + f"\n{c.body}"),
    Variant("sentence", "descriptive sentence in Polish",
            lambda c: f"Fragment dokumentu „{c.title}”" + (f", sekcja „{c.section}”" if c.section else "") + f".\n\n{c.body}"),
    Variant("suffix", "breadcrumb AFTER content (position)", lambda c: f"{c.body}\n\n{_bc(c)}"),
    Variant("llm_context", "LLM context (Contextual Retrieval) + content",
            lambda c: f"{c.llm_context}\n\n{c.body}", "llm_context"),
    Variant("breadcrumb+llm_context", "breadcrumb + LLM context",
            lambda c: f"{_bc(c)}\n{c.llm_context}\n\n{c.body}", "llm_context"),
    Variant("breadcrumb+summary", "breadcrumb + document summary (LLM)",
            lambda c: f"{_bc(c)}\n{c.doc_summary}\n\n{c.body}", "doc_summary"),
    Variant("doc2query", "content + 3 LLM generated questions  risk of loop: LLM also generated test queries",
            lambda c: f"{c.body}\n\nPytania: {c.questions}", "questions"),
]
VARIANT_BY_NAME = {v.name: v for v in VARIANTS}

P_CONTEXT = """<dokument>
{DOC}
</dokument>
Oto fragment tego dokumentu:
<fragment>
{CHUNK}
</fragment>
Napisz 1-2 krótkie zdania po polsku, które umieszczają ten fragment w kontekście całego dokumentu (czego dotyczy dokument i której jego części dotyczy fragment), aby ułatwić wyszukanie fragmentu. Nie powtarzaj treści fragmentu. Zwróć tylko te zdania."""
P_SUMMARY = """<dokument>
{DOC}
</dokument>
Napisz streszczenie tego dokumentu w 2 zdaniach po polsku (czego dotyczy, dla kogo). Zwróć tylko streszczenie."""
P_QUESTIONS = """Tytuł dokumentu: {TITLE}
<fragment>
{CHUNK}
</fragment>
Napisz 3 różne pytania, które mógłby zadać student AGH i na które odpowiada ten fragment. Zwróć je w jednej linii, oddzielone znakiem |. Nic więcej."""


def one_line(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def add_llm_fields(chunks: List[ChunkRec], docs: Dict[str, DocRec], llm, needs: set, log=print) -> None:
    if "doc_summary" in needs:
        summaries = {k: one_line(llm.complete(P_SUMMARY.replace("{DOC}", d.body[:MAX_DOC_CHARS]), "sum_v1"))
                     for k, d in docs.items()}
        for c in chunks:
            c.doc_summary = summaries[c.file]
    for n, c in enumerate(chunks, 1):
        d = docs[c.file]
        if "llm_context" in needs:
            c.llm_context = one_line(llm.complete(
                P_CONTEXT.replace("{DOC}", d.body[:MAX_DOC_CHARS]).replace("{CHUNK}", c.body), "ctx_v1"))
        if "questions" in needs:
            c.questions = one_line(llm.complete(
                P_QUESTIONS.replace("{TITLE}", c.title).replace("{CHUNK}", c.body), "q_v1"))
        if n % 200 == 0:
            log(f"   [LLM] {n}/{len(chunks)} chunks")


# =====================================================================
# 4. CACHE, EMBEDDINGS, LLM
# =====================================================================
def h(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()


class Cache:
    """SQLite key->bytes. We save vectors as float32 (6 KB instead of ~20 KB in JSON)."""

    def __init__(self, path):
        self.con = sqlite3.connect(str(path))
        self.con.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v BLOB)")
        self.hits: Counter = Counter()
        self.misses: Counter = Counter()

    def get(self, key: str) -> Optional[bytes]:
        row = self.con.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: bytes) -> None:
        self.con.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (key, value))

    def commit(self) -> None:
        self.con.commit()


class Embedder:
    def __init__(self, client, model: str, cache: Cache, batch: int = 128):
        self.client, self.model, self.cache, self.batch = client, model, cache
        self.api_calls = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        keys = [h("emb", self.model, t) for t in texts]
        missing: Dict[str, str] = {}
        for k, t in zip(keys, texts):
            if self.cache.get(k) is None:
                missing.setdefault(k, t)
        self.cache.hits["emb"] += len(texts) - len(missing)
        self.cache.misses["emb"] += len(missing)
        todo = list(missing.items())
        for s in range(0, len(todo), self.batch):
            part = todo[s:s + self.batch]
            resp = self.client.embeddings.create(model=self.model, input=[t or " " for _, t in part])
            for (k, _), d in zip(part, sorted(resp.data, key=lambda d: d.index)):
                self.cache.put(k, np.asarray(d.embedding, dtype=np.float32).tobytes())
            self.cache.commit()
            self.api_calls += 1
        mat = np.vstack([np.frombuffer(self.cache.get(k), dtype=np.float32) for k in keys])
        return mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)


class LLM:
    def __init__(self, client, model: str, cache: Cache):
        self.client, self.model, self.cache = client, model, cache
        self.api_calls = 0

    def complete(self, prompt: str, tag: str) -> str:
        key = h("llm", self.model, tag, prompt)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache.hits["llm"] += 1
            return cached.decode("utf-8")
        resp = self.client.chat.completions.create(model=self.model, temperature=0,
                                                   messages=[{"role": "user", "content": prompt}])
        text = (resp.choices[0].message.content or "").strip()
        self.cache.put(key, text.encode("utf-8"))
        self.cache.commit()
        self.cache.misses["llm"] += 1
        self.api_calls += 1
        return text


def load_settings() -> dict:
    try:
        from dotenv import load_dotenv
        load_dotenv(SCRIPT_DIR.parent / ".env")
    except ImportError:
        pass
    key = os.getenv("OPENAI_API_KEY")
    try:
        for p in (SCRIPT_DIR, SCRIPT_DIR.parent):
            if str(p) not in sys.path:
                sys.path.append(str(p))
        from app.core.config import settings
        key = getattr(settings, "OPENAI_API_KEY", None) or key
    except Exception:
        pass
    return {"openai_key": key}


# =====================================================================
# 5. RETRIEVERS (local, deterministic)
# =====================================================================
class DenseIndex:
    def __init__(self, vecs: np.ndarray):
        self.v = vecs

    def search(self, q: np.ndarray, k: int) -> List[int]:
        scores = self.v @ q
        return list(np.lexsort((np.arange(len(scores)), -scores))[:k])


_TOK = re.compile(r"\w+", re.U)


def bm25_tokens(text: str) -> List[str]:
    return [w[:BM25_PREFIX] for w in _TOK.findall(text.lower()) if len(w) > 1]


class BM25:
    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.inv: Dict[str, List[tuple]] = defaultdict(list)
        self.len = np.zeros(len(docs))
        for i, d in enumerate(docs):
            tf = Counter(bm25_tokens(d))
            self.len[i] = sum(tf.values())
            for t, f in tf.items():
                self.inv[t].append((i, f))
        self.avg = max(self.len.mean(), 1e-9)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.inv.items()}

    def search(self, query: str, k: int) -> List[int]:
        scores: Dict[int, float] = defaultdict(float)
        for t in set(bm25_tokens(query)):
            for i, f in self.inv.get(t, ()):
                scores[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return sorted(scores, key=lambda i: (-scores[i], i))[:k]


def rrf(lists: Sequence[Sequence[int]], k: int = 60) -> List[int]:
    score: Dict[int, float] = defaultdict(float)
    best: Dict[int, int] = {}
    for lst in lists:
        for r, i in enumerate(lst, 1):
            score[i] += 1.0 / (k + r)
            best[i] = min(best.get(i, r), r)
    return sorted(score, key=lambda i: (-score[i], best[i], i))


# =====================================================================
# 6. EVALUATION
# =====================================================================
@dataclass
class Run:
    ranks: np.ndarray
    p5: np.ndarray
    top3: List[List[str]] = field(default_factory=list)


def rank_files(chunk_ids: Sequence[int], chunks: List[ChunkRec], depth: int = 10) -> List[str]:
    seen, files = set(), []
    for i in chunk_ids:
        f = chunks[i].file
        if f not in seen:
            seen.add(f)
            files.append(f)
            if len(files) == depth:
                break
    return files


def run_evaluation(variants: List[Variant], retrievers: Sequence[str], chunks: List[ChunkRec], items: List[Item],
                   docs: Dict[str, DocRec], embedder, qvecs: np.ndarray, log=print) -> Dict[str, Dict[str, Run]]:
    runs: Dict[str, Dict[str, Run]] = {r: {} for r in retrievers}
    for v in variants:
        texts = [v.build(c) for c in chunks]
        dense = DenseIndex(embedder.embed(texts)) if ({"dense", "hybrid"} & set(retrievers)) else None
        bm25 = BM25(texts) if ({"bm25", "hybrid"} & set(retrievers)) else None
        log(f"   [+] {v.name}: {len(texts)} chunks")
        for ret in retrievers:
            ranks, p5, top3 = [], [], []
            for it, qv in zip(items, qvecs):
                if ret == "dense":
                    ids = dense.search(qv, POOL)
                elif ret == "bm25":
                    ids = bm25.search(it.query, POOL)
                else:
                    ids = rrf([dense.search(qv, POOL), bm25.search(it.query, POOL)])
                files = rank_files(ids, chunks)
                exp = set(it.expected)
                ranks.append(next((r for r, f in enumerate(files, 1) if f in exp), 0))
                top5 = ids[:5]
                p5.append(sum(chunks[i].file in exp for i in top5) / max(len(top5), 1))
                top3.append([docs[f].disp for f in files[:3]])
            runs[ret][v.name] = Run(np.array(ranks), np.array(p5), top3)
    return runs


# =====================================================================
# 7. STATISTICS
# =====================================================================
METRICS = ["Hit@1", "Hit@3", "Hit@5", "MRR@10", "P@5chunks"]


def metric_arrays(run: Run) -> Dict[str, np.ndarray]:
    r = run.ranks
    found = r > 0
    out = {f"Hit@{k}": (found & (r <= k)).astype(float) for k in (1, 3, 5)}
    out["MRR@10"] = np.where(found, 1.0 / np.maximum(r, 1), 0.0)
    out["P@5chunks"] = run.p5
    return out


class Boot:
    """Bootstrap by clusters (files). Same samples for all comparisons => paired tests."""

    def __init__(self, groups: Sequence[str], n_boot: int, seed: int):
        uniq, self.inv = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
        self.G = len(uniq)
        self.idx = np.random.default_rng(seed).integers(0, self.G, size=(n_boot, self.G))

    def dist(self, values: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        v = np.asarray(values, float)
        m = np.ones(len(v)) if mask is None else np.asarray(mask, float)
        num = np.bincount(self.inv, weights=v * m, minlength=self.G)
        den = np.bincount(self.inv, weights=m, minlength=self.G)
        n, d = num[self.idx].sum(1), den[self.idx].sum(1)
        return np.where(d > 0, n / np.where(d > 0, d, 1), np.nan)

    def ci(self, values, mask=None) -> dict:
        v = np.asarray(values, float)
        m = np.ones(len(v)) if mask is None else np.asarray(mask, float)
        d = self.dist(v, m)
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        return {"mean": float((v * m).sum() / max(m.sum(), 1)), "lo": float(lo), "hi": float(hi)}

    def paired(self, a, b, mask=None) -> dict:
        diff = np.asarray(a, float) - np.asarray(b, float)
        m = np.ones(len(diff)) if mask is None else np.asarray(mask, float)
        d = self.dist(diff, m)
        d = d[~np.isnan(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = min(1.0, max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / len(d)))
        return {"diff": float((diff * m).sum() / max(m.sum(), 1)), "lo": float(lo), "hi": float(hi), "p": float(p)}


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    out, running = {}, 0.0
    m = len(pvals)
    for i, (k, p) in enumerate(sorted(pvals.items(), key=lambda kv: kv[1])):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def title_mentioned(item: Item, docs: Dict[str, DocRec]) -> bool:
    """Does the query contain (most of the words from) the expected document title? Heuristics on 4-char stems."""
    q = stems(item.query)
    for f in item.expected:
        t = stems(docs[f].title) or stems(Path(docs[f].disp).stem)
        if t and len(t & q) / len(t) >= 0.5:
            return True
    return False


def analyze(runs, items: List[Item], docs: Dict[str, DocRec], n_boot: int, seed: int) -> dict:
    boot = Boot(["|".join(sorted(it.expected)) for it in items], n_boot, seed)
    types = [t for t in QUERY_TYPES if any(it.qtype == t for it in items)]
    types += sorted({it.qtype for it in items} - set(types))
    mention = np.array([title_mentioned(it, docs) for it in items])
    subgroups = {"wszystkie": np.ones(len(items), bool)}
    subgroups.update({f"typ: {t}": np.array([it.qtype == t for it in items]) for t in types})
    subgroups["tytuł w zapytaniu"] = mention
    subgroups["bez tytułu w zapytaniu"] = ~mention

    out = {"n_queries": len(items), "n_files": boot.G, "subgroup_sizes": {k: int(m.sum()) for k, m in subgroups.items()},
           "retrievers": {}}
    for ret, by_var in runs.items():
        arrays = {v: metric_arrays(r) for v, r in by_var.items()}
        base = arrays[BASELINE]
        res = {}
        for v, arr in arrays.items():
            e = {"metrics": {m: boot.ci(arr[m]) for m in METRICS}, "delta": {}, "subgroups": {}}
            if v != BASELINE:
                e["delta"] = {m: boot.paired(arr[m], base[m]) for m in METRICS}
                e["subgroups"] = {sg: {"n": int(msk.sum()),
                                       "mrr": boot.ci(arr["MRR@10"], msk)["mean"] if msk.any() else float("nan"),
                                       "delta": boot.paired(arr["MRR@10"], base["MRR@10"], msk) if msk.any() else None}
                                  for sg, msk in subgroups.items()}
            res[v] = e
        for m in METRICS:
            adj = holm({v: res[v]["delta"][m]["p"] for v in res if v != BASELINE})
            for v, p in adj.items():
                res[v]["delta"][m]["p_holm"] = p
        out["retrievers"][ret] = res
    return out


# =====================================================================
# 8. REPORT, CSV, PLOTS
# =====================================================================
def md_table(headers: List[str], rows: List[List[str]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def fmt_ci(m: dict, d: int = 3) -> str:
    return f"{m['mean']:.{d}f} [{m['lo']:.2f}–{m['hi']:.2f}]"


def fmt_delta(d: Optional[dict], key: str = "p_holm") -> str:
    if not d:
        return "–"
    star = "*" if d.get(key, d["p"]) < 0.05 else ""
    return f"{d['diff']:+.3f} [{d['lo']:+.2f}, {d['hi']:+.2f}]{star}"


def render_report(an: dict, variants: List[Variant], meta: dict) -> str:
    L = ["# Ewaluacja wzbogacania chunków o metadane", "",
         f"- zapytań: **{an['n_queries']}** (plików: {an['n_files']}), chunków w korpusie: **{meta['n_chunks']}**",
         f"- model embeddingów: `{meta['embed_model']}`, bootstrap: {meta['n_boot']} prób po plikach, seed {meta['seed']}",
         f"- baseline: `{BASELINE}`; `*` = różnica istotna po korekcie Holma (α = 0,05); przedziały to 95% CI", "",
         "## Warianty", "", md_table(["wariant", "opis"], [[f"`{v.name}`", v.desc] for v in variants]), ""]
    sizes = an["subgroup_sizes"]
    for ret, res in an["retrievers"].items():
        L += [f"## {RET_NAMES[ret]}", ""]
        rows = []
        for v in (x.name for x in variants):
            e = res[v]
            m = e["metrics"]
            d = e["delta"].get("MRR@10")
            rows.append([f"`{v}`", fmt_ci(m["Hit@1"], 2), fmt_ci(m["Hit@3"], 2), fmt_ci(m["Hit@5"], 2),
                         fmt_ci(m["MRR@10"]), fmt_ci(m["P@5chunks"], 2), fmt_delta(d) if d else "(baseline)",
                         f"{d['p_holm']:.3f}" if d else "–"])
        L += [md_table(["wariant", "Hit@1", "Hit@3", "Hit@5", "MRR@10", "P@5 chunków", "ΔMRR@10 vs raw", "p (Holm)"], rows), ""]
        sgs = list(sizes)
        rows = [[f"`{v}`"] + [fmt_delta(res[v]["subgroups"][sg]["delta"], "p") for sg in sgs]
                for v in (x.name for x in variants) if v != BASELINE]
        L += [f"**Gdzie pomaga** – ΔMRR@10 względem `raw` w podgrupach (surowe CI; `*` = p<0,05 bez korekty, "
              f"podgrupy są eksploracyjne):", "",
              md_table(["wariant"] + [f"{sg} (n={sizes[sg]})" for sg in sgs], rows), ""]
    L += ["## Jak czytać wyniki", "",
          f"- **Tytuł w zapytaniu: {sizes['tytuł w zapytaniu']} z {an['n_queries']}.** Pytania zawierające tytuł dokumentu "
          "zawyżają zysk z wstrzykiwania tytułu – realni użytkownicy rzadko go wpisują. Wiarygodniejszy jest wynik w kolumnie "
          "„bez tytułu w zapytaniu”.",
          "- Różnica, której CI obejmuje 0 (brak `*`), nie jest dowodem na poprawę – przy tej liczbie zapytań to może być szum.",
          "- `P@5 chunków` mówi, ile z 5 chunków, które trafią do LLM-a, pochodzi z właściwego dokumentu.",
          f"- p-value z bootstrapu ma rozdzielczość 1/{meta['n_boot']}; po korekcie Holma najmniejsze możliwe p to "
          f"{len(variants) - 1}/{meta['n_boot']} = {(len(variants) - 1) / meta['n_boot']:.4f} – identyczne p dla wielu wariantów "
          "oznacza po prostu „poza zasięgiem rozdzielczości”, a nie ten sam efekt.",
          "- BM25 jest lokalny (prefiksowy stemming), więc bezwzględne wartości różnią się od Elasticsearch z analizatorem "
          "polskim; porównanie wariantów między sobą pozostaje miarodajne.",
          "- Warianty LLM (`llm_context`, `doc2query`, ...) są oceniane na pytaniach wygenerowanych przez LLM – to faworyzuje "
          "metody generatywne. Potwierdź je na pytaniach pisanych ręcznie lub z logów."]
    return "\n".join(L)


def write_csv(path: Path, runs, items: List[Item], docs: Dict[str, DocRec], mention: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["retriever", "variant", "query", "type", "title_in_query", "expected", "rank", "p5_chunks", "top3"])
        for ret, by_var in runs.items():
            for v, run in by_var.items():
                for i, it in enumerate(items):
                    w.writerow([ret, v, it.query, it.qtype, int(mention[i]), "; ".join(it.expected_disp),
                                int(run.ranks[i]), f"{run.p5[i]:.2f}", " | ".join(run.top3[i])])


def make_plots(an: dict, variants: List[Variant], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rets = list(an["retrievers"])
    names = [v.name for v in variants]
    others = [n for n in names if n != BASELINE]

    fig, axes = plt.subplots(1, len(rets), figsize=(5.6 * len(rets), 0.42 * len(names) + 2), sharey=True, squeeze=False)
    for ax, ret in zip(axes[0], rets):
        m = [an["retrievers"][ret][n]["metrics"]["MRR@10"] for n in names]
        mean = np.array([x["mean"] for x in m])
        err = [mean - np.array([x["lo"] for x in m]), np.array([x["hi"] for x in m]) - mean]
        ax.barh(range(len(names)), mean, xerr=err, capsize=3, color=["#7f7f7f" if n == BASELINE else "#1f77b4" for n in names])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_title(ret)
        ax.set_xlabel("MRR@10 (95% CI)")
    fig.tight_layout()
    fig.savefig(out_dir / "wykres_mrr.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(rets), figsize=(5.6 * len(rets), 0.42 * len(others) + 2), sharey=True, squeeze=False)
    for ax, ret in zip(axes[0], rets):
        for y, n in enumerate(others):
            d = an["retrievers"][ret][n]["delta"]["MRR@10"]
            sig = d["p_holm"] < 0.05
            ax.errorbar(d["diff"], y, xerr=[[d["diff"] - d["lo"]], [d["hi"] - d["diff"]]], fmt="o", capsize=3,
                        color="#2ca02c" if sig and d["diff"] > 0 else "#d62728" if sig else "#7f7f7f")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(range(len(others)))
        ax.set_yticklabels(others)
        ax.invert_yaxis()
        ax.set_title(ret)
        ax.set_xlabel("ΔMRR@10 vs raw  (zielony/czerwony = istotne po Holmie)")
    fig.tight_layout()
    fig.savefig(out_dir / "wykres_delta_forest.png", dpi=200)
    plt.close(fig)

    sgs = list(an["subgroup_sizes"])
    fig, axes = plt.subplots(1, len(rets), figsize=(1.35 * len(sgs) * len(rets) + 2, 0.42 * len(others) + 3), sharey=True, squeeze=False)
    for ax, ret in zip(axes[0], rets):
        mat = np.array([[(an["retrievers"][ret][n]["subgroups"][sg]["delta"] or {"diff": np.nan})["diff"] for sg in sgs]
                        for n in others], dtype=float)
        lim = max(0.05, np.nanmax(np.abs(mat)))
        ax.imshow(mat, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(sgs)))
        ax.set_xticklabels([f"{s}\n(n={an['subgroup_sizes'][s]})" for s in sgs], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(others)))
        ax.set_yticklabels(others)
        ax.set_title(f"ΔMRR@10 vs raw – {ret}")
    fig.tight_layout()
    fig.savefig(out_dir / "wykres_podgrupy.png", dpi=200)
    plt.close(fig)


# =====================================================================
# 9. DRY-RUN AND EXECUTION
# =====================================================================
def dry_run(variants: List[Variant], chunks: List[ChunkRec], docs: Dict[str, DocRec], items: List[Item], bd) -> None:
    n = len(chunks)
    print(f"\nCorpus: {len(docs)} files, {n} chunks | test queries: {len(items)}")
    tok_raw = sum(bd.tokeny(c.body) for c in chunks)
    tot = 0
    print(f"\n{'variant':<26}{'tokens for embedding':>22}")
    for v in variants:
        if v.needs:
            extra = {"llm_context": 60, "doc_summary": 80, "questions": 45}[v.needs]
            t = tok_raw + n * (extra + 25)
            note = "  (estimate)"
        else:
            t = sum(bd.tokeny(v.build(c)) for c in chunks)
            note = ""
        tot += t
        print(f"{v.name:<26}{t:>22,}{note}")
    print(f"\nTotal ≈ {tot:,} tokens ≈ ${tot / 1e6 * PRICE_PER_1M_EMBED_USD:.3f} "
          f"(no cache; identical texts are counted once). Price from settings: ${PRICE_PER_1M_EMBED_USD}/1M.")
    needs = {v.needs for v in variants if v.needs}
    if needs:
        doc_tok = {k: bd.tokeny(d.body[:MAX_DOC_CHARS]) for k, d in docs.items()}
        ctx_in = sum(doc_tok[c.file] + bd.tokeny(c.body) + 150 for c in chunks)
        print(f"LLM: {n} calls/contextual variant (≈ {ctx_in / 1e6:.1f}M input tokens for llm_context), "
              f"{len(docs)} calls for summaries, {n} for doc2query. Results will be cached.")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Evaluation of chunk metadata enrichment")
    ap.add_argument("--data-dir", default=EVAL_DATA_DIR)
    ap.add_argument("--dataset", default=str(DATASET_PATH))
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--cache", default=str(CACHE_PATH))
    ap.add_argument("--variants", nargs="+", help="subset of variants (raw is added automatically)")
    ap.add_argument("--retrievers", nargs="+", default=list(RETRIEVERS), choices=RETRIEVERS)
    ap.add_argument("--llm", action="store_true", help="include variants with context generated by LLM (incurs cost)")
    ap.add_argument("--llm-model", default=LLM_MODEL)
    ap.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    return ap.parse_args(argv)


def select_variants(args) -> List[Variant]:
    if args.variants:
        unknown = [n for n in args.variants if n not in VARIANT_BY_NAME]
        if unknown:
            sys.exit(f"Unknown variants: {unknown}. Available: {list(VARIANT_BY_NAME)}")
        names = list(dict.fromkeys([BASELINE] + args.variants))
        chosen = [VARIANT_BY_NAME[n] for n in names]
        if any(v.needs for v in chosen) and not args.llm:
            sys.exit("Selected variants require LLM – add --llm.")
        return chosen
    return [v for v in VARIANTS if args.llm or not v.needs]


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.list:
        for v in VARIANTS:
            print(f"{v.name:<24}{'[LLM] ' if v.needs else '      '}{v.desc}")
        return
    bd = import_chunker()
    items = load_dataset(Path(args.dataset))
    chunks, docs = load_chunks(Path(args.data_dir), bd)
    validate(items, docs)
    variants = select_variants(args)
    mention = np.array([title_mentioned(it, docs) for it in items])
    print(f"Corpus: {len(docs)} files, {len(chunks)} chunks | queries: {len(items)} "
          f"({', '.join(f'{k}: {v}' for k, v in Counter(it.qtype for it in items).items())}) | "
          f"with title in query: {int(mention.sum())}")
    if args.dry_run:
        dry_run(variants, chunks, docs, items, bd)
        return

    from openai import OpenAI
    key = load_settings()["openai_key"]
    if not key:
        sys.exit("Missing OPENAI_API_KEY (environment variable / .env / settings).")
    client = OpenAI(api_key=key, max_retries=6)
    cache = Cache(args.cache)
    embedder = Embedder(client, EMBED_MODEL, cache)
    needs = {v.needs for v in variants if v.needs}
    if needs:
        print(f" LLM context generation ({args.llm_model}) – results are cached...")
        add_llm_fields(chunks, docs, LLM(client, args.llm_model, cache), needs)
    qvecs = embedder.embed([it.query for it in items])

    print("\n ENRICHMENT VARIANTS EVALUATION\n")
    runs = run_evaluation(variants, args.retrievers, chunks, items, docs, embedder, qvecs)
    an = analyze(runs, items, docs, args.n_boot, SEED)

    out_dir = Path(args.out) / time.strftime("run_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"n_chunks": len(chunks), "embed_model": EMBED_MODEL, "n_boot": args.n_boot, "seed": SEED}
    report = render_report(an, variants, meta)
    (out_dir / "raport.md").write_text(report, encoding="utf-8")
    (out_dir / "wyniki.json").write_text(json.dumps({"meta": meta, "analysis": an}, ensure_ascii=False, indent=1), encoding="utf-8")
    write_csv(out_dir / "per_query.csv", runs, items, docs, mention)
    try:
        make_plots(an, variants, out_dir)
    except ImportError:
        print(" matplotlib unavailable – skipped plotting")
    print(report)
    print(f"\n cache: API embeddings: {cache.misses['emb']} new / {cache.hits['emb']} cached | "
          f"LLM: {cache.misses['llm']} new / {cache.hits['llm']} cached")
    print(f" Saved report, per_query.csv, wyniki.json, and plots in: {out_dir}")


if __name__ == "__main__":
    main()