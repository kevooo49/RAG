"""Shared building blocks for the evaluate_*.py scripts.

Everything an evaluation needs, in one place, so the scripts only differ in WHAT they compare:

* `load_sample()`      - the sample from build_eval_sample.py (sample_manifest.json), chunked by the PRODUCTION
                         chunker (build_database.chunk_file) with the real collection settings (source, type,
                         legal headers). Chunks are cached.
* `load_questions()`   - eval_questions.json from generate_eval_questions.py (relevant_ids, evidence, flags).
* `Embedder`, `OpenAIEmbedder`, `EmbeddingCache` - vectors cached in SQLite by (model, text) so each text is paid
                         for once across all scripts and runs.
* `DenseIndex`         - exact cosine search (numpy, no HNSW approximation).
* `ESIndex`, `ANALYZERS` - Elasticsearch BM25 indexes with different Polish analyzers (stempel, morfologik...).
* `QueryExpander`      - the PRODUCTION query-expansion prompt (imported from app.services.rag_service), cached.
* `rrf()`              - the PRODUCTION reciprocal rank fusion (same tie-breaking).
* Metrics              - file level (Hit@k, MRR@10, nDCG@10 with several relevant files) and context level
                         (CtxHit@5: is a chunk that contains the evidence among the 5 chunks the LLM would get).
* Statistics           - cluster bootstrap by target document, paired differences, Holm correction.
* `write_run()`        - report.txt + results.json + per_query.csv in <out_root>/run_<timestamp>/.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR.parents[0] / "config"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(BACKEND_DIR / ".env")

from sites import load_or_exit  # noqa: E402  # type: ignore
import build_database as bd  # noqa: E402  (production chunker; also loads app.core.config.settings)
from generate_eval_questions import norm  # noqa: E402  (same normalization as the evidence check)

CACHE_DIR = SCRIPT_DIR / ".eval_cache"
DEFAULT_QUESTIONS = SCRIPT_DIR / "eval_questions.json"
CHUNK_POOL = 40         # production: rag_service.CHUNK_POOL
CONTEXT_K = 5           # production: rag_service.DEFAULT_TOP_K (chunks given to the LLM)
FILE_K = 10
RRF_K = bd.settings.RRF_K


# ==========================================
# 1. SAMPLE + CHUNKS
# ==========================================
@dataclass
class Chunk:
    id: str
    file_id: str            # production `plik`
    text: str               # what ES and the LLM get ([Źródło] Tytuł › Sekcja + content, with links)
    text_emb: str           # what is embedded (breadcrumb + content, URLs removed)
    meta: dict = field(default_factory=dict)


@dataclass
class Sample:
    eval_dir: Path
    manifest: dict
    files: List[dict]
    chunks: List[Chunk]
    fingerprint: str

    @property
    def file_ids(self) -> List[str]:
        return [f["id"] for f in self.files]


def _collection_cfgs(cfg, eval_dir: Path) -> Dict[str, dict]:
    """Collection folder name (as in the manifest) -> chunker config pointing INTO the sample."""
    out = {}
    for c in cfg.collections():
        name = Path(c["path"]).relative_to(cfg.data_root).as_posix()
        out[name] = {**c, "path": str(eval_dir / name)}
    return out


def sample_fingerprint(manifest: dict, extra: str = "") -> str:
    h = hashlib.sha256()
    h.update(manifest.get("created", "").encode())
    h.update(str(len(manifest.get("files", []))).encode())
    h.update(Path(bd.__file__).read_bytes())          # any chunker change invalidates the chunk cache
    h.update(extra.encode())
    return h.hexdigest()[:16]


def load_sample(eval_dir: Optional[Path] = None, quiet: bool = False) -> Sample:
    cfg = load_or_exit()
    eval_dir = Path(eval_dir or cfg.eval_dir).resolve()
    mpath = eval_dir / "sample_manifest.json"
    if not mpath.exists():
        sys.exit(f"No sample_manifest.json in {eval_dir} - run build_eval_sample.py first.")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    files = manifest["files"]
    fp = sample_fingerprint(manifest)
    cache = CACHE_DIR / f"chunks_{fp}.jsonl"
    chunks: List[Chunk] = []
    if cache.exists():
        try:
            with cache.open(encoding="utf-8", newline="\n") as fh:   # iterate on \n only (texts may hold \u2028)
                chunks = [Chunk(**json.loads(line)) for line in fh if line.strip()]
            if not quiet:
                print(f"   [=] {len(chunks)} chunks of {len(files)} files loaded from {cache.name}")
        except (ValueError, TypeError) as e:
            print(f"   [!] chunk cache unreadable ({e}) - chunking again")
            chunks = []
    if not chunks:
        cfgs = _collection_cfgs(cfg, eval_dir)
        t0 = time.time()
        for i, f in enumerate(files, 1):
            c = cfgs.get(f["collection"])
            if c is None:
                print(f"   [!] collection {f['collection']} not in sites.yaml - skipped {f['id']}")
                continue
            path = eval_dir / f["sample_path"]
            doc = bd.prepare_document(path, c)
            if doc is None:
                continue
            for cid, ch, md in bd.chunk_file(path, c, doc, rel=f["id"]):
                chunks.append(Chunk(id=cid, file_id=f["id"], text=ch.text, text_emb=ch.text_emb,
                                    meta={"tytul": md.get("tytul", ""), "sekcja": md.get("sekcja", ""),
                                          "zrodlo": md.get("zrodlo", ""), "typ": md.get("typ", "")}))
            if not quiet and i % 200 == 0:
                print(f"   chunking {i}/{len(files)}", end="\r")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            for c in chunks:
                fh.write(json.dumps(asdict(c), ensure_ascii=True) + "\n")       # ASCII: no stray line breaks
        os.replace(tmp, cache)
        if not quiet:
            print(f"   [+] chunked {len(files)} files -> {len(chunks)} chunks in {time.time() - t0:.0f}s")
    ids = [c.id for c in chunks]
    if len(set(ids)) != len(ids):                      # identical copies give identical chunk IDs
        seen: Dict[str, int] = defaultdict(int)
        for c in chunks:
            seen[c.id] += 1
            if seen[c.id] > 1:
                c.id = f"{c.id}~{seen[c.id]}"
    return Sample(eval_dir, manifest, files, chunks, fp)


# ==========================================
# 2. QUESTIONS
# ==========================================
@dataclass
class Question:
    qid: str
    text: str
    qtype: str
    doc_type: str
    source: str
    expected_id: str
    relevant: Set[str]
    evidence: str
    position: float
    generic: bool
    answer: str = ""


def load_questions(path: Path = DEFAULT_QUESTIONS, sample: Optional[Sample] = None, exclude_generic: bool = False,
                   types: Optional[Sequence[str]] = None, limit: int = 0) -> List[Question]:
    if not Path(path).exists():
        sys.exit(f"Missing {path} - run generate_eval_questions.py first.")
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = raw["questions"] if isinstance(raw, dict) else raw
    known = set(sample.file_ids) if sample else None
    out = []
    for r in raw:
        if "relevant_ids" not in r:
            sys.exit(f"{path} has no relevant_ids - generate it with generate_eval_questions.py.")
        q = Question(r["qid"], r["query"], r["type"], r.get("doc_type", ""), r.get("source", ""), r["expected_id"],
                     set(r["relevant_ids"]), r["evidence"], float(r.get("position", 0)),
                     "generic" in r.get("flags", []), r.get("answer", ""))
        if known is not None:
            q.relevant &= known
            if not q.relevant:
                continue
        if exclude_generic and q.generic:
            continue
        if types and q.qtype not in types:
            continue
        out.append(q)
    return out[:limit] if limit else out


class EvidenceMatcher:
    """Decides which chunks contain the answer of a question (context-level relevance).
    A chunk counts if it belongs to a relevant file AND contains the evidence quote, or at least `min_cover`
    of its word 3-grams (quotes can be split across two chunks)."""

    def __init__(self, sample: Sample, min_cover: float = 0.5):
        self.min_cover = min_cover
        self.by_file: Dict[str, List[Chunk]] = defaultdict(list)
        for c in sample.chunks:
            self.by_file[c.file_id].append(c)
        self._norm: Dict[str, str] = {}

    @staticmethod
    def _grams(t: str) -> Set[str]:
        w = t.split()
        return {" ".join(w[i:i + 3]) for i in range(max(1, len(w) - 2))}

    def relevant_chunks(self, q: Question) -> Set[str]:
        ev = q.evidence
        g_ev = self._grams(ev)
        out = set()
        for fid in q.relevant:
            for c in self.by_file.get(fid, []):
                t = self._norm.get(c.id)
                if t is None:
                    t = self._norm[c.id] = norm(c.text)
                if ev in t or len(g_ev & self._grams(t)) / max(1, len(g_ev)) >= self.min_cover:
                    out.add(c.id)
        return out


# ==========================================
# 3. EMBEDDINGS
# ==========================================
class EmbeddingCache:
    """SQLite: (model key, sha1(text)) -> float32 vector. Shared by all scripts."""

    def __init__(self, path: Path = CACHE_DIR / "embedding_cache.sqlite"):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (model TEXT, h TEXT, v BLOB, PRIMARY KEY (model, h))")

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def get_many(self, model: str, texts: Sequence[str]) -> Dict[str, np.ndarray]:
        keys = list({self.key(t) for t in texts})
        out = {}
        for i in range(0, len(keys), 500):
            part = keys[i:i + 500]
            q = f"SELECT h, v FROM emb WHERE model=? AND h IN ({','.join('?' * len(part))})"
            for h, v in self.db.execute(q, [model, *part]):
                out[h] = np.frombuffer(v, dtype=np.float32)
        return out

    def put_many(self, model: str, texts: Sequence[str], vecs: np.ndarray):
        self.db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?, ?)",
                            [(model, self.key(t), np.asarray(v, dtype=np.float32).tobytes())
                             for t, v in zip(texts, vecs)])
        self.db.commit()


class Embedder:
    """Interface: `name` (cache key, must change when the model/config changes) and `_encode(texts)`."""
    name = "base"
    batch = 64

    def _encode(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def encode(self, texts: Sequence[str], cache: Optional[EmbeddingCache] = None, desc: str = "",
               stats: Optional[dict] = None) -> np.ndarray:
        """L2-normalized vectors; missing ones are computed in batches and cached."""
        texts = list(texts)
        have = cache.get_many(self.name, texts) if cache else {}
        todo = list(dict.fromkeys(t for t in texts if EmbeddingCache.key(t) not in have))
        t0 = time.perf_counter()
        for i in range(0, len(todo), self.batch):
            part = todo[i:i + self.batch]
            vecs = np.asarray(self._encode(part), dtype=np.float32)
            if cache:
                cache.put_many(self.name, part, vecs)
            for t, v in zip(part, vecs):
                have[EmbeddingCache.key(t)] = v
            if desc:
                print(f"   {desc}: embedded {min(i + self.batch, len(todo))}/{len(todo)} new texts", end="\r")
        if todo and desc:
            print()
        if stats is not None:
            stats["new_texts"] = stats.get("new_texts", 0) + len(todo)
            stats["encode_s"] = stats.get("encode_s", 0.0) + (time.perf_counter() - t0)
        m = np.stack([have[EmbeddingCache.key(t)] for t in texts]) if texts else np.zeros((0, 1), np.float32)
        n = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(n == 0, 1, n)


class OpenAIEmbedder(Embedder):
    batch = 128

    def __init__(self, model: str = bd.settings.EMBED_MODEL, dimensions: Optional[int] = bd.settings.EMBED_DIMENSIONS):
        from openai import OpenAI
        self.client = OpenAI(max_retries=6)
        self.model, self.dimensions = model, dimensions
        self.name = f"openai:{model}:{dimensions or 'native'}"

    def _encode(self, texts):
        kw = {"dimensions": self.dimensions} if self.dimensions else {}
        texts = [t[:30000] or " " for t in texts]           # API rejects empty strings / > 8192 tokens
        r = self.client.embeddings.create(model=self.model, input=texts, **kw)
        return [d.embedding for d in sorted(r.data, key=lambda d: d.index)]


class DenseIndex:
    """Exact cosine search over normalized vectors."""

    def __init__(self, chunk_ids: List[str], matrix: np.ndarray):
        self.ids = chunk_ids
        self.m = matrix

    def search(self, qvec: np.ndarray, k: int) -> List[str]:
        s = self.m @ qvec
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        return [self.ids[i] for i in top[np.argsort(-s[top], kind="stable")]]


# ==========================================
# 4. ELASTICSEARCH
# ==========================================
# Analyzer variants. `requires` = substrings of installed ES plugin names (GET _cat/plugins), e.g. the Allegro
# plugin may report itself as "analysis-morfologik" or "elasticsearch-analysis-morfologik".
ANALYZERS: Dict[str, dict] = {
    "standard": {   # no stemming at all - shows what stemming/lemmatization brings
        "requires": [],
        "settings": {"analysis": {"analyzer": {"default": {"type": "standard"}}}},
    },
    "stempel": {    # PRODUCTION (build_database.ES_SETTINGS): Stempel stemmer + Polish stopwords
        "requires": ["stempel"],
        "settings": {"analysis": {"analyzer": {"default": {"type": "polish"}}}},
    },
    "morfologik": {  # as in the NLP course lab (apohllo/nlp, 2-fts): standard -> lowercase -> morfologik -> lowercase
        "requires": ["morfologik"],
        "settings": {"analysis": {
            "filter": {"same_pos_unique": {"type": "unique", "only_on_same_position": True}},
            "analyzer": {"default": {"type": "custom", "tokenizer": "standard",
                                     "filter": ["lowercase", "morfologik_stem", "lowercase", "same_pos_unique"]}}}},
    },
    "morfologik_stop": {  # + Polish stopword list (from the stempel plugin), closer to what `polish` does
        "requires": ["morfologik", "stempel"],
        "settings": {"analysis": {
            "filter": {"same_pos_unique": {"type": "unique", "only_on_same_position": True}},
            "analyzer": {"default": {"type": "custom", "tokenizer": "standard",
                                     "filter": ["lowercase", "polish_stop", "morfologik_stem", "lowercase",
                                                "same_pos_unique"]}}}},
    },
}


def es_client(url: Optional[str] = None):
    from elasticsearch import Elasticsearch
    return Elasticsearch(url or bd.settings.ES_URL, request_timeout=120)


def missing_plugins(es, required: Sequence[str]) -> List[str]:
    installed = es_plugins(es)
    return [r for r in required if not any(r in name for name in installed)]


def es_plugins(es) -> Set[str]:
    try:
        return {p["component"] for p in es.cat.plugins(format="json")}
    except Exception:
        return set()


class ESIndex:
    """One BM25 index of the sample's chunks (field `tekst` = chunk.text, as in production)."""

    def __init__(self, es, analyzer: str, sample: Sample, rebuild: bool = False):
        self.es, self.analyzer = es, analyzer
        spec = ANALYZERS[analyzer]
        fp = hashlib.sha1((sample.fingerprint + json.dumps(spec["settings"], sort_keys=True)).encode()).hexdigest()[:10]
        self.name = f"evalbm25_{analyzer}_{fp}"
        exists = es.indices.exists(index=self.name)
        if exists and not rebuild and es.count(index=self.name)["count"] == len(sample.chunks):
            print(f"   [=] ES index {self.name} reused")
            return
        if exists:
            es.indices.delete(index=self.name)
        # evalbm25_* indexes of other samples / subsets are kept; delete them by hand if disk space matters:
        #   curl -X DELETE "localhost:9200/evalbm25_*"
        es.indices.create(index=self.name, settings={**spec["settings"], "number_of_shards": 1,
                                                     "number_of_replicas": 0},
                          mappings={"properties": {"tekst": {"type": "text"}, "plik": {"type": "keyword"}}})
        from elasticsearch.helpers import bulk
        bulk(es, ({"_index": self.name, "_id": c.id, "_source": {"tekst": c.text, "plik": c.file_id}}
                  for c in sample.chunks), chunk_size=500)
        es.indices.refresh(index=self.name)
        print(f"   [+] ES index {self.name}: {es.count(index=self.name)['count']} chunks ({analyzer})")

    def search(self, query: str, k: int) -> List[str]:
        r = self.es.search(index=self.name, query={"match": {"tekst": query}}, size=k, source=False)
        return [h["_id"] for h in r["hits"]["hits"]]

    def analyze(self, text: str) -> List[str]:
        r = self.es.indices.analyze(index=self.name, text=text)
        return [t["token"] for t in r["tokens"]]


# ==========================================
# 5. QUERY EXPANSION (production prompt) + FUSION
# ==========================================
class QueryExpander:
    """Production query expansion: the prompt and the post-processing are imported from rag_service, the model is
    settings.EXPANSION_MODEL, temperature 0. Results (and the measured latency of the first call) are cached."""

    def __init__(self, path: Path = CACHE_DIR / "query_expansions.json"):
        from app.services import rag_service as rs
        from openai import OpenAI
        self.rs, self.client, self.path = rs, OpenAI(max_retries=4), path
        self.model = bd.settings.EXPANSION_MODEL
        self.salt = hashlib.sha1((self.model + rs._EXPANSION_PROMPT).encode()).hexdigest()[:10]
        self.cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def expand(self, query: str) -> Tuple[str, float]:
        key = f"{self.salt}|{query}"
        if key not in self.cache:
            t0 = time.perf_counter()
            r = self.client.chat.completions.create(
                model=self.model, temperature=0,
                messages=[{"role": "system", "content": self.rs._EXPANSION_PROMPT},
                          {"role": "user", "content": f"Oryginalne zapytanie: {query}"}])
            ms = (time.perf_counter() - t0) * 1000
            self.cache[key] = {"text": self.rs._sanitize_expansion(query, r.choices[0].message.content or ""),
                               "ms": ms}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=0), encoding="utf-8")
        e = self.cache[key]
        return e["text"], e["ms"]


def rrf(lists: Sequence[Sequence[str]], k: int = RRF_K) -> List[str]:
    """Same as rag_service._reciprocal_rank_fusion (score, then best rank, then id)."""
    scores: Dict[str, float] = defaultdict(float)
    best: Dict[str, int] = {}
    for lst in lists:
        for r, d in enumerate(lst, start=1):
            scores[d] += 1.0 / (r + k)
            best[d] = min(best.get(d, r), r)
    return [d for d, _ in sorted(scores.items(), key=lambda kv: (-kv[1], best[kv[0]], kv[0]))]


# ==========================================
# 6. METRICS
# ==========================================
@dataclass
class QResult:
    system: str
    qid: str
    rank: int                   # rank of the first relevant FILE (0 = not in top FILE_K)
    ndcg: float
    ctx_hit: bool               # evidence chunk among the first CONTEXT_K chunks
    ctx_rel: int                # how many of the first CONTEXT_K chunks come from a relevant file
    top_files: List[str]
    ms: float = 0.0             # retrieval time (without query embedding / expansion)
    llm_ms: float = 0.0         # expansion latency (first, uncached call)
    ctx_budget_hit: bool = False  # evidence chunk within a fixed character budget (set by evaluate_chunking)
    ctx_chars: int = 0          # characters in the first CONTEXT_K chunks (what the LLM would read)


def score_ranking(system: str, q: Question, chunk_ranking: List[str], chunk_file: Dict[str, str],
                  rel_chunks: Set[str], ms: float = 0.0, llm_ms: float = 0.0) -> QResult:
    files: List[str] = []
    for cid in chunk_ranking:
        f = chunk_file[cid]
        if f not in files:
            files.append(f)
        if len(files) >= FILE_K:
            break
    rank = next((i for i, f in enumerate(files, 1) if f in q.relevant), 0)
    dcg = sum(1 / math.log2(i + 1) for i, f in enumerate(files, 1) if f in q.relevant)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(q.relevant), FILE_K) + 1))
    ctx = chunk_ranking[:CONTEXT_K]
    return QResult(system, q.qid, rank, dcg / idcg if idcg else 0.0, any(c in rel_chunks for c in ctx),
                   sum(chunk_file[c] in q.relevant for c in ctx), files[:5], ms, llm_ms)


METRICS: Dict[str, Callable[[QResult], float]] = {
    "Hit@1": lambda r: float(r.rank == 1),
    "Hit@3": lambda r: float(0 < r.rank <= 3),
    "Hit@5": lambda r: float(0 < r.rank <= 5),
    "Hit@10": lambda r: float(0 < r.rank <= 10),
    "MRR@10": lambda r: 1.0 / r.rank if r.rank else 0.0,
    "nDCG@10": lambda r: r.ndcg,
    "CtxHit@5": lambda r: float(r.ctx_hit),
    "CtxPrec@5": lambda r: r.ctx_rel / CONTEXT_K,
    "CtxHit@B": lambda r: float(r.ctx_budget_hit),   # only meaningful where a script sets it (chunking)
}


def budget_context(chunk_ranking: Sequence[str], chunk_len: Dict[str, int], budget: int) -> List[str]:
    """Top chunks in rank order while their total length fits in `budget` characters (at least one chunk)."""
    out, used = [], 0
    for cid in chunk_ranking:
        n = chunk_len[cid]
        if out and used + n > budget:
            break
        out.append(cid)
        used += n
    return out


# ==========================================
# 7. STATISTICS
# ==========================================
class Bootstrap:
    """Cluster bootstrap: resample TARGET DOCUMENTS (questions of one document are not independent)."""

    def __init__(self, questions: List[Question], n: int = 2000, seed: int = 42):
        groups: Dict[str, List[int]] = defaultdict(list)
        for i, q in enumerate(questions):
            groups[q.expected_id].append(i)
        g = list(groups.values())
        rng = np.random.default_rng(seed)
        self.samples = [np.concatenate([g[j] for j in rng.integers(0, len(g), len(g))]) for _ in range(n)]

    def ci(self, v: np.ndarray) -> Tuple[float, float, float]:
        b = np.array([v[s].mean() for s in self.samples])
        return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    def paired(self, a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
        d = a - b
        bs = np.array([d[s].mean() for s in self.samples])
        p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
        return {"diff": float(d.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                "p": float(min(1.0, max(p, 1 / len(bs))))}


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    order = sorted(pvals, key=pvals.get)
    m, out, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def summarize(results: Dict[str, List[QResult]], questions: List[Question], baseline: str,
              boot: Bootstrap, metrics: Sequence[str] = ("Hit@1", "Hit@5", "MRR@10", "nDCG@10", "CtxHit@5",
                                                        "CtxPrec@5"),
              primary: Sequence[str] = ("MRR@10", "CtxHit@5")) -> dict:
    """Per system: mean + CI of every metric; vs baseline: paired diff + CI + p (+ Holm over systems)."""
    vals = {s: {m: np.array([METRICS[m](r) for r in rs]) for m in METRICS} for s, rs in results.items()}
    summ = {"n_questions": len(questions), "n_docs": len({q.expected_id for q in questions}),
            "baseline": baseline, "systems": {}}
    for s, rs in results.items():
        e = {m: dict(zip(("mean", "lo", "hi"), boot.ci(vals[s][m]))) for m in metrics}
        ms = np.array([r.ms for r in rs])
        e["retrieval_ms_median"] = float(np.median(ms))
        e["retrieval_ms_p95"] = float(np.percentile(ms, 95))
        llm = [r.llm_ms for r in rs if r.llm_ms]
        e["llm_ms_mean"] = float(np.mean(llm)) if llm else 0.0
        summ["systems"][s] = e
    if baseline in results:
        for m in primary:
            raw = {s: boot.paired(vals[s][m], vals[baseline][m]) for s in results if s != baseline}
            adj = holm({s: d["p"] for s, d in raw.items()})
            for s, d in raw.items():
                summ["systems"][s][f"d_{m}"] = {**d, "p_holm": adj[s]}
    return summ


def subgroup_table(results: Dict[str, List[QResult]], questions: List[Question], key: Callable[[Question], str],
                   metric: str = "MRR@10") -> Dict[str, Dict[str, Tuple[float, int]]]:
    idx: Dict[str, List[int]] = defaultdict(list)
    for i, q in enumerate(questions):
        idx[key(q)].append(i)
    return {s: {g: (float(np.mean([METRICS[metric](rs[i]) for i in ii])), len(ii)) for g, ii in sorted(idx.items())}
            for s, rs in results.items()}


# ==========================================
# 8. OUTPUT
# ==========================================
def fmt_ci(e: dict) -> str:
    return f"{e['mean']:.3f} [{e['lo']:.2f}-{e['hi']:.2f}]"


def fmt_diff(d: Optional[dict]) -> str:
    if not d:
        return "(baseline)"
    star = "*" if d["p_holm"] < 0.05 else " "
    return f"{d['diff']:+.3f} [{d['lo']:+.2f},{d['hi']:+.2f}] p={d['p_holm']:.3f}{star}"


def standard_report(title: str, summ: dict, results: Dict[str, List[QResult]], questions: List[Question],
                    names: Dict[str, str], notes: Sequence[str] = ()) -> List[str]:
    L = ["=" * 110, title, "=" * 110,
         f"Questions: {summ['n_questions']} | target documents: {summ['n_docs']} | baseline: {summ['baseline']} | "
         f"CI: cluster bootstrap by document, 95% | p: paired bootstrap, Holm-corrected over systems (* < 0.05)",
         f"File level: first relevant file (copies count) in top {FILE_K}. "
         f"Context level: CtxHit@5 = a chunk containing the evidence is among the {CONTEXT_K} chunks given to the LLM.",
         *notes, "",
         f"{'System':<34}{'Hit@1':>7}{'Hit@5':>7}{'MRR@10 [95% CI]':>22}{'CtxHit@5 [95% CI]':>22}{'ms med':>9}"
         f"{'LLM ms':>8}", "-" * 110]
    order = sorted(summ["systems"], key=lambda s: -summ["systems"][s]["MRR@10"]["mean"])
    for s in order:
        e = summ["systems"][s]
        L.append(f"{names.get(s, s)[:33]:<34}{e['Hit@1']['mean']:>7.3f}{e['Hit@5']['mean']:>7.3f}"
                 f"{fmt_ci(e['MRR@10']):>22}{fmt_ci(e['CtxHit@5']):>22}{e['retrieval_ms_median']:>9.0f}"
                 f"{e['llm_ms_mean']:>8.0f}")
    L += ["", f"Differences vs {summ['baseline']}:", f"{'System':<34}{'dMRR@10':>38}{'dCtxHit@5':>38}", "-" * 110]
    for s in order:
        e = summ["systems"][s]
        L.append(f"{names.get(s, s)[:33]:<34}{fmt_diff(e.get('d_MRR@10')):>38}{fmt_diff(e.get('d_CtxHit@5')):>38}")
    for label, key in (("question type", lambda q: q.qtype), ("document type", lambda q: q.doc_type),
                       ("position in document (thirds)", lambda q: ("start", "middle", "end")[min(2, int(q.position * 3))]),
                       ("generic (evidence in > 5 docs)", lambda q: "generic" if q.generic else "specific")):
        tab = subgroup_table(results, questions, key)
        groups = list(next(iter(tab.values())).keys())
        L += ["", f"MRR@10 by {label}:",
              f"{'System':<34}" + "".join(f"{g[:12] + ' (' + str(tab[order[0]][g][1]) + ')':>20}" for g in groups)]
        for s in order:
            L.append(f"{names.get(s, s)[:33]:<34}" + "".join(f"{tab[s][g][0]:>20.3f}" for g in groups))
    return L


def write_run(out_root: Path, report: List[str], summ: dict, results: Dict[str, List[QResult]],
              questions: List[Question], extra: Optional[dict] = None) -> Path:
    run = out_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    (run / "results.json").write_text(json.dumps({**summ, **(extra or {})}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    qmap = {q.qid: q for q in questions}
    with (run / "per_query.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["system", "qid", "type", "doc_type", "position", "generic", "query", "expected_id", "rank",
                    "ctx_hit", "ctx_rel", "ms", "top_files"])
        for s, rs in results.items():
            for r in rs:
                q = qmap[r.qid]
                w.writerow([s, r.qid, q.qtype, q.doc_type, f"{q.position:.3f}", int(q.generic), q.text,
                            q.expected_id, r.rank,
                            int(r.ctx_hit), r.ctx_rel, f"{r.ms:.1f}", " | ".join(f.rsplit("/", 1)[-1] for f in r.top_files)])
    print("\n".join(report))
    try:
        import eval_plots
        made = eval_plots.plot_run(run)
        print(f"\nPlots: {', '.join(p.name for p in made)}")
    except Exception as e:                       # a plotting problem must never lose the results
        print(f"\n[!] plots failed ({e}); redraw later with: python eval_plots.py {run}")
    print(f"Saved to {run}")
    return run