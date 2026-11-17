"""Embedding model evaluation: which model finds the answer best (dense retrieval only, optionally in the hybrid).

Data: the sample from build_eval_sample.py + questions from generate_eval_questions.py (see eval_common.py).
All models embed THE SAME production chunks (`text_emb` from build_database.chunk_file) and are searched exactly
(numpy cosine), so the only difference is the model.

Models (keys for --models):
    oai-small          text-embedding-3-small (production)
    oai-large          text-embedding-3-large
    bge-m3             BAAI/bge-m3 (local, sentence-transformers)
    silver-retriever   ipipan/silver-retriever-base-v1.1 (local) - Polish QA retriever; queries get the
                       "Pytanie: " prefix and passages the "title</s>text" form, as on the model card
    silver-v1          ipipan/silver-retriever-base-v1 (the previous release of the model)
    e5-base            intfloat/multilingual-e5-base (local; "query: " / "passage: " prefixes)
    spacy-lg           spaCy pl_core_news_lg - average of static word vectors (stopwords/punctuation skipped);
                       a classical non-neural baseline: what contextual models add over word vectors

Default: oai-small, oai-large, bge-m3, silver-retriever, spacy-lg. Baseline: oai-small.

Metrics: Hit@k / MRR@10 / nDCG@10 on files, CtxHit@5 on chunks (see eval_common). Plus cost-side numbers:
    build s      time to embed all chunks of the sample (measured when computed; cached afterwards)
    query ms     median time to embed ONE query (30 live calls: network for OpenAI, inference for local models)
    dim          vector size (index size grows linearly with it)

Options:
    --hybrid        also run RRF(model, BM25 stempel) for every model - does a better model still matter once
                    BM25 is fused in, as in production? (needs Elasticsearch)
    --subset N      use only the documents of the questions + N random others, FOR ALL MODELS of the run
                    (local models on CPU are slow: ~27k chunks can take hours; the cache keeps finished batches,
                    so an interrupted run continues where it stopped)
    --device cuda   for local models (default: cuda if available, else cpu)

Install for local models:  pip install sentence-transformers  |  pip install spacy &&
                           python -m spacy download pl_core_news_lg
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import eval_common as ec

OUT_ROOT = ec.SCRIPT_DIR / "embed_eva_stats"
BUILD_TIMES = ec.CACHE_DIR / "embedding_build_times.json"
N_LATENCY = 30


@dataclass
class Spec:
    key: str
    label: str
    kind: str                      # openai | st | spacy
    name: str
    query_tpl: str = "{q}"
    doc_tpl: str = "{text}"        # fields: text (= chunk.text_emb), title
    extra: dict = field(default_factory=dict)


MODELS: Dict[str, Spec] = {s.key: s for s in [
    Spec("oai-small", "OpenAI 3-small (prod)", "openai", "text-embedding-3-small"),
    Spec("oai-large", "OpenAI 3-large", "openai", "text-embedding-3-large"),
    Spec("bge-m3", "BGE-m3", "st", "BAAI/bge-m3"),
    Spec("silver-retriever", "Silver Retriever v1.1", "st", "ipipan/silver-retriever-base-v1.1",
         query_tpl="Pytanie: {q}", doc_tpl="{title}</s>{text}"),
    Spec("silver-v1", "Silver Retriever v1", "st", "ipipan/silver-retriever-base-v1",
         query_tpl="Pytanie: {q}", doc_tpl="{title}</s>{text}"),
    Spec("e5-base", "multilingual-E5-base", "st", "intfloat/multilingual-e5-base",
         query_tpl="query: {q}", doc_tpl="passage: {text}"),
    Spec("spacy-lg", "spaCy pl_core_news_lg (avg)", "spacy", "pl_core_news_lg"),
]}


class STEmbedder(ec.Embedder):
    batch = 32

    def __init__(self, model: str, device: Optional[str], max_len: int = 512):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model, device=device)
        self.m.max_seq_length = min(self.m.max_seq_length or max_len, max_len)
        self.name = f"st:{model}:{self.m.max_seq_length}"

    def _encode(self, texts):
        return self.m.encode(texts, batch_size=self.batch, normalize_embeddings=True, convert_to_numpy=True,
                             show_progress_bar=False)


class SpacyEmbedder(ec.Embedder):
    """Mean of static word vectors over content tokens (no stopwords, punctuation, numbers-only tokens)."""
    batch = 256

    def __init__(self, model: str):
        import spacy
        self.nlp = spacy.load(model, exclude=["parser", "ner", "tagger", "morphologizer", "lemmatizer",
                                              "attribute_ruler", "senter", "tok2vec", "trainable_lemmatizer"])
        self.nlp.max_length = 2_000_000
        self.dim = self.nlp.vocab.vectors.shape[1]
        if self.dim == 0:
            sys.exit(f"spaCy model {model} has no word vectors - use pl_core_news_md or _lg.")
        self.name = f"spacy:{model}:{self.nlp.meta.get('version', '?')}:content-avg"

    def _encode(self, texts):
        out = []
        for doc in self.nlp.pipe(texts, batch_size=64):
            vs = [t.vector for t in doc if t.has_vector and not (t.is_stop or t.is_punct or t.is_space or t.like_num)]
            out.append(np.mean(vs, axis=0) if vs else np.zeros(self.dim, dtype=np.float32))
        return np.stack(out)


def make_embedder(s: Spec, device: Optional[str]) -> ec.Embedder:
    if s.kind == "openai":
        dims = ec.bd.settings.EMBED_DIMENSIONS if s.name == ec.bd.settings.EMBED_MODEL else None
        return ec.OpenAIEmbedder(s.name, dims)
    if s.kind == "st":
        return STEmbedder(s.name, device)
    return SpacyEmbedder(s.name)


def pick_device(arg: str) -> Optional[str]:
    if arg != "auto":
        return arg
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return None


def restrict(sample: ec.Sample, qs: List[ec.Question], n_other: int, seed: int) -> ec.Sample:
    keep = set().union(*(q.relevant for q in qs))
    rest = sorted(set(sample.file_ids) - keep)
    keep |= set(random.Random(seed).sample(rest, min(n_other, len(rest))))
    files = [f for f in sample.files if f["id"] in keep]
    chunks = [c for c in sample.chunks if c.file_id in keep]
    return ec.Sample(sample.eval_dir, sample.manifest, files, chunks, f"{sample.fingerprint}-sub{n_other}-{seed}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="oai-small,oai-large,bge-m3,silver-retriever,spacy-lg")
    ap.add_argument("--baseline", default="oai-small")
    ap.add_argument("--questions", type=Path, default=ec.DEFAULT_QUESTIONS)
    ap.add_argument("--hybrid", action="store_true", help="also RRF(model, BM25 stempel)")
    ap.add_argument("--es-url", default=ec.bd.settings.ES_URL)
    ap.add_argument("--subset", type=int, default=0, help="documents of the questions + N others (0 = all)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--exclude-generic", action="store_true")
    ap.add_argument("--types", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    keys = [k for k in a.models.split(",") if k]
    bad = [k for k in keys if k not in MODELS]
    if bad:
        sys.exit(f"Unknown model(s) {bad}; available: {', '.join(MODELS)}")

    print("[1/4] Sample and questions")
    sample = ec.load_sample()
    qs = ec.load_questions(a.questions, sample, a.exclude_generic,
                           [t for t in a.types.split(",") if t] or None, a.limit)
    if a.subset:
        sample = restrict(sample, qs, a.subset, a.seed)
        print(f"   subset: {len(sample.files)} docs, {len(sample.chunks)} chunks (same for every model)")
    print(f"   {len(qs)} questions about {len({q.expected_id for q in qs})} documents")
    matcher = ec.EvidenceMatcher(sample)
    rel_chunks = {q.qid: matcher.relevant_chunks(q) for q in qs}
    chunk_file = {c.id: c.file_id for c in sample.chunks}
    ids = [c.id for c in sample.chunks]

    bm25 = {}
    if a.hybrid:
        es = ec.es_client(a.es_url)
        idx = ec.ESIndex(es, "stempel", sample)
        print("   BM25 (stempel) rankings...")
        bm25 = {q.qid: idx.search(q.text, ec.CHUNK_POOL) for q in qs}

    cache = ec.EmbeddingCache()
    build_times = json.loads(BUILD_TIMES.read_text(encoding="utf-8")) if BUILD_TIMES.exists() else {}
    device = pick_device(a.device)
    results: Dict[str, List[ec.QResult]] = {}
    info: Dict[str, dict] = {}
    names: Dict[str, str] = {}

    for n, key in enumerate(keys, 1):
        s = MODELS[key]
        print(f"[2/4] ({n}/{len(keys)}) {s.label} - {s.name}")
        try:
            emb = make_embedder(s, device)
        except (ImportError, OSError) as e:
            print(f"   [!] skipped: {e}")
            continue
        docs = [s.doc_tpl.format(text=c.text_emb, title=c.meta.get("tytul", "")) for c in sample.chunks]
        stats: dict = {}
        mat = emb.encode(docs, cache, "chunks", stats)
        bkey = f"{emb.name}|{sample.fingerprint}"
        if stats.get("new_texts") == len(set(docs)):
            build_times[bkey] = {"s": stats["encode_s"], "estimated": False}
        elif stats.get("new_texts"):
            build_times.setdefault(bkey, {"s": stats["encode_s"] * len(set(docs)) / stats["new_texts"],
                                          "estimated": True})
        BUILD_TIMES.write_text(json.dumps(build_times, indent=1), encoding="utf-8")
        dense = ec.DenseIndex(ids, mat)

        qtexts = [s.query_tpl.format(q=q.text) for q in qs]
        qmat = emb.encode(qtexts, cache, "queries")
        lat = []
        for t in qtexts[:N_LATENCY]:                        # live, uncached single-query latency
            t0 = time.perf_counter()
            emb._encode([t + " "])
            lat.append((time.perf_counter() - t0) * 1000)

        rs, rs_h = [], []
        for q, v in zip(qs, qmat):
            t0 = time.perf_counter()
            d = dense.search(v, ec.CHUNK_POOL * 2)
            ms = (time.perf_counter() - t0) * 1000
            rs.append(ec.score_ranking(key, q, d, chunk_file, rel_chunks[q.qid], ms))
            if a.hybrid:
                fused = ec.rrf([d[:ec.CHUNK_POOL], bm25[q.qid]])
                rs_h.append(ec.score_ranking(f"{key}+bm25", q, fused, chunk_file, rel_chunks[q.qid], ms))
        results[key], names[key] = rs, s.label
        if a.hybrid:
            results[f"{key}+bm25"], names[f"{key}+bm25"] = rs_h, f"{s.label} + BM25"
        bt = build_times.get(bkey, {})
        info[key] = {"model": s.name, "embedder": emb.name, "dim": int(mat.shape[1]),
                     "build_s": bt.get("s"), "build_estimated": bt.get("estimated"),
                     "query_ms_median": float(np.median(lat)), "query_ms_p95": float(np.percentile(lat, 95)),
                     "query_tpl": s.query_tpl, "doc_tpl": s.doc_tpl}
        print(f"   MRR@10={np.mean([ec.METRICS['MRR@10'](r) for r in rs]):.3f}  "
              f"CtxHit@5={np.mean([r.ctx_hit for r in rs]):.3f}  dim={mat.shape[1]}  "
              f"query={info[key]['query_ms_median']:.0f} ms")
        del mat, dense, emb

    if not results:
        sys.exit("No model could be evaluated.")
    print("[3/4] Statistics")
    baseline = a.baseline if a.baseline in results else next(iter(results))
    boot = ec.Bootstrap(qs, a.n_boot)
    summ = ec.summarize(results, qs, baseline, boot)

    print("[4/4] Report")
    notes = [f"Sample: {len(sample.files)} docs, {len(sample.chunks)} chunks{' (subset)' if a.subset else ''} | "
             f"device for local models: {device} | exact cosine search over all chunks"]
    rep = ec.standard_report("EMBEDDING MODELS EVALUATION (evaluate_embeddings.py)", summ, results, qs, names,
                             notes)
    rep += ["", "Cost side:", f"{'Model':<34}{'dim':>7}{'build s':>12}{'query ms (med/p95)':>22}  input format",
            "-" * 110]
    for key, e in info.items():
        b = "-" if e["build_s"] is None else f"{e['build_s']:.0f}{' est.' if e['build_estimated'] else ''}"
        rep.append(f"{names[key]:<34}{e['dim']:>7}{b:>12}{e['query_ms_median']:>13.0f} / {e['query_ms_p95']:<7.0f} "
                   f"q='{e['query_tpl']}' d='{e['doc_tpl']}'")
    rep.append("query ms: OpenAI = network round trip; local = inference on the device above.")
    ec.write_run(OUT_ROOT, rep, summ, results, qs, {"names": names, "models": info,
                                                    "params": {k: str(v) for k, v in vars(a).items()}})


if __name__ == "__main__":
    main()