"""Chunking evaluation: how the way documents are split affects retrieval - alone (dense) and in the hybrid.

Data: the sample from build_eval_sample.py + questions from generate_eval_questions.py (see eval_common.py).
Every strategy splits THE SAME documents of the sample; everything else is fixed (embedding model = production,
exact cosine search, BM25 = Elasticsearch `polish` analyzer, RRF as in production, no query expansion).

Strategies (keys for --strategies):
    production          the production chunker (build_database.chunk_file): Markdown structure, atomic tables,
                        legal headers, sentence overlap, breadcrumb header; MAX_TOKENS = 450
    production_<N>      the same chunker with MAX_TOKENS = N (MIN_TOKENS scaled proportionally), e.g.
                        production_200, production_300, production_650 - "should production use other sizes?"
    fixed_<N>           fixed window of N characters, 10% overlap, no header (e.g. fixed_500, fixed_1000)
    fixed_matched_title     fixed window of the production mean chunk length, document title prepended
    paragraphs_matched_title  paragraphs glued up to the production mean length, document title prepended

For non-production strategies the embedded text has URLs removed (as production does) and the BM25/LLM text is
the raw chunk (with the title line where the strategy has one).

Modes (--modes): dense (vector only), hybrid (RRF of vector + BM25 over the same chunks). Baseline: hybrid production.

Metrics: the standard ones (eval_common) plus
    CtxHit@B    evidence chunk inside a FIXED CHARACTER BUDGET (5 x the mean production chunk length): top chunks
                are taken in rank order while they fit. CtxHit@5 always takes 5 chunks, so it rewards big chunks
                (more text) - CtxHit@B compares strategies at the same amount of text given to the LLM.
    ctx chars   median characters in the first 5 chunks (what the LLM reads with the production top_k = 5)

Usage (from backend/scripts; Elasticsearch running for hybrid, OPENAI_API_KEY in backend/.env):
    python evaluate_chunking.py
    python evaluate_chunking.py --strategies production,production_300,fixed_500 --modes hybrid
Cost: each new strategy embeds the whole sample once with text-embedding-3-small (~$0.2-0.4 per strategy,
then cached; the production strategy is already cached by evaluate_search.py).
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import eval_common as ec

OUT_ROOT = ec.SCRIPT_DIR / "chunk_eva_stats"
DEFAULT_STRATEGIES = ("production,production_200,production_300,production_650,"
                      "fixed_500,fixed_1000,fixed_2000,fixed_matched_title,paragraphs_matched_title")


# ==========================================
# 1. SPLITTERS
# ==========================================
def chunk_fixed(text: str, size: int, overlap_ratio: float = 0.1) -> List[str]:
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


@dataclass
class Strategy:
    key: str
    label: str
    max_tokens: Optional[int] = None                       # production chunker variants
    split: Optional[Callable[[str, int], List[str]]] = None  # (body, matched_size) -> pieces
    title: bool = False


def make_strategy(key: str) -> Strategy:
    if key == "production":
        return Strategy(key, f"Production ({ec.bd.MAX_TOKENS} tok)", max_tokens=ec.bd.MAX_TOKENS)
    m = re.fullmatch(r"production_(\d+)", key)
    if m:
        return Strategy(key, f"Production ({m.group(1)} tok)", max_tokens=int(m.group(1)))
    m = re.fullmatch(r"fixed_(\d+)", key)
    if m:
        n = int(m.group(1))
        return Strategy(key, f"Fixed {n} chars", split=lambda t, _m, n=n: chunk_fixed(t, n))
    if key == "fixed_matched_title":
        return Strategy(key, "Fixed matched + title", split=lambda t, m: chunk_fixed(t, m), title=True)
    if key == "paragraphs_matched_title":
        return Strategy(key, "Paragraphs matched + title", split=lambda t, m: chunk_paragraphs(t, m), title=True)
    sys.exit(f"Unknown strategy {key}")


def production_chunks(sample: ec.Sample, max_tokens: int) -> List[ec.Chunk]:
    """Production chunker with another MAX_TOKENS (module globals are read at call time)."""
    if max_tokens == ec.bd.MAX_TOKENS:
        return sample.chunks
    cfgs = ec._collection_cfgs(ec.load_or_exit(), sample.eval_dir)
    old = (ec.bd.MAX_TOKENS, ec.bd.MIN_TOKENS)
    ec.bd.MAX_TOKENS = max_tokens
    ec.bd.MIN_TOKENS = max(30, round(old[1] * max_tokens / old[0]))
    out: List[ec.Chunk] = []
    try:
        for f in sample.files:
            c = cfgs.get(f["collection"])
            path = sample.eval_dir / f["sample_path"]
            doc = ec.bd.prepare_document(path, c) if c else None
            if doc is None:
                continue
            for n, (_cid, ch, md) in enumerate(ec.bd.chunk_file(path, c, doc, rel=f["id"])):
                out.append(ec.Chunk(f"p{max_tokens}:{f['id']}#{n}", f["id"], ch.text, ch.text_emb,
                                    {"tytul": md.get("tytul", "")}))
    finally:
        ec.bd.MAX_TOKENS, ec.bd.MIN_TOKENS = old
    return out


def simple_chunks(sample: ec.Sample, s: Strategy, matched: int) -> List[ec.Chunk]:
    cfgs = ec._collection_cfgs(ec.load_or_exit(), sample.eval_dir)
    out: List[ec.Chunk] = []
    for f in sample.files:
        c = cfgs.get(f["collection"])
        path = sample.eval_dir / f["sample_path"]
        doc = ec.bd.prepare_document(path, c) if c else None
        if doc is None:
            continue
        _meta, body, title = doc
        for n, piece in enumerate(s.split(body, matched)):
            text = f"{title}\n\n{piece}" if s.title else piece
            out.append(ec.Chunk(f"{s.key}:{f['id']}#{n}", f["id"], text, ec.bd.remove_urls(text), {"tytul": title}))
    return out


def strategy_sample(sample: ec.Sample, s: Strategy, matched: int) -> ec.Sample:
    chunks = production_chunks(sample, s.max_tokens) if s.max_tokens else simple_chunks(sample, s, matched)
    fp = hashlib.sha1(f"{sample.fingerprint}|{s.key}|{s.max_tokens}|{matched if s.split else ''}".encode()
                      ).hexdigest()[:16]
    return ec.Sample(sample.eval_dir, sample.manifest, sample.files, chunks, fp)


# ==========================================
# 2. MAIN
# ==========================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategies", default=DEFAULT_STRATEGIES)
    ap.add_argument("--modes", default="dense,hybrid")
    ap.add_argument("--baseline", default="hybrid:production")
    ap.add_argument("--questions", type=Path, default=ec.DEFAULT_QUESTIONS)
    ap.add_argument("--es-url", default=ec.bd.settings.ES_URL)
    ap.add_argument("--exclude-generic", action="store_true")
    ap.add_argument("--types", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    modes = [m for m in a.modes.split(",") if m]
    if set(modes) - {"dense", "hybrid"}:
        sys.exit("--modes: dense, hybrid")
    strategies = [make_strategy(k) for k in a.strategies.split(",") if k]

    print("[1/4] Sample and questions")
    base = ec.load_sample()
    qs = ec.load_questions(a.questions, base, a.exclude_generic,
                           [t for t in a.types.split(",") if t] or None, a.limit)
    print(f"   {len(qs)} questions about {len({q.expected_id for q in qs})} documents")
    matched = int(np.mean([len(c.text_emb) for c in base.chunks]))
    budget = ec.CONTEXT_K * int(np.mean([len(c.text) for c in base.chunks]))
    print(f"   production mean chunk: {matched} chars (embedded text) | context budget for CtxHit@B: {budget} chars")

    emb = ec.OpenAIEmbedder()
    cache = ec.EmbeddingCache()
    qvecs = emb.encode([q.text for q in qs], cache, "queries")
    es = ec.es_client(a.es_url) if "hybrid" in modes else None
    if es is not None and ec.missing_plugins(es, ec.ANALYZERS["stempel"]["requires"]):
        sys.exit("Elasticsearch has no stempel plugin (needed for the production analyzer).")

    results: Dict[str, List[ec.QResult]] = {}
    names: Dict[str, str] = {}
    stats: Dict[str, dict] = {}
    for n, s in enumerate(strategies, 1):
        print(f"[2/4] ({n}/{len(strategies)}) {s.label}")
        t0 = time.time()
        smp = strategy_sample(base, s, matched)
        lens = np.array([len(c.text) for c in smp.chunks])
        chunk_len = {c.id: len(c.text) for c in smp.chunks}
        chunk_file = {c.id: c.file_id for c in smp.chunks}
        print(f"   {len(smp.chunks)} chunks, mean {lens.mean():.0f} chars ({time.time() - t0:.0f}s)")
        matcher = ec.EvidenceMatcher(smp)
        rel = {q.qid: matcher.relevant_chunks(q) for q in qs}
        est: dict = {}
        mat = emb.encode([c.text_emb for c in smp.chunks], cache, "chunks", est)
        dense = ec.DenseIndex([c.id for c in smp.chunks], mat)
        es_idx = ec.ESIndex(es, "stempel", smp) if es is not None else None
        n_tok = sum(len(c.text_emb) for c in smp.chunks) / 3.0                  # rough, Polish ~3 chars/token
        stats[s.key] = {"label": s.label, "chunks": len(smp.chunks), "per_doc": len(smp.chunks) / len(smp.files),
                        "mean_chars": float(lens.mean()), "median_chars": float(np.median(lens)),
                        "p95_chars": float(np.percentile(lens, 95)), "embed_tokens_est": n_tok,
                        "no_evidence_chunk": sum(1 for v in rel.values() if not v)}
        for mode in modes:
            key = f"{mode}:{s.key}"
            rs = []
            for q, v in zip(qs, qvecs):
                t1 = time.perf_counter()
                d = dense.search(v, ec.CHUNK_POOL * 2)
                ranking = d if mode == "dense" else ec.rrf([d[:ec.CHUNK_POOL], es_idx.search(q.text, ec.CHUNK_POOL)])
                ms = (time.perf_counter() - t1) * 1000
                r = ec.score_ranking(key, q, ranking, chunk_file, rel[q.qid], ms)
                r.ctx_budget_hit = any(c in rel[q.qid] for c in ec.budget_context(ranking, chunk_len, budget))
                r.ctx_chars = sum(chunk_len[c] for c in ranking[:ec.CONTEXT_K])
                rs.append(r)
            results[key] = rs
            names[key] = f"{s.label} · {mode}"
            stats[s.key][f"ctx_chars_median_{mode}"] = float(np.median([r.ctx_chars for r in rs]))
            print(f"   {mode:<6} MRR@10={np.mean([ec.METRICS['MRR@10'](r) for r in rs]):.3f}  "
                  f"CtxHit@5={np.mean([r.ctx_hit for r in rs]):.3f}  CtxHit@B={np.mean([r.ctx_budget_hit for r in rs]):.3f}")
        del mat, dense

    print("[3/4] Statistics")
    baseline = a.baseline if a.baseline in results else next(iter(results))
    boot = ec.Bootstrap(qs, a.n_boot)
    metrics = ("Hit@1", "Hit@5", "MRR@10", "nDCG@10", "CtxHit@5", "CtxPrec@5", "CtxHit@B")
    summ = ec.summarize(results, qs, baseline, boot, metrics=metrics, primary=("MRR@10", "CtxHit@5", "CtxHit@B"))

    print("[4/4] Report")
    notes = [f"Sample: {len(base.files)} docs | embeddings: {emb.name} | BM25: ES `polish` | no query expansion | "
             f"CtxHit@B budget: {budget} chars (= {ec.CONTEXT_K} x mean production chunk)"]
    rep = ec.standard_report("CHUNKING EVALUATION (evaluate_chunking.py)", summ, results, qs, names, notes)
    rep += ["", f"Fixed text budget (CtxHit@B) and LLM context size, differences vs {baseline}:",
            f"{'System':<44}{'CtxHit@B [95% CI]':>22}{'dCtxHit@B':>38}{'ctx chars':>11}", "-" * 115]
    for k in sorted(summ["systems"], key=lambda k: -summ["systems"][k]["CtxHit@B"]["mean"]):
        e = summ["systems"][k]
        sk = k.split(":", 1)[1]
        rep.append(f"{names[k][:43]:<44}{ec.fmt_ci(e['CtxHit@B']):>22}{ec.fmt_diff(e.get('d_CtxHit@B')):>38}"
                   f"{stats[sk].get('ctx_chars_median_' + k.split(':')[0], 0):>11.0f}")
    rep += ["", "Strategy statistics:",
            f"{'Strategy':<30}{'chunks':>8}{'per doc':>9}{'mean ch':>9}{'median':>8}{'p95':>8}"
            f"{'~emb tok':>11}{'no-evid q':>11}", "-" * 94]
    for sk, st in stats.items():
        rep.append(f"{st['label'][:29]:<30}{st['chunks']:>8}{st['per_doc']:>9.1f}{st['mean_chars']:>9.0f}"
                   f"{st['median_chars']:>8.0f}{st['p95_chars']:>8.0f}{st['embed_tokens_est'] / 1e6:>10.1f}M"
                   f"{st['no_evidence_chunk']:>11}")
    rep.append("no-evid q: questions for which no chunk of that strategy contains the evidence quote "
               "(the quote is cut by chunk borders).")
    ec.write_run(OUT_ROOT, rep, summ, results, qs, {"names": names, "strategies": stats, "budget_chars": budget,
                                                    "params": {k: str(v) for k, v in vars(a).items()}})


if __name__ == "__main__":
    main()
