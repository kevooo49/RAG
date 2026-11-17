"""Search evaluation: vector vs BM25 (several Polish analyzers) vs hybrid (RRF) vs query expansion (vs reranker).

Data: the sample from build_eval_sample.py + questions from generate_eval_questions.py (see eval_common.py).
What is compared (keys for --only / --baseline):

    vector                       dense only (production embedding model, exact cosine)
    bm25_<analyzer>              Elasticsearch BM25 only, analyzer = standard | stempel | morfologik | morfologik_stop
    hybrid_<analyzer>            RRF(dense, BM25) - production fusion (pool 40 + 40, k=60)
    hybrid_<analyzer>_exp        the same on the query rewritten by the PRODUCTION expansion prompt
    hybrid_<analyzer>_exp_rerank + cross-encoder reranking of the top 30 (only with --rerank)

`hybrid_stempel_exp` is what production runs today (default baseline).

Analyzers (eval_common.ANALYZERS):
    standard          no stemming (lowercase only) - what stemming brings at all
    stempel           production: ES `polish` analyzer (Stempel stemmer + Polish stopwords)
    morfologik        dictionary lemmatizer, pipeline from the NLP course lab: standard -> lowercase ->
                      morfologik_stem -> lowercase (+ unique on the same position)
    morfologik_stop   the same + Polish stopword list

Morfologik needs the plugin in the Elasticsearch container (see Dockerfile_ES). Missing plugins are reported and
the analyzers that need them are skipped. A second ES (e.g. apohllo/elasticsearch-morfologik on port 9201) can be
given with --es-url-morfologik.

Metrics (per question): Hit@k / MRR@10 / nDCG@10 on FILES (every copy of the answer counts) and CtxHit@5 -
whether a chunk containing the evidence quote is among the 5 chunks production would pass to the LLM.

Usage (from backend/scripts; Elasticsearch must be running, OPENAI_API_KEY in backend/.env):
    python evaluate_search.py                                # everything except the reranker
    python evaluate_search.py --only bm25_stempel,bm25_morfologik,hybrid_stempel,hybrid_morfologik
    python evaluate_search.py --rerank --exclude-generic
Costs: chunk embeddings once (~$0.2 for 3-small on 2000 files, then cached) + 1 expansion call per question.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

import eval_common as ec

OUT_ROOT = ec.SCRIPT_DIR / "search_eva_stats"
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANK_POOL = 30
DEMO_TEXT = "Studentki składały wnioski o stypendia rektora w dziekanatach wydziałów; terminy zaliczeń w sesji letniej."


@dataclass
class System:
    key: str
    name: str
    run: Callable[["Ctx", ec.Question], tuple]     # -> (chunk ranking, retrieval ms, llm ms)


class Ctx:
    def __init__(self, sample, dense, es_idx, emb, cache, expander, reranker):
        self.sample, self.dense, self.es_idx, self.emb, self.cache = sample, dense, es_idx, emb, cache
        self.expander, self.reranker = expander, reranker
        self.text = {c.id: c.text for c in sample.chunks}
        self._qvec: Dict[str, np.ndarray] = {}

    def qvec(self, text: str) -> np.ndarray:
        if text not in self._qvec:
            self._qvec[text] = self.emb.encode([text], self.cache)[0]
        return self._qvec[text]

    def expanded(self, q: ec.Question):
        return self.expander.expand(q.text) if self.expander else (q.text, 0.0)


def timed(fn, *a):
    t0 = time.perf_counter()
    r = fn(*a)
    return r, (time.perf_counter() - t0) * 1000


def make_systems(analyzers: List[str], hybrid_analyzers: List[str], with_rerank: bool) -> List[System]:
    S = [System("vector", "Vector (dense)",
                lambda c, q: (*timed(c.dense.search, c.qvec(q.text), ec.CHUNK_POOL * 2), 0.0))]
    for a in analyzers:
        S.append(System(f"bm25_{a}", f"BM25 ({a})",
                        lambda c, q, a=a: (*timed(c.es_idx[a].search, q.text, ec.CHUNK_POOL * 2), 0.0)))

    def hybrid(c, text, a):
        d, t1 = timed(c.dense.search, c.qvec(text), ec.CHUNK_POOL)
        b, t2 = timed(c.es_idx[a].search, text, ec.CHUNK_POOL)
        return ec.rrf([d, b]), max(t1, t2)          # production runs both legs in parallel

    for a in hybrid_analyzers:
        S.append(System(f"hybrid_{a}", f"Hybrid RRF ({a})", lambda c, q, a=a: (*hybrid(c, q.text, a), 0.0)))

        def exp(c, q, a=a):
            text, llm_ms = c.expanded(q)
            return (*hybrid(c, text, a), llm_ms)
        S.append(System(f"hybrid_{a}_exp", f"Hybrid ({a}) + Expansion", exp))

        if with_rerank:
            def rr(c, q, a=a):
                text, llm_ms = c.expanded(q)
                fused, ms = hybrid(c, text, a)
                cands = fused[:RERANK_POOL]
                t0 = time.perf_counter()
                scores = c.reranker.predict([[q.text, c.text[i]] for i in cands], batch_size=16,
                                            show_progress_bar=False)
                order = [cands[i] for i in np.argsort(-np.asarray(scores), kind="stable")]
                return order + fused[RERANK_POOL:], ms + (time.perf_counter() - t0) * 1000, llm_ms
            S.append(System(f"hybrid_{a}_exp_rerank", f"Hybrid ({a}) + Exp + Re-Rank", rr))
    return S


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", type=Path, default=ec.DEFAULT_QUESTIONS)
    ap.add_argument("--analyzers", default="standard,stempel,morfologik,morfologik_stop",
                    help="BM25 analyzers to evaluate on their own")
    ap.add_argument("--hybrid-analyzers", default="stempel,morfologik",
                    help="analyzers used inside the hybrid variants")
    ap.add_argument("--only", default="", help="comma-separated system keys")
    ap.add_argument("--baseline", default="hybrid_stempel_exp")
    ap.add_argument("--rerank", action="store_true", help="add cross-encoder variants (sentence-transformers)")
    ap.add_argument("--embed-model", default=ec.bd.settings.EMBED_MODEL)
    ap.add_argument("--es-url", default=ec.bd.settings.ES_URL)
    ap.add_argument("--es-url-morfologik", default="", help="separate ES for morfologik analyzers (optional)")
    ap.add_argument("--rebuild-es", action="store_true")
    ap.add_argument("--exclude-generic", action="store_true")
    ap.add_argument("--types", default="", help="e.g. lexical,semantic")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    print("[1/5] Sample and questions")
    sample = ec.load_sample()
    qs = ec.load_questions(a.questions, sample, a.exclude_generic,
                           [t for t in a.types.split(",") if t] or None, a.limit)
    print(f"   {len(qs)} questions about {len({q.expected_id for q in qs})} documents")
    matcher = ec.EvidenceMatcher(sample)
    rel_chunks = {q.qid: matcher.relevant_chunks(q) for q in qs}
    no_ev = sum(1 for v in rel_chunks.values() if not v)
    if no_ev:
        print(f"   [!] {no_ev} questions: no chunk contains their evidence (CtxHit is 0 for them)")

    print("[2/5] Elasticsearch indexes")
    analyzers = [x for x in a.analyzers.split(",") if x]
    hybrid_an = [x for x in a.hybrid_analyzers.split(",") if x]
    clients = {"main": ec.es_client(a.es_url)}
    if a.es_url_morfologik:
        clients["morf"] = ec.es_client(a.es_url_morfologik)
    es_idx, skipped, demo = {}, {}, {}
    for an in dict.fromkeys(analyzers + hybrid_an):
        if an not in ec.ANALYZERS:
            sys.exit(f"Unknown analyzer {an}; available: {', '.join(ec.ANALYZERS)}")
        es = clients["morf"] if an.startswith("morfologik") and "morf" in clients else clients["main"]
        missing = ec.missing_plugins(es, ec.ANALYZERS[an]["requires"])
        if missing:
            skipped[an] = f"missing ES plugin(s): {', '.join(sorted(missing))}"
            print(f"   [!] {an} skipped - {skipped[an]}")
            continue
        es_idx[an] = ec.ESIndex(es, an, sample, a.rebuild_es)
        demo[an] = es_idx[an].analyze(DEMO_TEXT)
    analyzers = [x for x in analyzers if x in es_idx]
    hybrid_an = [x for x in hybrid_an if x in es_idx]

    systems = make_systems(analyzers, hybrid_an, a.rerank)
    if a.only:
        keep = set(a.only.split(","))
        systems = [s for s in systems if s.key in keep]
    if not systems:
        sys.exit("Nothing to evaluate.")
    keys = [s.key for s in systems]
    baseline = a.baseline if a.baseline in keys else keys[0]
    if baseline != a.baseline:
        print(f"   [!] baseline {a.baseline} not evaluated - using {baseline}")

    print(f"[3/5] Embeddings ({a.embed_model})")
    cache = ec.EmbeddingCache()
    emb = ec.OpenAIEmbedder(a.embed_model, ec.bd.settings.EMBED_DIMENSIONS if a.embed_model == ec.bd.settings.EMBED_MODEL
                            else None)
    stats: dict = {}
    mat = emb.encode([c.text_emb for c in sample.chunks], cache, "chunks", stats)
    dense = ec.DenseIndex([c.id for c in sample.chunks], mat)

    expander = ec.QueryExpander() if any(k.endswith(("_exp", "_rerank")) for k in keys) else None
    reranker = None
    if a.rerank:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(RERANKER_MODEL, max_length=512)

    print("[4/5] Running queries")
    ctx = Ctx(sample, dense, es_idx, emb, cache, expander, reranker)
    if expander:                                      # expand + embed everything up front (cached)
        for i, q in enumerate(qs, 1):
            ctx.expanded(q)
            if i % 50 == 0:
                print(f"   expansion {i}/{len(qs)}", end="\r")
        print()
    texts = [q.text for q in qs] + ([ctx.expanded(q)[0] for q in qs] if expander else [])
    for t, v in zip(texts, emb.encode(texts, cache, "queries")):
        ctx._qvec[t] = v
    chunk_file = {c.id: c.file_id for c in sample.chunks}
    results: Dict[str, List[ec.QResult]] = {}
    for s in systems:
        rs = []
        for q in qs:
            ranking, ms, llm_ms = s.run(ctx, q)
            rs.append(ec.score_ranking(s.key, q, ranking, chunk_file, rel_chunks[q.qid], ms, llm_ms))
        results[s.key] = rs
        print(f"   {s.name:<34} MRR@10={np.mean([ec.METRICS['MRR@10'](r) for r in rs]):.3f}  "
              f"CtxHit@5={np.mean([r.ctx_hit for r in rs]):.3f}")

    print("[5/5] Statistics")
    boot = ec.Bootstrap(qs, a.n_boot)
    summ = ec.summarize(results, qs, baseline, boot)
    names = {s.key: s.name for s in systems}
    notes = [f"Sample: {len(sample.files)} docs, {len(sample.chunks)} chunks | embeddings: {emb.name} | "
             f"expansion: {ec.bd.settings.EXPANSION_MODEL} (production prompt) | ES: {a.es_url}",
             "Latency: dense = numpy exact search (not Chroma HNSW); hybrid = max of the two legs "
             "(they run in parallel in production); LLM ms = expansion, first uncached call."]
    if skipped:
        notes.append("Skipped analyzers: " + "; ".join(f"{k} ({v})" for k, v in skipped.items()))
    report = ec.standard_report("SEARCH EVALUATION (evaluate_search.py)", summ, results, qs, names, notes)
    report += ["", "Analyzer demo - tokens produced for:", f"   \"{DEMO_TEXT}\""]
    report += [f"   {k:<16} {' '.join(v)}" for k, v in demo.items()]
    best = max(results, key=lambda s: summ["systems"][s]["MRR@10"]["mean"])
    misses = [(q, r) for q, r in zip(qs, results[best]) if r.rank == 0 or r.rank > 5]
    report += ["", f"Questions where the best system ({names[best]}) misses the top 5 ({len(misses)}), first 25:"]
    for q, r in misses[:25]:
        report += [f" - [{q.qtype}] {q.text}", f"     expected: {q.expected_id.rsplit('/', 1)[-1]} "
                   f"(+{len(q.relevant) - 1} copies) | got: {', '.join(f.rsplit('/', 1)[-1] for f in r.top_files[:3])}"]
    ec.write_run(OUT_ROOT, report, summ, results, qs,
                 {"names": names, "analyzer_demo": demo, "skipped": skipped, "embedding_stats": stats,
                  "params": {k: str(v) for k, v in vars(a).items()}})


if __name__ == "__main__":
    main()