"""Metadata (chunk header) evaluation: what context, added to each chunk, helps retrieval most.

Data: the sample from build_eval_sample.py + questions from generate_eval_questions.py (see eval_common.py).
Every variant uses THE SAME production chunks (build_database.chunk_file, MAX_TOKENS = 450); only the header put in
front of (or after) the chunk content changes. Production keeps two texts per chunk, and so does this evaluation:
    embedded text   header + content without URLs        -> vector search
    BM25 text       header + content with links          -> Elasticsearch (`polish`) and the LLM
A variant `E/B` uses header E for embeddings and header B for BM25; a plain name uses the same header for both.

Headers (title, section path and source come from the production chunker, URL from the file's frontmatter):
    raw                 nothing - the chunk content alone
    title               Tytuł
    section             Sekcja › Podsekcja                      (no title)
    breadcrumb          Tytuł › Sekcja › Podsekcja              (production: embedded text)
    prod_header         [Źródło] Tytuł › Sekcja › Podsekcja     (production: BM25 / LLM text)
    source_title        [Źródło] Tytuł                          (no sections)
    title_last_section  Tytuł - Ostatnia sekcja
    labelled            Dokument: Tytuł / Sekcja: … / Źródło: … (one field per line)
    breadcrumb_url      Tytuł › Sekcja + words from the URL path
    suffix              breadcrumb AFTER the content
    production          alias for breadcrumb/prod_header (what production does) - the baseline (hybrid)

Retrievers (--retrievers): hybrid (default; RRF dense + BM25 as in production), dense, bm25.

Extra breakdown: questions that contain the document title vs those that don't - a title in the header can only
help BM25 when the question uses its words.

Usage (from backend/scripts; Elasticsearch running, OPENAI_API_KEY in backend/.env):
    python evaluate_metadata.py
    python evaluate_metadata.py --retrievers hybrid,dense,bm25
    python evaluate_metadata.py --variants production,title,source_title,breadcrumb/title
Cost: every new embedded header embeds the sample once (~7-8M tokens, ~$0.15 with 3-small, then cached).
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple
from urllib.parse import unquote, urlparse

import numpy as np

import eval_common as ec

OUT_ROOT = ec.SCRIPT_DIR / "meta_eva_stats"
DEFAULT_VARIANTS = ("production,raw,title,section,breadcrumb,prod_header,source_title,title_last_section,"
                    "labelled,breadcrumb_url,suffix,breadcrumb/title,breadcrumb/source_title")
URL_STOP = {"www", "agh", "edu", "pl", "html", "htm", "php", "index", "pl_pl", "en", "page", "pages", "node"}


# ==========================================
# 1. CHUNK PARTS + HEADERS
# ==========================================
@dataclass
class Parts:
    chunk: ec.Chunk
    title: str
    section: str
    source: str
    url_words: str
    content_emb: str        # production embedded text without its breadcrumb line
    content_raw: str        # production BM25 text without its [Źródło] header line
    prod_header: str        # the exact production header line of the BM25 text


def url_words(url: str) -> str:
    if not url:
        return ""
    path = unquote(urlparse(str(url)).path)
    words = [w for w in re.split(r"[/\-_.,+?=&]+", path.lower()) if len(w) > 2 and not w.isdigit()]
    return " ".join(dict.fromkeys(w for w in words if w not in URL_STOP))


def split_parts(sample: ec.Sample) -> List[Parts]:
    cfgs = ec._collection_cfgs(ec.load_or_exit(), sample.eval_dir)
    urls = {}
    for f in sample.files:
        c = cfgs.get(f["collection"])
        doc = ec.bd.prepare_document(sample.eval_dir / f["sample_path"], c) if c else None
        if doc:
            urls[f["id"]] = doc[0].get("url") or doc[0].get("zrodlo") or ""
    out = []
    for c in sample.chunks:
        title, section = c.meta.get("tytul", ""), c.meta.get("sekcja", "")
        bc = " › ".join(x for x in (title, section) if x)
        emb = c.text_emb
        content_emb = emb
        for h in (ec.bd.build_context(c.meta.get("zrodlo", ""), title, ()), bc):   # production header, older format
            if h and emb.startswith(h + "\n\n"):
                content_emb = emb[len(h) + 2:]
                break
        head, _, body = c.text.partition("\n\n")
        out.append(Parts(c, title, section, c.meta.get("zrodlo", ""), url_words(urls.get(c.file_id, "")),
                         content_emb, body if body else c.text, head if body else ""))
    return out


def header(name: str, p: Parts) -> str:
    bc = " › ".join(x for x in (p.title, p.section) if x)
    last = p.section.split(" › ")[-1] if p.section else ""
    if name == "raw":
        return ""
    if name == "title":
        return p.title
    if name == "section":
        return p.section
    if name in ("breadcrumb", "suffix"):
        return bc
    if name == "prod_header":
        return p.prod_header
    if name == "source_title":
        return f"[{p.source}] {p.title}"
    if name == "title_last_section":
        return f"{p.title} - {last}" if last else p.title
    if name == "labelled":
        return "\n".join(x for x in (f"Dokument: {p.title}", f"Sekcja: {p.section}" if p.section else "",
                                    f"Źródło: {p.source}") if x)
    if name == "breadcrumb_url":
        return f"{bc}\n{p.url_words}" if p.url_words else bc
    raise KeyError(name)


HEADERS = ("raw", "title", "section", "breadcrumb", "prod_header", "source_title", "title_last_section", "labelled",
           "breadcrumb_url", "suffix")


def compose(name: str, h: str, content: str) -> str:
    if not h:
        return content
    return f"{content}\n\n{h}" if name == "suffix" else f"{h}\n\n{content}"


def emb_text(name: str, p: Parts) -> str:
    return compose(name, ec.bd.remove_urls(header(name, p)) if name == "breadcrumb_url" else header(name, p),
                   p.content_emb)


def bm25_text(name: str, p: Parts) -> str:
    if name == "prod_header":
        return p.chunk.text                         # exactly what production indexes
    return compose(name, header(name, p), p.content_raw)


def parse_variant(v: str) -> Tuple[str, str]:
    if v == "production":
        return "breadcrumb", "prod_header"
    e, _, b = v.partition("/")
    b = b or e
    for x in (e, b):
        if x not in HEADERS:
            sys.exit(f"Unknown header '{x}' in variant '{v}'; available: {', '.join(HEADERS)} (+ 'production')")
    return e, b


# ==========================================
# 2. MAIN
# ==========================================
def title_in_question(q: ec.Question, title: str) -> bool:
    words = [w for w in re.findall(r"\w+", re.sub(r"_[0-9a-f]{8}$", "", title.lower())) if len(w) > 3 and not w.isdigit()]
    if not words:
        return False
    ql = q.text.lower()
    return sum(w[:6] in ql for w in words) / len(words) >= 0.6


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default=DEFAULT_VARIANTS)
    ap.add_argument("--retrievers", default="hybrid", help="hybrid,dense,bm25")
    ap.add_argument("--baseline", default="hybrid:production")
    ap.add_argument("--questions", type=Path, default=ec.DEFAULT_QUESTIONS)
    ap.add_argument("--es-url", default=ec.bd.settings.ES_URL)
    ap.add_argument("--exclude-generic", action="store_true")
    ap.add_argument("--types", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    retrievers = [r for r in a.retrievers.split(",") if r]
    if set(retrievers) - {"hybrid", "dense", "bm25"}:
        sys.exit("--retrievers: hybrid, dense, bm25")
    variants = [v for v in a.variants.split(",") if v]
    pairs = {v: parse_variant(v) for v in variants}

    # systems: (key, label, dense header or None, bm25 header or None); duplicates of the same texts are skipped
    systems: List[Tuple[str, str, str, str]] = []
    seen = set()
    for r in retrievers:
        for v, (e, b) in pairs.items():
            sig = (r, e if r != "bm25" else None, b if r != "dense" else None)
            if sig in seen:
                continue
            seen.add(sig)
            if r == "hybrid":
                systems.append((f"hybrid:{v}", f"{v} · hybrid", e, b))
            elif r == "dense":
                systems.append((f"dense:{e}", f"{e} · dense", e, None))
            else:
                systems.append((f"bm25:{b}", f"{b} · bm25", None, b))
    need_e = sorted({s[2] for s in systems if s[2]})
    need_b = sorted({s[3] for s in systems if s[3]})

    print("[1/4] Sample and questions")
    sample = ec.load_sample()
    qs = ec.load_questions(a.questions, sample, a.exclude_generic,
                           [t for t in a.types.split(",") if t] or None, a.limit)
    print(f"   {len(qs)} questions about {len({q.expected_id for q in qs})} documents; "
          f"{len(systems)} systems, {len(need_e)} embedded headers, {len(need_b)} BM25 headers")
    parts = split_parts(sample)
    ids = [p.chunk.id for p in parts]
    chunk_file = {p.chunk.id: p.chunk.file_id for p in parts}
    matcher = ec.EvidenceMatcher(sample)
    rel = {q.qid: matcher.relevant_chunks(q) for q in qs}
    titles = {f["id"]: f["title"] for f in sample.files}
    tiq = {q.qid: title_in_question(q, titles.get(q.expected_id, "")) for q in qs}

    print("[2/4] Embeddings and BM25 indexes")
    emb, cache = ec.OpenAIEmbedder(), ec.EmbeddingCache()
    qvecs = emb.encode([q.text for q in qs], cache, "queries")
    dense: Dict[str, ec.DenseIndex] = {}
    for h in need_e:
        texts = [emb_text(h, p) for p in parts]
        dense[h] = ec.DenseIndex(ids, emb.encode(texts, cache, f"chunks [{h}]"))
    es = ec.es_client(a.es_url) if need_b else None
    if es is not None and ec.missing_plugins(es, ec.ANALYZERS["stempel"]["requires"]):
        sys.exit("Elasticsearch has no stempel plugin (needed for the production analyzer).")
    bm25: Dict[str, Dict[str, List[str]]] = {}
    for h in need_b:
        chunks = [ec.Chunk(p.chunk.id, p.chunk.file_id, bm25_text(h, p), p.chunk.text_emb, {}) for p in parts]
        fp = sample.fingerprint if h == "prod_header" else hashlib.sha1(f"{sample.fingerprint}|meta|{h}".encode()).hexdigest()[:16]
        idx = ec.ESIndex(es, "stempel", ec.Sample(sample.eval_dir, sample.manifest, sample.files, chunks, fp))
        bm25[h] = {q.qid: idx.search(q.text, ec.CHUNK_POOL * 2) for q in qs}

    print("[3/4] Rankings")
    dense_rank = {h: {q.qid: d.search(v, ec.CHUNK_POOL * 2) for q, v in zip(qs, qvecs)} for h, d in dense.items()}
    results: Dict[str, List[ec.QResult]] = {}
    names = {}
    for key, label, e, b in systems:
        rs = []
        for q in qs:
            if e and b:
                ranking = ec.rrf([dense_rank[e][q.qid][:ec.CHUNK_POOL], bm25[b][q.qid][:ec.CHUNK_POOL]])
            else:
                ranking = dense_rank[e][q.qid] if e else bm25[b][q.qid]
            rs.append(ec.score_ranking(key, q, ranking, chunk_file, rel[q.qid]))
        results[key], names[key] = rs, label
        print(f"   {label:<40} MRR@10={np.mean([ec.METRICS['MRR@10'](r) for r in rs]):.3f}  "
              f"CtxHit@5={np.mean([r.ctx_hit for r in rs]):.3f}")

    print("[4/4] Statistics and report")
    baseline = a.baseline if a.baseline in results else next(iter(results))
    boot = ec.Bootstrap(qs, a.n_boot)
    summ = ec.summarize(results, qs, baseline, boot)
    notes = [f"Sample: {len(sample.files)} docs, {len(parts)} production chunks | embeddings: {emb.name} | "
             f"BM25: ES `polish` | no query expansion",
             "Variant E/B = header E in the embedded text, header B in the BM25 text; production = breadcrumb/prod_header."]
    rep = ec.standard_report("METADATA (CHUNK HEADER) EVALUATION (evaluate_metadata.py)", summ, results, qs, names, notes)
    tab = ec.subgroup_table(results, qs, lambda q: "title in question" if tiq[q.qid] else "no title in question")
    groups = list(next(iter(tab.values())).keys())
    order = sorted(summ["systems"], key=lambda s: -summ["systems"][s]["MRR@10"]["mean"])
    rep += ["", "MRR@10 by whether the question contains the document title:",
            f"{'System':<34}" + "".join(f"{g + ' (' + str(tab[order[0]][g][1]) + ')':>28}" for g in groups)]
    for s in order:
        rep.append(f"{names[s][:33]:<34}" + "".join(f"{tab[s][g][0]:>28.3f}" for g in groups))
    ex = next((p for p in parts if p.section and p.url_words), parts[0])
    rep += ["", f"Header examples (chunk of {ex.chunk.file_id.rsplit('/', 1)[-1]}):"]
    for h in HEADERS:
        rep.append(f"   {h:<20} {header(h, ex)[:150].replace(chr(10), ' | ') or '(none)'}")
    ec.write_run(OUT_ROOT, rep, summ, results, qs,
                 {"names": names, "title_in_question": sum(tiq.values()), "no_tradeoff": True,
                  "header_examples": {h: header(h, ex) for h in HEADERS},
                  "params": {k: str(v) for k, v in vars(a).items()}})


if __name__ == "__main__":
    main()