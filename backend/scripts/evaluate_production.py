"""Production configuration evaluation: today's RAG setup vs the planned one, end to end, on the evaluation sample.

The two changes proposed by the component evaluations are tested together and one by one (2 x 2):

    system          embedded chunk header                   query expansion (gpt-4o-mini)
    current         Tytuł › Sekcja › Podsekcja (production)  yes (production)
    current_noexp   Tytuł › Sekcja › Podsekcja               no
    planned_exp     [Źródło] Tytuł                           yes
    planned         [Źródło] Tytuł                           no                 <- proposed production setup

Everything else is production: chunks (build_database.chunk_file), BM25 text "[Źródło] Tytuł › Sekcja" in
Elasticsearch (analyzer "polish" / stempel), text-embedding-3-small, RRF (k=60) over top 40 + top 40, top 5 chunks.
With expansion, the expanded query goes to BOTH search legs, as in rag_service.search_by_query.

Part 1 - retrieval, all questions, four systems (metrics from eval_common: Hit@k, MRR@10, CtxHit@5, ...).
Part 2 - answers, a stratified subset (default 300, the same as evaluate_models.py), systems `current` and
          `planned`: production prompt and context blocks (llm_service), answer model (default gpt-4o-mini, the
          production default), the same judge as evaluate_models.py (gpt-4.1). Generations and judgments are cached
          in the same SQLite file, so the planned answers of gpt-4o-mini from evaluate_models.py are reused for free.

Outputs (production_eva_stats/run_<timestamp>/): report.txt, results.json, per_query.csv (retrieval),
per_answer.csv (answers, both systems side by side), the standard retrieval plots and answers.png.

Usage (from backend/scripts, Elasticsearch running):
    python evaluate_production.py --dry-run          # retrieval part + cost estimate, no generation/judge calls
    python evaluate_production.py                    # everything
    python evaluate_production.py --skip-answers     # retrieval only
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import numpy as np

import eval_common as ec
import evaluate_metadata as em
import evaluate_models as emod

OUT_ROOT = ec.SCRIPT_DIR / "production_eva_stats"
SYSTEMS = {  # name: (embedded header, query expansion)
    "current": ("breadcrumb", True),
    "current_noexp": ("breadcrumb", False),
    "planned_exp": ("source_title", True),
    "planned": ("source_title", False),
}
NAMES = {
    "current": "current (breadcrumb + expansion)",
    "current_noexp": "breadcrumb, no expansion",
    "planned_exp": "[Źródło] Tytuł + expansion",
    "planned": "planned ([Źródło] Tytuł, no exp.)",
}
ANSWER_SYSTEMS = ("current", "planned")


# ==========================================
# 1. RETRIEVAL (all four systems)
# ==========================================
def retrieve(sample: ec.Sample, qs: List[ec.Question], es_url: str):
    """Returns parts, {system: {qid: chunk ranking}}, {system: {qid: retrieval ms}}, {qid: expansion ms}."""
    parts = em.split_parts(sample)
    ids = [p.chunk.id for p in parts]
    emb, cache = ec.OpenAIEmbedder(), ec.EmbeddingCache()
    dense = {h: ec.DenseIndex(ids, emb.encode([em.emb_text(h, p) for p in parts], cache, f"chunks [{h}]"))
             for h in sorted({h for h, _ in SYSTEMS.values()})}
    bm25 = ec.ESIndex(ec.es_client(es_url), "stempel", sample)
    expander = ec.QueryExpander()
    expanded, exp_ms = {}, {}
    for q in qs:
        expanded[q.qid], exp_ms[q.qid] = expander.expand(q.text)
    texts = {False: [q.text for q in qs], True: [expanded[q.qid] for q in qs]}
    qvecs = {e: emb.encode(t, cache, "queries" + (" (expanded)" if e else "")) for e, t in texts.items()}
    rankings: Dict[str, Dict[str, List[str]]] = {}
    times: Dict[str, Dict[str, float]] = {}
    for s, (h, e) in SYSTEMS.items():
        rankings[s], times[s] = {}, {}
        for q, t, v in zip(qs, texts[e], qvecs[e]):
            t0 = time.perf_counter()
            rankings[s][q.qid] = ec.rrf([dense[h].search(v, ec.CHUNK_POOL), bm25.search(t, ec.CHUNK_POOL)])
            times[s][q.qid] = (time.perf_counter() - t0) * 1000
    return parts, rankings, times, exp_ms


def contexts_from(sample: ec.Sample, parts, ranking: Dict[str, List[str]], qs: List[ec.Question]):
    """Top CONTEXT_K chunks as production passes them to the LLM (same construction as evaluate_models.py,
    so cached generations and judgments match)."""
    from app.services import llm_service as ls
    cfgs = ec._collection_cfgs(ec.load_or_exit(), sample.eval_dir)
    urls = {}
    for f in sample.files:
        c = cfgs.get(f["collection"])
        doc = ec.bd.prepare_document(sample.eval_dir / f["sample_path"], c) if c else None
        if doc:
            u = doc[0].get("url") or doc[0].get("zrodlo") or ""
            urls[f["id"]] = u if str(u).startswith("http") else ""
    by_id = {p.chunk.id: p for p in parts}
    out = {}
    for q in qs:
        chunks = [(by_id[c].chunk.text, {"score": 0.0, "meta": {"title": by_id[c].title, "source": by_id[c].source,
                                                                 "section": by_id[c].section,
                                                                 "url": urls.get(by_id[c].chunk.file_id, "")}})
                  for c in ranking[q.qid][:ec.CONTEXT_K]]
        out[q.qid] = ls._trim_context(chunks)
    return out


# ==========================================
# 2. ANSWERS (current vs planned)
# ==========================================
def answer_part(a, sample, parts, rankings, rel, sub: List[ec.Question]):
    from app.services import llm_service as ls
    from langchain_core.messages import HumanMessage, SystemMessage
    sysmsg = ls._get_system_prompt()
    prompts: Dict[str, Dict[str, tuple]] = {}
    hits: Dict[str, Dict[str, bool]] = {}
    for s in ANSWER_SYSTEMS:
        ctxs = contexts_from(sample, parts, rankings[s], sub)
        prompts[s] = {}
        hits[s] = {}
        for q in sub:
            ctx = ls._build_context(ctxs[q.qid])
            prompts[s][q.qid] = (ctx, f"{ctx}\n\nPytanie użytkownika: {q.text.strip()}")
            hits[s][q.qid] = any(c in rel[q.qid] for c in rankings[s][q.qid][:len(ctxs[q.qid])])

    store = emod.Store()
    cached = sum(store.get_gen(emod.Store.key("gen", a.model, sysmsg.split("Dzisiejsza data")[0] + prompts[s][q.qid][1]))
                 is not None for s in ANSWER_SYSTEMS for q in sub)
    todo = 2 * len(sub) - cached
    in_tok = np.mean([(len(sysmsg) + len(p[1])) / 3.0 for s in ANSWER_SYSTEMS for p in prompts[s].values()])
    gp = emod.PRICES.get(a.model)
    jp = emod.PRICES.get(a.judge, (2, 8))
    est = todo * (((in_tok * gp[0] + 250 * gp[1]) if gp else 0) + (in_tok + 900) * jp[0] + 180 * jp[1]) / 1e6
    print(f"   answers: {len(sub)} questions x {len(ANSWER_SYSTEMS)} systems, model {a.model}, judge {a.judge} | "
          f"{cached} generations already cached | estimated cost of the rest: ${est:.2f}")
    if a.dry_run:
        return None
    if not a.yes and input("Continue? [y/N] ").strip().lower() not in ("y", "yes", "t", "tak"):
        return None

    judge = emod.Judge(a.judge, store)

    def one(s, q):
        ctx, user = prompts[s][q.qid]
        msgs = [SystemMessage(content=sysmsg), HumanMessage(content=user)]
        try:
            ans, ttft, total = emod.generate(a.model, msgs, store, sysmsg.split("Dzisiejsza data")[0] + user)
            j = judge(q, ctx, ans)
        except Exception as e:                       # not cached -> retried on the next run
            return s, q, {"error": str(e)}
        exact_refusal = ls.NO_ANSWER_FALLBACK.rstrip(".") in ans
        return s, q, {**j, "refused": bool(j["refused"] or exact_refusal), "answer": ans, "ttft_s": ttft,
                      "total_s": total}

    rows: Dict[str, Dict[str, dict]] = {s: {} for s in ANSWER_SYSTEMS}
    t0 = time.time()
    workers = 1 if a.model == "bielik" else a.workers
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, s, q) for s in ANSWER_SYSTEMS for q in sub]
        for n, f in enumerate(as_completed(futs), 1):
            s, q, r = f.result()
            rows[s][q.qid] = r
            print(f"   {n}/{len(futs)}  ({time.time() - t0:.0f}s)".ljust(40), end="\r")
    print(f"   answers done in {time.time() - t0:.0f}s".ljust(40))
    ok = [q for q in sub if all("error" not in rows[s][q.qid] for s in ANSWER_SYSTEMS)]
    if len(ok) < len(sub):
        print(f"   [!] {len(sub) - len(ok)} questions dropped (generation/judge error) - compared on {len(ok)}")
    return {"rows": rows, "ok": ok, "hits": hits, "judge": judge}


def answer_stats(a, res, n_boot: int) -> dict:
    ok, rows, hits = res["ok"], res["rows"], res["hits"]
    boot = ec.Bootstrap(ok, n_boot)
    per = {s: emod.model_metrics([rows[s][q.qid] for q in ok]) for s in ANSWER_SYSTEMS}
    for s in ANSWER_SYSTEMS:
        per[s]["ctx_hit"] = np.array([float(hits[s][q.qid]) for q in ok])
    keys = ["ctx_hit", *emod.SCORES, "correct", "hallucination", "false_refusal", "right_refusal"]
    out = {"n_questions": len(ok), "model": a.model, "judge": a.judge, "systems": {}}
    for s in ANSWER_SYSTEMS:
        e = {k: dict(zip(("mean", "lo", "hi", "n"), emod.ci_nan(boot, per[s][k]))) for k in keys}
        e["refusal_rate"] = float(np.mean([rows[s][q.qid]["refused"] for q in ok]))
        e["total_s_median"] = float(np.median(per[s]["total_s"]))
        e["context_sufficient"] = dict(Counter(rows[s][q.qid]["context_sufficient"] for q in ok))
        out["systems"][s] = e
    raw = {k: emod.paired_nan(boot, per["planned"][k], per["current"][k])
           for k in ("ctx_hit", "correctness", "correct", "faithfulness", "citations", "hallucination")}
    adj = ec.holm({k: d["p"] for k, d in raw.items() if d})
    out["diff"] = {k: {**d, "p_holm": adj[k]} for k, d in raw.items() if d}
    c = {s: [rows[s][q.qid]["correctness"] for q in ok] for s in ANSWER_SYSTEMS}
    out["per_question"] = {"planned_better": int(sum(p > q for p, q in zip(c["planned"], c["current"]))),
                           "same": int(sum(p == q for p, q in zip(c["planned"], c["current"]))),
                           "current_better": int(sum(p < q for p, q in zip(c["planned"], c["current"])))}
    out["by_type"] = {s: {t: float(np.mean([rows[s][q.qid]["correctness"] / 2 for q in ok if q.qtype == t]))
                          for t in sorted({q.qtype for q in ok})} for s in ANSWER_SYSTEMS}
    out["by_doc_type"] = {s: {t: float(np.mean([rows[s][q.qid]["correctness"] / 2 for q in ok if q.doc_type == t]))
                              for t in sorted({q.doc_type for q in ok})} for s in ANSWER_SYSTEMS}
    return out


def answer_report(st: dict, exp_ms: Dict[str, float], judge) -> List[str]:
    W = 110
    L = ["", "=" * W, f"PART 2 - ANSWERS ({st['n_questions']} questions, model {st['model']}, judge {st['judge']})",
         "=" * W,
         f"{'System':<34}{'CtxHit@5':>10}{'Correctness':>20}{'Fully corr.':>12}{'Faithfulness':>20}{'Citations':>12}",
         "-" * W]
    for s in ANSWER_SYSTEMS:
        e = st["systems"][s]
        f = lambda k: f"{e[k]['mean']:.3f} [{e[k]['lo']:.2f}-{e[k]['hi']:.2f}]"
        L.append(f"{NAMES[s][:33]:<34}{e['ctx_hit']['mean']:>10.3f}{f('correctness'):>20}{e['correct']['mean']:>12.3f}"
                 f"{f('faithfulness'):>20}{e['citations']['mean']:>12.3f}")
    L += ["", "planned - current (paired cluster bootstrap, Holm over the metrics below, * p < 0.05):"]
    for k, d in st["diff"].items():
        L.append(f"   {k:<14}{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]  p={d['p_holm']:.3f}"
                 f"{'*' if d['p_holm'] < 0.05 else ''}")
    pq = st["per_question"]
    L += ["", f"Per question (correctness 0-2): planned better {pq['planned_better']}, same {pq['same']}, "
              f"current better {pq['current_better']}", "",
          f"{'System':<34}{'refusals':>10}{'right refusal':>16}{'false refusal':>16}{'hallucination':>15}"
          f"{'answer s med':>14}", "-" * W]
    for s in ANSWER_SYSTEMS:
        e = st["systems"][s]
        r = lambda k: f"{e[k]['mean']:.2f} (n={e[k]['n']})" if e[k]["n"] else "-"
        L.append(f"{NAMES[s][:33]:<34}{e['refusal_rate']:>10.2f}{r('right_refusal'):>16}{r('false_refusal'):>16}"
                 f"{e['hallucination']['mean']:>15.3f}{e['total_s_median']:>13.1f}s")
    for label, key in (("question type", "by_type"), ("document type", "by_doc_type")):
        groups = list(st[key]["current"])
        L += ["", f"Correctness by {label}:", f"{'System':<34}" + "".join(f"{g:>16}" for g in groups)]
        for s in ANSWER_SYSTEMS:
            L.append(f"{NAMES[s][:33]:<34}" + "".join(f"{st[key][s][g]:>16.3f}" for g in groups))
    ms = [v for v in exp_ms.values() if v]
    jp = emod.PRICES.get(st["judge"], (2, 8))
    jc = (judge.usage["input"] * jp[0] + judge.usage["output"] * jp[1]) / 1e6
    L += ["", f"Query expansion (current only): one extra gpt-4o-mini call before every search"
              + (f", mean {np.mean(ms):.0f} ms / median {np.median(ms):.0f} ms (first, uncached calls)" if ms else "")
              + ". The planned setup skips it for standalone questions.",
          f"Judge tokens this run (uncached calls): in {judge.usage['input']}, out {judge.usage['output']} (~${jc:.2f})"]
    return L


def plot_answers(run, st: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import eval_plots as ep
    metrics = [("ctx_hit", "Evidence in context"), ("correctness", "Correctness"), ("correct", "Fully correct"),
               ("faithfulness", "Faithfulness"), ("citations", "Citations")]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    y = np.arange(len(metrics))[::-1]
    for si, s in enumerate(ANSWER_SYSTEMS):
        off = (si - 0.5) * 0.28
        for yi, (k, _) in zip(y, metrics):
            e = st["systems"][s][k]
            ax.plot([e["lo"], e["hi"]], [yi + off] * 2, color=ep.SERIES[si], lw=2, alpha=0.6)
            ax.plot(e["mean"], yi + off, ep.MARKERS[si], ms=8, color=ep.SERIES[si], mec=ep.SURFACE, mew=1.5,
                    label=NAMES[s] if yi == y[0] else None)
            ax.annotate(f"{e['mean']:.2f}", (e["hi"], yi + off), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=8, color=ep.INK2)
    ax.set_yticks(y, [m[1] for m in metrics])
    lo = min(st["systems"][s][k]["lo"] for s in ANSWER_SYSTEMS for k, _ in metrics)
    ax.set_xlim(max(0.0, np.floor((lo - 0.05) * 10) / 10), 1.06)
    ax.set_xlabel("share / score (0-1), mean with 95% CI")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Answers: current vs planned ({st['n_questions']} questions, {st['model']})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5)
    fig.savefig(run / "answers.png")
    plt.close(fig)


# ==========================================
# 3. MAIN
# ==========================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default=str(ec.DEFAULT_QUESTIONS))
    ap.add_argument("--es-url", default=ec.bd.settings.ES_URL)
    ap.add_argument("--model", default="gpt-4o-mini", help="answer model")
    ap.add_argument("--judge", default="gpt-4.1")
    ap.add_argument("--n", type=int, default=300, help="questions in the answer part (equal share per type)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--skip-answers", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="retrieval + cost estimate, no generation/judge calls")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if a.model not in ec.bd.settings.ALLOWED_MODELS:
        sys.exit(f"Not in settings.ALLOWED_MODELS: {a.model}")

    print("[1/4] Sample and questions")
    from pathlib import Path
    sample = ec.load_sample()
    qs = ec.load_questions(Path(a.questions), sample)
    print(f"   {len(qs)} questions about {len({q.expected_id for q in qs})} documents")

    print("[2/4] Retrieval: 4 systems")
    parts, rankings, times, exp_ms = retrieve(sample, qs, a.es_url)
    chunk_file = {p.chunk.id: p.chunk.file_id for p in parts}
    matcher = ec.EvidenceMatcher(sample)
    rel = {q.qid: matcher.relevant_chunks(q) for q in qs}
    results = {s: [ec.score_ranking(s, q, rankings[s][q.qid], chunk_file, rel[q.qid], times[s][q.qid],
                                    exp_ms[q.qid] if SYSTEMS[s][1] else 0.0) for q in qs] for s in SYSTEMS}
    boot = ec.Bootstrap(qs, a.n_boot)
    summ = ec.summarize(results, qs, "current", boot)
    report = ec.standard_report(
        "PRODUCTION CONFIGURATION EVALUATION (evaluate_production.py) - PART 1: RETRIEVAL", summ, results, qs, NAMES,
        notes=["Systems: embedded header (breadcrumb = Tytuł › Sekcja, source_title = [Źródło] Tytuł) x query "
               "expansion (yes/no). BM25 text, chunks, model, RRF and top 5 = production.",
               "LLM ms = mean latency of the expansion call (first, uncached call); 0 = no expansion."])

    answers = None
    if not a.skip_answers:
        print("[3/4] Answers: current vs planned")
        sub = emod.pick_subset(qs, a.n, a.seed)
        res = answer_part(a, sample, parts, rankings, rel, sub)
        if res:
            answers = answer_stats(a, res, a.n_boot)
            report += answer_report(answers, exp_ms, res["judge"])

    print("[4/4] Report")
    run = ec.write_run(OUT_ROOT, report, summ, results, qs,
                       extra={"answers": answers, "params": {k: str(v) for k, v in vars(a).items()}})
    if answers:
        with (run / "per_answer.csv").open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh, delimiter=";")
            w.writerow(["qid", "type", "doc_type", "query", "reference",
                        *[f"{s}_{c}" for s in ANSWER_SYSTEMS for c in ("evidence_in_context", "answer", "correctness",
                                                                         "faithfulness", "citations", "refused",
                                                                         "rationale")]])
            for q in res["ok"]:
                cells = []
                for s in ANSWER_SYSTEMS:
                    r = res["rows"][s][q.qid]
                    cells += [int(res["hits"][s][q.qid]), r["answer"], r["correctness"], r["faithfulness"],
                              r["citations"], int(r["refused"]), r["rationale"]]
                w.writerow([q.qid, q.qtype, q.doc_type, q.text, q.answer, *cells])
        try:
            plot_answers(run, answers)
            print("Answers: per_answer.csv, answers.png")
        except Exception as e:
            print(f"[!] answers plot failed ({e})")


if __name__ == "__main__":
    main()