"""Answer-generation evaluation: which model answers best, given what the search really retrieves.

Data: the sample from build_eval_sample.py + questions from generate_eval_questions.py (see eval_common.py).
A stratified subset of questions (default 300: 100 per question type, fixed seed) is answered by EVERY model.

What is taken from production (app/services/llm_service.py), so the models are tested as users meet them:
    system prompt (_get_system_prompt), <source> context blocks (_build_context), context trimming
    (_trim_context, RAG_MAX_CONTEXT_CHARS), model clients (_get_llm: OpenAI, Bielik via Ollama, temperature 0.1),
    refusal sentence (NO_ANSWER_FALLBACK). Single-turn questions, no history.

Context = the top 5 chunks of the hybrid search over the sample:
    --retrieval planned   embedded header "[Źródło] Tytuł", BM25/LLM text as in production, no query expansion
                          (the configuration chosen by evaluate_search / evaluate_metadata) - default
    --retrieval current   production today: breadcrumb embedded header + query expansion

Judge (default gpt-4.1, temperature 0, structured output) sees the question, the reference answer and evidence quote
(from generate_eval_questions.py), the SAME context the model saw, and the model's answer. It returns:
    context_sufficient  yes / partial / no  - does the context contain what is needed to answer?
    correctness         0-2   agreement with the reference answer (2 = complete and correct)
    faithfulness        0-2   every claim supported by the context (0 = significant unsupported/invented content)
    citations           0-2   [n] markers present and pointing to sources that support the sentence (null if refusal)
    language            0-2   correct, natural Polish without stray English / reasoning out loud
    rationale           one or two sentences
Refusal is detected exactly (the production NO_ANSWER_FALLBACK sentence) and by the judge (answer says it does not
know). Good behaviour: answer when the context is sufficient, refuse when it is not.

Outputs (models_eva_stats/run_<timestamp>/): report.txt, results.json, per_answer.csv, plots, and
judge_review.csv - a random sample of judged answers with empty columns for YOUR scores. After filling it in:
    python evaluate_models.py --agreement models_eva_stats/run_.../judge_review.csv
prints the agreement between you and the judge (exact agreement + weighted Cohen's kappa) - a number for the thesis.

Usage (from backend/scripts; Elasticsearch running; Ollama running with the Bielik model for "bielik"):
    python evaluate_models.py --dry-run                     # subset, contexts and cost estimate, no LLM calls
    python evaluate_models.py                               # gpt-4o-mini, gpt-4o, bielik on 300 questions
    python evaluate_models.py --models gpt-4o-mini,gpt-4o --n 100
Generations and judgments are cached (SQLite) - an interrupted run continues; Bielik runs one request at a time.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sqlite3
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import eval_common as ec

OUT_ROOT = ec.SCRIPT_DIR / "models_eva_stats"
DB_PATH = ec.CACHE_DIR / "model_eval.sqlite"
SCORES = ("correctness", "faithfulness", "citations", "language")
PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00), "gpt-4.1": (2.00, 8.00), "gpt-4.1-mini": (0.40, 1.60)}

JUDGE_SYSTEM = """Jesteś surowym, rzetelnym oceniającym odpowiedzi asystenta AI uczelni AGH. Asystent miał odpowiedzieć \
WYŁĄCZNIE na podstawie fragmentów <source>. Oceniasz jedną odpowiedź.

Dostajesz: pytanie, odpowiedź wzorcową i cytat-dowód (z dokumentu, z którego powstało pytanie), fragmenty <source> \
(dokładnie te, które widział asystent) oraz odpowiedź asystenta.

Oceń:
- context_sufficient: czy fragmenty <source> zawierają informacje potrzebne do odpowiedzi na pytanie \
("yes" - w pełni, "partial" - częściowo, "no" - wcale). Oceniaj fragmenty, nie odpowiedź.
- refused: czy asystent odmówił odpowiedzi / stwierdził, że baza nie zawiera informacji (true/false).
- correctness (0-2): zgodność z odpowiedzią wzorcową. 2 = poprawna i kompletna (inne sformułowanie jest w porządku), \
1 = częściowo poprawna lub niekompletna, 0 = błędna, sprzeczna z wzorcem albo brak odpowiedzi (także odmowa).
- faithfulness (0-2): czy KAŻDE twierdzenie odpowiedzi wynika z fragmentów <source>. 2 = wszystko poparte, \
1 = drobne niepoparte szczegóły, 0 = istotne treści zmyślone lub spoza fragmentów. Odmowa = 2.
- citations (0-2 albo null): 2 = zdania oparte na źródłach mają znaczniki [n] wskazujące fragmenty, które je \
popierają, 1 = część znaczników brakuje lub wskazuje zły fragment, 0 = brak znaczników lub wskazują złe fragmenty. \
null, jeśli asystent odmówił.
- language (0-2): 2 = poprawna, naturalna polszczyzna, 1 = drobne błędy lub wtrącenia, \
0 = poważne błędy, inny język, myślenie na głos.
- rationale: 1-2 zdania uzasadnienia po polsku."""

JUDGE_SCHEMA = {
    "name": "answer_judgment", "strict": True,
    "schema": {"type": "object", "additionalProperties": False,
               "required": ["context_sufficient", "refused", "correctness", "faithfulness", "citations", "language",
                            "rationale"],
               "properties": {
                   "context_sufficient": {"type": "string", "enum": ["yes", "partial", "no"]},
                   "refused": {"type": "boolean"},
                   "correctness": {"type": "integer", "enum": [0, 1, 2]},
                   "faithfulness": {"type": "integer", "enum": [0, 1, 2]},
                   "citations": {"type": ["integer", "null"], "enum": [0, 1, 2, None]},
                   "language": {"type": "integer", "enum": [0, 1, 2]},
                   "rationale": {"type": "string"}}},
}


# ==========================================
# 1. CACHE
# ==========================================
class Store:
    def __init__(self, path: Path = DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()
        self.db.execute("CREATE TABLE IF NOT EXISTS gen (k TEXT PRIMARY KEY, answer TEXT, ttft REAL, total REAL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS judge (k TEXT PRIMARY KEY, j TEXT)")
        self.db.commit()

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()

    def get_gen(self, k):
        with self.lock:
            r = self.db.execute("SELECT answer, ttft, total FROM gen WHERE k=?", (k,)).fetchone()
        return r

    def put_gen(self, k, answer, ttft, total):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO gen VALUES (?,?,?,?)", (k, answer, ttft, total))
            self.db.commit()

    def get_judge(self, k):
        with self.lock:
            r = self.db.execute("SELECT j FROM judge WHERE k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else None

    def put_judge(self, k, j):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO judge VALUES (?,?)", (k, json.dumps(j, ensure_ascii=False)))
            self.db.commit()


# ==========================================
# 2. QUESTIONS + CONTEXTS
# ==========================================
def pick_subset(qs: List[ec.Question], n: int, seed: int) -> List[ec.Question]:
    """n questions, equal share per question type, fixed seed (the same subset for every model and run)."""
    by_type: Dict[str, List[ec.Question]] = defaultdict(list)
    for q in sorted(qs, key=lambda q: q.qid):
        by_type[q.qtype].append(q)
    rng = random.Random(seed)
    per = n // max(1, len(by_type))
    out = []
    for t in sorted(by_type):
        out += rng.sample(by_type[t], min(per, len(by_type[t])))
    rest = [q for q in qs if q not in out]
    out += rng.sample(rest, max(0, min(n - len(out), len(rest))))
    return sorted(out, key=lambda q: q.qid)


def build_contexts(sample: ec.Sample, qs: List[ec.Question], mode: str, es_url: str):
    """Returns {qid: [(chunk text, info), ...]} - top CONTEXT_K chunks as production would pass them - and
    {qid: ctx_hit} (a chunk with the evidence quote is in the context)."""
    import evaluate_metadata as em
    from app.services import llm_service as ls
    parts = em.split_parts(sample)
    cfgs = ec._collection_cfgs(ec.load_or_exit(), sample.eval_dir)
    urls = {}
    for f in sample.files:
        c = cfgs.get(f["collection"])
        doc = ec.bd.prepare_document(sample.eval_dir / f["sample_path"], c) if c else None
        if doc:
            u = doc[0].get("url") or doc[0].get("zrodlo") or ""
            urls[f["id"]] = u if str(u).startswith("http") else ""
    by_id = {p.chunk.id: p for p in parts}
    ids = [p.chunk.id for p in parts]
    emb, cache = ec.OpenAIEmbedder(), ec.EmbeddingCache()
    header = "source_title" if mode == "planned" else "breadcrumb"
    dense = ec.DenseIndex(ids, emb.encode([em.emb_text(header, p) for p in parts], cache, f"chunks [{header}]"))
    bm25 = ec.ESIndex(ec.es_client(es_url), "stempel", sample)          # production BM25 text (prod_header)
    expander = ec.QueryExpander() if mode == "current" else None
    texts = [expander.expand(q.text)[0] if expander else q.text for q in qs]
    qvecs = emb.encode(texts, cache, "queries")
    matcher = ec.EvidenceMatcher(sample)
    contexts, hits = {}, {}
    for q, t, v in zip(qs, texts, qvecs):
        ranking = ec.rrf([dense.search(v, ec.CHUNK_POOL), bm25.search(t, ec.CHUNK_POOL)])[:ec.CONTEXT_K]
        chunks = [(by_id[c].chunk.text, {"score": 0.0, "meta": {"title": by_id[c].title, "source": by_id[c].source,
                                                                 "section": by_id[c].section,
                                                                 "url": urls.get(by_id[c].chunk.file_id, "")}})
                  for c in ranking]
        contexts[q.qid] = ls._trim_context(chunks)
        rel = matcher.relevant_chunks(q)
        hits[q.qid] = any(c in rel for c in ranking[:len(contexts[q.qid])])
    return contexts, hits


# ==========================================
# 3. GENERATION + JUDGE
# ==========================================
def generate(model: str, messages: list, store: Store, prompt_key: str) -> Tuple[str, float, float]:
    from app.services import llm_service as ls
    k = Store.key("gen", model, prompt_key)
    cached = store.get_gen(k)
    if cached:
        return cached
    llm = ls._get_llm(model)
    last = None
    for attempt in range(3):
        try:
            t0, ttft, parts = time.perf_counter(), None, []
            for chunk in llm.stream(messages):
                txt = ls._extract_chunk_text(chunk)
                if txt:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    parts.append(txt)
            total = time.perf_counter() - t0
            answer = "".join(parts).strip()
            store.put_gen(k, answer, ttft or total, total)
            return answer, ttft or total, total
        except Exception as e:  # network / Ollama hiccup
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{model}: generation failed: {last}")


class Judge:
    def __init__(self, model: str, store: Store):
        from openai import OpenAI
        self.client, self.model, self.store = OpenAI(max_retries=5), model, store
        self.usage = Counter()
        self.lock = threading.Lock()

    def __call__(self, q: ec.Question, context: str, answer: str) -> dict:
        user = (f"PYTANIE:\n{q.text}\n\nODPOWIEDŹ WZORCOWA:\n{q.answer}\n\nCYTAT-DOWÓD:\n{q.evidence}\n\n"
                f"FRAGMENTY <source>:\n{context}\n\nODPOWIEDŹ ASYSTENTA:\n{answer}")
        k = Store.key("judge", self.model, JUDGE_SYSTEM, user)
        cached = self.store.get_judge(k)
        if cached:
            return cached
        r = self.client.chat.completions.create(
            model=self.model, temperature=0, response_format={"type": "json_schema", "json_schema": JUDGE_SCHEMA},
            messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}])
        with self.lock:
            self.usage["input"] += r.usage.prompt_tokens
            self.usage["output"] += r.usage.completion_tokens
        j = json.loads(r.choices[0].message.content)
        self.store.put_judge(k, j)
        return j


# ==========================================
# 4. STATISTICS + REPORT
# ==========================================
def model_metrics(rows: List[dict]) -> Dict[str, np.ndarray]:
    """Per-question arrays (same question order for every model)."""
    m = {}
    for s in SCORES:
        m[s] = np.array([r[s] / 2 if r[s] is not None else np.nan for r in rows], dtype=float)
    m["correct"] = np.array([float(r["correctness"] == 2) for r in rows])
    m["hallucination"] = np.array([float(r["faithfulness"] == 0) for r in rows])
    suff = np.array([r["context_sufficient"] == "yes" for r in rows])
    none = np.array([r["context_sufficient"] == "no" for r in rows])
    ref = np.array([r["refused"] for r in rows])
    m["false_refusal"] = np.where(suff, ref.astype(float), np.nan)        # refused although the answer was there
    m["right_refusal"] = np.where(none, ref.astype(float), np.nan)        # refused when the context had nothing
    m["total_s"] = np.array([r["total_s"] for r in rows])
    m["ttft_s"] = np.array([r["ttft_s"] for r in rows])
    return m


def ci_nan(boot: ec.Bootstrap, v: np.ndarray) -> Tuple[float, float, float, int]:
    ok = ~np.isnan(v)
    if not ok.any():
        return float("nan"), float("nan"), float("nan"), 0
    means = [np.nanmean(v[s]) for s in boot.samples]
    return float(np.nanmean(v)), float(np.nanpercentile(means, 2.5)), float(np.nanpercentile(means, 97.5)), int(ok.sum())


def paired_nan(boot: ec.Bootstrap, a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    if np.isnan(d).all():
        return {}
    bs = np.array([np.nanmean(d[s]) for s in boot.samples])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return {"diff": float(np.nanmean(d)), "lo": float(np.nanpercentile(bs, 2.5)),
            "hi": float(np.nanpercentile(bs, 97.5)), "p": float(min(1.0, max(p, 1 / len(bs))))}


def weighted_kappa(a: Sequence[int], b: Sequence[int], k: int = 3) -> float:
    """Quadratic-weighted Cohen's kappa for ordinal scores 0..k-1."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    if len(a) == 0:
        return float("nan")
    o = np.zeros((k, k))
    for x, y in zip(a, b):
        o[x, y] += 1
    w = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    den = (w * e).sum()
    return float(1 - (w * o).sum() / den) if den else float("nan")


def agreement(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig"), delimiter=";"))
    print(f"{path}: {len(rows)} rows")
    for s in ("correctness", "faithfulness", "citations", "language"):
        pairs = [(int(r[f"judge_{s}"]), int(r[f"my_{s}"])) for r in rows
                 if r.get(f"my_{s}", "").strip() in ("0", "1", "2") and r.get(f"judge_{s}", "") in ("0", "1", "2")]
        if not pairs:
            print(f"   {s:<13} no ratings")
            continue
        j, m = zip(*pairs)
        exact = np.mean([x == y for x, y in pairs])
        within1 = np.mean([abs(x - y) <= 1 for x, y in pairs])
        print(f"   {s:<13} n={len(pairs):>3}  exact agreement={exact:.2f}  within 1={within1:.2f}  "
              f"weighted kappa={weighted_kappa(j, m):.2f}  (judge mean {np.mean(j):.2f}, yours {np.mean(m):.2f})")


def plot_models(run: Path, summ: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import eval_plots as ep
    models = list(summ["models"])
    metrics = [("correctness", "Correctness (0-1)"), ("faithfulness", "Faithfulness (0-1)"),
               ("citations", "Citations (0-1)"), ("language", "Language (0-1)")]
    fig, axes = plt.subplots(1, len(metrics) + 1, figsize=(16, 0.55 * len(models) + 1.8), sharey=True)
    y = np.arange(len(models))[::-1]
    for ax, (m, title) in zip(axes, metrics):
        for yi, mod in zip(y, models):
            e = summ["models"][mod][m]
            if e["n"] == 0:
                continue
            ax.plot([e["lo"], e["hi"]], [yi, yi], color=ep.ACCENT, lw=2, solid_capstyle="round")
            ax.plot(e["mean"], yi, "o", ms=8, color=ep.ACCENT, mec=ep.SURFACE, mew=2)
            ax.annotate(f"{e['mean']:.2f}", (e["hi"], yi), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=ep.INK2)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 1.12)
        ax.grid(axis="y", visible=False)
    ax = axes[-1]
    for yi, mod in zip(y, models):
        t = summ["models"][mod]["total_s_median"]
        ax.barh(yi, t, height=0.35, color=ep.MUTED)
        ax.annotate(f"{t:.1f} s", (t, yi), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8.5,
                    color=ep.INK2)
    ax.set_xscale("log")
    ax.set_title("Answer time, median (log)", fontsize=10)
    ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, models)
    fig.suptitle(f"{summ['n_questions']} questions, judge {summ['judge']} - mean with 95% CI (cluster bootstrap)",
                 x=0.01, ha="left", fontsize=9, color=ep.INK2, y=0.995)
    fig.savefig(run / "models.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(models) + 1.6))
    labels = [("right_refusal", "refuses when context has nothing (higher = better)"),
              ("false_refusal", "refuses although the answer was there (lower = better)"),
              ("hallucination", "invented content, faithfulness 0 (lower = better)")]
    off = np.linspace(-0.22, 0.22, len(labels))
    for li, (m, lab) in enumerate(labels):
        for yi, mod in zip(y, models):
            e = summ["models"][mod][m]
            if e["n"] == 0:
                continue
            ax.plot([e["lo"], e["hi"]], [yi + off[li]] * 2, color=ep.SERIES[li], lw=2, alpha=0.6)
            ax.plot(e["mean"], yi + off[li], ep.MARKERS[li], ms=7.5, color=ep.SERIES[li], mec=ep.SURFACE, mew=1.5,
                    label=lab if yi == y[0] else None)
    ax.set_yticks(y, models)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("share of answers")
    ax.grid(axis="y", visible=False)
    ax.set_title("Refusals and hallucinations")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5)
    fig.savefig(run / "refusals.png")
    plt.close(fig)


# ==========================================
# 5. MAIN
# ==========================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="gpt-4o-mini,gpt-4o,bielik")
    ap.add_argument("--baseline", default="gpt-4o-mini")
    ap.add_argument("--judge", default="gpt-4.1")
    ap.add_argument("--n", type=int, default=300, help="questions (equal share per question type)")
    ap.add_argument("--retrieval", default="planned", choices=["planned", "current"])
    ap.add_argument("--questions", type=Path, default=ec.DEFAULT_QUESTIONS)
    ap.add_argument("--es-url", default=ec.bd.settings.ES_URL)
    ap.add_argument("--workers", type=int, default=4, help="parallel requests for OpenAI models and the judge")
    ap.add_argument("--review", type=int, default=60, help="judged answers exported for your own scoring")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--agreement", type=Path, help="judge_review.csv with your scores -> agreement with the judge")
    a = ap.parse_args()

    if a.agreement:
        agreement(a.agreement)
        return
    from app.services import llm_service as ls
    models = [m for m in a.models.split(",") if m]
    bad = [m for m in models if m not in ec.bd.settings.ALLOWED_MODELS]
    if bad:
        sys.exit(f"Not in settings.ALLOWED_MODELS: {bad}")

    print("[1/5] Sample, questions, contexts")
    sample = ec.load_sample()
    qs = pick_subset(ec.load_questions(a.questions, sample), a.n, a.seed)
    print(f"   {len(qs)} questions ({dict(Counter(q.qtype for q in qs))}) about {len({q.expected_id for q in qs})} docs")
    contexts, hits = build_contexts(sample, qs, a.retrieval, a.es_url)
    print(f"   retrieval: {a.retrieval} | evidence chunk in the context for {sum(hits.values())}/{len(qs)} questions")
    sysmsg = ls._get_system_prompt()
    prompts = {}
    for q in qs:
        ctx = ls._build_context(contexts[q.qid])
        prompts[q.qid] = (ctx, f"{ctx}\n\nPytanie użytkownika: {q.text.strip()}")

    in_tok = np.mean([(len(sysmsg) + len(p[1])) / 3.0 for p in prompts.values()])
    est = {m: len(qs) * (in_tok * PRICES[m][0] + 250 * PRICES[m][1]) / 1e6 for m in models if m in PRICES}
    j_est = len(qs) * len(models) * ((in_tok + 900) * PRICES.get(a.judge, (2, 8))[0] + 180 * PRICES.get(a.judge, (2, 8))[1]) / 1e6
    print(f"   ~{in_tok:.0f} input tokens per prompt | estimated cost: "
          + ", ".join(f"{m} ${c:.2f}" for m, c in est.items()) + f", judge {a.judge} ${j_est:.2f} "
          f"(cached answers/judgments are free)" + (" | bielik: local, ~1-2 min per answer" if "bielik" in models else ""))
    if a.dry_run:
        print("Dry run - no LLM calls.")
        return
    if not a.yes and input("Continue? [y/N] ").strip().lower() not in ("y", "yes", "t", "tak"):
        return

    from langchain_core.messages import HumanMessage, SystemMessage
    store = Store()
    judge = Judge(a.judge, store)
    rows: Dict[str, Dict[str, dict]] = {m: {} for m in models}

    def one(model, q):
        ctx, user = prompts[q.qid]
        msgs = [SystemMessage(content=sysmsg), HumanMessage(content=user)]
        try:
            ans, ttft, total = generate(model, msgs, store, sysmsg.split("Dzisiejsza data")[0] + user)
        except RuntimeError as e:
            return model, q, {"error": str(e)}
        try:
            j = judge(q, ctx, ans)
        except Exception as e:                      # not cached -> retried on the next run
            return model, q, {"error": f"judge: {e}"}
        exact_refusal = ls.NO_ANSWER_FALLBACK.rstrip(".") in ans
        return model, q, {**j, "refused": bool(j["refused"] or exact_refusal), "answer": ans,
                          "ttft_s": ttft, "total_s": total}

    print("[2/5] Generation + judging")
    for model in models:
        workers = 1 if model == "bielik" else a.workers
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(one, model, q) for q in qs]
            for n, f in enumerate(as_completed(futs), 1):
                m, q, r = f.result()
                rows[m][q.qid] = r
                print(f"   {model:<12} {n}/{len(qs)}  ({time.time() - t0:.0f}s)", end="\r")
        errs = sum("error" in r for r in rows[model].values())
        print(f"   {model:<12} done in {time.time() - t0:.0f}s" + (f"  [!] {errs} generation errors" if errs else ""))

    ok = [q for q in qs if all("error" not in rows[m][q.qid] for m in models)]
    if len(ok) < len(qs):
        print(f"   [!] {len(qs) - len(ok)} questions dropped (a model failed on them) - all models compared on {len(ok)}")
    print("[3/5] Statistics")
    boot = ec.Bootstrap(ok, a.n_boot)
    per = {m: model_metrics([rows[m][q.qid] for q in ok]) for m in models}
    keys = list(SCORES) + ["correct", "hallucination", "false_refusal", "right_refusal"]
    summ = {"n_questions": len(ok), "judge": a.judge, "retrieval": a.retrieval, "baseline": a.baseline,
            "ctx_hit_rate": float(np.mean([hits[q.qid] for q in ok])), "models": {}}
    for m in models:
        e = {k: dict(zip(("mean", "lo", "hi", "n"), ci_nan(boot, per[m][k]))) for k in keys}
        e["total_s_median"] = float(np.median(per[m]["total_s"]))
        e["ttft_s_median"] = float(np.median(per[m]["ttft_s"]))
        e["refusal_rate"] = float(np.mean([rows[m][q.qid]["refused"] for q in ok]))
        summ["models"][m] = e
    if a.baseline in models:
        for k in ("correctness", "faithfulness", "citations", "language"):
            raw = {m: paired_nan(boot, per[m][k], per[a.baseline][k]) for m in models if m != a.baseline}
            adj = ec.holm({m: d["p"] for m, d in raw.items() if d})
            for m, d in raw.items():
                if d:
                    summ["models"][m][f"d_{k}"] = {**d, "p_holm": adj[m]}
    suff = Counter(rows[models[0]][q.qid]["context_sufficient"] for q in ok)
    summ["context_sufficient"] = dict(suff)

    print("[4/5] Report")
    W = 112
    L = ["=" * W, "ANSWER GENERATION EVALUATION (evaluate_models.py)", "=" * W,
         f"Questions: {len(ok)} (stratified by type, seed {a.seed}) | target documents: {len({q.expected_id for q in ok})} | "
         f"retrieval: {a.retrieval} | judge: {a.judge} | baseline: {a.baseline}",
         "Production prompt, context blocks and model clients from app/services/llm_service.py; scores 0-2 shown as 0-1.",
         f"Evidence chunk in the context: {summ['ctx_hit_rate']:.1%} of questions | judge: context sufficient "
         f"yes {suff.get('yes', 0)} / partial {suff.get('partial', 0)} / no {suff.get('no', 0)} "
         f"(judged on {models[0]}'s rows; the context is the same for every model)", "",
         f"{'Model':<14}{'Correctness':>20}{'Fully correct':>15}{'Faithfulness':>20}{'Citations':>20}{'Language':>20}",
         "-" * W]
    for m in models:
        e = summ["models"][m]
        f = lambda k: f"{e[k]['mean']:.3f} [{e[k]['lo']:.2f}-{e[k]['hi']:.2f}]"
        L.append(f"{m:<14}{f('correctness'):>20}{e['correct']['mean']:>15.3f}{f('faithfulness'):>20}"
                 f"{f('citations'):>20}{f('language'):>20}")
    L += ["", f"Differences vs {a.baseline} (paired bootstrap, Holm over models, * p < 0.05):",
          f"{'Model':<14}" + "".join(f"{k:>24}" for k in ("correctness", "faithfulness", "citations", "language")),
          "-" * W]
    for m in models:
        cells = []
        for k in ("correctness", "faithfulness", "citations", "language"):
            d = summ["models"][m].get(f"d_{k}")
            cells.append("(baseline)" if m == a.baseline else
                         (f"{d['diff']:+.3f} p={d['p_holm']:.3f}{'*' if d['p_holm'] < 0.05 else ' '}" if d else "-"))
        L.append(f"{m:<14}" + "".join(f"{c:>24}" for c in cells))
    L += ["", "Refusals, hallucinations, speed:",
          f"{'Model':<14}{'refusals':>10}{'right refusal':>16}{'false refusal':>16}{'hallucination':>15}"
          f"{'TTFT med':>11}{'total med':>11}", "-" * W]
    for m in models:
        e = summ["models"][m]
        r = lambda k: f"{e[k]['mean']:.2f} (n={e[k]['n']})" if e[k]["n"] else "-"
        L.append(f"{m:<14}{e['refusal_rate']:>10.2f}{r('right_refusal'):>16}{r('false_refusal'):>16}"
                 f"{e['hallucination']['mean']:>15.3f}{e['ttft_s_median']:>10.1f}s{e['total_s_median']:>10.1f}s")
    L += ["right refusal = share of refusals among questions whose context had nothing (judge: no);",
          "false refusal = share of refusals among questions whose context was sufficient (judge: yes)."]
    by = lambda key: {g: [i for i, q in enumerate(ok) if key(q) == g] for g in sorted({key(q) for q in ok})}
    for label, key in (("question type", lambda q: q.qtype), ("document type", lambda q: q.doc_type),
                       ("evidence chunk in context", lambda q: "in context" if hits[q.qid] else "not in context")):
        groups = by(key)
        L += ["", f"Correctness by {label}:", f"{'Model':<14}" + "".join(f"{g + ' (' + str(len(ii)) + ')':>22}"
                                                                         for g, ii in groups.items())]
        for m in models:
            L.append(f"{m:<14}" + "".join(f"{np.nanmean(per[m]['correctness'][ii]):>22.3f}" for ii in groups.values()))
    gen_cost = {m: c for m, c in est.items()}
    jc = (judge.usage["input"] * PRICES.get(a.judge, (2, 8))[0] + judge.usage["output"] * PRICES.get(a.judge, (2, 8))[1]) / 1e6
    L += ["", f"Judge tokens this run (uncached calls): in {judge.usage['input']}, out {judge.usage['output']} (~${jc:.2f}); "
          f"estimated generation cost for {len(qs)} questions: " + ", ".join(f"{m} ${c:.2f}" for m, c in gen_cost.items())]

    run = OUT_ROOT / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "report.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    (run / "results.json").write_text(json.dumps({**summ, "params": {k: str(v) for k, v in vars(a).items()}},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    with (run / "per_answer.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["qid", "model", "type", "doc_type", "query", "reference", "evidence_in_context", "answer",
                    "context_sufficient", "refused", *SCORES, "rationale", "ttft_s", "total_s"])
        for q in ok:
            for m in models:
                r = rows[m][q.qid]
                w.writerow([q.qid, m, q.qtype, q.doc_type, q.text, q.answer, int(hits[q.qid]), r["answer"],
                            r["context_sufficient"], int(r["refused"]), *[r[s] for s in SCORES], r["rationale"],
                            f"{r['ttft_s']:.2f}", f"{r['total_s']:.2f}"])
    rng = random.Random(a.seed)
    pick = rng.sample([(q, m) for q in ok for m in models], min(a.review, len(ok) * len(models)))
    with (run / "judge_review.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["qid", "model", "query", "reference", "context", "answer", "judge_rationale",
                    *[f"judge_{s}" for s in SCORES], *[f"my_{s}" for s in SCORES], "note"])
        for q, m in sorted(pick, key=lambda x: (x[0].qid, x[1])):
            r = rows[m][q.qid]
            w.writerow([q.qid, m, q.text, q.answer, prompts[q.qid][0][:6000], r["answer"], r["rationale"],
                        *["" if r[s] is None else r[s] for s in SCORES], "", "", "", "", ""])
    try:
        plot_models(run, summ)
        plots = "models.png, refusals.png"
    except Exception as e:
        plots = f"failed ({e})"
    print("[5/5] Done")
    print("\n".join(L))
    print(f"\nPlots: {plots}\nSaved to {run}\nRate the answers in judge_review.csv (my_* columns, 0-2), then run:\n"
          f"   python evaluate_models.py --agreement {run / 'judge_review.csv'}")


if __name__ == "__main__":
    main()