"""Final validation: hand-written questions against the running production system (full database).

The questions go through the real HTTP API (POST /api/chat, the same endpoint the frontend uses), so everything is
tested end to end: Docker setup, full Elasticsearch + Chroma indexes, query handling, prompt, model. Nothing is scored
automatically - YOU rate the answers; the script collects them, makes them easy to read and summarises your ratings.

Steps (backend running, from backend/scripts):

    1. python validate_final.py pytania_walidacja.csv
         asks every question, saves answers + the 5 sources, writes to validation_stats/run_<timestamp>/:
           przeglad.html   question, answer, the 5 source chunks (with links) - read this while rating
           ocena.csv       one row per question with empty rating columns - fill these in (Excel, ';')
       interrupted? rerun with --resume validation_stats/run_<timestamp> (answered questions are skipped)

    2. fill in ocena.csv:
           ocena_odpowiedzi  0 = wrong / made up / pointless refusal, 1 = partly right or incomplete, 2 = correct
                             (category "brak": 2 = refused or said the information is not available, 0 = made it up)
           ocena_zrodel      0 = no source has the answer, 1 = part of the answer is in the sources, 2 = all of it
                             (leave empty for "brak")
           halucynacja       1 = the answer states something that is not in the sources (otherwise 0 / empty)
           uwagi             free text

    3. python validate_final.py --summarize validation_stats/run_<timestamp>
         report.txt, results.json, wyniki.png and pomylki.html (every miss with its diagnosis and sources)

Questions file (CSV, ';', UTF-8; Excel "CSV UTF-8" works): columns id; kategoria; pytanie; po_pytaniu; uwagi.
`po_pytaniu` = id of an earlier question this one follows in the same conversation (e.g. "A do kiedy?"): the earlier
question and the system's answer to it are sent as history, like the frontend does.
"""
from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_ROOT = SCRIPT_DIR / "validation_stats"
CATEGORIES = {   # order of the report; descriptions for the report header
    "fakt": "pojedynczy fakt ze strony",
    "regulamin": "zasady z regulaminów i zarządzeń",
    "tabela": "wartość z tabeli (terminy, opłaty, liczby)",
    "sylabus": "treść sylabusa",
    "jednostka": "informacje o wydziale lub jednostce",
    "osoba": "osoby i kontakty",
    "potoczne": "język potoczny, skróty",
    "wiele": "odpowiedź z kilku źródeł",
    "rozmowa": "pytanie kontynuujące rozmowę",
    "angielski": "pytanie po angielsku",
    "brak": "odpowiedzi nie ma w bazie (oczekiwana odmowa)",
}
RATING_COLS = ["ocena_odpowiedzi", "ocena_zrodel", "halucynacja", "uwagi"]
RRF_K = 60
# The frontend (frontend/src/main.ts) starts every chat with this assistant message and sends it as history, so in
# the UI even the first question has history and goes through the backend's history reformulation (LLM rewrite +
# keywords). The script sends it too, unless --no-greeting (= the bare API path: no history, no rewrite).
GREETING = "Cześć! Jestem asystentem Akademii Górniczo-Hutniczej. W czym mogę Ci pomóc?"


def single_leg(sources: List[dict]) -> bool:
    """True if the context came from ONE search leg only (dense or BM25 failed, e.g. a Chroma timeout):
    rag_service then continues with the other leg, and the RRF scores of the top chunks are exactly
    1/(k+1), 1/(k+2), ... With both legs working, chunks found by both score higher than 1/(k+1)."""
    if not sources:
        return False
    return all(abs((s.get("score") or 0) - 1 / (RRF_K + i)) < 1e-9 for i, s in enumerate(sources, 1))


def no_answer_text() -> str:
    """NO_ANSWER_FALLBACK read from llm_service.py without importing the app."""
    src = (SCRIPT_DIR.parent / "app" / "services" / "llm_service.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "NO_ANSWER_FALLBACK" for t in node.targets):
            return node.value.value
    return "nie potrafię odpowiedzieć na to pytanie"


# ==========================================
# 1. QUESTIONS AND THE API
# ==========================================
def load_questions(path: Path) -> List[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        sample = fh.read(2048)
        fh.seek(0)
        rows = list(csv.DictReader(fh, delimiter=";" if sample.count(";") >= sample.count(",") else ","))
    qs, seen = [], set()
    for n, r in enumerate(rows, 1):
        r = {k.strip().lower(): (v or "").strip() for k, v in r.items() if k}
        if not r.get("pytanie"):
            continue
        qid = r.get("id") or str(n)
        if qid in seen:
            sys.exit(f"Duplicate id {qid} in {path}")
        seen.add(qid)
        cat = r.get("kategoria", "").lower() or "fakt"
        if cat not in CATEGORIES:
            print(f"   [!] question {qid}: unknown category {cat!r} (known: {', '.join(CATEGORIES)})")
        qs.append({"id": qid, "kategoria": cat, "pytanie": r["pytanie"], "po_pytaniu": r.get("po_pytaniu", ""),
                   "uwagi": r.get("uwagi", "")})
    ids = {q["id"] for q in qs}
    for q in qs:
        if q["po_pytaniu"] and q["po_pytaniu"] not in ids:
            sys.exit(f"Question {q['id']}: po_pytaniu={q['po_pytaniu']} does not exist")
    return qs


def ask(url: str, model: str, query: str, history: List[dict], timeout: float) -> dict:
    """One POST /api/chat, reading the SSE stream like the frontend. Returns answer, sources, errors, timings."""
    body = json.dumps({"query": query, "model": model, "history": history}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    out = {"answer": "", "sources": [], "errors": [], "ttft_s": None}
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev["type"] == "token":
                if out["ttft_s"] is None:
                    out["ttft_s"] = time.perf_counter() - t0
                out["answer"] += ev["content"]
            elif ev["type"] == "sources":
                out["sources"] = ev["content"]
            elif ev["type"] == "error":
                out["errors"].append(ev["content"])
    out["total_s"] = time.perf_counter() - t0
    return out


def run(a):
    qs = load_questions(a.questions)
    run_dir = a.resume or OUT_ROOT / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    store = run_dir / "answers.jsonl"
    done: Dict[str, dict] = {}
    if store.exists():
        for line in store.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if not r.get("errors"):
                done[r["id"]] = r
    if a.redo:
        redo = set(done) if a.redo.strip().lower() == "all" else {x.strip() for x in a.redo.split(",") if x.strip()}
        unknown = redo - {q["id"] for q in qs}
        if unknown:
            sys.exit(f"--redo: unknown ids {', '.join(sorted(unknown))}")
        print(f"   --redo: asking {len(redo & set(done))} answered questions again (their ratings are cleared)")
        for i in redo:
            done.pop(i, None)
    degraded = {i for i, r in done.items() if single_leg(r["sources"])}
    if degraded:
        print(f"   {len(degraded)} earlier answers came from one search leg only - asking them again: "
              f"{', '.join(sorted(degraded, key=lambda x: (len(x), x)))}")
        for i in degraded:
            del done[i]
    pfile = run_dir / "params.json"
    if a.resume and pfile.exists():
        prev_greeting = json.loads(pfile.read_text(encoding="utf-8")).get("greeting", False)
        if prev_greeting != (not a.no_greeting):
            sys.exit(f"This run was made {'with' if prev_greeting else 'without'} the frontend greeting in the history; "
                     f"resume it the same way ({'drop' if prev_greeting else 'add'} --no-greeting) or start a new run.")
    (run_dir / "params.json").write_text(json.dumps({"questions": str(a.questions), "model": a.model, "url": a.url,
                                                     "greeting": not a.no_greeting,
                                                     "started": datetime.now().isoformat(timespec="seconds")},
                                                    ensure_ascii=False, indent=2), encoding="utf-8")
    todo = [q for q in qs if q["id"] not in done]
    print(f"{len(qs)} questions ({dict(Counter(q['kategoria'] for q in qs))}) | {len(done)} already answered | "
          f"model {a.model} | {a.url} | history: " + ("none (bare API)" if a.no_greeting else "frontend greeting (as in the UI)"))
    print(f"   rate limit {a.per_minute}/min -> about {len(todo) * 60 / a.per_minute / 60:.0f} min")
    no_answer = no_answer_text()
    by_id = {q["id"]: q for q in qs}
    gap = 60.0 / a.per_minute
    last = 0.0
    with store.open("a", encoding="utf-8") as fh:
        for n, q in enumerate(todo, 1):
            history = [] if a.no_greeting else [{"role": "assistant", "content": GREETING}]
            if q["po_pytaniu"]:                       # earlier turn of the same conversation (frontend format)
                prev = done.get(q["po_pytaniu"])
                if not prev:
                    print(f"   [!] {q['id']}: question {q['po_pytaniu']} it follows has no answer yet - skipped")
                    continue
                history += [{"role": "user", "content": by_id[q["po_pytaniu"]]["pytanie"]},
                            {"role": "assistant", "content": prev["answer"][:3000] or no_answer}]
            for attempt in range(4):
                time.sleep(max(0.0, last + gap - time.time()))
                last = time.time()
                try:
                    res = ask(a.url, a.model, q["pytanie"], history, a.timeout)
                    if single_leg(res["sources"]) and attempt < 3:
                        print(f"   [{q['id']}] context from one search leg only (the other failed or found nothing) - retry in {a.retry_wait:.0f} s")
                        time.sleep(a.retry_wait)
                        continue
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 429 and attempt < 3:
                        print(f"   429 rate limit - waiting {gap * 2:.0f}s")
                        time.sleep(gap * 2)
                        continue
                    res = {"answer": "", "sources": [], "errors": [f"HTTP {e.code}"], "ttft_s": None, "total_s": 0}
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                    if attempt < 3:
                        time.sleep(5)
                        continue
                    res = {"answer": "", "sources": [], "errors": [str(e)], "ttft_s": None, "total_s": 0}
            row = {**q, **res, "refused": no_answer.rstrip(".") in res["answer"], "model": a.model,
                   "single_leg": single_leg(res["sources"]), "asked": datetime.now().isoformat(timespec="seconds")}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if not res["errors"]:
                done[q["id"]] = row
            flag = ("ERROR " + "; ".join(res["errors"]) if res["errors"] else
                    "ONE SEARCH LEG ONLY" if row["single_leg"] else "refused" if row["refused"] else "")
            print(f"   {n}/{len(todo)}  [{q['id']}] {q['pytanie'][:60]:<60} {res['total_s']:5.1f}s  {flag}")
    rows = [done[q["id"]] for q in qs if q["id"] in done]
    if todo or not (run_dir / "ocena.csv").exists():
        write_review(run_dir, rows, reasked={q["id"] for q in todo})
    else:
        print("   nothing new - ocena.csv left as it is")
    bad = [r["id"] for r in rows if single_leg(r["sources"])]
    if bad:
        print(f"[!] still from one search leg only: {', '.join(bad)} - check the backend log, then --resume")
    missing = len(qs) - len(rows)
    print(f"\nSaved to {run_dir}" + (f"\n[!] {missing} questions without an answer - rerun with --resume {run_dir}"
                                       if missing else ""))
    print("Read przeglad.html, rate the answers in ocena.csv, then run:\n"
          f"   python validate_final.py --summarize {run_dir}")


# ==========================================
# 2. FILES FOR RATING
# ==========================================
def src_label(s: dict) -> str:
    m = s.get("meta", {})
    return f"{m.get('source', '')} | {m.get('title', '')}" + (f" › {m['section']}" if m.get("section") else "")


def write_review(run_dir: Path, rows: List[dict], reasked=frozenset()):
    path = run_dir / "ocena.csv"
    old = {}
    if path.exists():                                 # keep ratings already typed in (rerun / resume)
        with path.open(encoding="utf-8-sig", newline="") as fh:
            old = {r["id"]: r for r in csv.DictReader(fh, delimiter=";")}
    try:
        fh = path.open("w", encoding="utf-8-sig", newline="")
    except PermissionError:
        sys.exit(f"[!] Cannot write {path} - it is open in Excel. Close it and run the same command again "
                 f"(answers are saved; nothing will be asked twice).")
    with fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["id", "kategoria", "pytanie", "odpowiedz", "odmowa", "zrodla", *RATING_COLS])
        for r in rows:
            prev = {} if r["id"] in reasked else old.get(r["id"], {})   # new answer -> old rating no longer applies
            w.writerow([r["id"], r["kategoria"], r["pytanie"], r["answer"], int(r["refused"]),
                        " || ".join(f"[{i}] {src_label(s)}" for i, s in enumerate(r["sources"], 1)),
                        *[prev.get(c, "") for c in RATING_COLS]])
    write_html(run_dir / "przeglad.html", "Walidacja końcowa — odpowiedzi do oceny", rows)


def write_html(path: Path, title: str, rows: List[dict], extra: Dict[str, str] = None):
    e = html.escape
    parts = [f"""<!DOCTYPE html><html lang="pl"><head><meta charset="utf-8"><title>{e(title)}</title><style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:1000px;margin:2em auto;padding:0 1em;color:#1f2328}}
.q{{border:1px solid #d0d7de;border-radius:8px;padding:1em 1.2em;margin:1.2em 0}}
.h{{font-weight:600;font-size:16px}} .cat{{background:#eef2f6;border-radius:4px;padding:1px 6px;font-size:12px;
margin-right:6px}} .ans{{white-space:pre-wrap;background:#f6f8fa;padding:.7em;border-radius:6px;margin:.6em 0}}
.ref{{color:#9a6700}} .diag{{color:#cf222e;font-weight:600}} details{{margin:.3em 0}} summary{{cursor:pointer;
color:#0969da}} .src{{white-space:pre-wrap;font-size:13px;background:#fbfbfb;border-left:3px solid #d0d7de;
padding:.4em .7em}} .meta{{color:#57606a;font-size:13px}}</style></head><body><h1>{e(title)}</h1>
<p class="meta">{len(rows)} pytań. Kliknij źródło, żeby zobaczyć chunk przekazany do modelu.</p>"""]
    for r in rows:
        follow = f" <span class='meta'>(kontynuacja pytania {e(r['po_pytaniu'])})</span>" if r.get("po_pytaniu") else ""
        parts.append(f"<div class='q' id='q{e(r['id'])}'><div class='h'><span class='cat'>{e(r['kategoria'])}</span>"
                     f"{e(r['id'])}. {e(r['pytanie'])}{follow}</div>")
        if extra and r["id"] in extra:
            parts.append(extra[r["id"]])
        parts.append(f"<div class='ans'>{e(r['answer'])}</div>")
        if single_leg(r["sources"]):
            parts.append("<div class='diag'>Uwaga: kontekst tylko z jednej ścieżki wyszukiwania "
                         "(druga zawiodła) — zadaj pytanie ponownie (--resume)</div>")
        parts.append(f"<div class='meta'>{'<span class=ref>odmowa</span> · ' if r['refused'] else ''}"
                     f"pierwszy token {r['ttft_s'] or 0:.1f} s · całość {r['total_s'] or 0:.1f} s</div>")
        for i, s in enumerate(r["sources"], 1):
            url = s.get("meta", {}).get("url", "")
            link = f" — <a href='{e(url)}' target='_blank'>otwórz</a>" if url.startswith("http") else ""
            parts.append(f"<details><summary>[{i}] {e(src_label(s))}</summary>{link}"
                         f"<div class='src'>{e(s.get('text', ''))}</div></details>")
        parts.append("</div>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


# ==========================================
# 3. SUMMARY OF YOUR RATINGS
# ==========================================
def num(x):
    x = (x or "").strip().replace(",", ".")
    try:
        return float(x)
    except ValueError:
        return None


def ci(values: List[float], n_boot: int = 2000, seed: int = 0):
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n_boot))
    return sum(values) / len(values), means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def diagnose(r: dict) -> str:
    """Where a miss comes from, by your two ratings."""
    a, s = r["ocena_odpowiedzi"], r["ocena_zrodel"]
    if r["kategoria"] == "brak":
        return "zmyślona odpowiedź na pytanie spoza bazy" if a == 0 else "odpowiedź częściowo zmyślona"
    if r["refused"] and s is not None and s >= 1:
        return "fałszywa odmowa (odpowiedź była w źródłach)"
    if s == 0:
        return "wyszukiwanie (żadne źródło nie zawiera odpowiedzi)"
    if s == 1:
        return "wyszukiwanie częściowe (źródła zawierają tylko część odpowiedzi)"
    if s == 2:
        return "generowanie (odpowiedź była w źródłach, model ją pominął lub przekręcił)"
    return "brak oceny źródeł"


def summarize(run_dir: Path):
    answers = {}
    for line in (run_dir / "answers.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if not r.get("errors"):
            answers[r["id"]] = r
    with (run_dir / "ocena.csv").open(encoding="utf-8-sig", newline="") as fh:
        rated = list(csv.DictReader(fh, delimiter=";"))
    rows, unrated = [], []
    for g in rated:
        r = answers.get(g["id"])
        if not r:
            continue
        r = {**r, "ocena_odpowiedzi": num(g.get("ocena_odpowiedzi")), "ocena_zrodel": num(g.get("ocena_zrodel")),
             "halucynacja": num(g.get("halucynacja")) or 0.0, "uwagi_oceny": g.get("uwagi", "")}
        (rows if r["ocena_odpowiedzi"] is not None else unrated).append(r)
    if not rows:
        sys.exit("No ratings in ocena.csv yet (column ocena_odpowiedzi is empty).")
    degraded = [r["id"] for r in rows if single_leg(r["sources"])]
    params = json.loads((run_dir / "params.json").read_text(encoding="utf-8"))
    answerable = [r for r in rows if r["kategoria"] != "brak"]
    brak = [r for r in rows if r["kategoria"] == "brak"]

    def stats(rs):
        a = [r["ocena_odpowiedzi"] / 2 for r in rs]
        s = [r["ocena_zrodel"] / 2 for r in rs if r["ocena_zrodel"] is not None]
        return {"n": len(rs), "answer": ci(a), "correct": ci([float(r["ocena_odpowiedzi"] == 2) for r in rs]),
                "sources": ci(s), "sources_full": ci([float(r["ocena_zrodel"] == 2) for r in rs
                                                     if r["ocena_zrodel"] is not None]),
                "hallucination": ci([float(r["halucynacja"] >= 1) for r in rs]),
                "refused": sum(r["refused"] for r in rs) / len(rs)}

    overall = stats(answerable) if answerable else None
    per_cat = {c: stats([r for r in rows if r["kategoria"] == c]) for c in CATEGORIES
               if any(r["kategoria"] == c for r in rows)}
    for c in sorted({r["kategoria"] for r in rows} - set(per_cat)):
        per_cat[c] = stats([r for r in rows if r["kategoria"] == c])
    misses = [r for r in rows if r["ocena_odpowiedzi"] < 2]
    diag = Counter(diagnose(r) for r in misses)
    # answer quality given the sources: separates retrieval from generation
    by_src = defaultdict(list)
    for r in answerable:
        if r["ocena_zrodel"] is not None:
            by_src[int(r["ocena_zrodel"])].append(r["ocena_odpowiedzi"] / 2)

    W = 104
    f = lambda t: f"{t[0]:.2f} [{t[1]:.2f}-{t[2]:.2f}]" if t[0] == t[0] else "-"
    L = ["=" * W, "FINAL VALIDATION (validate_final.py) - hand-written questions, full database, production API",
         "=" * W,
         f"Run: {run_dir.name} | model {params['model']} | {len(rows)} rated questions"
         + (f" ({len(unrated)} not rated yet)" if unrated else ""),
         "Ratings by the author: answer 0-2, sources 0-2 (shown as 0-1); 95% CI: bootstrap over questions."]
    if degraded:
        L.append(f"[!] {len(degraded)} answers used one search leg only (the other failed): {', '.join(degraded)} "
                 f"- rerun with --resume to ask them again")
    L.append("")
    if overall:
        L += [f"Answerable questions ({overall['n']}, all categories except 'brak'):",
              f"   answer score          {f(overall['answer'])}",
              f"   fully correct         {f(overall['correct'])}",
              f"   sources score         {f(overall['sources'])}",
              f"   answer fully in sources {f(overall['sources_full'])}",
              f"   hallucination         {f(overall['hallucination'])}",
              f"   refusals              {overall['refused']:.2f}", ""]
    if brak:
        ok = [float(r["ocena_odpowiedzi"] == 2) for r in brak]
        L += [f"Questions without an answer in the database ('brak', {len(brak)}): handled correctly "
              f"{f(ci(ok))}, refused {sum(r['refused'] for r in brak)}/{len(brak)}", ""]
    L += [f"{'Category':<12}{'n':>4}{'answer':>20}{'fully correct':>16}{'sources':>20}{'halluc.':>9}{'refused':>9}",
          "-" * W]
    for c, st in per_cat.items():
        L.append(f"{c:<12}{st['n']:>4}{f(st['answer']):>20}{st['correct'][0]:>16.2f}{f(st['sources']):>20}"
                 f"{st['hallucination'][0]:>9.2f}{st['refused']:>9.2f}")
    L += ["", "Answer score by sources score (answerable questions) - where the misses come from:"]
    for k in sorted(by_src):
        v = by_src[k]
        L.append(f"   sources {k} ({['none', 'part', 'all'][k]} of the answer)  n={len(v):>3}  answer {sum(v) / len(v):.2f}")
    L += ["", f"Misses (answer < 2): {len(misses)}"]
    for d, n in diag.most_common():
        L.append(f"   {n:>3}  {d}")
    tt = [r["ttft_s"] for r in rows if r.get("ttft_s")]
    tot = [r["total_s"] for r in rows if r.get("total_s")]
    if tt:
        L += ["", f"Latency (end to end, through the API): first token median {median(tt):.1f} s, "
                  f"whole answer median {median(tot):.1f} s, max {max(tot):.1f} s"]
    kinds = Counter(s.get("meta", {}).get("source", "?") for r in rows for s in r["sources"])
    L += ["", "Most frequent sources in the contexts: "
              + ", ".join(f"{k} {v}" for k, v in kinds.most_common(8))]
    report = "\n".join(L)
    print(report)
    (run_dir / "report.txt").write_text(report, encoding="utf-8")
    (run_dir / "results.json").write_text(json.dumps(
        {"run": run_dir.name, "params": params, "n_rated": len(rows), "n_unrated": len(unrated),
         "overall": overall, "per_category": per_cat, "misses": dict(diag),
         "answer_by_sources": {k: sum(v) / len(v) for k, v in by_src.items()}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    extra = {}
    for r in misses:
        src = "-" if r["ocena_zrodel"] is None else f"{r['ocena_zrodel']:.0f}"
        extra[r["id"]] = (f"<div class='diag'>{html.escape(diagnose(r))} — odpowiedź {r['ocena_odpowiedzi']:.0f}/2, "
                          f"źródła {src}/2</div>"
                          + (f"<div class='meta'>uwagi: {html.escape(r['uwagi_oceny'])}</div>"
                             if r["uwagi_oceny"] else ""))
    write_html(run_dir / "pomylki.html", "Walidacja końcowa — pomyłki", misses, extra)
    try:
        plot(run_dir, per_cat)
        print(f"\nSaved: report.txt, results.json, pomylki.html, wyniki.png -> {run_dir}")
    except Exception as e:
        print(f"[!] plot failed ({e}); report saved to {run_dir}")


def plot(run_dir: Path, per_cat: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import eval_plots as ep
    cats = list(per_cat)[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(cats) + 1.4))
    for si, (key, label) in enumerate((("answer", "answer score"), ("sources", "sources score"))):
        off = (si - 0.5) * 0.3
        for yi, c in enumerate(cats):
            m, lo, hi = per_cat[c][key]
            if m != m:
                continue
            ax.plot([lo, hi], [yi + off] * 2, color=ep.SERIES[si], lw=2, alpha=0.6)
            ax.plot(m, yi + off, ep.MARKERS[si], ms=7, color=ep.SERIES[si], mec=ep.SURFACE, mew=1.5,
                    label=label if yi == 0 else None)
    ax.set_yticks(range(len(cats)), [f"{c} (n={per_cat[c]['n']})" for c in cats])
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("score 0-1 (your ratings), mean with 95% CI")
    ax.grid(axis="y", visible=False)
    ax.set_title("Final validation by question category")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5)
    fig.savefig(run_dir / "wyniki.png", bbox_inches="tight")
    plt.close(fig)


def load_answers(run_dir: Path) -> Dict[str, dict]:
    out = {}
    for line in (run_dir / "answers.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if not r.get("errors"):
            out[r["id"]] = r
    return out


def compare(run_a: Path, run_b: Path):
    """How much the retrieved sources differ between two runs of the same questions (e.g. before and after the
    vector index finished building). Sources are compared as documents (url, otherwise source + title)."""
    A, B = load_answers(run_a), load_answers(run_b)
    key = lambda s: s.get("meta", {}).get("url") or f"{s.get('meta', {}).get('source')}|{s.get('meta', {}).get('title')}"
    common = [i for i in A if i in B]
    rows = []
    for i in common:
        da, db = {key(s) for s in A[i]["sources"]}, {key(s) for s in B[i]["sources"]}
        j = len(da & db) / len(da | db) if da | db else 1.0
        rows.append((j, i))
    rows.sort()
    same = sum(j == 1.0 for j, _ in rows)
    print(f"{len(common)} questions in both runs | identical source documents: {same} | "
          f"mean overlap (Jaccard) {sum(j for j, _ in rows) / max(1, len(rows)):.2f}")
    print(f"{'id':>5}  {'overlap':>7}  question")
    for j, i in rows:
        if j < 1.0:
            print(f"{i:>5}  {j:>7.2f}  {A[i]['pytanie'][:80]}")


def copy_ratings(src: Path, dst: Path):
    A, B = load_answers(src), load_answers(dst)
    def read(p: Path) -> Dict[str, dict]:
        with p.open(encoding="utf-8-sig", newline="") as fh:
            return {r["id"]: r for r in csv.DictReader(fh, delimiter=";")}
    ra, rb = read(src / "ocena.csv"), read(dst / "ocena.csv")
    key = lambda r: (r["answer"].strip(), [(s.get("meta", {}).get("url"), s.get("text")) for s in r["sources"]])
    n = 0
    for i, row in rb.items():
        if i in A and i in B and i in ra and ra[i].get("ocena_odpowiedzi") and not row.get("ocena_odpowiedzi") \
                and key(A[i]) == key(B[i]):
            for c in RATING_COLS:
                row[c] = ra[i].get(c, "")
            n += 1
    cols = ["id", "kategoria", "pytanie", "odpowiedz", "odmowa", "zrodla", *RATING_COLS]
    try:
        fh = (dst / "ocena.csv").open("w", encoding="utf-8-sig", newline="")
    except PermissionError:
        sys.exit(f"[!] {dst / 'ocena.csv'} is open in Excel - close it and run again.")
    with fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter=";", extrasaction="ignore")
        w.writeheader()
        w.writerows(rb.values())
    print(f"Copied ratings for {n} questions (identical answer and sources); the rest of {dst.name} is yours to rate.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("questions", nargs="?", type=Path, help="questions CSV (id;kategoria;pytanie;po_pytaniu;uwagi)")
    ap.add_argument("--url", default="http://localhost:8000/api/chat")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--per-minute", type=float, default=4.5, help="requests per minute (backend limit: 5/minute)")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--retry-wait", type=float, default=30, help="seconds before re-asking a one-leg answer")
    ap.add_argument("--resume", type=Path, help="continue an interrupted run (its folder)")
    ap.add_argument("--summarize", type=Path, help="run folder with a rated ocena.csv -> report")
    ap.add_argument("--no-greeting", action="store_true",
                    help="send questions without the frontend greeting (bare API path: no history -> no rewrite)")
    ap.add_argument("--copy-ratings", type=Path, nargs=2, metavar=("FROM_RUN", "TO_RUN"),
                    help="copy ratings to TO_RUN for questions whose answer AND sources are identical in both runs")
    ap.add_argument("--redo", help="with --resume: ask these question ids again (comma list, or 'all')")
    ap.add_argument("--compare", type=Path, nargs=2, metavar=("RUN_A", "RUN_B"),
                    help="compare the retrieved sources of two runs, question by question")
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
    elif a.copy_ratings:
        copy_ratings(*a.copy_ratings)
    elif a.summarize:
        summarize(a.summarize)
    elif a.questions:
        run(a)
    else:
        ap.error("give the questions CSV, or --summarize <run folder>")


if __name__ == "__main__":
    main()