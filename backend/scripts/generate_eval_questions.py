"""Generates the evaluation question set from the sample built by `build_eval_sample.py`.

How the questions are made:

* **Questions come from anywhere in the document.** Every target is split into passages (paragraph-aligned
  windows); for each question a passage is drawn from a different part of the document (start / middle / end).
* **One question per style per document**: `lexical` (exact terms, names, codes),
  `semantic` (paraphrase, avoids the passage's distinctive words), `colloquial` (student chat style).
* **The question is grounded in a quote.** The model must return `evidence` - a verbatim quote from the passage that
  contains the answer. The quote is verified against the passage; questions with an unverifiable or too-short quote
  are rejected.
* **Several correct documents.** `relevant_ids` lists EVERY document in the sample whose text contains the evidence
  quote (copies of the same syllabus, the same regulation on two sites...). `n_relevant` > `--generic-threshold`
  flags questions that are probably ambiguous (`"generic"` in `flags`).
* **Unusable passages are skipped.** The model can answer `usable=false` (no concrete fact). The script then tries
  other passages; a target that yields no question at all is replaced by a background document of the same
  (source, type) that passes the same quality filter as targets.
* **Reference answer included.** `answer` is written from the passage, so the set can also be used to evaluate
  answer generation (no separate golden-answer run needed).

Output (`--out`, default `eval_questions.json` next to this script):
    {"meta": {...parameters, counts...}, "questions": [ {qid, query, type, answer, evidence, expected_id,
     relevant_ids, n_relevant, flags, source, doc_type, title, passage, passage_start, passage_end,
     doc_chars, position, replacement_for, model}, ... ]}
Progress is appended to `<out>.progress.jsonl` - an interrupted run continues where it stopped.
`--review N` also writes `<out stem>_review.csv` with N random questions for the manual check (Excel-friendly).

Usage (from backend/scripts):
    python generate_eval_questions.py --dry-run          # passages + cost estimate, no API calls
    python generate_eval_questions.py --review 50        # asks for confirmation, then generates
    python generate_eval_questions.py --model gpt-4.1-mini --yes

Dependencies: openai, python-dotenv, numpy, PyYAML. Requires OPENAI_API_KEY (backend/.env).
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import build_eval_sample as bes  # same folder: read_body, target_quality, Doc

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from sites import load_or_exit  # noqa: E402  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
STYLES = ("lexical", "semantic", "colloquial")
PRICES = {  # USD per 1M tokens (input, output) - check https://openai.com/api/pricing before a big run
    "gpt-4.1": (2.00, 8.00), "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60),
}
CHARS_PER_TOKEN = 3.0        # rough for Polish text
OUTPUT_TOKENS_PER_Q = 180

_PARAGRAPH = re.compile(r"(?:[^\n]|\n(?![ \t]*\n))+")
_NORM_EMPH = re.compile(r"[*_`~]+")       # emphasis markers: removed
_NORM_SEP = re.compile(r"[#|>]+|-{3,}")    # headers, table pipes and rules, quotes: become spaces
_WS = re.compile(r"\s+")


# ==========================================
# 1. PASSAGES
# ==========================================
def norm(text: str) -> str:
    """Normalization used for evidence matching (quote vs passage, quote vs every sample document)."""
    t = bes._MD_LINK.sub(lambda m: m.group(1), text)
    t = _NORM_SEP.sub(" ", _NORM_EMPH.sub("", t.lower().replace("\u00a0", " ")))
    t = t.replace("„", '"').replace("”", '"').replace("’", "'").replace("–", "-").replace("—", "-")
    return _WS.sub(" ", t).strip()


def is_linky(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    bad = 0
    for ln in lines:
        if "](" in ln or "http" in ln:
            rest = bes._LIST_MARK.sub("", bes._URL.sub("", bes._MD_LINK.sub("", ln)))
            if len(re.sub(r"[\W_]+", "", rest)) < 20:
                bad += 1
    return bad / len(lines) > 0.5


def candidate_windows(body: str, min_chars: int, max_chars: int) -> List[Tuple[int, int]]:
    """Paragraph-aligned windows [start, end) of roughly min..max chars. Link-heavy windows are dropped."""
    blocks = []
    for m in _PARAGRAPH.finditer(body):
        if not m.group(0).strip():
            continue
        start, end = m.start(), m.end()
        while end - start > max_chars:      # long paragraph (typical for PDF conversions): split it
            cut = -1
            for sep in ("\n", ". ", "; ", " "):
                cut = body.rfind(sep, start + min_chars, start + max_chars)
                if cut != -1:
                    cut += len(sep)
                    break
            cut = cut if cut != -1 else start + max_chars
            blocks.append((start, cut))
            start = cut
        blocks.append((start, end))
    out = []
    for i, (start, end) in enumerate(blocks):
        j = i
        while end - start < min_chars and j + 1 < len(blocks):
            j += 1
            end = blocks[j][1]
        if end - start > max_chars:                       # cut long windows at a line/word boundary
            cut = body.rfind("\n", start + min_chars, start + max_chars)
            if cut == -1:
                cut = body.rfind(" ", start + min_chars, start + max_chars)
            end = cut if cut != -1 else start + max_chars
        text = body[start:end]
        if len(norm(text)) >= min(200, len(norm(body))) and not is_linky(text):
            out.append((start, end))
        if end >= len(body) and end - start >= min_chars:
            break
    return list(dict.fromkeys(out))


def choose_passages(windows: List[Tuple[int, int]], k: int, doc_len: int, rng: random.Random,
                    used: set) -> List[Tuple[int, int]]:
    """k non-overlapping windows, one from each k-th part of the document when possible."""
    free = [w for w in windows if w not in used]
    chosen: List[Tuple[int, int]] = []

    def overlaps(w):
        return any(w[0] < c[1] and c[0] < w[1] for c in chosen)

    for s in range(k):
        lo, hi = s * doc_len / k, (s + 1) * doc_len / k
        cands = [w for w in free if lo <= w[0] < hi and not overlaps(w)]
        if cands:
            chosen.append(rng.choice(cands))
    rest = [w for w in free if w not in chosen and not overlaps(w)]
    rng.shuffle(rest)
    for w in rest:
        if len(chosen) >= k:
            break
        if not overlaps(w):
            chosen.append(w)
    return sorted(chosen)


# ==========================================
# 2. PROMPT + LLM
# ==========================================
STYLE_RULES = {
    "lexical": "PRECYZYJNE - użyj dokładnych terminów, nazw własnych, kodów, kwot lub sygnatur z fragmentu.",
    "semantic": "OPISOWE - zapytaj o sens, warunek, procedurę lub cel własnymi słowami; unikaj charakterystycznych "
                "słów i nazw własnych z fragmentu (zostaw tylko to, co konieczne, żeby pytanie było jednoznaczne).",
    "colloquial": "POTOCZNE - tak, jak student pisze na czacie: luźno, skrótami, czasem bez polskich znaków. "
                  "Pisz naturalnie, nie upychaj na siłę slangu.",
}

SYSTEM_PROMPT = """Jesteś autorem zbioru testowego dla asystenta AI Akademii Górniczo-Hutniczej (AGH). Asystent odpowiada \
na pytania studentów, kandydatów i pracowników na podstawie ok. 100 tysięcy stron i dokumentów uczelni \
(strony jednostek, regulaminy, zarządzenia, sylabusy przedmiotów).

Dostajesz fragmenty JEDNEGO dokumentu. Dla każdego fragmentu napisz pytanie w wymaganym stylu.

Zasady:
1. Odpowiedź na pytanie musi wynikać WYŁĄCZNIE z danego fragmentu: konkretny fakt (liczba, termin, kwota, warunek, \
osoba, miejsce, kolejność kroków procedury).
2. Pytanie musi być zrozumiałe bez widzenia dokumentu. Osoba pyta asystenta CAŁEJ uczelni, więc podaj kontekst \
potrzebny do jednoznaczności (np. nazwę przedmiotu i kierunku, jednostkę, rodzaj świadczenia, rok akademicki), \
ale nie cytuj tytułu dokumentu, jeśli styl tego nie wymaga. NIE używaj odniesień typu „ten przedmiot”, \
„tego kursu”, „na tym kierunku”, „w tym projekcie”, „ta osoba” - zawsze nazwij przedmiot, kierunek, projekt \
lub jednostkę. Pytanie o osobę musi zawierać jej imię i nazwisko albo funkcję jednoznaczną w skali uczelni \
(np. „prodziekan ds. kształcenia Wydziału Fizyki i Informatyki Stosowanej”) - dotyczy to także stylu semantic.
3. Nie pytaj o nawigację, linki, strukturę strony, układ tabel, pola formularzy, metadane, cel samego pisma \
ani o to, „co zawiera dokument/fragment”.
4. Jeśli fragment nie zawiera konkretnego faktu, o który ktoś realnie by zapytał, ustaw usable=false \
(pozostałe pola mogą być wtedy puste).
5. answer: 1-2 zdania, wyłącznie na podstawie fragmentu.
6. evidence: DOSŁOWNY cytat z fragmentu (40-300 znaków), który zawiera odpowiedź. Kopiuj znak w znak, \
bez skracania i bez wielokropków.
7. Pisz po polsku.

Style:
- lexical: {lexical}
- semantic: {semantic}
- colloquial: {colloquial}""".format(**STYLE_RULES)

RESPONSE_SCHEMA = {
    "name": "eval_questions",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["items"],
        "properties": {"items": {"type": "array", "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["passage", "style", "usable", "question", "answer", "evidence"],
            "properties": {
                "passage": {"type": "integer"},
                "style": {"type": "string", "enum": list(STYLES)},
                "usable": {"type": "boolean"},
                "question": {"type": "string"},
                "answer": {"type": "string"},
                "evidence": {"type": "string"},
            }}}},
    },
}


def build_user_prompt(doc: dict, passages: List[str], styles: List[str]) -> str:
    parts = [f"Dokument: {doc['title']}", f"Źródło: {doc['source']} (typ: {doc['type']})", ""]
    for n, (p, s) in enumerate(zip(passages, styles), 1):
        parts += [f"=== FRAGMENT {n} | wymagany styl: {s} ===", p.strip(), ""]
    parts.append(f"Zwróć dokładnie {len(passages)} elementów w `items` - po jednym na fragment, "
                 f"z polem passage = numer fragmentu.")
    return "\n".join(parts)


class LLM:
    def __init__(self, model: str, temperature: float):
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(SCRIPT_DIR.parent / ".env")
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            sys.exit("Missing OPENAI_API_KEY (backend/.env).")
        self.client = OpenAI()
        self.model, self.temperature = model, temperature
        self.usage = Counter()
        self.lock = threading.Lock()

    def ask(self, user_prompt: str) -> List[dict]:
        last = None
        for attempt in range(4):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, temperature=self.temperature,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": user_prompt}],
                    response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA})
                with self.lock:
                    self.usage["input"] += r.usage.prompt_tokens
                    self.usage["output"] += r.usage.completion_tokens
                    self.usage["calls"] += 1
                return json.loads(r.choices[0].message.content)["items"]
            except Exception as e:  # network, rate limit, malformed JSON
                last = e
                time.sleep(2 ** attempt * 2)
        raise RuntimeError(f"LLM call failed 4x: {last}")


# ==========================================
# 3. PER-DOCUMENT GENERATION
# ==========================================
def locate_evidence(evidence: str, passage: str, min_len: int) -> Optional[str]:
    """Returns the normalized evidence if it is (almost) verbatim in the passage, else None."""
    ev, pa = norm(evidence), norm(passage)
    if len(ev) < min_len:
        return None
    if ev in pa:
        return ev
    m = difflib.SequenceMatcher(None, pa, ev, autojunk=False).find_longest_match(0, len(pa), 0, len(ev))
    if m.size >= max(min_len, 0.8 * len(ev)):
        return pa[m.a:m.a + m.size].strip()
    return None


# A question that ENDS with a demonstrative reference ("... z tego przedmiotu?", "... na tym kierunku?") names
# nothing and cannot be answered across the whole university. Mid-sentence uses ("w tej komisji odznaczeń na IET")
# are usually followed by the missing context, so only the trailing form is rejected.
_DEICTIC_NOUNS = (r"przedmio\w*|kurs\w*|kierunk\w*|kierunek|zaj[eę]\w*|komisj\w*|projek\w*|dokumen\w*|"
                  r"regulamin\w*|fragmen\w*|stron\w*|studi\w*|lab\w*|[cć]wicze\w*|wydzia\w*|katedr\w*|"
                  r"program\w*|osob\w*|egzamin\w*|stypendi\w*|szkoleni\w*|warsztat\w*|konferencj\w*|"
                  r"uchwa\w*|zarz[aą]dzeni\w*|specjalno\w*|jednost\w*|pi[sś]m\w*")
_DEICTIC_END = re.compile(r"\b(ten|ta|to|te|tego|tej|tym|tych|t[aą]|tymi|temu)\s+(" + _DEICTIC_NOUNS + r")\W*$",
                          re.I | re.U)
_HTML = re.compile(r"</?\s*(t[hdr]|table|tbody|thead|strong|span|div|p|br)\b", re.I)


def check_question(question: str, evidence: str) -> Optional[str]:
    """Rejection reason for a question that passed generation, or None. Also used to re-check saved progress."""
    if len(question.strip()) < 15:
        return "short_question"
    if _DEICTIC_END.search(question.strip()):
        return "deictic_reference"
    if _HTML.search(evidence):
        return "html_evidence"
    return None


def generate_for_doc(doc: dict, eval_dir: Path, llm: Optional[LLM], a, dry: bool = False,
                     keep: Optional[List[dict]] = None, prev: Optional[dict] = None) -> dict:
    """keep = already accepted questions of this doc (re-check after a filter change): only the missing styles
    are generated again; prev = the saved progress record (its counters are carried over)."""
    meta, body = bes.read_body(eval_dir / doc["sample_path"])
    windows = candidate_windows(body, a.passage_min, a.passage_max)
    rec = {"doc_id": doc["id"], "status": "failed", "questions": list(keep or []), "attempts": 0,
           "rejections": Counter((prev or {}).get("rejections", {})),
           "replacement_for": doc.get("_replacement_for", (prev or {}).get("replacement_for"))}
    if not windows:
        rec["rejections"]["no_passages"] += 1
        return rec
    rng = random.Random(f"{a.seed}|{doc['id']}")
    missing = list(STYLES[:a.per_doc]) if a.per_doc <= 3 else list(STYLES) * (a.per_doc // 3 + 1)
    missing = missing[:a.per_doc]
    for q in rec["questions"]:
        if q["type"] in missing:
            missing.remove(q["type"])
    rng.shuffle(missing)
    used: set = {(q["passage_start"], q["passage_end"]) for q in rec["questions"]}
    if rec["questions"] and not missing:
        rec["status"] = "ok"
        rec["rejections"] = dict(rec["rejections"])
        return rec

    while missing and rec["attempts"] < a.max_attempts:
        rec["attempts"] += 1
        chosen = choose_passages(windows, len(missing), len(body), rng, used)
        if not chosen:                              # short doc: re-use passages for the remaining styles
            used.clear()
            chosen = choose_passages(windows, len(missing), len(body), rng, used)
        while len(chosen) < len(missing):           # fewer passages than styles -> several styles per passage
            chosen.append(chosen[len(chosen) % max(1, len(chosen))])
        styles = missing[:len(chosen)]
        used.update(chosen)
        texts = [body[s:e] for s, e in chosen]
        if dry:
            rec["dry_passages"] = [{"style": st, "start": s, "end": e, "chars": e - s}
                                   for st, (s, e) in zip(styles, chosen)]
            rec["prompt_chars"] = len(SYSTEM_PROMPT) + len(build_user_prompt(doc, texts, styles))
            rec["status"] = "dry"
            return rec
        try:
            items = llm.ask(build_user_prompt(doc, texts, styles))
        except RuntimeError as e:
            rec["rejections"]["api_error"] += 1
            rec["error"] = str(e)[:300]
            break
        got = set()
        for it in items:
            n = it.get("passage", 0) - 1
            if not (0 <= n < len(chosen)) or n in got:
                rec["rejections"]["bad_passage_index"] += 1
                continue
            style = styles[n]
            if not it.get("usable"):
                rec["rejections"]["not_usable"] += 1
                continue
            q = it.get("question", "").strip()
            ev = locate_evidence(it.get("evidence", ""), texts[n], a.min_evidence)
            if ev is None:
                rec["rejections"]["evidence_not_found"] += 1
                continue
            reason = check_question(q, ev)
            if reason:
                rec["rejections"][reason] += 1
                continue
            got.add(n)
            s, e = chosen[n]
            rec["questions"].append({
                "query": q, "type": style, "answer": it.get("answer", "").strip(), "evidence": ev,
                "passage": texts[n], "passage_start": s, "passage_end": e,
                "position": round(s / max(1, len(body)), 3),
            })
        missing = [st for i, st in enumerate(styles) if i not in got] + missing[len(chosen):]
    if rec["questions"]:
        rec["status"] = "ok" if not missing else "partial"
    rec["rejections"] = dict(rec["rejections"])
    return rec


# ==========================================
# 4. REPLACEMENTS, RELEVANCE, OUTPUT
# ==========================================
def replacement_pool(files: List[dict], eval_dir: Path, a) -> Dict[Tuple[str, str], List[dict]]:
    rng = random.Random(f"{a.seed}|replacements")
    pool = defaultdict(list)
    for f in files:
        if f["role"] == "background" and f["chars"] >= a.min_target_chars:
            pool[(f["source"], f["type"])].append(f)
    for k in pool:
        pool[k].sort(key=lambda f: f["id"])
        rng.shuffle(pool[k])
    return pool


def passes_quality(f: dict, eval_dir: Path, a) -> bool:
    d = bes.Doc(id=f["id"], path=str(eval_dir / f["sample_path"]), rel_out=f["sample_path"],
                collection=f["collection"], source=f["source"], type=f["type"], size=0, mtime_ns=0,
                chars=f["chars"], title=f["title"])
    return bes.target_quality(d, SimpleNamespace(min_target_chars=a.min_target_chars, max_link_ratio=0.5)) is None


def relevance(questions: List[dict], files: List[dict], eval_dir: Path) -> None:
    print("   matching evidence against all sample documents...")
    texts = {}
    for f in files:
        try:
            texts[f["id"]] = norm(bes.read_body(eval_dir / f["sample_path"])[1])
        except OSError:
            pass
    for q in questions:
        rel = [i for i, t in texts.items() if q["evidence"] in t]
        if q["expected_id"] not in rel:          # should not happen; keep the target relevant anyway
            rel.append(q["expected_id"])
            q["flags"].append("target_not_matched")
        q["relevant_ids"] = sorted(rel)
        q["n_relevant"] = len(rel)


def load_progress(path: Path) -> Dict[str, dict]:
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["doc_id"]] = r
    return done


def write_review(questions: List[dict], path: Path, n: int, seed: int):
    rng = random.Random(f"{seed}|review")
    pick = rng.sample(questions, min(n, len(questions)))
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["qid", "type", "query", "answer", "evidence", "title", "source", "n_relevant",
                    "OK (1/0)", "problem (ambiguous / not_answerable / unnatural / wrong_answer / other)", "note"])
        for q in sorted(pick, key=lambda q: q["qid"]):
            w.writerow([q["qid"], q["type"], q["query"], q["answer"], q["evidence"], q["title"], q["source"],
                        q["n_relevant"], "", "", ""])
    print(f"   review sheet: {path} ({len(pick)} questions)")


def estimate(recs: List[dict], model: str, per_doc: int):
    chars = sum(r.get("prompt_chars", 0) for r in recs)
    tin = chars / CHARS_PER_TOKEN
    tout = len(recs) * per_doc * OUTPUT_TOKENS_PER_Q
    price = PRICES.get(model)
    cost = (tin * price[0] + tout * price[1]) / 1e6 if price else None
    return tin, tout, cost


def main():
    cfg = load_or_exit()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, default=cfg.eval_dir)
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR / "eval_questions.json")
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--per-doc", type=int, default=3, help="questions per target (styles rotate)")
    ap.add_argument("--passage-min", type=int, default=600, help="min passage length (chars)")
    ap.add_argument("--passage-max", type=int, default=1800, help="max passage length (chars)")
    ap.add_argument("--min-evidence", type=int, default=30, help="min normalized evidence length (chars)")
    ap.add_argument("--max-attempts", type=int, default=3, help="LLM calls per document")
    ap.add_argument("--max-replacements", type=int, default=3, help="replacement docs tried per failed target")
    ap.add_argument("--min-target-chars", type=int, default=800)
    ap.add_argument("--generic-threshold", type=int, default=5,
                    help="evidence found in more docs than this -> flag 'generic' (probably ambiguous)")
    ap.add_argument("--retry-partial", action="store_true",
                    help="also top up docs that got fewer than --per-doc questions in an earlier run")
    ap.add_argument("--limit", type=int, default=0, help="only the first N targets (for a test run)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--review", type=int, default=0, help="write a CSV with N random questions to check by hand")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    eval_dir = a.eval_dir.resolve()
    manifest_path = eval_dir / "sample_manifest.json"
    if not manifest_path.exists():
        sys.exit(f"No sample_manifest.json in {eval_dir} - run build_eval_sample.py first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest["files"]
    targets = sorted([f for f in files if f["role"] == "target"], key=lambda f: f["id"])
    if a.limit:
        targets = targets[:a.limit]
    print(f"Sample: {len(files)} docs, {len(targets)} targets, {a.per_doc} questions each, model {a.model}")

    # ---- plan + cost estimate (no API) ----
    plan = [generate_for_doc(t, eval_dir, None, a, dry=True) for t in targets]
    no_pass = [r["doc_id"] for r in plan if r["status"] != "dry"]
    tin, tout, cost = estimate([r for r in plan if r["status"] == "dry"], a.model, a.per_doc)
    pos = Counter(min(2, int(p["start"] / max(1, t["chars"]) * 3)) for r, t in zip(plan, targets)
                  for p in r.get("dry_passages", []))
    print(f"Passages: {sum(len(r.get('dry_passages', [])) for r in plan)} "
          f"(start/middle/end of doc: {pos[0]}/{pos[1]}/{pos[2]}); targets without usable passages: {len(no_pass)}")
    est = f"~${cost:.2f}" if cost is not None else "unknown price for this model"
    print(f"Estimated first pass: ~{tin / 1e3:.0f}k input + ~{tout / 1e3:.0f}k output tokens -> {est} "
          f"(retries and replacements add roughly 10-30%)")
    if a.dry_run:
        print("Dry run - no API calls.")
        return
    if not a.yes and input("Continue? [y/N] ").strip().lower() not in ("y", "yes", "t", "tak"):
        return

    llm = LLM(a.model, a.temperature)
    progress_path = a.out.with_name(a.out.name + ".progress.jsonl")
    done = load_progress(progress_path)
    lock = threading.Lock()

    def save(rec):
        with lock:
            with progress_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[rec["doc_id"]] = rec

    # ---- phase 1: targets (parallel) ----
    def api_failed(r):
        return r["status"] == "failed" and "api_error" in r.get("rejections", {})

    # Re-check saved questions against the current filters; docs that lose questions are redone
    # for the missing styles only (accepted questions are kept as they are).
    redo: Dict[str, List[dict]] = {}
    dropped = Counter()
    for doc_id, r in done.items():
        if r["status"] not in ("ok", "partial"):
            continue
        good = []
        for q in r["questions"]:
            reason = check_question(q["query"], q["evidence"])
            if reason:
                dropped[reason] += 1
            else:
                good.append(q)
        if len(good) < len(r["questions"]) or r["status"] == "partial" and a.retry_partial:
            redo[doc_id] = good
    if dropped or redo:
        print(f"Re-check of saved questions: dropped {dict(dropped)}; {len(redo)} docs will be topped up")
    todo = [t for t in targets if t["id"] not in done or api_failed(done[t["id"]]) or t["id"] in redo]
    print(f"Phase 1: {len(todo)} targets to process ({len(targets) - len(todo)} already done)")
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(generate_for_doc, t, eval_dir, llm, a, False, redo.get(t["id"]), done.get(t["id"])): t
                for t in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            save(fut.result())
            print(f"   {n}/{len(todo)}  calls={llm.usage['calls']}  "
                  f"tokens in/out={llm.usage['input']}/{llm.usage['output']}", end="\r")
    print()

    # replacement docs that lost questions in the re-check
    by_id_all = {f["id"]: f for f in files}
    for doc_id in [d for d in redo if d not in {t["id"] for t in targets} and d in by_id_all]:
        save(generate_for_doc({**by_id_all[doc_id]}, eval_dir, llm, a, False, redo[doc_id], done[doc_id]))

    # ---- phase 2: replacements for targets with no question (sequential = deterministic) ----
    pool = replacement_pool(files, eval_dir, a)
    taken = {r["doc_id"] for r in done.values()}
    failed = [t for t in targets if done.get(t["id"], {}).get("status") == "failed"
              and not api_failed(done[t["id"]])]
    replaced_for = {r.get("replacement_for") for r in done.values() if r.get("status") in ("ok", "partial")}
    for t in failed:
        if t["id"] in replaced_for:
            continue
        cands = pool.get((t["source"], t["type"]), [])
        tried = 0
        while cands and tried < a.max_replacements:
            c = cands.pop(0)
            if c["id"] in taken or not passes_quality(c, eval_dir, a):
                continue
            tried += 1
            taken.add(c["id"])
            rec = generate_for_doc({**c, "_replacement_for": t["id"]}, eval_dir, llm, a)
            save(rec)
            if rec["status"] in ("ok", "partial"):
                print(f"   replaced {t['id'].rsplit('/', 1)[-1]} -> {c['id'].rsplit('/', 1)[-1]}")
                break

    # ---- collect ----
    by_id = {f["id"]: f for f in files}
    questions = []
    for r in sorted(done.values(), key=lambda r: r["doc_id"]):
        if r["status"] not in ("ok", "partial"):
            continue
        f = by_id.get(r["doc_id"])
        if f is None:
            continue
        for q in sorted(r["questions"], key=lambda q: STYLES.index(q["type"]) if q["type"] in STYLES else 9):
            questions.append({**q, "expected_id": f["id"], "source": f["source"], "doc_type": f["type"],
                              "title": f["title"], "doc_chars": f["chars"], "dup_group": f["dup_group"],
                              "replacement_for": r.get("replacement_for"), "model": a.model, "flags": []})
    for i, q in enumerate(questions, 1):
        q["qid"] = f"q{i:04d}"
    relevance(questions, files, eval_dir)
    for q in questions:
        if q["n_relevant"] > a.generic_threshold:
            q["flags"].append("generic")

    rej = Counter()
    for r in done.values():
        rej.update(r.get("rejections", {}))
    status = Counter(r["status"] for r in done.values())
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "sample_manifest_created": manifest.get("created"),
        "params": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        "n_questions": len(questions),
        "by_type": dict(Counter(q["type"] for q in questions)),
        "by_doc_type": dict(Counter(q["doc_type"] for q in questions)),
        "doc_status": dict(status),
        "rejections": dict(rej),
        "n_relevant_hist": dict(sorted(Counter(min(q["n_relevant"], 10) for q in questions).items())),
        "flag_generic": sum("generic" in q["flags"] for q in questions),
        "position_thirds": dict(Counter(min(2, int(q["position"] * 3)) for q in questions)),
        "usage": dict(llm.usage),
    }
    a.out.write_text(json.dumps({"meta": meta, "questions": questions}, ensure_ascii=False, indent=1),
                     encoding="utf-8")
    print(f"\nSaved {len(questions)} questions to {a.out}")
    for k in ("by_type", "by_doc_type", "doc_status", "rejections", "n_relevant_hist", "flag_generic",
              "position_thirds", "usage"):
        print(f"   {k}: {meta[k]}")
    if a.review:
        write_review(questions, a.out.with_name(a.out.stem + "_review.csv"), a.review, a.seed)


if __name__ == "__main__":
    main()