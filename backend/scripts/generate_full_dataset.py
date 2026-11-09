"""
Builds an evaluation dataset that covers the WHOLE indexed database (companion of evaluate_production_full.py).

test_dataset_balanced.json is built from the ~200-file sample, so it only probes ~30 files. This script instead
samples files from what is really in the Elasticsearch index - the same `plik` values retrieval returns - balanced
across sources (`zrodlo`: one stratum per site / syllabus set / ...), and asks an LLM for three questions per file
(lexical / semantic / colloquial - same prompt strategy as generate_qa_dataset.py).

Each row carries the exact indexed path (`expected_plik`), so ground truth is unambiguous even when the same file
name exists in several folders (e.g. syllabi of different academic years).

    python scripts/generate_full_dataset.py --per-source 3                 # ~3 files x 3 questions per source
    python scripts/generate_full_dataset.py --per-source 10 --yes --seed 7
    python scripts/evaluate_production_full.py --dataset scripts/full_db_dataset.json

Costs one chat-completion call per sampled file (default model gpt-4o, ~2.5k input tokens each); the number of
calls is printed first and must be confirmed (or pass --yes). Interrupted runs resume from <out>.progress.jsonl.

Caveats worth keeping in mind when reading the results:
    * Questions are written by an LLM that has seen the document, so some leak document wording; the
      `semantic` type is instructed to avoid it, `lexical` is meant to use it.
    * Sampling is balanced per source, NOT proportional to corpus size: huge sources (syllabi) are down-weighted
      relative to their share of the index. Use per-source breakdowns in the report, not only the global mean.
    * Only the first 5000 characters of a file are shown to the LLM, so questions come from the document's start.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR.parents[1] / "config"))

import evaluate_production_full as ev
from app.core.config import settings
from app.services import rag_service

DEFAULT_OUT = SCRIPT_DIR / "full_db_dataset.json"
QUESTION_TYPES = ("lexical", "semantic", "colloquial")
MAX_CHARS_FOR_LLM = 5000


def build_prompt(file_name: str, text: str) -> str:
    return f"""
Jesteś inżynierem AI budującym system ewaluacyjny RAG dla uczelni AGH. Twój cel to wygenerowanie 3 pytań testowych na podstawie poniższego fragmentu dokumentu.
Pytania muszą celowo faworyzować różne algorytmy wyszukiwania, aby zbiór był zbalansowany.

Wygeneruj dokładnie 3 pytania, po jednym dla każdej strategii:
1. "lexical" (Precyzyjne): Użyj dokładnych, unikalnych terminów, sygnatur, kodów, kwot lub nazw własnych z tekstu. (Ma na celu faworyzowanie Elasticsearch).
2. "semantic" (Opisowe): Zapytaj o dany proces, warunek lub cel, ale CELOWO NIE UŻYWAJ głównych słów kluczowych ani nazw własnych z tekstu. Użyj silnych parafraz i omówień. (Ma na celu faworyzowanie ChromaDB).
3. "colloquial" (Potoczne): Zapytaj o konkretne informacje, używając slangu studenckiego, potocznych słów, skrótów i naturalnego, niedbałego języka (np. stypa, wuef, warunek, dziekanat, hajs, akademik, oblać). (Ma na celu faworyzowanie RRF/Hybrydy).

Każde pytanie musi dać się zrozumieć BEZ znajomości dokumentu (nie pisz "w tym dokumencie", "według tekstu").

Zwróć wynik WYŁĄCZNIE jako poprawny obiekt JSON zawierający tablicę "pytania" w następującym formacie:
{{
  "pytania": [
    {{"query": "pytanie leksykalne...", "type": "lexical"}},
    {{"query": "pytanie semantyczne...", "type": "semantic"}},
    {{"query": "pytanie potoczne...", "type": "colloquial"}}
  ]
}}

Nazwa pliku (tylko kontekst): {file_name}
Tekst źródłowy:
{text}
"""


def plik_to_disk(plik: str, cfg) -> Optional[Path]:
    """`plik` = <id_prefix>/<collection folder>/<relative path> (see build_database.collect_files) -> real file."""
    prefix = cfg.id_prefix + "/"
    if not plik.startswith(prefix):
        return None
    path = cfg.data_root / plik[len(prefix):]
    return path if path.is_file() else None

_FRONTMATTER = re.compile(r"\A\ufeff?---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.S)


def strip_frontmatter(content: str) -> str:
    m = _FRONTMATTER.match(content)
    return (content[m.end():] if m else content).strip()


def make_text_loader(cfg) -> Callable[[str], Optional[str]]:
    def load(plik: str) -> Optional[str]:
        path = plik_to_disk(plik, cfg)
        if path is None:
            return None
        return strip_frontmatter(path.read_text(encoding="utf-8-sig").replace("\r\n", "\n"))
    return load


def sample_files(inventory: Dict[str, dict], per_source: int, rng: random.Random, min_chars: int,
                 load_text: Callable[[str], Optional[str]]) -> List[Tuple[str, str, str]]:
    """[(source, plik, text)] - up to `per_source` readable, long-enough, content-distinct files per source."""
    by_source: Dict[str, List[str]] = defaultdict(list)
    for plik, entry in inventory.items():
        by_source[entry.get("zrodlo") or "(brak)"].append(plik)

    chosen, seen_content = [], set()
    for source in sorted(by_source):
        files = sorted(by_source[source])
        rng.shuffle(files)
        got = 0
        for plik in files:
            if got >= per_source:
                break
            text = load_text(plik)
            if text is None or len(text) < min_chars:
                continue
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if digest in seen_content:
                continue
            seen_content.add(digest)
            chosen.append((source, plik, text))
            got += 1
    return chosen


def parse_questions(raw: str) -> List[Dict[str, str]]:
    """Valid, distinct questions with a known type; anything else the LLM produced is dropped."""
    data = json.loads(raw)
    items = data.get("pytania") if isinstance(data, dict) else data
    out, seen_types = [], set()
    for it in items or []:
        if not isinstance(it, dict):
            continue
        q, t = str(it.get("query") or "").strip(), str(it.get("type") or "").strip().lower()
        if t in QUESTION_TYPES and t not in seen_types and 10 <= len(q) <= 400:
            seen_types.add(t)
            out.append({"query": q, "type": t})
    return out


def make_llm_ask(model: str) -> Callable[[str], str]:
    from openai import OpenAI
    client = OpenAI(api_key=settings.OPENAI_API_KEY, max_retries=4)

    def ask(prompt: str) -> str:
        resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                              temperature=0.8, response_format={"type": "json_object"})
        return resp.choices[0].message.content.strip()
    return ask


def load_progress(path: Path) -> List[dict]:
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def generate(sample: Sequence[Tuple[str, str, str]], ask: Callable[[str], str], progress_path: Path) -> List[dict]:
    rows = load_progress(progress_path)
    done = {r["expected_plik"] for r in rows}
    todo = [s for s in sample if s[1] not in done]
    print(f"   {len(done)} files already done, {len(todo)} to go.")
    for i, (source, plik, text) in enumerate(todo, 1):
        name = ev.basename(plik)
        try:
            questions = parse_questions(ask(build_prompt(name, text[:MAX_CHARS_FOR_LLM])))
        except Exception as e:
            print(f"   [!] {name}: {type(e).__name__}: {e} - skipped")
            continue
        new = [{"query": q["query"], "type": q["type"], "expected_file": name, "expected_plik": plik, "source": source}
               for q in questions]
        with open(progress_path, "a", encoding="utf-8") as f:
            for r in new:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        rows += new
        print(f"      [{i}/{len(todo)}] {source[:28]:<28} {name[:50]}", end="\r")
    print()
    return rows


def main(argv: Optional[Sequence[str]] = None, ask: Optional[Callable[[str], str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Sample files from the live index and generate evaluation questions.")
    ap.add_argument("--per-source", type=int, default=3, help="files sampled per source (zrodlo)")
    ap.add_argument("--min-chars", type=int, default=800, help="skip files with a shorter body (nothing to ask about)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation before calling the LLM")
    ap.add_argument("--refresh-inventory", action="store_true")
    for flag in ("--chroma-host", "--chroma-port", "--es-url", "--es-index", "--chroma-collection"):
        ap.add_argument(flag, type=int if flag == "--chroma-port" else str)
    a = ap.parse_args(argv)
    ev.apply_overrides(argparse.Namespace(**{k: v for k, v in vars(a).items() if k in
                                             ("chroma_host", "chroma_port", "es_url", "es_index", "chroma_collection")}))

    from sites import load_or_exit # type: ignore
    cfg = load_or_exit()

    print(" [1/3] Reading the file list from the live index...")
    es = rag_service._es_client()
    n_docs = int(rag_service._get_body(es.count(index=settings.ES_INDEX))["count"])
    inventory, _ = ev.load_inventory(es, settings.ES_INDEX, n_docs, a.refresh_inventory)
    print(f"       {len(inventory):,} files in '{settings.ES_INDEX}'")

    print(" [2/3] Sampling (balanced per source)...")
    sample = sample_files(inventory, a.per_source, random.Random(a.seed), a.min_chars, make_text_loader(cfg))
    if not sample:
        sys.exit(f"No readable files found under {cfg.data_root}. Is the knowledge base on this machine (RAG_DATA_ROOT)?")
    sources = len({s for s, _, _ in sample})
    print(f"       {len(sample)} files from {sources} sources -> up to {len(sample) * len(QUESTION_TYPES)} questions")

    if not a.yes:
        if not sys.stdin.isatty():
            sys.exit("Refusing to call the LLM non-interactively without --yes.")
        if input(f"       This makes {len(sample)} calls to {a.model}. Continue? [y/N] ").strip().lower() != "y":
            sys.exit("Cancelled.")

    print(" [3/3] Generating questions...")
    rows = generate(sample, ask or make_llm_ask(a.model), Path(str(a.out) + ".progress.jsonl"))
    a.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n Saved {len(rows)} questions for {len({r['expected_plik'] for r in rows})} files to {a.out}")


if __name__ == "__main__":
    main()
