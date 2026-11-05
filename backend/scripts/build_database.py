#!/usr/bin/env python3
"""
Build the RAG vector database from Markdown files (AGH) - incremental version.

    python build_database.py                    # index (only new/modified files)
    python build_database.py --dry-run          # stats + cost estimate, NO API usage
    python build_database.py --preview file.md  # show how a file will be chunked (NO API)
    python build_database.py --search "query" -k 5 [--source "Biblioteka Główna AGH"]
    python build_database.py --rebuild          # clear Chroma and Elasticsearch and build from scratch
    python build_database.py --rebuild-es       # restore ES index from files (no OpenAI API, free)

Safeguards: missing collection directories abort the process (--allow-missing-dirs), and massive deletion
of chunks (> MAX_DELETED_PERCENT % of files) requires --confirm-deletion.

What's different from a simple RecursiveCharacterTextSplitter:
  * We split by Markdown STRUCTURE (headers -> blocks: paragraph / list / table / code),
    not by characters; sentences are split considering Polish abbreviations (ul., godz., r., ust.).
  * TABLES are atomic; oversized ones are split by rows, repeating the header row in each part.
  * Each chunk gets a contextual header: [Source] Title › Section › Subsection
    (without this, "Deadline: October 5" is meaningless).
  * Text sent for embedding has URLs removed, but the DB keeps the full text (with links for citations).
  * Indexing is incremental and idempotent: changing one file = re-embedding only
    modified chunks; running twice doesn't duplicate data.
  * API key from env (OPENAI_API_KEY), retry, token batching, model check in the collection.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.core.config import settings

# ==========================================
# SETTINGS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
DB_DIR = str((BASE_DIR / "../../data/Baza_Wektorowa_Chroma_ASYM").resolve())
MANIFEST_PATH = os.path.join(DB_DIR, "_manifest_indeksu.json")

COLLECTION_NAME = settings.CHROMA_COLLECTION
ES_INDEX = settings.ES_INDEX
ES_URL = settings.ES_URL
MODEL = settings.EMBED_MODEL
DIMENSIONS = settings.EMBED_DIMENSIONS

FORMAT_VERSION = "asym-v2"
MAX_DELETED_PERCENT = 20

PRICE_PER_1M_TOKENS_USD = 0.02

MAX_TOKENS = 450
MIN_TOKENS = 120
TOKEN_MARGIN = 25
SENTENCE_OVERLAP = 1
MAX_INTRO_TOKENS = 80
MIN_DOC_CHARS = 80

MAX_BATCH_ELEMENTS = 256
MAX_BATCH_TOKENS = 200_000

COLLECTIONS = [
    {"path": "../../data/Baza_Wiedzy_AGH/Rekrutacja_Strony", "source": "Rekrutacja AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/Rekrutacja_Dokumenty_Przetworzone", "source": "Rekrutacja AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/Biblioteka_Strony", "source": "Biblioteka Główna AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/Biblioteka_Dokumenty_Przetworzone", "source": "Biblioteka Główna AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/CSS_Strony", "source": "Centrum Spraw Studenckich AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/CSS_Dokumenty_Przetworzone", "source": "Centrum Spraw Studenckich AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/Miasteczko_Strony", "source": "Miasteczko Studenckie AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/Miasteczko_Dokumenty_Przetworzone", "source": "Miasteczko Studenckie AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/Centrum_Karier_Strony", "source": "Centrum Karier AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/Centrum_Karier_Dokumenty_Przetworzone", "source": "Centrum Karier AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/Centrum_Spraw_Miedzynarodowych_Strony", "source": "Centrum Spraw Międzynarodowych AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/Centrum_Spraw_Miedzynarodowych_Dokumenty_Przetworzone", "source": "Centrum Spraw Międzynarodowych AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/SWFiS_Strony", "source": "SWFiS AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/SWFiS_Dokumenty_Przetworzone", "source": "SWFiS AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/Cyfronet_Strony", "source": "Cyfronet AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/Cyfronet_Dokumenty_Przetworzone", "source": "Cyfronet AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/SJO_Strony", "source": "SJO AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/SJO_Dokumenty_Przetworzone", "source": "SJO AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/KA_Dokumenty_Przetworzone", "source": "KA AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/KA_Strony", "source": "KA AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/CKiM_Strony", "source": "CKiM AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/CKiM_Dokumenty_Przetworzone", "source": "CKiM AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/CRI_Strony", "source": "CRI AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/CRI_Dokumenty_Przetworzone", "source": "CRI AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/SD_Strony", "source": "SD AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/SD_Dokumenty_Przetworzone", "source": "SD AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/Podyplomowe_Strony", "source": "Podyplomowe AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/Podyplomowe_Dokumenty_Przetworzone", "source": "Podyplomowe AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/Fundacja_Strony", "source": "Fundacja AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/Fundacja_Dokumenty_Przetworzone", "source": "Fundacja AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/Samorzad_Strony", "source": "Samorzad AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/Samorzad_Dokumenty_Przetworzone", "source": "Samorzad AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/EAIIB_Strony", "source": "EAIIB AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/RPS_Strony", "source": "RPS AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/RPS_Dokumenty_Przetworzone", "source": "RPS AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/WH_Strony", "source": "WH AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/WH_Dokumenty_Przetworzone", "source": "WH AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/Odlewnictwo_Strony", "source": "Odlewnictwo AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/WNIG_Strony", "source": "WNIG AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/WNIG_Dokumenty_Przetworzone", "source": "WNIG AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/Zarz_Strony", "source": "Zarz AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/Zarz_Dokumenty_Przetworzone", "source": "Zarz AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/WIMIC_Strony", "source": "WIMIC AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/WIMIC_Dokumenty_Przetworzone", "source": "WIMIC AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/WILGZ_Strony", "source": "WILGZ AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/WILGZ_Dokumenty_Przetworzone", "source": "WILGZ AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/Metal_Strony", "source": "Metal AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/Metal_Dokumenty_Przetworzone", "source": "Metal AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/WGGIOS_Strony", "source": "WGGIOS AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/WGGIOS_Dokumenty_Przetworzone", "source": "WGGIOS AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/IET_Strony", "source": "IET AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/IET_Dokumenty_Przetworzone", "source": "IET AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/WGGIIS_Strony", "source": "WGGIIS AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/WGGIIS_Dokumenty_Przetworzone", "source": "WGGIIS AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/WGGIIS_Studiuj_Strony", "source": "WGGIIS_Studiuj AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/WGGIIS_Studiuj_Dokumenty_Przetworzone", "source": "WGGIIS_Studiuj AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/Informatyka_Strony", "source": "Informatyka AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/Informatyka_Dokumenty_Przetworzone", "source": "Informatyka AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/Spacetech_Strony", "source": "Spacetech AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/Spacetech_Dokumenty_Przetworzone", "source": "Spacetech AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/WMS_Strony", "source": "WMS AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/WMS_Dokumenty_Przetworzone", "source": "WMS AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/FIS_Strony", "source": "FIS AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/FIS_Dokumenty_Przetworzone", "source": "FIS AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/WEIP_Strony", "source": "WEIP AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/WEIP_Dokumenty_Przetworzone", "source": "WEIP AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/WMN_Strony", "source": "WMN AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/WMN_Dokumenty_Przetworzone", "source": "WMN AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/Historia_Strony", "source": "Historia AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/IMIR_Strony", "source": "IMIR AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/IMIR_Dokumenty_Przetworzone", "source": "IMIR AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/AZS_Strony", "source": "AZS AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/AZS_Dokumenty_Przetworzone", "source": "AZS AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/COK_Strony", "source": "COK AGH", "type": "strona"},
    {"path": "../../data/Baza_Wiedzy_AGH/COK_Dokumenty_Przetworzone", "source": "COK AGH", "type": "dokument", "legal_headers": True},
    {"path": "../../data/Baza_Wiedzy_AGH/Sylabusy_Markdown/2021_2022", "source": "Sylabusy AGH", "type": "sylabus"},
    {"path": "../../data/Baza_Wiedzy_AGH/Sylabusy_Markdown/2022_2023", "source": "Sylabusy AGH", "type": "sylabus"},
    {"path": "../../data/Baza_Wiedzy_AGH/Sylabusy_Markdown/2023_2024", "source": "Sylabusy AGH", "type": "sylabus"},
    {"path": "../../data/Baza_Wiedzy_AGH/Sylabusy_Markdown/2024_2025", "source": "Sylabusy AGH", "type": "sylabus"},
    {"path": "../../data/Baza_Wiedzy_AGH/Sylabusy_Markdown/2025_2026", "source": "Sylabusy AGH", "type": "sylabus"},
    {"path": "../../data/Baza_Wiedzy_AGH/Sylabusy_Markdown/2026_2027", "source": "Sylabusy AGH", "type": "sylabus"},
    {"path": "../../data/Baza_Wiedzy_AGH/Skos_Strony", "source": "Skos AGH", "type": "strona"},
]

# ==========================================
# TOKEN COUNTING
# ==========================================
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    TOKENIZER = "tiktoken"

    def get_tokens(text):
        return len(_ENC.encode(text, disallowed_special=()))
except ImportError:
    TOKENIZER = "approximation"

    def get_tokens(text):
        return -(-len(text) * 10 // 28)


# ==========================================
# LOADING FILE + FRONTMATTER
# ==========================================
_FRONTMATTER = re.compile(r"\A\ufeff?---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.S)


def parse_frontmatter(content):
    """Frontmatter only at the START of the file with '---' on a separate line."""
    m = _FRONTMATTER.match(content)
    if not m:
        return {}, content.strip()
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as e:
        print(f"   ⚠️ Invalid YAML in frontmatter ({str(e).splitlines()[0]}) - ignoring metadata")
        meta = {}
    return (meta if isinstance(meta, dict) else {}), content[m.end():].strip()


def clean_metadata(meta):
    """Chroma accepts only str/int/float/bool: lists -> text, dates -> ISO, None/empty -> ignored."""
    result = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (datetime, date)):
            v = v.isoformat()
        elif isinstance(v, (list, tuple, set)):
            v = ", ".join(str(x) for x in v)
        elif isinstance(v, dict):
            v = json.dumps(v, ensure_ascii=False)
        elif not isinstance(v, (bool, int, float, str)):
            v = str(v)
        if isinstance(v, str):
            v = v.strip()[:500]
            if not v:
                continue
        result[str(k)] = v
    return result


def normalize(t):
    t = re.sub(r"[*_`#]|\[([^\]]*)\]\([^)]*\)", lambda m: m.group(1) or "", t)
    return re.sub(r"\W+", " ", t.lower()).strip()


def remove_urls(text):
    """Text for EMBEDDING: without URLs and bold markers (noise for the vector)."""
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<?https?://\S+>?", "", text)
    return re.sub(r"[ \t]{2,}", " ", text.replace("**", " ").replace("__", " "))


# ==========================================
# MARKDOWN PARSER -> BLOCKS
# ==========================================
@dataclass
class Block:
    type: str
    text: str
    path: tuple
    intro: str = ""


RE_HEADER = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
RE_LEGAL = re.compile(r"^(Rozdział\s+[IVXLCDM\d]+[a-z]?|§\s*\d+[a-z]?|Art\.\s*\d+[a-z]?)(?=[\s.]|$)", re.I)
RE_LIST = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+")
RE_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
RE_FENCE = re.compile(r"^\s*(```|~~~)")
RE_HRULE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
RE_BOLD_LINE = re.compile(r"^(?:\*\*|__)(?P<t>[^*_].*?)(?:\*\*|__)$")


def bold_as_header(line):
    """CMSs often make 'headers' by bolding: **1. Who can apply for a scholarship?**"""
    m = RE_BOLD_LINE.match(line.strip())
    if not m or "**" in m.group("t") or "__" in m.group("t"):
        return None
    t = m.group("t").strip()
    if t.endswith("!") or len(t) > 200 or len(t) < 4:
        return None
    if len(t.split()) < 3 and not re.match(r"\d+[.)]\s", t):
        return None
    return t.rstrip(":").strip()


def is_structure_start(line, legal):
    return bool(RE_HEADER.match(line) or RE_FENCE.match(line) or line.lstrip().startswith("|")
                or RE_LIST.match(line) or (legal and RE_LEGAL.match(line.strip())))


def parse_blocks(body, title, legal=False, bold_headers=True):
    lines = body.split("\n")
    n = len(lines)
    stack = []
    blocks = []
    title_norm = normalize(title)

    def get_path():
        return tuple(t for _, t, _ in stack if normalize(t) != title_norm)

    def push(level, text, block_type=""):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, text, block_type))

    def push_bold(text):
        """Bold line = subsection of current TRUE header; only replaces previous bold."""
        while stack and stack[-1][2] == "bold":
            stack.pop()
        push(stack[-1][0] + 1 if stack else 2, text, "bold")

    i = 0
    while i < n:
        line = lines[i]
        if not line.strip() or RE_HRULE.match(line):
            i += 1
            continue

        m = RE_FENCE.match(line)
        if m:
            marker, j = m.group(1), i + 1
            while j < n and not lines[j].lstrip().startswith(marker):
                j += 1
            blocks.append(Block("code", "\n".join(lines[i:j + 1]), get_path()))
            i = j + 1
            continue

        m = RE_HEADER.match(line)
        if m:
            text = re.sub(r"\*\*|__|`", "", re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", m.group(2))).strip()
            push(len(m.group(1)), text)
            i += 1
            continue

        if line.lstrip().startswith("|"):
            j = i
            while j < n and lines[j].lstrip().startswith("|"):
                j += 1
            blocks.append(Block("table", "\n".join(l.rstrip() for l in lines[i:j]), get_path()))
            i = j
            continue

        if RE_LIST.match(line):
            j = i + 1
            while j < n:
                if lines[j].strip():
                    if RE_LIST.match(lines[j]) or lines[j].startswith((" ", "\t")):
                        j += 1
                        continue
                    break
                k = j + 1
                while k < n and not lines[k].strip():
                    k += 1
                if k < n and (RE_LIST.match(lines[k]) or lines[k].startswith((" ", "\t"))):
                    j = k
                    continue
                break
            blocks.append(Block("list", "\n".join(l.rstrip() for l in lines[i:j] if l.strip()), get_path()))
            i = j
            continue

        if legal:
            mp = RE_LEGAL.match(line.strip())
            if mp:
                label = re.sub(r"\s+", " ", mp.group(1)).strip()
                if label.lower().startswith("rozdz"):
                    full = re.sub(r"\s+", " ", line).strip()
                    push(2, full if len(full) <= 100 else label)
                else:
                    push(3, label)
        j = i + 1
        while j < n and lines[j].strip() and not is_structure_start(lines[j], legal):
            j += 1
        if bold_headers and j == i + 1:
            header = bold_as_header(line)
            if header:
                push_bold(header)
                i = j
                continue
        blocks.append(Block("paragraph", "\n".join(l.strip() for l in lines[i:j]), get_path()))
        i = j

    result, idx = [], 0
    while idx < len(blocks):
        b = blocks[idx]
        if b.type == "paragraph" and idx + 1 < len(blocks) and blocks[idx + 1].path == b.path:
            bare = re.sub(r"[*_\s]+$", "", re.sub(r"^[*_\s]+", "", b.text))
            nxt = blocks[idx + 1]
            a = bare.endswith(":") and nxt.type in ("table", "list", "code")
            bb = bare.endswith(("!", ":")) and len(bare.split()) <= 6 and "\n" not in bare
            if (a or bb) and get_tokens(b.text) <= MAX_INTRO_TOKENS and not nxt.intro:
                nxt.intro = b.text
                idx += 1
                continue
        result.append(b)
        idx += 1
    return result


# ==========================================
# SPLITTING OVERSIZED BLOCKS
# ==========================================
ABBREVIATIONS = {
    "np", "tj", "tzn", "tzw", "itd", "itp", "ul", "al", "pl", "os", "godz", "ok", "nr", "poz", "ust", "art",
    "pkt", "lit", "zał", "dz", "prof", "dr", "mgr", "inż", "hab", "ds", "wg", "zob", "por", "im", "bud", "pok",
    "tel", "str", "rys", "tab", "ww", "r", "w", "wyd", "red", "tłum", "ewent", "min", "maks", "max", "śr",
    "ur", "zm", "ks", "płk", "gen", "kpt", "mjr", "sp", "zs", "ogł", "woj", "pow", "gm", "m.in", "j.w",
}
RE_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[\"„(\[]?[A-ZĄĆĘŁŃÓŚŹŻ0-9§])")


def split_into_sentences(text):
    result, start = [], 0
    for m in RE_SENTENCE_END.finditer(text):
        before = text[start:m.start()]
        words = before.split()
        if words and words[-1].endswith("."):
            w = words[-1].rstrip(".").lower().strip("„\"([")
            is_year_r = w == "r" and len(words) >= 2 and words[-2].isdigit() and len(words[-2]) == 4
            if not is_year_r and (w in ABBREVIATIONS or len(w) == 1 or w.isdigit()):
                continue
        result.append(before.strip())
        start = m.end()
    rest = text[start:].strip()
    if rest:
        result.append(rest)
    return result


def hard_split(text, budget):
    parts, current, tok = [], [], 0
    for word in text.split():
        t = get_tokens(word) + 1
        if current and tok + t > budget:
            parts.append(" ".join(current))
            current, tok = [], 0
        current.append(word)
        tok += t
    if current:
        parts.append(" ".join(current))
    return parts


def split_paragraph(text, budget):
    sentences = []
    for line in text.split("\n"):
        sentences += split_into_sentences(line)
    sentences = [z for s in sentences for z in ([s] if get_tokens(s) <= budget else hard_split(s, budget))]
    parts, current, tok = [], [], 0
    for z in sentences:
        t = get_tokens(z) + 1
        if current and tok + t > budget:
            parts.append(" ".join(current))
            tail = current[-SENTENCE_OVERLAP:] if SENTENCE_OVERLAP else []
            tail_tok = sum(get_tokens(x) + 1 for x in tail)
            current, tok = (tail, tail_tok) if tail_tok <= budget * 0.25 else ([], 0)
        current.append(z)
        tok += t
    if current:
        parts.append(" ".join(current))
    return parts


def split_list(text, budget):
    lines = text.split("\n")
    indents = [len(RE_LIST.match(l).group(1)) for l in lines if RE_LIST.match(l)]
    minimum = min(indents) if indents else 0
    elements = []
    for l in lines:
        m = RE_LIST.match(l)
        if m and len(m.group(1)) == minimum or not elements:
            elements.append([l])
        else:
            elements[-1].append(l)
    elements = ["\n".join(e) for e in elements]
    parts, current, tok = [], [], 0
    for e in elements:
        t = get_tokens(e) + 1
        if t > budget:
            if current:
                parts.append("\n".join(current))
                current, tok = [], 0
            parts += split_paragraph(e, budget)
            continue
        if current and tok + t > budget:
            parts.append("\n".join(current))
            current, tok = [], 0
        current.append(e)
        tok += t
    if current:
        parts.append("\n".join(current))
    return parts


def split_table(text, budget):
    """By rows, with REPEATED header in each part."""
    lines = text.split("\n")
    if len(lines) >= 2 and RE_TABLE_SEP.match(lines[1]):
        header, rows = lines[:2], lines[2:]
    else:
        header, rows = [], lines
    header_tok = get_tokens("\n".join(header)) + 1
    parts, current, tok = [], [], header_tok

    def close_part():
        nonlocal current, tok
        if current:
            parts.append("\n".join(header + current))
        current, tok = [], header_tok

    for r in rows:
        t = get_tokens(r) + 1
        if header_tok + t > budget:
            close_part()
            parts += ["\n".join(header + [k]) for k in hard_split(r, max(budget - header_tok, 20))]
            continue
        if current and tok + t > budget:
            close_part()
        current.append(r)
        tok += t
    close_part()
    return parts


def split_code(text, budget):
    parts, current, tok = [], [], 0
    for l in text.split("\n"):
        t = get_tokens(l) + 1
        if current and tok + t > budget:
            parts.append("\n".join(current))
            current, tok = [], 0
        current.append(l)
        tok += t
    if current:
        parts.append("\n".join(current))
    return parts


def break_down_block(b, budget):
    bud = max(budget - (get_tokens(b.intro) + 1 if b.intro else 0), 40)
    if get_tokens(b.text) <= bud:
        parts = [b.text]
    else:
        parts = {"table": split_table, "list": split_list, "code": split_code}.get(b.type, split_paragraph)(b.text, bud)
    return [f"{b.intro}\n\n{c}" if b.intro else c for c in parts]


# ==========================================
# ASSEMBLING CHUNKS
# ==========================================
@dataclass
class TextChunk:
    text: str
    text_emb: str
    section: str
    tokens: int
    has_table: bool
    tokens_emb: int = 0


def build_context(source, title, path):
    return f"[{source}] {title}" + (" › " + " › ".join(path) if path else "")


def get_common_prefix(paths):
    pref = list(paths[0])
    for p in paths[1:]:
        k = 0
        while k < min(len(pref), len(p)) and pref[k] == p[k]:
            k += 1
        pref = pref[:k]
    return tuple(pref)


def render_parts(parts):
    """Returns (path_prefix, content). When chunk covers multiple sections, their headers are inline."""
    prefix = get_common_prefix([p[0] for p in parts])
    lines, prev = [], prefix
    for path, text in parts:
        changed = False
        for d in range(len(prefix), len(path)):
            if changed or d >= len(prev) or prev[d] != path[d]:
                lines.append("#" * min(6, d + 2) + " " + path[d])
                changed = True
        prev = path
        lines.append(text)
    return prefix, "\n\n".join(lines)


def assemble_chunks(blocks, title, source):
    groups, current = [], None

    for b in blocks:
        overhead_b = get_tokens(build_context(source, title, b.path)) + 2
        for part in break_down_block(b, MAX_TOKENS - overhead_b - TOKEN_MARGIN):
            t = get_tokens(part) + 1
            if current:
                new_sec = current["parts"][-1][0] != b.path
                overhead = max(current["overhead"], overhead_b)
                exceeds = current["tok"] + t + overhead + TOKEN_MARGIN > MAX_TOKENS
                if exceeds or (new_sec and current["tok"] >= MIN_TOKENS):
                    groups.append(current)
                    current = None
            if current is None:
                current = {"parts": [], "tok": 0, "overhead": 0, "table": False}
            current["parts"].append((b.path, part))
            current["tok"] += t
            current["overhead"] = max(current["overhead"], overhead_b)
            current["table"] = current["table"] or b.type == "table"
    if current:
        groups.append(current)

    if len(groups) >= 2 and groups[-1]["tok"] < MIN_TOKENS // 2:
        a, z = groups[-2], groups[-1]
        if a["tok"] + z["tok"] + max(a["overhead"], z["overhead"]) + TOKEN_MARGIN <= MAX_TOKENS:
            a["parts"] += z["parts"]
            a["tok"] += z["tok"]
            a["overhead"] = max(a["overhead"], z["overhead"])
            a["table"] = a["table"] or z["table"]
            groups.pop()

    chunks = []
    for g in groups:
        prefix, content = render_parts(g["parts"])

        context_txt = build_context(source, title, prefix)
        text = f"{context_txt}\n\n{content}"
        
        bc_parts = [x for x in (title, " › ".join(prefix)) if x]
        breadcrumb = " › ".join(bc_parts)
        text_vector = f"{breadcrumb}\n\n{remove_urls(content)}" if breadcrumb else remove_urls(content)
        
        chunks.append(TextChunk(text=text, text_emb=text_vector,
                                section=" › ".join(prefix), tokens=get_tokens(text), has_table=g["table"], tokens_emb=get_tokens(text_vector)))
    return chunks


# ==========================================
# DOCUMENT -> CHUNKS WITH METADATA
# ==========================================
def prepare_document(file_path, cfg):
    """Returns (meta, body, title) or None if file is empty/too short."""
    content = Path(file_path).read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    meta, body = parse_frontmatter(content)
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    body = re.sub(r"<br\s*/?>", " ", body, flags=re.I)
    if len(body) < MIN_DOC_CHARS:
        return None
    title = str(meta.get("tytul") or "").strip()
    if not title:
        m = re.search(r"^#\s+(.+)$", body, re.M)
        title = m.group(1).strip() if m else Path(file_path).stem
    return meta, body, title


def get_file_signature(meta, body, title, cfg):
    """Changes when content OR chunking parameters OR url change."""
    params = [MAX_TOKENS, MIN_TOKENS, TOKEN_MARGIN, SENTENCE_OVERLAP, MAX_INTRO_TOKENS, TOKENIZER, MODEL, DIMENSIONS,
              cfg["source"], cfg["type"], bool(cfg.get("legal_headers")),
              cfg.get("bold_as_headers", True), 4, FORMAT_VERSION]
    url = meta.get("url") or meta.get("zrodlo") or ""
    return hashlib.sha1(f"{url}\x00{body}\x00{title}\x00{params}".encode("utf-8")).hexdigest()


def chunk_file(file_path, cfg, doc=None, rel=None):
    doc = doc or prepare_document(file_path, cfg)
    if doc is None:
        return []
    meta, body, title = doc
    extra_meta = clean_metadata(meta)
    if "zrodlo" in extra_meta and str(extra_meta["zrodlo"]).startswith("http"):
        extra_meta["url"] = extra_meta.pop("zrodlo")
    blocks = parse_blocks(body, title, legal=bool(cfg.get("legal_headers")),
                          bold_headers=cfg.get("bold_as_headers", True))
    chunks = assemble_chunks(blocks, title, cfg["source"])

    rel = rel or Path(file_path).as_posix()
    extra_meta = clean_metadata(meta)
    result, seen = [], {}
    for nr, c in enumerate(chunks):
        k = c.text_emb + "\x00" + c.text
        seen[k] = seen.get(k, 0) + 1
        cid = hashlib.sha1(f"{rel}|{seen[k]}|{k}|{MODEL}|{DIMENSIONS}".encode("utf-8")).hexdigest()[:32]
        md = {**extra_meta,
              "zrodlo": cfg["source"], "kolekcja": Path(cfg["path"]).name, "typ": cfg["type"],
              "tytul": title, "sekcja": c.section, "plik": rel,
              "chunk_nr": nr, "chunk_z": len(chunks), "tokeny": c.tokens, "tabela": c.has_table}
        result.append((cid, c, md))
    return result


# ==========================================
# FETCHING FILE LIST, MANIFEST
# ==========================================
def resolve_path(path):
    p = Path(path)
    return p if p.is_absolute() else (BASE_DIR / p).resolve()


def validate_config():
    seen = []
    for i, c in enumerate(COLLECTIONS):
        if not isinstance(c, dict) or not {"path", "source", "type"} <= set(c):
            sys.exit(f"COLLECTIONS[{i}] is invalid (required keys: path, source, type): {c!r}")
        cat = resolve_path(c["path"])
        for other in seen:
            if cat == other or cat in other.parents or other in cat.parents:
                sys.exit(f"COLLECTIONS: overlapping paths ({cat} and {other}) - same files would be indexed twice.")
        seen.append(cat)


def collect_files(allow_missing=False):
    """Returns [(cfg, read_path, rel)]. `rel` is the old manifest key (path as in COLLECTIONS),
    so existing manifest and chunk IDs remain valid."""
    validate_config()
    files, missing = [], []
    for cfg in COLLECTIONS:
        cat = resolve_path(cfg["path"])
        if not cat.exists():
            missing.append(str(cat))
            continue
        files += [(cfg, p, (Path(cfg["path"]) / p.relative_to(cat)).as_posix()) for p in sorted(cat.rglob("*.md"))]
    if missing:
        for k in missing:
            print(f"⚠️ Missing directory: {k}")
        if not allow_missing:
            sys.exit("Aborted: missing collection dirs would trigger DELETION of their chunks from index. "
                     "Fix paths or use --allow-missing-dirs.")
    return files


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"pliki": {}}


def save_manifest(manifest):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    tmp = MANIFEST_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    os.replace(tmp, MANIFEST_PATH)


# ==========================================
# OPENAI + CHROMA (lazy imports)
# ==========================================
def create_openai_client():
    api_key = settings.OPENAI_API_KEY

    if not api_key:
        sys.exit(
            "Missing OPENAI_API_KEY in .env file."
        )

    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        max_retries=6
    )


def perform_embedding(client, texts):
    """Embeddings batched by element count and total tokens."""
    result, batch, tok = [], [], 0

    def send_batch():
        nonlocal batch, tok
        if batch:
            kwargs = {"dimensions": DIMENSIONS} if DIMENSIONS else {}
            res = client.embeddings.create(model=MODEL, input=batch, **kwargs)
            result.extend(d.embedding for d in sorted(res.data, key=lambda d: d.index))
        batch, tok = [], 0

    for t in texts:
        n = get_tokens(t)
        if batch and (len(batch) >= MAX_BATCH_ELEMENTS or tok + n > MAX_BATCH_TOKENS):
            send_batch()
        batch.append(t)
        tok += n
    send_batch()
    return result


def open_collection(rebuild=False):
    import chromadb
    client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    col = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "model_embeddingow": MODEL, "wymiary": DIMENSIONS or 0},
    )
    saved_model = (col.metadata or {}).get("model_embeddingow")
    if saved_model and (saved_model != MODEL or (col.metadata or {}).get("wymiary", 0) != (DIMENSIONS or 0)):
        sys.exit(f"Collection was built with model '{saved_model}', but script uses '{MODEL}'. "
                 f"Vectors from different models are not comparable - run with --rebuild.")
    return col


# ==========================================
# ELASTICSEARCH (lazy)
# ==========================================
ES_SETTINGS = {"analysis": {"analyzer": {"default": {"type": "polish"}}}}
ES_MAPPING = {"properties": {
    "tekst": {"type": "text"}, "tytul": {"type": "text"}, "sekcja": {"type": "text"},
    "zrodlo": {"type": "keyword"}, "url": {"type": "keyword", "ignore_above": 2048},
    "plik": {"type": "keyword", "ignore_above": 1024},
}}


def open_es():
    from elasticsearch import Elasticsearch
    return Elasticsearch(ES_URL)


def setup_es_index(es, rebuild=False):
    if rebuild and es.indices.exists(index=ES_INDEX):
        es.indices.delete(index=ES_INDEX)
    if not es.indices.exists(index=ES_INDEX):
        es.indices.create(index=ES_INDEX, settings=ES_SETTINGS, mappings=ES_MAPPING)


def bulk_es(es, actions):
    """bulk with error checking: 404 on deletion is OK; any other error ABORTS."""
    if not actions:
        return
    from elasticsearch.helpers import bulk
    _, errors = bulk(es, actions, raise_on_error=False, stats_only=False)
    errors = [e for e in errors if not ("delete" in e and e["delete"].get("status") == 404)]
    if errors:
        raise RuntimeError(f"Elasticsearch: {len(errors)} bulk operations failed, e.g., {errors[0]}")


def create_es_document(c, md):
    return {"tekst": c.text, "tytul": md.get("tytul", ""), "zrodlo": md.get("zrodlo", ""),
            "sekcja": md.get("sekcja", ""), "url": md.get("url") or "", "plik": md.get("plik", "")}


def build_index_actions(chunks):
    return [{"_op_type": "index", "_index": ES_INDEX, "_id": cid, "_source": create_es_document(c, md)} for cid, c, md in chunks]


def remove_ids(col, ids, es):
    ids = list(ids)
    for i in range(0, len(ids), 500):
        col.delete(ids=ids[i:i + 500])
    bulk_es(es, [{"_op_type": "delete", "_index": ES_INDEX, "_id": i} for i in ids])


def check_es_consistency(es, manifest):
    """Chunk texts live ONLY in ES. If ES was cleared (e.g., docker compose down -v),
    indexing would think everything is up to date, leaving the app with empty texts."""
    expected = sum(len(v["ids"]) for v in manifest["pliki"].values())
    if not expected:
        return
    es.indices.refresh(index=ES_INDEX)
    in_es = es.count(index=ES_INDEX)["count"]
    if in_es < expected:
        print(f"⚠️ DISCREPANCY: manifest has {expected} chunks, but Elasticsearch only has {in_es}. "
              f"Run:  python build_database.py --rebuild-es   (no API costs)")


def rebuild_es(allow_missing_dirs=False):
    """Restores ES index from files: NO OpenAI API and no Chroma modification. Unchanged files
    get the same IDs as in Chroma; others are skipped (standard indexing will handle them)."""
    manifest = load_manifest()
    if not manifest["pliki"]:
        sys.exit("Manifest is empty - run standard indexing first.")
    files = collect_files(allow_missing=allow_missing_dirs)
    es = open_es()
    setup_es_index(es, rebuild=True)
    chunk_count, skipped = 0, []
    for cfg, path, rel in files:
        entry, doc = manifest["pliki"].get(rel), prepare_document(path, cfg)
        if not entry or doc is None or entry["sygnatura"] != get_file_signature(doc[1], doc[2], cfg):
            skipped.append(rel)
            continue
        chunks = chunk_file(path, cfg, doc, rel=rel)
        if [cid for cid, _, _ in chunks] != entry["ids"]:
            skipped.append(rel)
            continue
        bulk_es(es, build_index_actions(chunks))
        chunk_count += len(chunks)
    es.indices.refresh(index=ES_INDEX)
    print(f"✅ ES rebuilt: {chunk_count} chunks from files matching manifest.")
    if skipped:
        print(f"⚠️ Skipped {len(skipped)} files modified since last indexing "
              f"(e.g., {skipped[0]}) - run standard indexing to update them.")


# ==========================================
# INDEXING (incremental)
# ==========================================
def run_indexing(dry_run=False, rebuild=False, confirm_deletion=False, allow_missing_dirs=False):
    files = collect_files(allow_missing=dry_run or allow_missing_dirs)
    if rebuild and os.path.exists(MANIFEST_PATH):
        os.remove(MANIFEST_PATH)
    manifest = load_manifest()
    col = client = es = None
    if not dry_run:
        col = open_collection(rebuild)
        client = create_openai_client()
        es = open_es()
        setup_es_index(es, rebuild)
        if not rebuild:
            check_es_consistency(es, manifest)

            expected = sum(len(v["ids"]) for v in manifest["pliki"].values())
            if expected > 0 and col.count() == 0:
                print("WARNING: ChromaDB is empty, but the manifest claims that data exists. Forcing a full reindexing!")
                manifest["pliki"] = {}

    print(f"Indexing: {len(files)} Markdown files | tokenizer: {TOKENIZER}"
          + ("  (approximation - install tiktoken)" if TOKENIZER != "tiktoken" else ""))

    current = {rel for _, _, rel in files}
    deleted = [r for r in manifest["pliki"] if r not in current]
    limit = max(5, len(manifest["pliki"]) * MAX_DELETED_PERCENT // 100)
    if len(deleted) > limit and not dry_run and not confirm_deletion:
        sys.exit(f"Aborted: attempting to delete {len(deleted)} of {len(manifest['pliki'])} files (safeguard limit: {limit}), "
                 f"e.g., {deleted[0]}.\nIf intentional, run again with --confirm-deletion; "
                 f"otherwise, check paths in COLLECTIONS.")
    for rel in deleted:
        if not dry_run:
            remove_ids(col, manifest["pliki"][rel]["ids"], es)
            del manifest["pliki"][rel]
    if deleted:
        print(f"Deleted files: {len(deleted)} (chunks {'to delete' if dry_run else 'deleted from DB'})")
        if not dry_run:
            save_manifest(manifest)

    stats = {"unchanged": 0, "processed": 0, "empty": 0, "chunks": 0, "new_chunks": 0, "new_tokens": 0}
    sizes, with_table = [], 0
    buffer = []
    buffer_chunks = 0

    def flush_buffer():
        nonlocal buffer, buffer_chunks
        if not buffer:
            return
        to_emb = [x for (_, _, _, _, new_chunks, _) in buffer for x in new_chunks]
        if to_emb:
            vectors = perform_embedding(client, [c.text_emb for _, c, _ in to_emb])
            for i in range(0, len(to_emb), 256):
                part = to_emb[i:i + 256]
                col.upsert(ids=[cid for cid, _, _ in part],
                           embeddings=vectors[i:i + 256],
                           documents=[c.text for _, c, _ in part],
                           metadatas=[md for _, _, md in part])
                bulk_es(es, build_index_actions(part))

        all_kept = [x for (_, _, _, _, _, kept) in buffer for x in kept]
        if all_kept:
            for i in range(0, len(all_kept), 2000):
                batch = all_kept[i:i + 2000]
                col.update(ids=[cid for cid, _, _ in batch], metadatas=[md for _, _, md in batch])
                bulk_es(es, build_index_actions(batch))
                print(f"    ⚡ Odzyskiwanie linków: {min(i+2000, len(all_kept))}/{len(all_kept)}...")
        for rel, sig, all_ids, old_ids, _, _ in buffer:
            orphans = set(old_ids) - set(all_ids)
            if orphans:
                remove_ids(col, orphans, es)
            manifest["pliki"][rel] = {"sygnatura": sig, "ids": all_ids}
        save_manifest(manifest)
        if to_emb:
            print(f"    💾 Zapisano {len(to_emb)} nowych wektorów.")   
        buffer, buffer_chunks = [], 0

    try:
        for cfg, path, rel in files:
            doc = prepare_document(path, cfg)
            if doc is None:
                stats["empty"] += 1
                continue
            sig = get_file_signature(doc[0], doc[1], doc[2], cfg)
            old = manifest["pliki"].get(rel, {})
            if old.get("sygnatura") == sig:
                stats["unchanged"] += 1
                continue

            chunks = chunk_file(path, cfg, doc, rel=rel)
            old_ids = set(old.get("ids", []))
            new_chunks = [(cid, c, md) for cid, c, md in chunks if cid not in old_ids]
            stats["processed"] += 1
            stats["chunks"] += len(chunks)
            stats["new_chunks"] += len(new_chunks)
            stats["new_tokens"] += sum(c.tokens_emb for _, c, _ in new_chunks)
            sizes += [c.tokens for _, c, _ in chunks]
            with_table += sum(c.has_table for _, c, _ in chunks)

            if dry_run:
                continue
            kept = [(cid, c, md) for cid, c, md in chunks if cid in old_ids]
            buffer.append((rel, sig, [cid for cid, _, _ in chunks], old_ids, new_chunks, kept))
            buffer_chunks += len(new_chunks)
            if buffer_chunks >= 256:
                flush_buffer()
        flush_buffer()
    except KeyboardInterrupt:
        print("\nAborted - what was saved is consistent; run again to continue.")
        return

    generate_report(stats, sizes, with_table, dry_run)


def generate_report(stats, sizes, with_table, dry_run):
    print("\n" + "=" * 56)
    print("📊 " + ("ESTIMATE (dry-run, nothing sent to API)" if dry_run else "INDEXING REPORT"))
    print("=" * 56)
    print(f"Unchanged files (skipped):        {stats['unchanged']}")
    print(f"Processed files:                  {stats['processed']}   (empty/too short: {stats['empty']})")
    print(f"Chunks from these files:          {stats['chunks']}   (with tables: {with_table})")
    print(f"Chunks to (re)embed:              {stats['new_chunks']}")
    if sizes:
        r = sorted(sizes)
        p = lambda q: r[min(len(r) - 1, int(len(r) * q))]
        print(f"Chunk sizes [tokens]:             min {r[0]} | median {p(0.5)} | p90 {p(0.9)} | max {r[-1]}")
        tiny = sum(1 for x in r if x < 60)
        print(f"Very small chunks (<60 tok.):     {tiny} ({100 * tiny // len(r)}%)")
    cost = stats["new_tokens"] / 1e6 * PRICE_PER_1M_TOKENS_USD
    print(f"Tokens for embedding:             {stats['new_tokens']:,}  ≈ ${cost:.3f} (price from settings: ${PRICE_PER_1M_TOKENS_USD}/1M)")
    print("=" * 56)


# ==========================================
# TOOLS: PREVIEW AND SEARCH TEST
# ==========================================
def preview_chunking(file_path):
    p = Path(file_path)
    cfg = next((c for c in COLLECTIONS if resolve_path(c["path"]) in p.resolve().parents),
               {"path": str(p.parent), "source": "Test Source", "type": "strona"})
    result = chunk_file(p, cfg)
    if not result:
        print("File empty / too short.")
        return
    print(f"File: {p} | collection: {cfg['source']} | chunks: {len(result)} | tokenizer: {TOKENIZER}\n")
    for cid, c, md in result:
        print("━" * 70)
        print(f"CHUNK {md['chunk_nr'] + 1}/{md['chunk_z']} | {c.tokens} tok. | section: {c.section or '(start)'}"
              f"{' | TABLE' if c.has_table else ''}")
        print("━" * 70)
        print(c.text)
        print()


def search_test(query, k=5, source=None):
    client = create_openai_client()
    col = open_collection()
    vector = perform_embedding(client, [query])[0]
    res = col.query(query_embeddings=[vector], n_results=k, where={"zrodlo": source} if source else None,
                    include=["documents", "metadatas", "distances"])
    for i, (doc, md, dist) in enumerate(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), 1):
        print("━" * 70)
        print(f"#{i}  similarity {1 - dist:.3f} | {md.get('zrodlo')} | {md.get('tytul')}"
              f"{' › ' + md['sekcja'] if md.get('sekcja') else ''}")
        print(f"    {md.get('url') or md.get('plik')}")
        print("    " + doc.replace("\n", "\n    ")[:700] + ("…" if len(doc) > 700 else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build RAG vector database (Markdown -> Chroma)")
    ap.add_argument("--dry-run", action="store_true", help="count chunks and cost, no API calls")
    ap.add_argument("--rebuild", action="store_true", help="delete Chroma collection and ES index, build from scratch")
    ap.add_argument("--rebuild-es", action="store_true", help="restore ES index from files (no OpenAI API)")
    ap.add_argument("--confirm-deletion", action="store_true", help="allow massive chunk deletion (safeguard override)")
    ap.add_argument("--allow-missing-dirs", action="store_true", help="continue despite missing collection directories")
    ap.add_argument("--preview", metavar="FILE.md", help="show how a file will be chunked")
    ap.add_argument("--search", metavar="QUERY", help="test database search")
    ap.add_argument("-k", type=int, default=5, help="number of --search results")
    ap.add_argument("--source", help="source filter for --search, e.g., 'Biblioteka Główna AGH'")
    a = ap.parse_args()
    if a.preview:
        preview_chunking(a.preview)
    elif a.search:
        search_test(a.search, a.k, a.source)
    elif a.rebuild_es:
        rebuild_es(allow_missing_dirs=a.allow_missing_dirs)
    else:
        run_indexing(dry_run=a.dry_run, rebuild=a.rebuild, confirm_deletion=a.confirm_deletion,
                     allow_missing_dirs=a.allow_missing_dirs)