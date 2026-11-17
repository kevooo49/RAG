"""Builds the evaluation sample from the full knowledge base.

Properties of the sample:

* **Reproducible**: everything is driven by `--seed`; the parameters are saved in the manifest.
* **Same files as production**: the population is exactly what `build_database.py` indexes
  (`sites.yaml -> collections()`, `*.md`, body >= MIN_DOC_CHARS). Raw/temp/_meta folders are never sampled.
* **No name collisions**: the sample mirrors the folder layout of `data_root`, and every file is identified by
  the same ID production uses in the `plik` field (e.g. `../../data/Baza_Wiedzy_AGH/CSS_Strony/x.md`).
  A question dataset written against these IDs therefore works on the sample AND on the full index.
* **Duplicates are known**: exact and near-duplicate documents (MinHash, Jaccard on word 5-grams) are grouped
  over the WHOLE base. If a target has copies, the copies are put into the sample too, and evaluation can count
  a hit on any member of the group as correct.
* **Hard distractors**: for every target, up to `--neighbors` documents with the same normalized title but
  different content (same course in another year/faculty, same regulation on another unit's site) are added.
* **Realistic background**: the rest of the sample is drawn proportionally to the real composition of the base
  (source x type), so syllabi are as common in the sample as they are in production.

Targets (documents questions will be written about) are drawn per stratum (source x type) with an allocation
proportional to sqrt(stratum size): big sources get more targets, small ones are not drowned out.

Output (`--out`, default = paths.eval_dir from sites.yaml):
    <out>/<collection folder>/.../*.md   copies of the sampled files (layout as in data_root)
    <out>/sample_manifest.json           parameters, counts, one record per file (id, role, source, type, group...)
    <out>/dup_groups_full.json           duplicate groups over the WHOLE base (for evaluation on the full index)

Usage (from backend/scripts):
    python build_eval_sample.py --dry-run                 # scan + allocation + statistics, copies nothing
    python build_eval_sample.py --out ../../data/Ewaluacja_Probka
    python build_eval_sample.py --size 3000 --targets 400 --seed 7 --out ...

The first run reads every file of the base (~100k) and computes MinHash signatures; the result is cached in
`.eval_cache/inventory.*` and reused as long as file sizes/mtimes do not change.

Dependencies: numpy, PyYAML (both already used by the project).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from sites import load_or_exit  # noqa: E402  # type: ignore

SCRIPT_VERSION = "eval-sample"
CACHE_DIR = Path(__file__).resolve().parent / ".eval_cache"
CACHE_STEM = CACHE_DIR / "inventory"

MIN_DOC_CHARS = 80          # same as build_database.MIN_DOC_CHARS (shorter files are not indexed)
SHINGLE = 5                 # words per shingle
NUM_PERM = 128              # MinHash permutations
LSH_BANDS, LSH_ROWS = 16, 8  # 16 x 8 = 128; candidate threshold ~ (1/16)^(1/8) = 0.71 Jaccard
MAX_BUCKET_FULL = 2000      # bigger LSH buckets are compared against a random subset only
BUCKET_SUBSAMPLE = 200

_FRONTMATTER = re.compile(r"\A﻿?---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.S)
_URL = re.compile(r"https?://\S+|www\.\S+")
_WORD = re.compile(r"\w+", re.U)


# ==========================================
# 1. INVENTORY (scan + signatures, cached)
# ==========================================
@dataclass
class Doc:
    id: str                 # production `plik` (id_path + relative path), POSIX
    path: str               # absolute path on disk
    rel_out: str            # path inside the sample: <collection folder>/<relative path>
    collection: str         # collection folder name, e.g. CSS_Strony, Sylabusy_Markdown/2024_2025
    source: str
    type: str               # strona / dokument / sylabus
    size: int
    mtime_ns: int
    chars: int = 0          # body length (after frontmatter and HTML comments)
    title: str = ""
    title_key: str = ""     # normalized title used for "hard neighbour" matching
    sha: str = ""           # sha1 of normalized body (exact duplicates)


def read_body(path: Path) -> Tuple[dict, str]:
    content = path.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n")
    meta: dict = {}
    m = _FRONTMATTER.match(content)
    if m:
        try:
            meta = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
        content = content[m.end():]
    body = re.sub(r"<!--.*?-->", "", content, flags=re.S)
    body = re.sub(r"<br\s*/?>", " ", body, flags=re.I).strip()
    return (meta if isinstance(meta, dict) else {}), body


def doc_title(meta: dict, body: str, path: Path) -> str:
    """Same rule as build_database.prepare_document."""
    title = str(meta.get("tytul") or "").strip()
    if not title:
        m = re.search(r"^#\s+(.+)$", body, re.M)
        title = m.group(1).strip() if m else path.stem
    return title


def title_key(title: str) -> str:
    t = title.lower()
    t = re.sub(r"_[0-9a-f]{8}$", "", t)              # crawler hash suffix in file-name titles
    t = re.sub(r"\d+", " ", t)                       # years, semester numbers, versions
    t = re.sub(r"[^\w]+", " ", t, flags=re.U)
    return re.sub(r"\s+", " ", t).strip()


def normalize_for_hash(body: str) -> List[str]:
    return _WORD.findall(_URL.sub(" ", body.lower()))


def minhash(tokens: List[str], perm_a: np.ndarray, perm_b: np.ndarray, token_cache: Dict[str, int]) -> np.ndarray:
    """MinHash over word 5-gram shingles. Hashing is vectorized; uint64 overflow is intentional (wrap-around)."""
    ids = np.fromiter((token_cache.setdefault(t, zlib.crc32(t.encode("utf-8"))) for t in tokens),
                      dtype=np.uint64, count=len(tokens))
    if len(ids) == 0:
        return np.full(NUM_PERM, np.iinfo(np.uint32).max, dtype=np.uint32)
    with np.errstate(over="ignore"):
        if len(ids) < SHINGLE:
            sh = np.array([np.bitwise_xor.reduce(ids * np.uint64(0x9E3779B97F4A7C15))], dtype=np.uint64)
        else:
            n = len(ids) - SHINGLE + 1
            sh = np.zeros(n, dtype=np.uint64)
            for k in range(SHINGLE):
                sh = sh * np.uint64(1099511628211) ^ ids[k:k + n]
        sh = np.unique(sh)
        sig = np.full(NUM_PERM, np.iinfo(np.uint64).max, dtype=np.uint64)
        for start in range(0, len(sh), 20000):
            block = sh[start:start + 20000]
            hv = perm_a[:, None] * block[None, :] + perm_b[:, None]
            np.minimum(sig, hv.min(axis=1), out=sig)
    return (sig >> np.uint64(32)).astype(np.uint32)


def list_population(cfg) -> List[Doc]:
    docs = []
    data_root = cfg.data_root
    for col in cfg.collections():
        cat = Path(col["path"])
        if not cat.is_dir():
            print(f"   [!] collection folder missing, skipped: {cat}")
            continue
        col_name = cat.relative_to(data_root).as_posix()
        for p in sorted(cat.rglob("*.md")):
            st = p.stat()
            rel = p.relative_to(cat).as_posix()
            docs.append(Doc(id=f"{col['id_path']}/{rel}", path=str(p), rel_out=f"{col_name}/{rel}",
                            collection=col_name, source=col["source"], type=col["type"],
                            size=st.st_size, mtime_ns=st.st_mtime_ns))
    return docs


def build_inventory(cfg, refresh: bool) -> Tuple[List[Doc], np.ndarray]:
    """Returns (indexable docs, MinHash signatures [n, NUM_PERM] uint32). Uses the cache when possible."""
    t0 = time.time()
    population = list_population(cfg)
    print(f"   {len(population)} .md files in {len(cfg.collections())} indexed collections")

    cached: Dict[str, Tuple[dict, np.ndarray]] = {}
    meta_path, sig_path = CACHE_STEM.with_suffix(".json"), CACHE_STEM.with_suffix(".npy")
    if not refresh and meta_path.exists() and sig_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            sigs = np.load(sig_path)
            if meta.get("version") == SCRIPT_VERSION and len(meta["docs"]) == len(sigs):
                for rec, sig in zip(meta["docs"], sigs):
                    cached[rec["id"]] = (rec, sig)
        except (OSError, ValueError, KeyError) as e:
            print(f"   [!] cache unreadable ({e}), rebuilding")

    rng = np.random.default_rng(12345)          # fixed: signatures must not depend on --seed
    perm_a = rng.integers(1, 2**63, NUM_PERM, dtype=np.uint64) | np.uint64(1)
    perm_b = rng.integers(0, 2**63, NUM_PERM, dtype=np.uint64)
    token_cache: Dict[str, int] = {}

    kept, sigs_out, reused = [], [], 0
    for i, d in enumerate(population, 1):
        c = cached.get(d.id)
        if c and c[0]["size"] == d.size and c[0]["mtime_ns"] == d.mtime_ns:
            rec, sig = c
            if rec["chars"] >= MIN_DOC_CHARS:
                d.chars, d.title, d.title_key, d.sha = rec["chars"], rec["title"], rec["title_key"], rec["sha"]
                kept.append(d)
                sigs_out.append(sig)
            reused += 1
            continue
        try:
            meta, body = read_body(Path(d.path))
        except OSError as e:
            print(f"\n   [!] cannot read {d.path}: {e}")
            continue
        d.chars = len(body)
        tokens = normalize_for_hash(body)
        d.sha = hashlib.sha1(" ".join(tokens).encode("utf-8")).hexdigest()
        d.title = doc_title(meta, body, Path(d.path))
        d.title_key = title_key(d.title)
        sig = minhash(tokens, perm_a, perm_b, token_cache)
        cached[d.id] = (asdict(d), sig)
        if d.chars >= MIN_DOC_CHARS:
            kept.append(d)
            sigs_out.append(sig)
        if i % 500 == 0:
            print(f"   scanned {i}/{len(population)} (reused from cache: {reused})", end="\r")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ids_now = {d.id for d in population}
    recs = [(r, s) for k, (r, s) in cached.items() if k in ids_now]
    meta_path.write_text(json.dumps({"version": SCRIPT_VERSION, "docs": [r for r, _ in recs]},
                                    ensure_ascii=False), encoding="utf-8")
    np.save(sig_path, np.stack([s for _, s in recs]) if recs else np.zeros((0, NUM_PERM), np.uint32))

    print(f"\n   {len(kept)} indexable documents (>= {MIN_DOC_CHARS} chars); "
          f"{reused} from cache; {time.time() - t0:.0f}s")
    return kept, (np.stack(sigs_out) if sigs_out else np.zeros((0, NUM_PERM), np.uint32))


# ==========================================
# 2. DUPLICATE GROUPS
# ==========================================
class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def duplicate_groups(docs: List[Doc], sigs: np.ndarray, threshold: float, seed: int) -> List[int]:
    """Returns group index per doc. Exact duplicates (same normalized text) + near-duplicates whose MinHash
    Jaccard estimate >= threshold (candidates from LSH banding)."""
    n = len(docs)
    uf = UnionFind(n)
    by_sha = defaultdict(list)
    for i, d in enumerate(docs):
        by_sha[d.sha].append(i)
    for members in by_sha.values():
        for j in members[1:]:
            uf.union(members[0], j)

    rng = random.Random(seed)
    compared = 0
    for b in range(LSH_BANDS):
        band = sigs[:, b * LSH_ROWS:(b + 1) * LSH_ROWS]
        buckets = defaultdict(list)
        for i, key in enumerate(map(bytes, band)):
            buckets[key].append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            # one representative per already-merged group keeps big exact-duplicate clusters cheap
            reps = list({uf.find(i): i for i in members}.values())
            if len(reps) < 2:
                continue
            S = sigs[reps]
            for a in range(len(reps) - 1):
                if len(reps) > MAX_BUCKET_FULL:
                    others = np.array(rng.sample(range(a + 1, len(reps)), min(BUCKET_SUBSAMPLE, len(reps) - a - 1)))
                else:
                    others = np.arange(a + 1, len(reps))
                if len(others) == 0:
                    continue
                sim = (S[others] == S[a]).mean(axis=1)
                compared += len(others)
                for o in others[sim >= threshold]:
                    uf.union(reps[a], reps[int(o)])
        print(f"   LSH band {b + 1}/{LSH_BANDS} ({compared} pair comparisons)", end="\r")
    print()
    roots = [uf.find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return [remap[r] for r in roots]


# ==========================================
# 3. SAMPLING
# ==========================================
def allocate(caps: Dict[Tuple[str, str], int], total: int, power: float, minimum: int,
             weights: Optional[Dict[Tuple[str, str], int]] = None,
             floors: Optional[Dict[Tuple[str, str], int]] = None) -> Dict[Tuple[str, str], int]:
    """Largest-remainder allocation of `total` over strata.
    weight = weights[k]**power (default: caps[k]**power); every stratum gets at least `minimum`
    (and at least floors[k]) but never more than caps[k]."""
    weights = weights or caps
    floors = floors or {}
    alloc = {k: min(c, max(minimum, floors.get(k, 0))) for k, c in caps.items()}
    rest = total - sum(alloc.values())
    if rest <= 0:
        return alloc
    w = {k: weights[k] ** power for k, c in caps.items() if c > alloc[k]}
    while rest > 0 and w:
        tot = sum(w.values())
        quota = {k: rest * x / tot for k, x in w.items()}
        add = {k: min(int(q), caps[k] - alloc[k]) for k, q in quota.items()}
        given = sum(add.values())
        if given == 0:          # hand out the remainder by the largest fractional parts
            for k in sorted(quota, key=lambda k: quota[k] - int(quota[k]), reverse=True)[:rest]:
                add[k] = 1
            given = sum(add.values())
        for k, a in add.items():
            alloc[k] += a
        rest -= given
        w = {k: x for k, x in w.items() if caps[k] > alloc[k]}
    return alloc


# Pages that exist on almost every site and carry no askable content (footer, sitemap, accessibility
# statement, cookie/privacy boilerplate, home page = news feed). They stay in the sample as background,
# because production indexes them too - they just never become targets.
BOILERPLATE = re.compile(
    r"(^_?stopka|^_?footer|^_?header|^_?menu|mapa[-_ ]?(serwisu|strony)|sitemap|"
    r"deklaracja[-_ ]dost[eę]pno|polityka[-_ ]prywatno|cookies?$|^index$|^strona[-_ ]g[lł][oó]wna$)", re.I)
_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_LIST_MARK = re.compile(r"^\s*([-*+]|\d+[.)])\s+")


def target_quality(d: "Doc", a) -> Optional[str]:
    """None if the document is a good question target, otherwise the rejection reason."""
    stem = re.sub(r"_[0-9a-f]{8}$", "", Path(d.path).stem)
    if BOILERPLATE.search(stem) or BOILERPLATE.search(d.title.strip()):
        return "boilerplate"
    try:
        _meta, body = read_body(Path(d.path))
    except OSError:
        return "unreadable"
    lines = [ln for ln in body.splitlines() if ln.strip()]
    link_lines = 0
    for ln in lines:
        if "](" in ln or "http" in ln:
            rest = _LIST_MARK.sub("", _URL.sub("", _MD_LINK.sub("", ln)))
            if len(re.sub(r"[\W_]+", "", rest)) < 20:
                link_lines += 1
    if lines and link_lines / len(lines) > a.max_link_ratio:
        return "link_list"
    prose = _URL.sub(" ", _MD_LINK.sub(lambda m: m.group(1), body))
    prose = re.sub(r"[#*_>|`~-]+", " ", prose)
    if len(re.sub(r"\s+", " ", prose).strip()) < a.min_target_chars:
        return "too_little_text"
    return None


@dataclass
class Selection:
    roles: Dict[int, str] = field(default_factory=dict)        # doc index -> role
    target_of: Dict[int, int] = field(default_factory=dict)    # neighbour/duplicate -> its target
    rejected: Counter = field(default_factory=Counter)          # target rejection reason -> count
    rejected_examples: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))


def select(docs: List[Doc], groups: List[int], a, rng: random.Random) -> Selection:
    sel = Selection()
    members = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)

    # --- targets: one per duplicate group, stratified by (source, type) ---
    canon = {}
    for g, idx in members.items():
        canon[g] = min(idx, key=lambda i: (docs[i].type != "dokument", docs[i].id))   # stable choice
    eligible = defaultdict(list)
    for g, i in canon.items():
        d = docs[i]
        if d.chars >= a.min_target_chars and (a.max_target_chars <= 0 or d.chars <= a.max_target_chars):
            eligible[(d.source, d.type)].append(i)
    exempt = {t.strip() for t in a.cap_exempt_types.split(",") if t.strip()}
    weights = {k: len(v) for k, v in eligible.items()}
    caps = {k: len(v) if (a.max_targets_per_stratum <= 0 or k[1] in exempt)
            else min(len(v), a.max_targets_per_stratum) for k, v in eligible.items()}
    order = {k: rng.sample(sorted(v), len(v)) for k, v in sorted(eligible.items())}
    taken: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    pos: Dict[Tuple[str, str], int] = defaultdict(int)
    # Draw in random order, skip low-quality docs; if a stratum runs out of good docs, its cap drops to what
    # it got and the remaining targets are re-allocated over the other strata.
    while True:
        need = min(a.targets, sum(caps.values()))
        alloc = allocate(caps, need, a.alloc_power, a.min_per_stratum, weights,
                         {k: len(v) for k, v in taken.items()})
        short = False
        for k in sorted(alloc):
            while len(taken[k]) < alloc[k] and pos[k] < len(order[k]):
                i = order[k][pos[k]]
                pos[k] += 1
                reason = target_quality(docs[i], a)
                if reason:
                    sel.rejected[reason] += 1
                    if len(sel.rejected_examples[reason]) < 8:
                        sel.rejected_examples[reason].append(docs[i].id)
                    continue
                taken[k].append(i)
            if len(taken[k]) < alloc[k]:
                caps[k] = len(taken[k])
                short = True
        if not short:
            break
    for k, idx in taken.items():
        for i in idx:
            sel.roles[i] = "target"

    # --- duplicates of targets (all copies, capped) ---
    for t in [i for i, r in sel.roles.items() if r == "target"]:
        copies = [j for j in members[groups[t]] if j != t]
        for j in rng.sample(copies, min(len(copies), a.max_group)):
            sel.roles.setdefault(j, "duplicate")
            sel.target_of.setdefault(j, t)

    # --- hard neighbours: same normalized title, different duplicate group ---
    by_title = defaultdict(list)
    for i, d in enumerate(docs):
        if d.title_key:
            by_title[d.title_key].append(i)
    for t in [i for i, r in sel.roles.items() if r == "target"]:
        cands = [j for j in by_title.get(docs[t].title_key, []) if groups[j] != groups[t] and j not in sel.roles]
        for j in rng.sample(cands, min(len(cands), a.neighbors)):
            sel.roles[j] = "neighbor"
            sel.target_of[j] = t

    # --- background: proportional to the real composition of the base ---
    remaining = a.size - len(sel.roles)
    if remaining > 0:
        pool = defaultdict(list)
        for i, d in enumerate(docs):
            if i not in sel.roles:
                pool[(d.source, d.type)].append(i)
        sizes = {k: len(v) for k, v in pool.items()}
        alloc = allocate(sizes, min(remaining, sum(sizes.values())), 1.0, 0)
        for k in sorted(alloc):
            for i in rng.sample(sorted(pool[k]), alloc[k]):
                sel.roles[i] = "background"
    return sel


# ==========================================
# 4. OUTPUT
# ==========================================
def summarize(docs, groups, sel) -> dict:
    pop = Counter((d.source, d.type) for d in docs)
    by_role = Counter(sel.roles.values())
    strata = defaultdict(lambda: Counter())
    for i, r in sel.roles.items():
        strata[f"{docs[i].source} | {docs[i].type}"][r] += 1
    group_sizes = Counter(groups)
    return {
        "population_docs": len(docs),
        "population_by_type": dict(Counter(d.type for d in docs)),
        "duplicate_groups_with_copies": sum(1 for c in group_sizes.values() if c > 1),
        "docs_in_duplicate_groups": sum(c for c in group_sizes.values() if c > 1),
        "sample_docs": len(sel.roles),
        "sample_by_role": dict(by_role),
        "sample_by_type": dict(Counter(docs[i].type for i in sel.roles)),
        "target_rejections": dict(sel.rejected),
        "target_rejection_examples": {k: v for k, v in sel.rejected_examples.items()},
        "strata": {k: {"population": pop[tuple(k.split(" | "))], **v} for k, v in sorted(strata.items())},
    }


def print_summary(s: dict):
    print("\n" + "=" * 78)
    print(f"Population: {s['population_docs']} docs  {s['population_by_type']}")
    print(f"Duplicate groups (>1 member): {s['duplicate_groups_with_copies']} "
          f"covering {s['docs_in_duplicate_groups']} docs")
    print(f"Sample: {s['sample_docs']} docs  roles={s['sample_by_role']}  types={s['sample_by_type']}")
    if s["target_rejections"]:
        print(f"Rejected as targets (stay available as background): {s['target_rejections']}")
        for reason, ex in s["target_rejection_examples"].items():
            print(f"   {reason}: " + ", ".join(e.rsplit('/', 1)[-1] for e in ex[:4]))
    print("-" * 78)
    print(f"{'source | type':<52}{'pop':>8}{'tgt':>5}{'dup':>5}{'nbr':>5}{'bg':>6}")
    for k, v in s["strata"].items():
        print(f"{k[:51]:<52}{v['population']:>8}{v.get('target', 0):>5}{v.get('duplicate', 0):>5}"
              f"{v.get('neighbor', 0):>5}{v.get('background', 0):>6}")
    print("=" * 78)


def write_sample(out: Path, docs, groups, sel, summary, args):
    gid = {}
    for i in sel.roles:
        gid.setdefault(groups[i], f"g{len(gid):05d}")
    members = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)

    files = []
    for i in sorted(sel.roles, key=lambda i: docs[i].id):
        d = docs[i]
        dst = out / d.rel_out
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(d.path, dst)
        files.append({
            "id": d.id, "sample_path": d.rel_out, "collection": d.collection, "source": d.source,
            "type": d.type, "role": sel.roles[i], "title": d.title, "chars": d.chars,
            "dup_group": gid[groups[i]],
            "dup_group_size_full_db": len(members[groups[i]]),
            "target": docs[sel.target_of[i]].id if i in sel.target_of else None,
        })

    sample_groups = defaultdict(list)
    for f in files:
        sample_groups[f["dup_group"]].append(f["id"])
    manifest = {
        "version": SCRIPT_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "id_prefix_note": "id == production `plik` field (id_path from sites.yaml + path inside the collection)",
        "summary": summary,
        "dup_groups_in_sample": {g: ids for g, ids in sample_groups.items() if len(ids) > 1},
        "files": files,
    }
    (out / "sample_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    full = [[docs[i].id for i in idx] for idx in members.values() if len(idx) > 1]
    (out / "dup_groups_full.json").write_text(
        json.dumps({"version": SCRIPT_VERSION, "threshold": args.dup_threshold, "groups": full},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nWritten {len(files)} files + sample_manifest.json + dup_groups_full.json to {out}")


# ==========================================
# 5. CLI
# ==========================================
def main():
    cfg = load_or_exit()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=cfg.eval_dir, help="output folder (must be empty or missing)")
    ap.add_argument("--size", type=int, default=2000, help="total number of documents in the sample")
    ap.add_argument("--targets", type=int, default=300, help="documents questions will be written about")
    ap.add_argument("--neighbors", type=int, default=2, help="hard distractors (same title) per target")
    ap.add_argument("--max-group", type=int, default=10, help="max copies of a target taken from its dup group")
    ap.add_argument("--min-target-chars", type=int, default=800, help="shorter docs are never targets")
    ap.add_argument("--max-target-chars", type=int, default=0, help="0 = no upper limit")
    ap.add_argument("--max-targets-per-stratum", type=int, default=12,
                    help="cap on targets per (source, type); 0 = no cap")
    ap.add_argument("--cap-exempt-types", default="sylabus",
                    help="comma-separated types the cap does not apply to")
    ap.add_argument("--max-link-ratio", type=float, default=0.5,
                    help="reject targets whose non-empty lines are mostly bare links (menus, sitemaps)")
    ap.add_argument("--alloc-power", type=float, default=0.5,
                    help="target allocation ~ stratum_size**power (0 = equal, 1 = proportional)")
    ap.add_argument("--min-per-stratum", type=int, default=1, help="minimum targets per (source, type)")
    ap.add_argument("--dup-threshold", type=float, default=0.85, help="MinHash Jaccard to treat docs as copies")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--refresh-inventory", action="store_true", help="ignore the scan cache")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, copy nothing")
    args = ap.parse_args()

    if args.targets > args.size:
        sys.exit("--targets cannot be larger than --size")
    out = args.out.resolve()
    if not args.dry_run and out.exists() and any(out.iterdir()):
        sys.exit(f"Output folder is not empty: {out}\nChoose another --out or empty it yourself "
                 f"(the script never deletes anything).")
    if not cfg.data_root.is_dir():
        sys.exit(f"data_root does not exist: {cfg.data_root}")

    print(f"[1/3] Scanning {cfg.data_root}")
    docs, sigs = build_inventory(cfg, args.refresh_inventory)
    if not docs:
        sys.exit("No indexable documents found.")

    print(f"[2/3] Grouping duplicates (Jaccard >= {args.dup_threshold})")
    groups = duplicate_groups(docs, sigs, args.dup_threshold, args.seed)

    print("[3/3] Sampling")
    sel = select(docs, groups, args, random.Random(args.seed))
    summary = summarize(docs, groups, sel)
    print_summary(summary)

    if args.dry_run:
        print("\nDry run - nothing copied.")
        return
    write_sample(out, docs, groups, sel, summary, args)
    if out != cfg.eval_dir.resolve():
        print(f"Remember to point paths.eval_dir in config/sites.yaml (or RAG_EVAL_DIR) at {out}")


if __name__ == "__main__":
    main()