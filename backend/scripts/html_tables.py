"""HTML tables in parsed documents -> Markdown tables.

LlamaParse writes complex tables (merged cells) as raw HTML (<table><tr><td rowspan=..>) inside its Markdown.
The production chunker only understands Markdown tables (it keeps them atomic and repeats the header row when a
table has to be split), so HTML tables were chunked as plain text, with tags, and showed up in the LLM context.

`convert_html_tables(text)` rewrites every <table> into a Markdown table:
    * rowspan  -> the value is repeated in every spanned row (each row stays self-contained after splitting)
    * colspan  -> the value goes into the first spanned column, the others stay empty
    * header   -> the first row (with <th> or not - Markdown needs one)
    * cells    -> <strong>/<b> -> **x**, <em>/<i> -> *x*, <br> -> space, <sup>x</sup> -> ^x, other tags dropped,
                  HTML entities decoded, "|" escaped, LaTeX ($...$) kept as is
    * columns and rows that are empty everywhere are dropped; tables LlamaParse left unclosed are handled
Also: <hr/> -> "---", stray <br/> -> line break (but -> space inside an existing Markdown table row).

Used by document_parser.py for new documents. For documents already parsed (no new LlamaParse calls):
    python html_tables.py --dry-run                 # what would change, per folder
    python html_tables.py                           # rewrite files; originals copied to <folder>_html_backup/
    python html_tables.py --file some.md --print    # show the result for one file
    python html_tables.py --restore                 # put the originals from <folder>_html_backup/ back
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import warnings
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

# A table cell that holds only a URL makes bs4 warn "looks like a URL" - harmless here, we parse cell text.
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

_TABLE_OPEN = re.compile(r"<table\b[^>]*>", re.I)
_TABLE_CLOSE = re.compile(r"</table\s*>", re.I)
_TR_CLOSE = re.compile(r"</tr\s*>", re.I)
# rows LlamaParse emitted outside any <table>: consecutive lines starting with <tr
_ORPHAN_ROWS = re.compile(r"(?:^[ \t]*<tr\b.*(?:\n|$))+", re.I | re.M)


def _cell_text(cell) -> str:
    html = cell.decode_contents()
    html = re.sub(r"<br\s*/?>", " ", html, flags=re.I)
    html = re.sub(r"<(strong|b)\b[^>]*>(.*?)</\1\s*>", lambda m: f"**{m.group(2).strip()}**" if m.group(2).strip() else "",
                  html, flags=re.I | re.S)
    html = re.sub(r"<(em|i)\b[^>]*>(.*?)</\1\s*>", lambda m: f"*{m.group(2).strip()}*" if m.group(2).strip() else "",
                  html, flags=re.I | re.S)
    html = re.sub(r"<sup\b[^>]*>(.*?)</sup\s*>", r"^\1", html, flags=re.I | re.S)
    text = BeautifulSoup(html, "html.parser").get_text("")        # drops remaining tags, decodes entities
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\*\*\s*\*\*", "", text)                        # empty bold left by nested markup
    return text.replace("|", r"\|")


def table_to_markdown(table_html: str) -> str:
    soup = BeautifulSoup(table_html, "html.parser")
    table = soup.find("table") or soup
    rows = [tr for tr in table.find_all("tr") if tr.find_parent("table") is table or table is soup]
    grid: List[List[Optional[str]]] = []
    pending: dict = {}                                             # column -> (text, rows left)
    for tr in rows:
        cells = [c for c in tr.find_all(["td", "th"], recursive=False)]
        row: List[Optional[str]] = []
        col = 0

        def fill_pending():
            nonlocal col
            while col in pending:
                text, left = pending[col]
                row.append(text)
                if left <= 1:
                    del pending[col]
                else:
                    pending[col] = (text, left - 1)
                col += 1

        for c in cells:
            fill_pending()
            text = _cell_text(c)
            try:
                rs = max(1, int(c.get("rowspan", 1) or 1))
                cs = max(1, int(c.get("colspan", 1) or 1))
            except ValueError:
                rs, cs = 1, 1
            for k in range(cs):
                val = text if k == 0 else ""
                row.append(val)
                if rs > 1:
                    pending[col] = (val, rs - 1)
                col += 1
        fill_pending()
        grid.append(row)

    width = max((len(r) for r in grid), default=0)
    if width == 0:
        return ""
    grid = [r + [""] * (width - len(r)) for r in grid]
    grid = [r for r in grid if any(x for x in r)]
    if not grid:
        return ""
    keep = [j for j in range(width) if any(r[j] for r in grid)]
    grid = [[r[j] for j in keep] for r in grid]
    header, body = grid[0], grid[1:]
    out = ["| " + " | ".join(h or " " for h in header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(x for x in r) + " |" for r in body]
    return "\n".join(out)


def _table_spans(text: str) -> List[Tuple[int, int]]:
    """(start, end) of every top-level <table>...</table>; an unclosed table ends after its last </tr>."""
    spans, pos = [], 0
    while True:
        m = _TABLE_OPEN.search(text, pos)
        if not m:
            return spans
        start, depth, i = m.start(), 0, m.start()
        end = None
        while True:
            o = _TABLE_OPEN.search(text, i)
            c = _TABLE_CLOSE.search(text, i)
            if c is None:
                break
            if o is not None and o.start() < c.start():
                depth += 1
                i = o.end()
            else:
                depth -= 1
                i = c.end()
                if depth == 0:
                    end = c.end()
                    break
        if end is None:                                            # unclosed: up to the last </tr> of this table
            nxt = _TABLE_OPEN.search(text, m.end())
            limit = nxt.start() if nxt else len(text)
            trs = list(_TR_CLOSE.finditer(text, m.end(), limit))
            end = trs[-1].end() if trs else m.end()
        spans.append((start, end))
        pos = end


_BR = re.compile(r"<br\s*/?>", re.I)


def _fix_br(line: str) -> str:
    """<br> -> line break in normal text, but -> space inside a Markdown table row: a line break there ends the
    row and breaks the whole table (LlamaParse writes "Numer cyklu<br/>1" inside header cells)."""
    if "<" not in line:
        return line
    if line.lstrip().startswith("|"):
        return _BR.sub(" ", line)
    return _BR.sub("\n", line)


def convert_html_tables(text: str) -> str:
    low = text.lower()
    if "<table" not in low and "<tr" not in low and "<hr" not in low and "<br" not in low:
        return text
    out, last = [], 0
    for s, e in _table_spans(text):
        out.append(text[last:s])
        md = table_to_markdown(text[s:e])
        out.append(f"\n\n{md}\n\n" if md else "\n")
        last = e
    out.append(text[last:])
    text = "".join(out)
    text = _ORPHAN_ROWS.sub(lambda m: f"\n\n{table_to_markdown('<table>' + m.group(0) + '</table>')}\n\n", text)
    text = re.sub(r"<hr\s*/?>", "\n---\n", text, flags=re.I)
    text = "\n".join(_fix_br(line) for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text)


# ==========================================
# CLI: fix already parsed documents
# ==========================================
def processed_folders() -> List[Path]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
    from sites import load_or_exit  # type: ignore
    cfg = load_or_exit()
    out = []
    for key, site in cfg.sites.items():
        if site.get("documents"):
            p = Path(cfg.document_dirs(key)["processed"])
            if p.is_dir():
                out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, help="convert a single file")
    ap.add_argument("--print", action="store_true", help="with --file: print the result instead of writing")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--restore", action="store_true", help="copy the originals from <folder>_html_backup/ back")
    a = ap.parse_args()

    if a.restore:
        n = 0
        for folder in processed_folders():
            backup = folder.with_name(folder.name + "_html_backup")
            for b in sorted(backup.glob("*.md")) if backup.is_dir() else []:
                shutil.copy2(b, folder / b.name)
                n += 1
        print(f"Restored {n} files from the backups (backups kept). Now run the conversion again.")
        return

    if a.file:
        text = a.file.read_text(encoding="utf-8")
        new = convert_html_tables(text)
        if a.print or a.dry_run:
            print(new)
        else:
            a.file.write_text(new, encoding="utf-8")
            print(f"rewritten: {a.file}")
        return

    total_files = total_tables = 0
    for folder in processed_folders():
        changed = []
        for p in sorted(folder.glob("*.md")):
            text = p.read_text(encoding="utf-8")
            n = len(_TABLE_OPEN.findall(text))
            if not n:
                continue
            new = convert_html_tables(text)
            if new != text:
                changed.append((p, new, n))
        if not changed:
            continue
        tables = sum(n for _, _, n in changed)
        left = sum(len(_TABLE_OPEN.findall(new)) for _, new, _ in changed)
        print(f"{folder.name:<58}{len(changed):>5} files {tables:>6} tables" + (f"  [!] {left} left" if left else ""))
        total_files += len(changed)
        total_tables += tables
        if a.dry_run:
            continue
        backup = folder.with_name(folder.name + "_html_backup")
        backup.mkdir(exist_ok=True)
        for p, new, _ in changed:
            if not (backup / p.name).exists():
                shutil.copy2(p, backup / p.name)
            p.write_text(new, encoding="utf-8")
    print(f"\n{'Would rewrite' if a.dry_run else 'Rewrote'} {total_files} files, {total_tables} tables."
          + ("" if a.dry_run else " Originals: <folder>_html_backup/ (delete when you are happy with the result)."))


if __name__ == "__main__":
    main()