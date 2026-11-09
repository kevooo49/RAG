"""Loader for config/sites.yaml - shared by the scrapers and backend/scripts.

Usage (from any script, any working directory):

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[N] / "config"))   # N = levels up to repo root
    from sites import load_config
    cfg = load_config()

Environment overrides (handy for CI / another machine, no code or YAML edits needed):
    RAG_SITES_FILE   path to an alternative sites.yaml
    RAG_DATA_ROOT    folder with the knowledge base (overrides paths.data_root)
    RAG_EVAL_DIR     folder with the evaluation sample (overrides paths.eval_dir)
    RAG_INDEX_DIR    folder for the indexer manifest (overrides paths.index_dir)

Relative paths in the YAML are resolved against the repository root, never against the current
working directory, so scripts behave the same wherever they are launched from.
"""
from __future__ import annotations

import os
import re
import sys
from copy import deepcopy
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONFIG_DIR.parent
DEFAULT_SITES_FILE = CONFIG_DIR / "sites.yaml"

COLLECTION_TYPES = {"strona", "dokument", "sylabus"}
SITE_KEYS = {"prefix", "start_url", "source", "pages", "documents", "crawl"}
LAYOUT_KEYS = {"pages", "raw_documents", "processed_documents", "temp_documents"}


class ConfigError(ValueError):
    pass


def _resolve(value: str | os.PathLike, base: Path = REPO_ROOT) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return p if p.is_absolute() else (base / p).resolve()


class SitesConfig:
    def __init__(self, raw: dict, source_file: Path):
        self.source_file = source_file
        self.raw = raw
        paths = raw.get("paths") or {}
        if "data_root" not in paths:
            raise ConfigError("sites.yaml: paths.data_root is required")

        self.data_root = _resolve(os.getenv("RAG_DATA_ROOT") or paths["data_root"])
        self.eval_dir = _resolve(os.getenv("RAG_EVAL_DIR") or paths.get("eval_dir", "data/Ewaluacja_Probka"))
        self.index_dir = _resolve(os.getenv("RAG_INDEX_DIR") or paths.get("index_dir", "data/Baza_Wektorowa_Chroma_ASYM"))
        self.id_prefix = str(paths.get("id_prefix") or self.data_root.name).rstrip("/")

        self.layout = {**{
            "pages": "{prefix}_Strony",
            "raw_documents": "{prefix}_Dokumenty_Surowe",
            "processed_documents": "{prefix}_Dokumenty_Przetworzone",
            "temp_documents": "{prefix}_Dokumenty_Tymczasowe",
        }, **(raw.get("layout") or {})}
        self.crawl_defaults = raw.get("crawl_defaults") or {}
        self.sites: dict[str, dict] = raw.get("sites") or {}
        self.extra_collections: list[dict] = raw.get("extra_collections") or []
        self._validate()

    def _validate(self):
        bad_layout = set(self.layout) - LAYOUT_KEYS
        if bad_layout:
            raise ConfigError(f"sites.yaml: unknown layout keys {sorted(bad_layout)}")
        for name, tmpl in self.layout.items():
            if "{prefix}" not in tmpl:
                raise ConfigError(f"sites.yaml: layout.{name} must contain '{{prefix}}'")

        prefixes = {}
        for key, site in self.sites.items():
            if not isinstance(site, dict):
                raise ConfigError(f"sites.{key}: must be a mapping")
            unknown = set(site) - SITE_KEYS
            if unknown:
                raise ConfigError(f"sites.{key}: unknown keys {sorted(unknown)} (allowed: {sorted(SITE_KEYS)})")
            if not site.get("prefix"):
                raise ConfigError(f"sites.{key}: 'prefix' is required (it names the folders)")
            if re.search(r"[\\/]", site["prefix"]):
                raise ConfigError(f"sites.{key}: prefix must not contain slashes")
            if (site.get("pages") or site.get("documents")) and not site.get("source"):
                raise ConfigError(f"sites.{key}: 'source' is required when pages/documents are indexed")
            if "crawl" in site and not site.get("start_url"):
                raise ConfigError(f"sites.{key}: 'start_url' is required when 'crawl' is present")
            if site.get("start_url") and not re.match(r"https?://", site["start_url"]):
                raise ConfigError(f"sites.{key}: start_url must start with http:// or https://")
            if site["prefix"] in prefixes:
                raise ConfigError(f"sites.{key}: prefix {site['prefix']!r} already used by {prefixes[site['prefix']]!r}")
            prefixes[site["prefix"]] = key

        for i, c in enumerate(self.extra_collections):
            if not isinstance(c, dict) or not {"path", "source", "type"} <= set(c):
                raise ConfigError(f"extra_collections[{i}]: required keys: path, source, type")
            if c["type"] not in COLLECTION_TYPES:
                raise ConfigError(f"extra_collections[{i}]: type must be one of {sorted(COLLECTION_TYPES)}")

    def _site(self, key: str) -> dict:
        try:
            return self.sites[key]
        except KeyError:
            raise ConfigError(f"Unknown site {key!r}. Available: {', '.join(sorted(self.sites))}") from None

    def folder_name(self, key: str, kind: str) -> str:
        return self.layout[kind].format(prefix=self._site(key)["prefix"])

    def folder(self, key: str, kind: str) -> Path:
        return self.data_root / self.folder_name(key, kind)

    def document_dirs(self, key: str) -> dict:
        """Folders for document_parser.py: raw -> (PyMuPDF / LlamaParse) -> processed."""
        site = self._site(key)
        return {
            "raw": self.folder(key, "raw_documents"),
            "processed": self.folder(key, "processed_documents"),
            "temp": self.folder(key, "temp_documents"),
            "source_domain": site.get("start_url", ""),
        }

    def crawl_sites(self) -> list[str]:
        return sorted(k for k, s in self.sites.items() if "crawl" in s)

    def crawl_profile(self, key: str) -> dict:
        """Complete crawler config for one site (defaults + overrides) in the shape scrape_html.crawl() expects."""
        site = self._site(key)
        if "crawl" not in site:
            raise ConfigError(f"Site {key!r} has no 'crawl' section - it cannot be scraped")
        prof = deepcopy(self.crawl_defaults)
        prof.update(deepcopy(site["crawl"]))
        langs = prof.get("allowed_languages")
        prof["allowed_languages"] = tuple(langs) if langs is not None else None
        prof["start_url"] = site["start_url"]
        prof["folder"] = str(self.folder(key, "pages"))
        prof["documents_folder"] = str(self.folder(key, "raw_documents"))
        return prof

    def collections(self) -> list[dict]:
        """Collections for build_database.py.

        path     - absolute folder to read
        id_path  - stable prefix for chunk IDs / manifest keys (independent of where the data lives on disk)
        """
        out = []
        for key, site in self.sites.items():
            if site.get("pages"):
                name = self.folder_name(key, "pages")
                out.append({"path": str(self.data_root / name), "id_path": f"{self.id_prefix}/{name}",
                            "source": site["source"], "type": "strona"})
            docs = site.get("documents")
            if docs:
                name = self.folder_name(key, "processed_documents")
                entry = {"path": str(self.data_root / name), "id_path": f"{self.id_prefix}/{name}",
                         "source": site["source"], "type": "dokument"}
                opts = docs if isinstance(docs, dict) else {}
                entry["legal_headers"] = bool(opts.get("legal_headers", True))
                if "bold_as_headers" in opts:
                    entry["bold_as_headers"] = bool(opts["bold_as_headers"])
                out.append(entry)
        for c in self.extra_collections:
            rel = str(c["path"]).strip("/\\")
            entry = {"path": str(self.data_root / rel), "id_path": f"{self.id_prefix}/{rel}",
                     "source": c["source"], "type": c["type"]}
            for opt in ("legal_headers", "bold_as_headers"):
                if opt in c:
                    entry[opt] = bool(c[opt])
            out.append(entry)
        return out


def load_config(path: str | os.PathLike | None = None) -> SitesConfig:
    file = _resolve(path or os.getenv("RAG_SITES_FILE") or DEFAULT_SITES_FILE)
    if not file.is_file():
        raise ConfigError(f"Config file not found: {file}")
    try:
        raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"{file}: invalid YAML - {e}") from e
    return SitesConfig(raw, file)


def load_or_exit(path=None) -> SitesConfig:
    """For CLI scripts: print a readable message instead of a traceback."""
    try:
        return load_config(path)
    except ConfigError as e:
        sys.exit(f"Configuration error: {e}")


if __name__ == "__main__":
    cfg = load_or_exit(sys.argv[1] if len(sys.argv) > 1 else None)
    cols = cfg.collections()
    print(f"OK: {cfg.source_file}")
    print(f"  data_root : {cfg.data_root}  ({'exists' if cfg.data_root.is_dir() else 'MISSING'})")
    print(f"  eval_dir  : {cfg.eval_dir}")
    print(f"  index_dir : {cfg.index_dir}")
    print(f"  sites     : {len(cfg.sites)} ({len(cfg.crawl_sites())} crawlable)")
    print(f"  indexed   : {len(cols)} collections "
          f"({sum(c['type'] == 'strona' for c in cols)} pages, {sum(c['type'] == 'dokument' for c in cols)} documents, "
          f"{sum(c['type'] == 'sylabus' for c in cols)} syllabi)")
    missing = [Path(c["path"]).name for c in cols if not Path(c["path"]).is_dir()]
    if missing:
        print(f"  not on disk yet ({len(missing)}): {', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}")
