"""Tests for config/sites.py + config/sites.yaml (no network, no API keys)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from sites import ConfigError, SitesConfig, load_config  # noqa: E402


def make(**over):
    raw = {"paths": {"data_root": "data/kb", "id_prefix": "legacy/kb"},
           "crawl_defaults": {"min_chars": 300, "allowed_languages": ["pl"]},
           "sites": {"x": {"prefix": "X", "start_url": "https://x.example/", "source": "X Source",
                           "pages": True, "documents": {"legal_headers": True},
                           "crawl": {"excluded_paths": ["/news"], "min_chars": 50}}},
           "extra_collections": [{"path": "Syl/2025", "source": "Syllabi", "type": "sylabus"}]}
    raw.update(over)
    return SitesConfig(raw, Path("memory.yaml"))


def test_real_config_is_valid_and_ids_are_unique():
    cfg = load_config()
    cols = cfg.collections()
    assert cols
    assert len({c["id_path"] for c in cols}) == len(cols)
    assert len({c["path"] for c in cols}) == len(cols)
    assert all(Path(c["path"]).is_absolute() for c in cols)


def test_id_path_is_independent_of_data_root(monkeypatch, tmp_path):
    a = make().collections()
    monkeypatch.setenv("RAG_DATA_ROOT", str(tmp_path / "elsewhere"))
    b = make().collections()
    assert [c["id_path"] for c in a] == [c["id_path"] for c in b]      # chunk IDs must not move with the data
    assert a[0]["path"] != b[0]["path"]


def test_crawl_profile_merges_defaults_and_overrides():
    p = make().crawl_profile("x")
    assert p["min_chars"] == 50 and p["excluded_paths"] == ["/news"]
    assert p["allowed_languages"] == ("pl",)
    assert p["folder"].endswith("X_Strony") and p["documents_folder"].endswith("X_Dokumenty_Surowe")


def test_collections_shape():
    cols = make().collections()
    assert [c["type"] for c in cols] == ["strona", "dokument", "sylabus"]
    assert cols[1]["legal_headers"] is True and cols[1]["id_path"] == "legacy/kb/X_Dokumenty_Przetworzone"
    assert cols[2]["id_path"] == "legacy/kb/Syl/2025"


@pytest.mark.parametrize("site, msg", [
    ({"start_url": "https://x/"}, "prefix"),
    ({"prefix": "A/B", "source": "s"}, "slashes"),
    ({"prefix": "A", "pages": True}, "source"),
    ({"prefix": "A", "crawl": {}}, "start_url"),
    ({"prefix": "A", "start_url": "ftp://x"}, "http"),
    ({"prefix": "A", "typo_key": 1}, "unknown keys"),
])
def test_invalid_site_is_rejected(site, msg):
    with pytest.raises(ConfigError, match=msg):
        make(sites={"bad": site})


def test_duplicate_prefix_rejected():
    s = {"prefix": "A", "source": "s"}
    with pytest.raises(ConfigError, match="already used"):
        make(sites={"one": s, "two": dict(s)})


def test_unknown_site_and_site_without_crawl():
    cfg = make(sites={"m": {"prefix": "M", "source": "s", "pages": True}})
    with pytest.raises(ConfigError, match="Unknown site"):
        cfg.crawl_profile("nope")
    with pytest.raises(ConfigError, match="no 'crawl'"):
        cfg.crawl_profile("m")
