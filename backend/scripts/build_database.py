#!/usr/bin/env python3
"""
Budowa bazy wektorowej RAG z plików Markdown (AGH) – wersja przyrostowa.

    python build_database.py                      # indeksuj (tylko nowe/zmienione pliki)
    python build_database.py --dry-run            # statystyki + szacunek kosztu, BEZ API
    python build_database.py --podglad plik.md    # pokaż, jak plik zostanie pocięty (BEZ API)
    python build_database.py --szukaj "jak przedłużyć książkę?" -k 5 [--zrodlo "Biblioteka Główna AGH"]
    python build_database.py --przebuduj          # wyczyść bazę i zbuduj od zera

Co jest inaczej niż w prostym RecursiveCharacterTextSplitter:
  * dzielimy po STRUKTURZE markdowna (nagłówki -> bloki: akapit / lista / tabela / kod),
    nie po znakach; zdania dzielimy z uwzględnieniem polskich skrótów (ul., godz., r., ust.),
  * TABELE są atomowe; zbyt duże dzielimy po wierszach i powtarzamy w każdej części wiersz nagłówkowy,
  * każdy chunk dostaje nagłówek kontekstowy: [Źródło] Tytuł › Sekcja › Podsekcja
    (bez tego "Termin: 5 października" nie wiadomo czego dotyczy),
  * do embeddingu idzie tekst bez adresów URL, a w bazie zostaje pełny (z linkami do cytowania),
  * indeksowanie jest przyrostowe i idempotentne: zmiana jednego pliku = ponowny embedding tylko
    zmienionych chunków; uruchomienie dwa razy nie duplikuje danych,
  * klucz API z env (OPENAI_API_KEY), retry, batching po tokenach, kontrola modelu w kolekcji.
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
from dotenv import load_dotenv
import yaml

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# ==========================================
# USTAWIENIA
# ==========================================
KATALOG_BAZY = "../../data/Baza_Wektorowa_Chroma"
NAZWA_KOLEKCJI = "agh_wiedza"
MANIFEST = os.path.join(KATALOG_BAZY, "_manifest_indeksu.json")

MODEL = "text-embedding-3-small"   # dla polskiego warto porównać z "text-embedding-3-large" (patrz --szukaj)
WYMIARY = None                     # np. 1024 – mniejsze wektory (model 3-* to wspiera); None = domyślne
CENA_ZA_1M_TOKENOW_USD = 0.02      # tylko do szacunku w --dry-run; sprawdź aktualny cennik OpenAI

# Rozmiar chunków (w tokenach, z nagłówkiem kontekstowym)
MAX_TOKENOW = 450
MIN_TOKENOW = 120                  # krótszy chunk dokleja kolejną sekcję zamiast kończyć się sam
ZAPAS_TOKENOW = 25                 # na śródtekstowe nagłówki przy łączeniu sekcji
NAKLADKA_ZDAN = 1                  # nakładka TYLKO gdy dzielimy długi akapit (w prozie ma sens)
MAX_TOKENOW_WSTEPU = 80            # zdanie-wstęp ("Terminy składania wniosków:") sklejane z tabelą/listą
MIN_ZNAKOW_DOKUMENTU = 80          # krótsze pliki pomijamy

MAX_BATCH_ELEMENTOW = 256
MAX_BATCH_TOKENOW = 200_000        # limit API to 300k tokenów na żądanie

# Kolekcje: "zrodlo" trafia do metadanych (filtrowanie) i do nagłówka kontekstowego chunków.
# "naglowki_prawne": traktuj "§ 5", "Art. 12", "Rozdział II" jako nagłówki (regulaminy, uchwały, zarządzenia).
# "pogrubienia_jako_naglowki" (domyślnie True): samotna pogrubiona linia (**1. Kto może...?**) = nagłówek.
KOLEKCJE = [
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Rekrutacja_Strony", "zrodlo": "Rekrutacja AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Rekrutacja_Dokumenty_Przetworzone", "zrodlo": "Rekrutacja AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Biblioteka_Strony", "zrodlo": "Biblioteka Główna AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Biblioteka_Dokumenty_Przetworzone", "zrodlo": "Biblioteka Główna AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/CSS_Strony", "zrodlo": "Centrum Spraw Studenckich AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/CSS_Dokumenty_Przetworzone", "zrodlo": "Centrum Spraw Studenckich AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Miasteczko_Strony", "zrodlo": "Miasteczko Studenckie AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Miasteczko_Dokumenty_Przetworzone", "zrodlo": "Miasteczko Studenckie AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Centrum_Karier_Strony", "zrodlo": "Centrum Karier AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Centrum_Karier_Dokumenty_Przetworzone", "zrodlo": "Centrum Karier AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Centrum_Spraw_Miedzynarodowych_Strony", "zrodlo": "Centrum Spraw Międzynarodowych AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Centrum_Spraw_Miedzynarodowych_Dokumenty_Przetworzone", "zrodlo": "Centrum Spraw Międzynarodowych AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/SWFiS_Strony", "zrodlo": "SWFiS AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/SWFiS_Dokumenty_Przetworzone", "zrodlo": "SWFiS AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Sylabusy_Markdown", "zrodlo": "Sylabusy AGH", "typ": "sylabus"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Cyfronet_Strony", "zrodlo": "Cyfronet AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Cyfronet_Dokumenty_Przetworzone", "zrodlo": "Cyfronet AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/SJO_Strony", "zrodlo": "SJO AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/SJO_Dokumenty_Przetworzone", "zrodlo": "SJO AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/KA_Dokumenty_Przetworzone", "zrodlo": "KA AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/KA_Strony", "zrodlo": "KA AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/CKiM_Strony", "zrodlo": "CKiM AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/CKiM_Dokumenty_Przetworzone", "zrodlo": "CKiM AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/CRI_Strony", "zrodlo": "CRI AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/CRI_Dokumenty_Przetworzone", "zrodlo": "CRI AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/SD_Strony", "zrodlo": "SD AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/SD_Dokumenty_Przetworzone", "zrodlo": "SD AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Podyplomowe_Strony", "zrodlo": "Podyplomowe AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Podyplomowe_Dokumenty_Przetworzone", "zrodlo": "Podyplomowe AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Fundacja_Strony", "zrodlo": "Fundacja AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Fundacja_Dokumenty_Przetworzone", "zrodlo": "Fundacja AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Samorzad_Strony", "zrodlo": "Samorzad AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Samorzad_Dokumenty_Przetworzone", "zrodlo": "Samorzad AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/EAIIB_Strony", "zrodlo": "EAIIB AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/RPS_Strony", "zrodlo": "RPS AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/RPS_Dokumenty_Przetworzone", "zrodlo": "RPS AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/WH_Strony", "zrodlo": "WH AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/WH_Dokumenty_Przetworzone", "zrodlo": "WH AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Odlewnictwo_Strony", "zrodlo": "Odlewnictwo AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Odlewnictwo_Dokumenty_Przetworzone", "zrodlo": "Odlewnictwo AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/WNIG_Strony", "zrodlo": "WNIG AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/WNIG_Dokumenty_Przetworzone", "zrodlo": "WNIG AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Zarz_Strony", "zrodlo": "Zarz AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Zarz_Dokumenty_Przetworzone", "zrodlo": "Zarz AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/WIMIC_Strony", "zrodlo": "WIMIC AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/WIMIC_Dokumenty_Przetworzone", "zrodlo": "WIMIC AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/WILGZ_Strony", "zrodlo": "WILGZ AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/WILGZ_Dokumenty_Przetworzone", "zrodlo": "WILGZ AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Metal_Strony", "zrodlo": "Metal AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Metal_Dokumenty_Przetworzone", "zrodlo": "Metal AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/WGGIOS_Strony", "zrodlo": "WGGIOS AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/WGGIOS_Dokumenty_Przetworzone", "zrodlo": "WGGIOS AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/IET_Strony", "zrodlo": "IET AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/IET_Dokumenty_Przetworzone", "zrodlo": "IET AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/WGGIIS_Strony", "zrodlo": "WGGIIS AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/WGGIIS_Dokumenty_Przetworzone", "zrodlo": "WGGIIS AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/WGGIIS_Studiuj_Strony", "zrodlo": "WGGIIS_Studiuj AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/WGGIIS_Studiuj_Dokumenty_Przetworzone", "zrodlo": "WGGIIS_Studiuj AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Informatyka_Strony", "zrodlo": "Informatyka AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Informatyka_Dokumenty_Przetworzone", "zrodlo": "Informatyka AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Spacetech_Strony", "zrodlo": "Spacetech AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Spacetech_Dokumenty_Przetworzone", "zrodlo": "Spacetech AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/WMS_Strony", "zrodlo": "WMS AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/WMS_Dokumenty_Przetworzone", "zrodlo": "WMS AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/FIS_Strony", "zrodlo": "FIS AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/FIS_Dokumenty_Przetworzone", "zrodlo": "FIS AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/WEIP_Strony", "zrodlo": "WEIP AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/WEIP_Dokumenty_Przetworzone", "zrodlo": "WEIP AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/WMN_Strony", "zrodlo": "WMN AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/WMN_Dokumenty_Przetworzone", "zrodlo": "WMN AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/Historia_Strony", "zrodlo": "Historia AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/IMIR_Strony", "zrodlo": "IMIR AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/IMIR_Dokumenty_Przetworzone", "zrodlo": "IMIR AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/AZS_Strony", "zrodlo": "AZS AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/AZS_Dokumenty_Przetworzone", "zrodlo": "AZS AGH", "typ": "dokument", "naglowki_prawne": True},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/COK_Strony", "zrodlo": "COK AGH", "typ": "strona"},
    {"sciezka": "../../data/Baza_Wiedzy_AGH/COK_Dokumenty_Przetworzone", "zrodlo": "COK AGH", "typ": "dokument", "naglowki_prawne": True}
]

# ==========================================
# LICZENIE TOKENÓW
# ==========================================
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")   # tokenizer modeli text-embedding-3-*
    TOKENIZER = "tiktoken"

    def tokeny(tekst):
        return len(_ENC.encode(tekst, disallowed_special=()))
except ImportError:
    TOKENIZER = "przyblizenie"

    def tokeny(tekst):
        return -(-len(tekst) * 10 // 28)          # ~2,8 znaku/token dla polskiego (zaniżone = bezpieczne)


# ==========================================
# WCZYTYWANIE PLIKU + FRONTMATTER
# ==========================================
_FRONTMATTER = re.compile(r"\A\ufeff?---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.S)


def parsuj_frontmatter(zawartosc):
    """Frontmatter tylko z POCZĄTKU pliku i z '---' na osobnej linii (split('---') łamie się
    na tytułach typu 'A --- B' i na poziomych liniach w treści)."""
    m = _FRONTMATTER.match(zawartosc)
    if not m:
        return {}, zawartosc.strip()
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as e:
        print(f"   ⚠️ Niepoprawny YAML we frontmatterze ({str(e).splitlines()[0]}) – ignoruję metadane")
        meta = {}
    return (meta if isinstance(meta, dict) else {}), zawartosc[m.end():].strip()


def czysc_meta(meta):
    """Chroma przyjmuje tylko str/int/float/bool: listy -> tekst, daty -> ISO, None/puste -> pomijamy."""
    wynik = {}
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
        wynik[str(k)] = v
    return wynik


def normalizuj(t):
    t = re.sub(r"[*_`#]|\[([^\]]*)\]\([^)]*\)", lambda m: m.group(1) or "", t)
    return re.sub(r"\W+", " ", t.lower()).strip()


def bez_url(tekst):
    """Tekst do EMBEDDINGU: bez adresów linków i znaczników pogrubienia (szum dla wektora)."""
    tekst = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", tekst)
    tekst = re.sub(r"<?https?://\S+>?", "", tekst)
    return re.sub(r"[ \t]{2,}", " ", tekst.replace("**", " ").replace("__", " "))


# ==========================================
# PARSER MARKDOWNA -> BLOKI
# ==========================================
@dataclass
class Blok:
    typ: str            # akapit | lista | tabela | kod
    tekst: str
    sciezka: tuple      # ścieżka nagłówków, np. ("Stypendia", "Terminy")
    wstep: str = ""     # akapit kończący się ':' sklejony z następną tabelą/listą


RE_NAGLOWEK = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
RE_PRAWNY = re.compile(r"^(Rozdział\s+[IVXLCDM\d]+[a-z]?|§\s*\d+[a-z]?|Art\.\s*\d+[a-z]?)(?=[\s.]|$)", re.I)
RE_LISTA = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+")
RE_SEPARATOR_TABELI = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
RE_OGRODZENIE = re.compile(r"^\s*(```|~~~)")
RE_POZIOMA_LINIA = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
RE_POGRUBIONA_LINIA = re.compile(r"^(?:\*\*|__)(?P<t>[^*_].*?)(?:\*\*|__)$")


def pogrubienie_jako_naglowek(linia):
    """CMS-y często robią "nagłówki" pogrubieniem: **1. Kto może ubiegać się o stypendium?**
    Zwraca tekst nagłówka albo None. Odrzuca krótkie okrzyki typu **Uwaga!**, **Ważne!**."""
    m = RE_POGRUBIONA_LINIA.match(linia.strip())
    if not m or "**" in m.group("t") or "__" in m.group("t"):
        return None
    t = m.group("t").strip()
    if t.endswith("!") or len(t) > 200 or len(t) < 4:
        return None
    if len(t.split()) < 3 and not re.match(r"\d+[.)]\s", t):
        return None
    return t.rstrip(":").strip()


def czy_poczatek_struktury(linia, prawne):
    return bool(RE_NAGLOWEK.match(linia) or RE_OGRODZENIE.match(linia) or linia.lstrip().startswith("|")
                or RE_LISTA.match(linia) or (prawne and RE_PRAWNY.match(linia.strip())))


def parsuj_bloki(body, tytul, prawne=False, pogrubienia=True):
    linie = body.split("\n")
    n = len(linie)
    stos = []           # [(poziom, tekst, rodzaj)]  rodzaj: "" | "bold" (pseudo-nagłówek z pogrubienia)
    bloki = []
    tyt_norm = normalizuj(tytul)

    def sciezka():
        return tuple(t for _, t, _ in stos if normalizuj(t) != tyt_norm)

    def wejdz(poziom, tekst, rodzaj=""):
        while stos and stos[-1][0] >= poziom:
            stos.pop()
        stos.append((poziom, tekst, rodzaj))

    def wejdz_pogrubiony(tekst):
        """Pogrubiona linia = podsekcja bieżącego PRAWDZIWEGO nagłówka; zastępuje tylko poprzednią pogrubioną."""
        while stos and stos[-1][2] == "bold":
            stos.pop()
        wejdz(stos[-1][0] + 1 if stos else 2, tekst, "bold")

    i = 0
    while i < n:
        linia = linie[i]
        if not linia.strip() or RE_POZIOMA_LINIA.match(linia):   # puste linie i poziome kreski (---) pomijamy
            i += 1
            continue

        m = RE_OGRODZENIE.match(linia)
        if m:                                           # blok kodu
            znacznik, j = m.group(1), i + 1
            while j < n and not linie[j].lstrip().startswith(znacznik):
                j += 1
            bloki.append(Blok("kod", "\n".join(linie[i:j + 1]), sciezka()))
            i = j + 1
            continue

        m = RE_NAGLOWEK.match(linia)
        if m:
            tekst = re.sub(r"\*\*|__|`", "", re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", m.group(2))).strip()
            wejdz(len(m.group(1)), tekst)
            i += 1
            continue

        if linia.lstrip().startswith("|"):              # tabela
            j = i
            while j < n and linie[j].lstrip().startswith("|"):
                j += 1
            bloki.append(Blok("tabela", "\n".join(l.rstrip() for l in linie[i:j]), sciezka()))
            i = j
            continue

        if RE_LISTA.match(linia):                       # lista (z zagnieżdżeniami i ciągłością)
            j = i + 1
            while j < n:
                if linie[j].strip():
                    if RE_LISTA.match(linie[j]) or linie[j].startswith((" ", "\t")):
                        j += 1
                        continue
                    break
                k = j + 1                               # pojedyncza pusta linia wewnątrz listy
                while k < n and not linie[k].strip():
                    k += 1
                if k < n and (RE_LISTA.match(linie[k]) or linie[k].startswith((" ", "\t"))):
                    j = k
                    continue
                break
            bloki.append(Blok("lista", "\n".join(l.rstrip() for l in linie[i:j] if l.strip()), sciezka()))
            i = j
            continue

        # akapit (z opcjonalnym nagłówkiem prawnym: "§ 5.", "Art. 3", "Rozdział II")
        if prawne:
            mp = RE_PRAWNY.match(linia.strip())
            if mp:
                etykieta = re.sub(r"\s+", " ", mp.group(1)).strip()
                if etykieta.lower().startswith("rozdz"):        # "Rozdział II Opłaty" – tytuł rozdziału jest częścią etykiety
                    pelna = re.sub(r"\s+", " ", linia).strip()
                    wejdz(2, pelna if len(pelna) <= 100 else etykieta)
                else:
                    wejdz(3, etykieta)
        j = i + 1
        while j < n and linie[j].strip() and not czy_poczatek_struktury(linie[j], prawne):
            j += 1
        if pogrubienia and j == i + 1:
            naglowek = pogrubienie_jako_naglowek(linia)
            if naglowek:
                wejdz_pogrubiony(naglowek)
                i = j
                continue
        bloki.append(Blok("akapit", "\n".join(l.strip() for l in linie[i:j]), sciezka()))
        i = j

    # Wstęp doklejany do następnego bloku (i powtarzany, gdy blok trzeba podzielić):
    #  a) zdanie kończące się ':'  ->  tabela / lista / kod   ("Terminy składania wniosków:")
    #  b) krótka etykieta z '!' lub ':' -> dowolny blok          ("**Ważne!**", "Uwaga:")
    wynik, idx = [], 0
    while idx < len(bloki):
        b = bloki[idx]
        if b.typ == "akapit" and idx + 1 < len(bloki) and bloki[idx + 1].sciezka == b.sciezka:
            goly = re.sub(r"[*_\s]+$", "", re.sub(r"^[*_\s]+", "", b.tekst))
            kolejny = bloki[idx + 1]
            a = goly.endswith(":") and kolejny.typ in ("tabela", "lista", "kod")
            bb = goly.endswith(("!", ":")) and len(goly.split()) <= 6 and "\n" not in goly
            if (a or bb) and tokeny(b.tekst) <= MAX_TOKENOW_WSTEPU and not kolejny.wstep:
                kolejny.wstep = b.tekst
                idx += 1
                continue
        wynik.append(b)
        idx += 1
    return wynik


# ==========================================
# DZIELENIE ZA DUŻYCH BLOKÓW
# ==========================================
SKROTY = {
    "np", "tj", "tzn", "tzw", "itd", "itp", "ul", "al", "pl", "os", "godz", "ok", "nr", "poz", "ust", "art",
    "pkt", "lit", "zał", "dz", "prof", "dr", "mgr", "inż", "hab", "ds", "wg", "zob", "por", "im", "bud", "pok",
    "tel", "str", "rys", "tab", "ww", "r", "w", "wyd", "red", "tłum", "ewent", "min", "maks", "max", "śr",
    "ur", "zm", "ks", "płk", "gen", "kpt", "mjr", "sp", "zs", "ogł", "woj", "pow", "gm", "m.in", "j.w",
}
RE_KONIEC_ZDANIA = re.compile(r"(?<=[.!?…])\s+(?=[\"„(\[]?[A-ZĄĆĘŁŃÓŚŹŻ0-9§])")


def podziel_na_zdania(tekst):
    """Dzieli na zdania, NIE tnąc po skrótach (ul., godz., 2026 r., ust. 3) i numeracji ('§ 5. 1. Tekst')."""
    wynik, start = [], 0
    for m in RE_KONIEC_ZDANIA.finditer(tekst):
        przed = tekst[start:m.start()]
        slowa = przed.split()
        if slowa and slowa[-1].endswith("."):
            s = slowa[-1].rstrip(".").lower().strip("„\"([")
            rok_r = s == "r" and len(slowa) >= 2 and slowa[-2].isdigit() and len(slowa[-2]) == 4   # "2026 r. Wniosek"
            if not rok_r and (s in SKROTY or len(s) == 1 or s.isdigit()):
                continue
        wynik.append(przed.strip())
        start = m.end()
    reszta = tekst[start:].strip()
    if reszta:
        wynik.append(reszta)
    return wynik


def twardy_podzial(tekst, budzet):
    """Ostateczność (np. jedno zdanie dłuższe niż chunk): dzielimy po słowach."""
    czesci, biez, tok = [], [], 0
    for slowo in tekst.split():
        t = tokeny(slowo) + 1
        if biez and tok + t > budzet:
            czesci.append(" ".join(biez))
            biez, tok = [], 0
        biez.append(slowo)
        tok += t
    if biez:
        czesci.append(" ".join(biez))
    return czesci


def dziel_akapit(tekst, budzet):
    zdania = []
    for linia in tekst.split("\n"):
        zdania += podziel_na_zdania(linia)
    zdania = [z for s in zdania for z in ([s] if tokeny(s) <= budzet else twardy_podzial(s, budzet))]
    czesci, biez, tok = [], [], 0
    for z in zdania:
        t = tokeny(z) + 1
        if biez and tok + t > budzet:
            czesci.append(" ".join(biez))
            ogon = biez[-NAKLADKA_ZDAN:] if NAKLADKA_ZDAN else []
            ogon_tok = sum(tokeny(x) + 1 for x in ogon)
            biez, tok = (ogon, ogon_tok) if ogon_tok <= budzet * 0.25 else ([], 0)
        biez.append(z)
        tok += t
    if biez:
        czesci.append(" ".join(biez))
    return czesci


def dziel_liste(tekst, budzet):
    linie = tekst.split("\n")
    wciecia = [len(RE_LISTA.match(l).group(1)) for l in linie if RE_LISTA.match(l)]
    minimalne = min(wciecia) if wciecia else 0
    elementy = []
    for l in linie:
        m = RE_LISTA.match(l)
        if m and len(m.group(1)) == minimalne or not elementy:
            elementy.append([l])
        else:
            elementy[-1].append(l)
    elementy = ["\n".join(e) for e in elementy]
    czesci, biez, tok = [], [], 0
    for e in elementy:
        t = tokeny(e) + 1
        if t > budzet:                                  # gigantyczny punkt listy -> jak akapit
            if biez:
                czesci.append("\n".join(biez))
                biez, tok = [], 0
            czesci += dziel_akapit(e, budzet)
            continue
        if biez and tok + t > budzet:
            czesci.append("\n".join(biez))
            biez, tok = [], 0
        biez.append(e)
        tok += t
    if biez:
        czesci.append("\n".join(biez))
    return czesci


def dziel_tabele(tekst, budzet):
    """Po wierszach, z POWTÓRZONYM nagłówkiem w każdej części (wiersz bez nagłówka jest bez sensu)."""
    linie = tekst.split("\n")
    if len(linie) >= 2 and RE_SEPARATOR_TABELI.match(linie[1]):
        naglowek, wiersze = linie[:2], linie[2:]
    else:
        naglowek, wiersze = [], linie
    nag_tok = tokeny("\n".join(naglowek)) + 1
    czesci, biez, tok = [], [], nag_tok

    def zamknij():
        nonlocal biez, tok
        if biez:
            czesci.append("\n".join(naglowek + biez))
        biez, tok = [], nag_tok

    for w in wiersze:
        t = tokeny(w) + 1
        if nag_tok + t > budzet:                        # pojedynczy wiersz większy niż chunk
            zamknij()
            czesci += ["\n".join(naglowek + [k]) for k in twardy_podzial(w, max(budzet - nag_tok, 20))]
            continue
        if biez and tok + t > budzet:
            zamknij()
        biez.append(w)
        tok += t
    zamknij()
    return czesci


def dziel_kod(tekst, budzet):
    czesci, biez, tok = [], [], 0
    for l in tekst.split("\n"):
        t = tokeny(l) + 1
        if biez and tok + t > budzet:
            czesci.append("\n".join(biez))
            biez, tok = [], 0
        biez.append(l)
        tok += t
    if biez:
        czesci.append("\n".join(biez))
    return czesci


def rozbij_blok(b, budzet):
    """Zwraca listę tekstów (każdy mieści się w budżecie); wstęp jest powtarzany w każdej części."""
    bud = max(budzet - (tokeny(b.wstep) + 1 if b.wstep else 0), 40)
    if tokeny(b.tekst) <= bud:
        czesci = [b.tekst]
    else:
        czesci = {"tabela": dziel_tabele, "lista": dziel_liste, "kod": dziel_kod}.get(b.typ, dziel_akapit)(b.tekst, bud)
    return [f"{b.wstep}\n\n{c}" if b.wstep else c for c in czesci]


# ==========================================
# SKŁADANIE CHUNKÓW
# ==========================================
@dataclass
class Chunk:
    tekst: str            # to, co trafia do bazy i do LLM (z nagłówkiem kontekstowym i linkami)
    tekst_emb: str        # to, co jest zamieniane na wektor (bez URL-i)
    sekcja: str
    tok: int
    ma_tabele: bool


def kontekst(zrodlo, tytul, sciezka):
    return f"[{zrodlo}] {tytul}" + (" › " + " › ".join(sciezka) if sciezka else "")


def wspolny_prefiks(sciezki):
    pref = list(sciezki[0])
    for s in sciezki[1:]:
        k = 0
        while k < min(len(pref), len(s)) and pref[k] == s[k]:
            k += 1
        pref = pref[:k]
    return tuple(pref)


def renderuj(czesci):
    """Zwraca (prefiks_sciezki, treść). Gdy chunk obejmuje kilka sekcji, ich nagłówki wracają śródtekstowo."""
    prefiks = wspolny_prefiks([c[0] for c in czesci])
    linie, poprz = [], prefiks
    for sciezka, tekst in czesci:
        zmiana = False
        for d in range(len(prefiks), len(sciezka)):
            if zmiana or d >= len(poprz) or poprz[d] != sciezka[d]:
                linie.append("#" * min(6, d + 2) + " " + sciezka[d])
                zmiana = True
        poprz = sciezka
        linie.append(tekst)
    return prefiks, "\n\n".join(linie)


def zloz_chunki(bloki, tytul, zrodlo):
    grupy, biezacy = [], None

    for b in bloki:
        narzut_b = tokeny(kontekst(zrodlo, tytul, b.sciezka)) + 2
        for czesc in rozbij_blok(b, MAX_TOKENOW - narzut_b - ZAPAS_TOKENOW):
            t = tokeny(czesc) + 1
            if biezacy:
                nowa_sekcja = biezacy["czesci"][-1][0] != b.sciezka
                narzut = max(biezacy["narzut"], narzut_b)
                przekroczy = biezacy["tok"] + t + narzut + ZAPAS_TOKENOW > MAX_TOKENOW
                if przekroczy or (nowa_sekcja and biezacy["tok"] >= MIN_TOKENOW):
                    grupy.append(biezacy)
                    biezacy = None
            if biezacy is None:
                biezacy = {"czesci": [], "tok": 0, "narzut": 0, "tabela": False}
            biezacy["czesci"].append((b.sciezka, czesc))
            biezacy["tok"] += t
            biezacy["narzut"] = max(biezacy["narzut"], narzut_b)
            biezacy["tabela"] = biezacy["tabela"] or b.typ == "tabela"
    if biezacy:
        grupy.append(biezacy)

    # malutki ogon na końcu dokumentu dokleja się do poprzedniego chunku, jeśli się zmieści
    if len(grupy) >= 2 and grupy[-1]["tok"] < MIN_TOKENOW // 2:
        a, z = grupy[-2], grupy[-1]
        if a["tok"] + z["tok"] + max(a["narzut"], z["narzut"]) + ZAPAS_TOKENOW <= MAX_TOKENOW:
            a["czesci"] += z["czesci"]
            a["tok"] += z["tok"]
            a["narzut"] = max(a["narzut"], z["narzut"])
            a["tabela"] = a["tabela"] or z["tabela"]
            grupy.pop()

    chunki = []
    for g in grupy:
        prefiks, tresc = renderuj(g["czesci"])
        kontekst_txt = kontekst(zrodlo, tytul, prefiks)
        tekst = f"{kontekst_txt}\n\n{tresc}"
        chunki.append(Chunk(tekst=tekst, tekst_emb=f"{kontekst_txt}\n\n{bez_url(tresc)}",
                            sekcja=" › ".join(prefiks), tok=tokeny(tekst), ma_tabele=g["tabela"]))
    return chunki


# ==========================================
# DOKUMENT -> CHUNKI Z METADANYMI
# ==========================================
def przygotuj_dokument(sciezka_pliku, cfg):
    """Zwraca (meta, body, tytul) albo None, gdy plik jest pusty/za krótki."""
    zawartosc = Path(sciezka_pliku).read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    meta, body = parsuj_frontmatter(zawartosc)
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    body = re.sub(r"<br\s*/?>", " ", body, flags=re.I)
    if len(body) < MIN_ZNAKOW_DOKUMENTU:
        return None
    tytul = str(meta.get("tytul") or "").strip()
    if not tytul:
        m = re.search(r"^#\s+(.+)$", body, re.M)
        tytul = m.group(1).strip() if m else Path(sciezka_pliku).stem
    return meta, body, tytul


def sygnatura_pliku(body, tytul, cfg):
    """Zmienia się, gdy zmieni się treść LUB sposób cięcia (parametry/tokenizer/model) – wtedy plik jest przetwarzany ponownie.
    Daty pobrania z frontmattera celowo NIE wchodzą – ponowny crawl bez zmian treści nie kosztuje ani grosza."""
    param = [MAX_TOKENOW, MIN_TOKENOW, ZAPAS_TOKENOW, NAKLADKA_ZDAN, MAX_TOKENOW_WSTEPU, TOKENIZER, MODEL, WYMIARY,
             cfg["zrodlo"], cfg["typ"], bool(cfg.get("naglowki_prawne")),
             cfg.get("pogrubienia_jako_naglowki", True), 4]
    return hashlib.sha1(f"{body}\x00{tytul}\x00{param}".encode("utf-8")).hexdigest()


def chunkuj_plik(sciezka_pliku, cfg, dok=None):
    dok = dok or przygotuj_dokument(sciezka_pliku, cfg)
    if dok is None:
        return []
    meta, body, tytul = dok
    bloki = parsuj_bloki(body, tytul, prawne=bool(cfg.get("naglowki_prawne")),
                         pogrubienia=cfg.get("pogrubienia_jako_naglowki", True))
    chunki = zloz_chunki(bloki, tytul, cfg["zrodlo"])

    rel = Path(sciezka_pliku).as_posix()
    dodatkowe = czysc_meta(meta)
    wynik, widziane = [], {}
    for nr, c in enumerate(chunki):
        k = c.tekst_emb
        widziane[k] = widziane.get(k, 0) + 1            # identyczny chunk w tym samym pliku -> inny id
        cid = hashlib.sha1(f"{rel}|{widziane[k]}|{k}|{MODEL}|{WYMIARY}".encode("utf-8")).hexdigest()[:32]
        md = {**dodatkowe,
              "zrodlo": cfg["zrodlo"], "kolekcja": Path(cfg["sciezka"]).name, "typ": cfg["typ"],
              "tytul": tytul, "sekcja": c.sekcja, "plik": rel,
              "chunk_nr": nr, "chunk_z": len(chunki), "tokeny": c.tok, "tabela": c.ma_tabele}
        wynik.append((cid, c, md))
    return wynik


# ==========================================
# POBIERANIE LISTY PLIKÓW, MANIFEST
# ==========================================
def zbierz_pliki():
    pliki = []
    for cfg in KOLEKCJE:
        kat = Path(cfg["sciezka"])
        if not kat.exists():
            print(f"⚠️ Brak katalogu: {kat}")
            continue
        pliki += [(cfg, p) for p in sorted(kat.rglob("*.md"))]
    return pliki


def wczytaj_manifest():
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as f:
            return json.load(f)
    return {"pliki": {}}


def zapisz_manifest(manifest):
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    os.replace(tmp, MANIFEST)


# ==========================================
# OPENAI + CHROMA (importy leniwe: --podglad i --dry-run działają bez nich)
# ==========================================
load_dotenv()
def utworz_klienta_openai():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        sys.exit(
            "Brak zmiennej OPENAI_API_KEY w pliku .env.\n"
            "Dodaj w .env:\n"
            "OPENAI_API_KEY=sk-..."
        )

    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        max_retries=6
    )


def embedduj(klient, teksty):
    """Embeddingi w batchach ograniczonych liczbą elementów i sumą tokenów."""
    wynik, batch, tok = [], [], 0

    def wyslij():
        nonlocal batch, tok
        if batch:
            kw = {"dimensions": WYMIARY} if WYMIARY else {}
            odp = klient.embeddings.create(model=MODEL, input=batch, **kw)
            wynik.extend(d.embedding for d in sorted(odp.data, key=lambda d: d.index))
        batch, tok = [], 0

    for t in teksty:
        n = tokeny(t)
        if batch and (len(batch) >= MAX_BATCH_ELEMENTOW or tok + n > MAX_BATCH_TOKENOW):
            wyslij()
        batch.append(t)
        tok += n
    wyslij()
    return wynik


def otworz_kolekcje(przebuduj=False):
    import chromadb
    klient = chromadb.PersistentClient(path=KATALOG_BAZY)
    if przebuduj:
        try:
            klient.delete_collection(NAZWA_KOLEKCJI)
        except Exception:
            pass
    kol = klient.get_or_create_collection(
        NAZWA_KOLEKCJI,
        metadata={"hnsw:space": "cosine", "model_embeddingow": MODEL, "wymiary": WYMIARY or 0},
    )
    zapisany = (kol.metadata or {}).get("model_embeddingow")
    if zapisany and (zapisany != MODEL or (kol.metadata or {}).get("wymiary", 0) != (WYMIARY or 0)):
        sys.exit(f"Kolekcja była zbudowana modelem '{zapisany}', a w skrypcie jest '{MODEL}'. "
                 f"Wektory różnych modeli nie są porównywalne – uruchom z --przebuduj.")
    return kol


def usun_id(kol, ids):
    ids = list(ids)
    for i in range(0, len(ids), 500):
        kol.delete(ids=ids[i:i + 500])
    akcje_es = [{"_op_type": "delete", "_index": NAZWA_KOLEKCJI, "_id": i} for i in ids]
    try:
        bulk(es, akcje_es, ignore_status=[404])
    except Exception:
        pass


es = Elasticsearch("http://localhost:9200")
try:
    es.indices.get(index=NAZWA_KOLEKCJI)
except Exception:
    es.indices.create(
        index=NAZWA_KOLEKCJI, 
        settings={
            "analysis": {
                "analyzer": {
                    "default": {
                        "type": "polish"
                    }
                }
            }
        }
    )

# ==========================================
# INDEKSOWANIE (przyrostowe)
# ==========================================
def indeksuj(dry_run=False, przebuduj=False):
    pliki = zbierz_pliki()
    if przebuduj and os.path.exists(MANIFEST):
        os.remove(MANIFEST)
    manifest = wczytaj_manifest()
    kol = klient = None
    if not dry_run:
        kol = otworz_kolekcje(przebuduj)
        klient = utworz_klienta_openai()

    print(f"🚀 Indeksowanie: {len(pliki)} plików Markdown | tokenizer: {TOKENIZER}"
          + ("  (⚠️ przybliżenie – zainstaluj tiktoken)" if TOKENIZER != "tiktoken" else ""))

    # 1) pliki usunięte z dysku -> usuń ich chunki
    obecne = {p.as_posix() for _, p in pliki}
    usuniete = [r for r in manifest["pliki"] if r not in obecne]
    for rel in usuniete:
        if not dry_run:
            usun_id(kol, manifest["pliki"][rel]["ids"])
            del manifest["pliki"][rel]
    if usuniete:
        print(f"🗑️  Usunięte pliki: {len(usuniete)} (chunki {'do usunięcia' if dry_run else 'usunięte z bazy'})")
        if not dry_run:
            zapisz_manifest(manifest)

    stat = {"bez_zmian": 0, "przetworzone": 0, "puste": 0, "chunki": 0, "chunki_nowe": 0, "tokeny_nowe": 0}
    rozmiary, z_tabela = [], 0
    bufor = []      # [(rel, sygnatura, wszystkie_ids, stare_ids, [(cid, chunk, md)] do embeddingu, [(cid, md)] zachowane)]
    bufor_chunkow = 0

    def zrzuc_bufor():
        nonlocal bufor, bufor_chunkow
        if not bufor:
            return
        do_emb = [x for (_, _, _, _, nowe, _) in bufor for x in nowe]
        if do_emb:
            wektory = embedduj(klient, [c.tekst_emb for _, c, _ in do_emb])
            for i in range(0, len(do_emb), 256):
                czesc = do_emb[i:i + 256]
                wektory_czesc = wektory[i:i + 256]
                kol.upsert(ids=[cid for cid, _, _ in czesc],
                           embeddings=wektory_czesc,
                           documents=[c.tekst for _, c, _ in czesc],
                           metadatas=[md for _, _, md in czesc])

                akcje_es = []
                for (cid, c, md) in czesc:
                    akcje_es.append({
                        "_op_type": "index",
                        "_index": NAZWA_KOLEKCJI,
                        "_id": cid,
                        "_source": {
                            "tekst": c.tekst,
                            "tytul": md.get("tytul", ""),
                            "zrodlo": md.get("zrodlo", ""),
                            "sekcja": md.get("sekcja", ""),
                            "url": md.get("url") or md.get("plik")
                        }
                    })
                bulk(es, akcje_es)

        for rel, sygn, wszystkie_ids, stare_ids, _, zachowane in bufor:
            for i in range(0, len(zachowane), 256):      # niezmienione chunki: tylko odświeżenie metadanych (bez embeddingu)
                kol.update(ids=[c for c, _ in zachowane[i:i + 256]], metadatas=[m for _, m in zachowane[i:i + 256]])
            osierocone = set(stare_ids) - set(wszystkie_ids)
            if osierocone:
                usun_id(kol, osierocone)
            manifest["pliki"][rel] = {"sygnatura": sygn, "ids": wszystkie_ids}
        zapisz_manifest(manifest)          # po każdym zrzucie: przerwanie (Ctrl+C) niczego nie psuje
        print(f"   💾 zapisano {len(do_emb)} nowych chunków ({stat['przetworzone']} plików przetworzonych)")
        bufor, bufor_chunkow = [], 0

    try:
        for cfg, sciezka in pliki:
            rel = sciezka.as_posix()
            dok = przygotuj_dokument(sciezka, cfg)
            if dok is None:
                stat["puste"] += 1
                continue
            sygn = sygnatura_pliku(dok[1], dok[2], cfg)
            stary = manifest["pliki"].get(rel, {})
            if stary.get("sygnatura") == sygn:
                stat["bez_zmian"] += 1
                continue

            chunki = chunkuj_plik(sciezka, cfg, dok)
            stare_ids = set(stary.get("ids", []))
            nowe = [(cid, c, md) for cid, c, md in chunki if cid not in stare_ids]   # niezmienione chunki nie kosztują
            stat["przetworzone"] += 1
            stat["chunki"] += len(chunki)
            stat["chunki_nowe"] += len(nowe)
            stat["tokeny_nowe"] += sum(c.tok for _, c, _ in nowe)
            rozmiary += [c.tok for _, c, _ in chunki]
            z_tabela += sum(c.ma_tabele for _, c, _ in chunki)

            if dry_run:
                continue
            zachowane = [(cid, md) for cid, _, md in chunki if cid in stare_ids]
            bufor.append((rel, sygn, [cid for cid, _, _ in chunki], stare_ids, nowe, zachowane))
            bufor_chunkow += len(nowe)
            if bufor_chunkow >= 256:
                zrzuc_bufor()
        zrzuc_bufor()
    except KeyboardInterrupt:
        print("\n⏹️  Przerwano – to, co zostało zapisane, jest spójne; uruchom ponownie, aby kontynuować.")
        return

    raport(stat, rozmiary, z_tabela, dry_run)


def raport(stat, rozmiary, z_tabela, dry_run):
    print("\n" + "=" * 56)
    print("📊 " + ("SZACUNEK (dry-run, nic nie wysłano do API)" if dry_run else "RAPORT INDEKSOWANIA"))
    print("=" * 56)
    print(f"Pliki bez zmian (pominięte):      {stat['bez_zmian']}")
    print(f"Pliki przetworzone:               {stat['przetworzone']}   (puste/za krótkie: {stat['puste']})")
    print(f"Chunki z tych plików:             {stat['chunki']}   (w tym z tabelami: {z_tabela})")
    print(f"Chunki do (re)embeddingu:         {stat['chunki_nowe']}")
    if rozmiary:
        r = sorted(rozmiary)
        p = lambda q: r[min(len(r) - 1, int(len(r) * q))]
        print(f"Rozmiar chunków [tokeny]:         min {r[0]} | mediana {p(0.5)} | p90 {p(0.9)} | max {r[-1]}")
        male = sum(1 for x in r if x < 60)
        print(f"Bardzo małe chunki (<60 tok.):    {male} ({100 * male // len(r)}%)")
    koszt = stat["tokeny_nowe"] / 1e6 * CENA_ZA_1M_TOKENOW_USD
    print(f"Tokeny do embeddingu:             {stat['tokeny_nowe']:,}  ≈ ${koszt:.3f} (cena z ustawień: ${CENA_ZA_1M_TOKENOW_USD}/1M)")
    print("=" * 56)


# ==========================================
# NARZĘDZIA: PODGLĄD CIĘCIA I TEST WYSZUKIWANIA
# ==========================================
def podglad(sciezka_pliku):
    p = Path(sciezka_pliku)
    cfg = next((c for c in KOLEKCJE if Path(c["sciezka"]).as_posix() in p.as_posix()),
               {"sciezka": str(p.parent), "zrodlo": "Źródło testowe", "typ": "strona"})
    wynik = chunkuj_plik(p, cfg)
    if not wynik:
        print("Plik pusty / za krótki.")
        return
    print(f"Plik: {p} | kolekcja: {cfg['zrodlo']} | chunków: {len(wynik)} | tokenizer: {TOKENIZER}\n")
    for cid, c, md in wynik:
        print("━" * 70)
        print(f"CHUNK {md['chunk_nr'] + 1}/{md['chunk_z']} | {c.tok} tok. | sekcja: {c.sekcja or '(początek)'}"
              f"{' | TABELA' if c.ma_tabele else ''}")
        print("━" * 70)
        print(c.tekst)
        print()


def szukaj(zapytanie, k=5, zrodlo=None):
    klient = utworz_klienta_openai()
    kol = otworz_kolekcje()
    wektor = embedduj(klient, [zapytanie])[0]
    odp = kol.query(query_embeddings=[wektor], n_results=k, where={"zrodlo": zrodlo} if zrodlo else None,
                    include=["documents", "metadatas", "distances"])
    for i, (doc, md, dist) in enumerate(zip(odp["documents"][0], odp["metadatas"][0], odp["distances"][0]), 1):
        print("━" * 70)
        print(f"#{i}  podobieństwo {1 - dist:.3f} | {md.get('zrodlo')} | {md.get('tytul')}"
              f"{' › ' + md['sekcja'] if md.get('sekcja') else ''}")
        print(f"    {md.get('url') or md.get('plik')}")
        print("    " + doc.replace("\n", "\n    ")[:700] + ("…" if len(doc) > 700 else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Budowa bazy wektorowej RAG (Markdown -> Chroma)")
    ap.add_argument("--dry-run", action="store_true", help="policz chunki i koszt, bez wywołań API")
    ap.add_argument("--przebuduj", action="store_true", help="usuń kolekcję i zbuduj od zera")
    ap.add_argument("--podglad", metavar="PLIK.md", help="pokaż, jak plik zostanie pocięty")
    ap.add_argument("--szukaj", metavar="ZAPYTANIE", help="test wyszukiwania w bazie")
    ap.add_argument("-k", type=int, default=5, help="liczba wyników --szukaj")
    ap.add_argument("--zrodlo", help="filtr źródła dla --szukaj, np. 'Biblioteka Główna AGH'")
    a = ap.parse_args()
    if a.podglad:
        podglad(a.podglad)
    elif a.szukaj:
        szukaj(a.szukaj, a.k, a.zrodlo)
    else:
        indeksuj(dry_run=a.dry_run, przebuduj=a.przebuduj)