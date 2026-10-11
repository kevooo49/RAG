import chromadb
from elasticsearch import Elasticsearch
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict, Any, Tuple

from app.core.config import settings

# ============================================================================
# Inicjalizacja połączeń z bazami (wzorzec Singleton per moduł)
# ============================================================================
try:
    es_client = Elasticsearch(settings.ES_URL)
    chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
    kolekcja_chroma = chroma_client.get_collection(name=settings.ES_INDEX)
    
    # Przekazanie klucza jawnie z konfiguracji
    embeddings_model = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY
    )
except Exception as e:
    print(f"⚠️ Błąd inicjalizacji baz danych w RAG Service: {e}")
    # W większym projekcie wylądowałoby to w systemie logowania


def wyszukaj_hybrydowo(pytanie: str, top_k: int = 5) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Wykonuje wyszukiwanie semantyczne (wektorowe) i leksykalne (BM25),
    a następnie łączy wyniki algorytmem Reciprocal Rank Fusion (RRF).
    Zwraca listę tupli: (treść_dokumentu, metadane_i_wynik)
    """
    
    # 1. ChromaDB (Semantyka)
    wektor = embeddings_model.embed_query(pytanie)
    wyniki_chroma = kolekcja_chroma.query(
        query_embeddings=[wektor], 
        n_results=top_k, 
        include=["documents", "metadatas"]
    )
    
    # 2. Elasticsearch (Słowa kluczowe)
    wyniki_es = es_client.search(
        index=settings.ES_INDEX, 
        query={"match": {"tekst": pytanie}}, 
        size=top_k
    )
    
    # 3. Reciprocal Rank Fusion (RRF)
    rejestr = {}
    rrf_k = settings.RRF_K
    
    # Przetwarzanie rankingu ChromaDB
    if wyniki_chroma.get("documents") and wyniki_chroma["documents"][0]:
        for rank, (doc, md) in enumerate(zip(wyniki_chroma["documents"][0], wyniki_chroma["metadatas"][0])):
            if doc not in rejestr:
                rejestr[doc] = {"score": 0.0, "meta": md}
            rejestr[doc]["score"] += 1.0 / (rank + 1 + rrf_k)
            
    # Przetwarzanie rankingu Elasticsearch
    if "hits" in wyniki_es and "hits" in wyniki_es["hits"]:
        for rank, hit in enumerate(wyniki_es["hits"]["hits"]):
            doc = hit["_source"].get("tekst", "")
            md = {
                "tytul": hit["_source"].get("tytul", ""),
                "zrodlo": hit["_source"].get("zrodlo", ""),
                "sekcja": hit["_source"].get("sekcja", ""),
                "url": hit["_source"].get("url", "")
            }
            if doc not in rejestr:
                rejestr[doc] = {"score": 0.0, "meta": md}
            rejestr[doc]["score"] += 1.0 / (rank + 1 + rrf_k)
            
    # Sortowanie po nowej, złączonej punktacji RRF
    posortowane = sorted(rejestr.items(), key=lambda x: x[1]["score"], reverse=True)
    
    return posortowane[:top_k]