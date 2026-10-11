from elasticsearch import Elasticsearch
from chromadb.api.models.Collection import Collection
from app.core.databases import db_manager

def get_es_client() -> Elasticsearch:
    """Dependency dostarczające klienta Elasticsearch"""
    if not db_manager.es_client:
        raise RuntimeError("Brak połączenia z Elasticsearch.")
    return db_manager.es_client

def get_chroma_collection() -> Collection:
    """Dependency dostarczające kolekcję z ChromaDB"""
    if not db_manager.chroma_collection:
        raise RuntimeError("Brak połączenia z ChromaDB.")
    return db_manager.chroma_collection