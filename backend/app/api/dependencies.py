from elasticsearch import Elasticsearch
from chromadb.api.models.Collection import Collection
from app.core.databases import db_manager

def get_es_client() -> Elasticsearch:
    """Dependency providing an Elasticsearch client"""
    if not db_manager.es_client:
        raise RuntimeError("No connection to Elasticsearch.")
    return db_manager.es_client

def get_chroma_collection() -> Collection:
    """Dependency providing ChromaDB collection"""
    if not db_manager.chroma_collection:
        raise RuntimeError("No connection to ChromaDB.")
    return db_manager.chroma_collection