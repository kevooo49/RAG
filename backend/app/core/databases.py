import chromadb
from elasticsearch import Elasticsearch
from app.core.config import settings
from app.core.logging import logger

class DatabaseManager:
    """Database connection management class (Singleton)"""
    def __init__(self):
        self.es_client: Elasticsearch | None = None
        self.chroma_client = None
        self.chroma_collection = None

    def connect(self):
        try:
            logger.info("Connecting to Elasticsearch...")
            self.es_client = Elasticsearch(settings.ES_URL)
            
            logger.info("Connecting to z ChromaDB...")
            self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
            self.chroma_collection = self.chroma_client.get_collection(name=settings.ES_INDEX)
            
            logger.info("Database connections successfully established.")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise e

db_manager = DatabaseManager()
db_manager.connect()