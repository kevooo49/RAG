import chromadb
from elasticsearch import Elasticsearch
from app.core.config import settings
from app.core.logging import logger

class DatabaseManager:
    """Klasa zarządzająca połączeniami z bazami danych (Singleton)"""
    def __init__(self):
        self.es_client: Elasticsearch | None = None
        self.chroma_client = None
        self.chroma_collection = None

    def connect(self):
        try:
            logger.info("Nawiązywanie połączenia z Elasticsearch...")
            self.es_client = Elasticsearch(settings.ES_URL)
            
            logger.info("Nawiązywanie połączenia z ChromaDB...")
            self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
            self.chroma_collection = self.chroma_client.get_collection(name=settings.ES_INDEX)
            
            logger.info("✅ Połączenia z bazami danych ustanowione pomyślnie.")
        except Exception as e:
            logger.error(f"❌ Błąd inicjalizacji baz danych: {e}")
            raise e

# Globalna instancja menedżera bazy danych
db_manager = DatabaseManager()
# Inicjujemy połączenie przy imporcie
db_manager.connect()