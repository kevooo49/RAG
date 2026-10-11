import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Klucze API (wymagane, wczytywane automatycznie z .env)
    OPENAI_API_KEY: str
    LLAMA_CLOUD_API_KEY: str = ""  # Opcjonalne w samym API, używane przez LlamaParse

    # Konfiguracja baz danych
    ES_INDEX: str = "agh_wiedza"
    ES_URL: str = "http://localhost:9200"
    
    # Ścieżka do bazy wektorowej (zakładając uruchamianie serwera z katalogu backend/)
    CHROMA_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/Baza_Wektorowa_Chroma"))

    # Ustawienia modelu RAG
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    RRF_K: int = 60

    # Wskazanie, gdzie znajduje się plik .env
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Ignoruje dodatkowe zmienne w .env, których tu nie zdefiniowaliśmy
    )

# Tworzymy globalną instancję ustawień, którą będziemy importować w innych plikach
settings = Settings()