import os
from typing import Set, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
ENV_PATH = os.path.join(BACKEND_DIR, ".env")

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    LLAMA_CLOUD_API_KEY: str = ""

    ES_INDEX: str = "agh_wiedza_asym"
    ES_URL: str = "http://localhost:9200"
    
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION: str = "agh_wiedza_asym"

    ALLOWED_MODELS: Set[str] = {"gpt-4o-mini", "bielik", "gpt-4o"}
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    BIELIK_MODEL: str = "SpeakLeash/bielik-11b-v3.0-instruct:Q4_K_M"
    BIELIK_NUM_CTX: int = 8192
    EXPANSION_MODEL: str = "gpt-4o-mini"
    
    EMBED_MODEL: str = "text-embedding-3-small"
    EMBED_DIMENSIONS: Optional[int] = None
    RRF_K: int = 60
    RAG_MIN_DENSE_SIM: Optional[float] = None
    RAG_MAX_CONTEXT_CHARS: int = 12000

    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore" 
    )

settings = Settings()