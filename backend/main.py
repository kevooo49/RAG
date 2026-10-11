from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Importujemy nasz wydzielony router czatu
from app.api.routers import chat

# Inicjalizacja głównej aplikacji
app = FastAPI(
    title="AGH RAG API",
    description="Wirtualny Asystent Wiedzy AGH - Architektura Mikroserwisowa",
    version="2.0.0"
)

# Konfiguracja CORS (pozwala na komunikację z frontendem, np. Vite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # W produkcji zmień na konkretny adres, np. ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rejestracja endpointów w głównej aplikacji
app.include_router(chat.router)