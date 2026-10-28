from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.api.routers import chat
from app.services import rag_service

from app.core.limiter import limiter

@asynccontextmanager
async def lifespan(app: FastAPI):
    rag_service.inicjalizuj()
    yield

app = FastAPI(
    title="AGH RAG API",
    description="Wirtualny Asystent Wiedzy AGH",
    version="2.0.0",
    lifespan=lifespan
)

app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Gdy ktoś pyta zbyt często (Błąd 429)."""
    return JSONResponse(
        status_code=429,
        content={"detail": "Zbyt wiele zapytań. Odczekaj chwilę przed zadaniem kolejnego pytania."}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Gdy historia lub pytanie przekroczy dozwolony limit (Błąd 422)."""
    return JSONResponse(
        status_code=422,
        content={"detail": "Przekroczono limit długości konwersacji lub zapytania. Odśwież stronę, aby zacząć nowy czat."}
    )

app.include_router(chat.router)

@app.get("/health", tags=["Monitoring"])
def health_check():
    """Zwraca status działania baz danych (Chroma, ES)."""
    return rag_service.sprawdz_zaleznosci()