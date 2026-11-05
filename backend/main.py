import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from slowapi.errors import RateLimitExceeded
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
from app.api.routers import chat
from app.services import rag_service
from app.core.limiter import limiter
from app.core.logging import setup_logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    rag_service.initialize()
    yield

app = FastAPI(
    title="AGH RAG API",
    description="Wirtualny Asystent Wiedzy AGH",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

app.state.limiter = limiter

ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:4173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """When someone asks too often (Error 429)."""
    return JSONResponse(
        status_code=429,
        content={"detail": "Zbyt wiele zapytań. Odczekaj chwilę przed zadaniem kolejnego pytania."}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """When the history or question exceeds the permitted limit (Error 422)."""
    return JSONResponse(
        status_code=422,
        content={"detail": "Przekroczono limit długości konwersacji lub zapytania. Odśwież stronę, aby zacząć nowy czat."}
    )

app.include_router(chat.router)

@app.get("/health", tags=["Monitoring"])
def health_check():
    """Returns the operational status of the databases (Chroma, ES)."""
    return rag_service.check_dependencies()