from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatRequestPayload
from app.services.llm_service import generuj_odpowiedz_stream

from app.core.limiter import limiter

router = APIRouter(
    prefix="/api/chat",
    tags=["Chat"]
)

@router.post("")
@limiter.limit("5/minute")
async def chat_endpoint(request: Request, payload: ChatRequestPayload):
    """
    Główny endpoint czatu. 
    Przyjmuje zapytanie użytkownika i zwraca asynchroniczny strumień Server-Sent Events (SSE).
    """
    return StreamingResponse(
        generuj_odpowiedz_stream(payload), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )