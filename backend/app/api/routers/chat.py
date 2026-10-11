from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatRequestPayload
from app.services.llm_service import generuj_odpowiedz_stream

router = APIRouter(
    prefix="/api/chat",
    tags=["Chat"]
)

@router.post("")
async def chat_endpoint(payload: ChatRequestPayload):
    """
    Główny endpoint czatu. 
    Przyjmuje zapytanie użytkownika i zwraca asynchroniczny strumień Server-Sent Events (SSE).
    """
    return StreamingResponse(
        generuj_odpowiedz_stream(payload), 
        media_type="text/event-stream"
    )