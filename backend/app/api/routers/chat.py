from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatRequestPayload
from app.services.llm_service import generate_streamed_response

from app.core.limiter import limiter

router = APIRouter(
    prefix="/api/chat",
    tags=["Chat"]
)

@router.post("")
@limiter.limit("5/minute")
async def chat_endpoint(request: Request, payload: ChatRequestPayload):
    """
    Main chat endpoint. 
    Accepts a user query and returns an asynchronous stream of Server-Sent Events (SSE).
    """
    return StreamingResponse(
        generate_streamed_response(payload), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )