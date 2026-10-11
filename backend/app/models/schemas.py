from pydantic import BaseModel, Field
from typing import List

class MessageHistory(BaseModel):
    role: str = Field(
        ..., 
        description="Rola nadawcy wiadomości (np. 'user', 'assistant')."
    )
    content: str = Field(
        ..., 
        description="Treść wiadomości."
    )

class ChatRequestPayload(BaseModel):
    query: str = Field(
        ..., 
        description="Aktualne pytanie zadane przez użytkownika.",
        example="Gdzie znajdę dziekanat WIET?"
    )
    model: str = Field(
        default="gpt-4o", 
        description="Identyfikator wybranego modelu językowego OpenAI.",
        example="gpt-4o-mini"
    )
    history: List[MessageHistory] = Field(
        default_factory=list, 
        description="Opcjonalna historia dotychczasowej konwersacji w celu zachowania kontekstu."
    )