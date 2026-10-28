from pydantic import BaseModel, Field
from typing import List, Literal

class MessageHistory(BaseModel):
    role: Literal["user", "assistant"] = Field(
        ..., 
        description="Rola nadawcy wiadomości (tylko 'user' lub 'assistant')."
    )
    content: str = Field(
        ..., 
        max_length=3000,
        description="Treść wiadomości."
    )

class ChatRequestPayload(BaseModel):
    query: str = Field(
        ..., 
        max_length=1000,
        description="Aktualne pytanie zadane przez użytkownika.",
        examples=["Gdzie znajdę dziekanat WIEiT?"]
    )
    model: str = Field(
        default="gpt-4o",
        description="Identyfikator wybranego modelu językowego.",
        examples=["gpt-4o-mini", "bielik", "gpt-4o"]
    )
    history: List[MessageHistory] = Field(
        default_factory=list, 
        max_length=100,
        description="Opcjonalna historia dotychczasowej konwersacji w celu zachowania kontekstu."
    )