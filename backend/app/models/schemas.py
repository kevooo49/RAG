from pydantic import BaseModel, Field
from typing import List, Literal

class MessageHistory(BaseModel):
    role: Literal["user", "assistant"] = Field(
        ..., 
        description="The role of the message sender (either 'user' or 'assistant')."
    )
    content: str = Field(
        ..., 
        min_length=1,
        pattern=r"\S",
        max_length=3000,
        description="Message content."
    )

class ChatRequestPayload(BaseModel):
    query: str = Field(
        ..., 
        min_length=1,
        pattern=r"\S",
        max_length=1000,
        description="Current question asked by a user.",
        examples=["Gdzie znajdę dziekanat WIEiT?"]
    )
    model: str = Field(
        default="gpt-4o",
        description="Identifier of the selected language model.",
        examples=["gpt-4o-mini", "bielik", "gpt-4o"]
    )
    history: List[MessageHistory] = Field(
        default_factory=list, 
        max_length=100,
        description="Optional conversation history to maintain context."
    )