import json
import asyncio
import datetime
from typing import AsyncGenerator

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import settings
from app.models.schemas import ChatRequestPayload, MessageHistory
from app.services.rag_service import wyszukaj_hybrydowo


async def przeformuluj_zapytanie(query: str, history: list[MessageHistory]) -> str:
    """
    Na podstawie historii rozmowy przeformułowuje pytanie tak,
    aby było samodzielne (Context Memory). Działa W PEŁNI asynchronicznie.
    """
    if not history:
        return query

    historia_tekst = "\n".join([f"{m.role.upper()}: {m.content}" for m in history[-4:]])
    
    llm_fast = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0,
        openai_api_key=settings.OPENAI_API_KEY
    )
    
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "Na podstawie historii rozmowy i ostatniego pytania sformułuj jedno, samodzielne zapytanie do bazy wiedzy. Zwróć TYLKO przeformułowane zapytanie, bez żadnych komentarzy i wstępów."),
        ("user", f"HISTORIA:\n{historia_tekst}\n\nPYTANIE: {query}")
    ])
    
    # ainvoke - asynchroniczne wywołanie zapobiegające blokowaniu serwera (Event Loop)
    response = await llm_fast.ainvoke(rewrite_prompt.format_messages())
    return response.content


async def generuj_odpowiedz_stream(payload: ChatRequestPayload) -> AsyncGenerator[str, None]:
    """
    Główny generator Server-Sent Events (SSE). Pobiera kontekst, wysyła źródła,
    a następnie strumieniuje odpowiedź modelu słowo po słowie.
    """
    
    # 1. Przygotowanie samodzielnego zapytania
    zapytanie_bazy = await przeformuluj_zapytanie(payload.query, payload.history)

    # 2. Wyciągnięcie odpowiednich fragmentów z bazy hybrydowej (RAG)
    zwycieskie_fragmenty = wyszukaj_hybrydowo(zapytanie_bazy)
    kontekst_tekst = "\n\n---\n\n".join([doc for doc, _ in zwycieskie_fragmenty])
    
    # 3. Przygotowanie struktury źródeł JSON dla frontendu
    zrodla_json = [
        {"text": doc, "score": info["score"], "meta": info["meta"]} 
        for doc, info in zwycieskie_fragmenty
    ]

    # Najpierw wysyłamy źródła w formacie zjadliwym dla SSE
    yield f"data: {json.dumps({'type': 'sources', 'content': zrodla_json})}\n\n"
    
    # 4. Inicjalizacja modelu z zachowaniem ustawień wybranych przez użytkownika na frontendzie
    llm = ChatOpenAI(
        model=payload.model, 
        temperature=0.1, 
        streaming=True,
        openai_api_key=settings.OPENAI_API_KEY
    )

    dzisiejsza_data = datetime.datetime.now().strftime("%Y-%m-%d")
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "Jesteś profesjonalnym asystentem uczelni AGH. Odpowiadasz na pytania "
            "opierając się WYŁĄCZNIE na dostarczonym poniżej kontekście z bazy wiedzy.\n\n"
            f"Dzisiejsza data to: {dzisiejsza_data}. Masz pełne prawo wykonywać na podstawie kontekstu "
            "proste obliczenia matematyczne (np. wiek uczelni) oraz wyciągać logiczne wnioski (jeśli coś nie jest powiedziane wprost), "
            "pod warunkiem, że dane wejściowe do tych obliczeń pochodzą z poniższego kontekstu.\n\n"
            "ZASADA BEZWZGLĘDNA: Jeśli odpowiedź wymaga faktu (np. definicji, daty, nazwiska), "
            "którego w ogóle nie ma w kontekście, nie używaj wiedzy ogólnej. "
            "Odpowiedz dokładnie: 'Przepraszam, ale nie potrafię odpowiedzieć na to pytanie w oparciu o udostępnioną mi bazę wiedzy AGH.'\n\n"
            "KONTEKST BAZY WIEDZY:\n{kontekst}"
        )),
        ("user", "{pytanie}")
    ])
    
    chain = prompt_template | llm
    
    # 5. Strumieniowanie tokenów za pomocą astream()
    async for chunk in chain.astream({"kontekst": kontekst_tekst, "pytanie": payload.query}):
        if chunk.content:
            yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"
        # Pozwala serwerowi FastAPI "odetchnąć" i obsłużyć inne requesty w tle
        await asyncio.sleep(0)
        
    yield "data: [DONE]\n\n"