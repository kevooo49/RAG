import json
import os
import sys
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1] / "config"))
from sites import load_or_exit  # noqa: E402 # type: ignore

DOCS_DIR = load_or_exit().eval_dir
INPUT_FILE = str(SCRIPT_DIR / "test_dataset_balanced.json")
OUTPUT_FILE = str(SCRIPT_DIR / "llm_golden_dataset.json")

load_dotenv(SCRIPT_DIR.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

def get_o3_model():
    """Initialization of the o3 model. Reasoning models do not accept a temperature parameter."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY. Check the .env file.")
    
    return ChatOpenAI(
        model="o3",
        openai_api_key=api_key,
        max_retries=3,
        timeout=60
    )

def build_prompt(query: str, context: str) -> str:
    """It creates a strict prompt for the model, forcing it to use ONLY the provided context."""
    return f"""Jesteś bezstronnym i precyzyjnym systemem eksperckim tworzącym zbiory testowe (Ground Truth).
Twoim zadaniem jest wygenerowanie idealnej, rzetelnej i zwięzłej odpowiedzi na pytanie użytkownika, 
bazując WYŁĄCZNIE na dostarczonym tekście źródłowym. 

ZASADY:
1. Odpowiadaj tylko na podstawie tekstu źródłowego. Jeśli tekst nie zawiera pełnej odpowiedzi, napisz to, co z niego wynika.
2. Zignoruj własną wiedzę o świecie i uczelni AGH.
3. Odpowiedź ma być zwięzła i bezpośrednia (maksymalnie 3-4 zdania).
4. Zachowaj profesjonalny ton, niezależnie od tego jak potocznie sformułowane jest pytanie.

PYTANIE UŻYTKOWNIKA:
{query}

TEKST ŹRÓDŁOWY:
{context}

IDEALNA ODPOWIEDŹ:"""

def load_progress() -> list:
    """Loads the current progress to avoid reprocessing the same queries."""
    if Path(OUTPUT_FILE).exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_progress(data: list) -> None:
    """Saves the updated list to the JSON file."""
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    if not Path(INPUT_FILE).exists():
        log.error(f"Input file not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    golden_dataset = load_progress()
    processed_queries = {item["query"] for item in golden_dataset}
    
    llm = get_o3_model()
    
    log.info(f"Queries to process: {len(dataset)}. Already done: {len(processed_queries)}.")

    for i, item in enumerate(dataset, 1):
        query = item["query"]
        if query in processed_queries:
            continue
            
        filename = item["expected_file"]
        file_path = DOCS_DIR / filename
        
        if not file_path.exists():
            log.warning(f"[{i}/{len(dataset)}] File not found on disk: {filename}. Skipping.")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            context = f.read()

        prompt = build_prompt(query, context)
        
        log.info(f"[{i}/{len(dataset)}] Generating o3 response for: '{query[:40]}...'")
        
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            golden_answer = response.content.strip()
            
            golden_item = {
                "query": query,
                "type": item["type"],
                "expected_file": filename,
                "context": context,
                "golden_answer": golden_answer
            }
            
            golden_dataset.append(golden_item)
            save_progress(golden_dataset)
            
            time.sleep(1)
            
        except Exception as e:
            log.error(f"Error while generating response: {e}")
            log.info("Interrupting loop. Run the script again to resume.")
            break
            
    log.info("Dataset generation completed!")

if __name__ == "__main__":
    main()