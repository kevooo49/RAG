import logging
import sys

def setup_logging():
    # Tworzymy główny logger dla naszej aplikacji
    logger = logging.getLogger("agh_rag")
    logger.setLevel(logging.INFO)
    
    # Definiujemy format (Czas | Poziom | Plik:Funkcja:Linia - Wiadomość)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Wypisywanie logów do konsoli
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Zabezpieczenie przed wielokrotnym dodawaniem handlerów
    if not logger.handlers:
        logger.addHandler(console_handler)
        
    return logger

logger = setup_logging()