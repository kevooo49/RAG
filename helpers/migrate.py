
import chromadb
import json
import os
import time
from datetime import timedelta

# =========================
# KONFIGURACJA
# =========================

SOURCE_PATH = "./data/Baza_Wektorowa_Chroma_ASYM"
COLLECTION_NAME = "agh_wiedza_asym"

TARGET_HOST = "127.0.0.1"
TARGET_PORT = 8001

BATCH_SIZE = 2000
MAX_RETRIES = 4
CHECKPOINT_FILE = "./chroma_migration_checkpoint.json"

# =========================
# POMOCNICZE
# =========================

def format_time(seconds):
    return str(timedelta(seconds=max(0, int(seconds))))


def save_checkpoint(next_offset, total):
    """Zapisuje postęp atomowo, aby łatwiej wznowić migrację."""
    temp_file = CHECKPOINT_FILE + ".tmp"

    checkpoint = {
        "source_path": os.path.abspath(SOURCE_PATH),
        "collection": COLLECTION_NAME,
        "total": total,
        "next_offset": next_offset,
    }

    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f)

    os.replace(temp_file, CHECKPOINT_FILE)


def load_checkpoint(total):
    """Wczytuje postęp tylko dla tej samej bazy i kolekcji."""
    if not os.path.exists(CHECKPOINT_FILE):
        return 0

    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    if (
        checkpoint.get("source_path") != os.path.abspath(SOURCE_PATH)
        or checkpoint.get("collection") != COLLECTION_NAME
        or checkpoint.get("total") != total
    ):
        raise RuntimeError(
            "Checkpoint nie pasuje do aktualnej bazy lub liczby "
            "rekordów. Zmień nazwę pliku checkpoint albo sprawdź "
            "konfigurację przed kontynuacją."
        )

    offset = checkpoint["next_offset"]

    if not isinstance(offset, int) or not 0 <= offset <= total:
        raise RuntimeError("Nieprawidłowy offset w checkpoint.")

    return offset


def retry_operation(operation, description):
    """Ponawia operację po błędzie, z rosnącym opóźnieniem."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return operation()
        except Exception:
            if attempt == MAX_RETRIES:
                raise

            delay = 2 ** attempt
            print(
                f"\nBłąd: {description}. "
                f"Ponowienie {attempt + 1}/{MAX_RETRIES} "
                f"za {delay} s..."
            )
            time.sleep(delay)


# =========================
# MIGRACJA
# =========================

def main():
    print("=" * 65)
    print("MIGRACJA CHROMADB")
    print("=" * 65)

    print("\n[1/4] Otwieranie lokalnej bazy...")

    old_client = chromadb.PersistentClient(path=SOURCE_PATH)
    old_col = old_client.get_collection(COLLECTION_NAME)
    total = old_col.count()

    print(f"Źródłowa kolekcja: {COLLECTION_NAME}")
    print(f"Liczba rekordów:   {total:,}")

    print("\n[2/4] Łączenie z serwerem Docker...")

    new_client = chromadb.HttpClient(
        host=TARGET_HOST,
        port=TARGET_PORT,
    )

    new_col = new_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata=old_col.metadata,
    )

    initial_target_count = new_col.count()

    print(f"Rekordy w celu przed migracją: {initial_target_count:,}")

    if initial_target_count:
        print(
            "Uwaga: kolekcja docelowa nie jest pusta. "
            "Istniejące rekordy mogą zostać nadpisane przez upsert."
        )

    print("\n[3/4] Kopiowanie danych...")

    offset = load_checkpoint(total)
    started_at = time.monotonic()
    copied_this_run = 0

    print(f"Wznawianie od offsetu: {offset:,} / {total:,}")

    while offset < total:
        batch_started = time.monotonic()
        limit = min(BATCH_SIZE, total - offset)

        data = retry_operation(
            lambda: old_col.get(
                limit=limit,
                offset=offset,
                include=["embeddings", "documents", "metadatas"],
            ),
            "odczyt partii ze źródła",
        )

        ids = data["ids"]

        if not ids:
            raise RuntimeError(
                f"Źródło zwróciło pustą partię przy offsecie {offset}. "
                "Migracja została przerwana, aby uniknąć pominięcia danych."
            )

        def upsert_batch():
            kwargs = {
                "ids": ids,
                "embeddings": data["embeddings"],
            }

            if all(x is not None for x in data["documents"]):
                kwargs["documents"] = data["documents"]

            if all(x is not None for x in data["metadatas"]):
                kwargs["metadatas"] = data["metadatas"]

            new_col.upsert(**kwargs)

        retry_operation(upsert_batch, "zapis partii do celu")

        result = retry_operation(
            lambda: new_col.get(
                ids=ids,
                include=["metadatas"],
            ),
            "weryfikacja partii",
        )

        found_ids = set(result["ids"])
        missing_ids = [item for item in ids if item not in found_ids]

        if missing_ids:
            raise RuntimeError(
                f"Weryfikacja nie powiodła się: "
                f"{len(missing_ids)} brakujących identyfikatorów. "
                "Checkpoint nie został przesunięty."
            )

        offset += len(ids)
        copied_this_run += len(ids)

        save_checkpoint(offset, total)

        elapsed = time.monotonic() - started_at
        speed = copied_this_run / elapsed if elapsed > 0 else 0
        remaining = (total - offset) / speed if speed > 0 else 0
        percent = offset / total * 100 if total else 100
        batch_time = time.monotonic() - batch_started

        print(
            f"\rPostęp: {percent:6.2f}% | "
            f"{offset:,}/{total:,} | "
            f"{speed:,.0f} rekordów/s | "
            f"partia: {batch_time:.1f} s | "
            f"pozostało: {format_time(remaining)}",
            end="",
            flush=True,
        )

    print("\n\n[4/4] Kontrola końcowa...")

    target_count = new_col.count()

    print(f"Źródło: {total:,}")
    print(f"Cel:    {target_count:,}")

    if target_count < total:
        raise RuntimeError(
            "Liczba rekordów w celu jest mniejsza niż w źródle. "
            "Nie usuwaj checkpointu; sprawdź przyczynę."
        )

    if target_count > total:
        print(
            "UWAGA: cel zawiera więcej rekordów niż źródło. "
            "Dodatkowe rekordy nie zostały usunięte."
        )

    os.remove(CHECKPOINT_FILE)

    print("\n" + "=" * 65)
    print("MIGRACJA ZAKOŃCZONA")
    print(f"Zweryfikowane identyfikatory źródłowe: {total:,}")
    print(f"Czas tej sesji: {format_time(time.monotonic() - started_at)}")
    print("Wszystkie partie zostały zapisane i sprawdzone.")
    print("=" * 65)


if __name__ == "__main__":
    main()
