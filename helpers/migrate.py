import chromadb
import json
import os
import time
from datetime import timedelta

# =========================
# CONFIGURATION
# =========================

SOURCE_PATH = "../data/Baza_Wektorowa_Chroma_ASYM"
COLLECTION_NAME = "agh_wiedza_asym"

TARGET_HOST = "127.0.0.1"
TARGET_PORT = 8001

BATCH_SIZE = 2000
MAX_RETRIES = 4
CHECKPOINT_FILE = "./chroma_migration_checkpoint.json"

# =========================
# UTILITIES
# =========================

def format_time(seconds):
    return str(timedelta(seconds=max(0, int(seconds))))


def save_checkpoint(next_offset, total):
    """Saves progress atomically to easily resume the migration."""
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
    """Loads progress only for the same database and collection."""
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
            "The checkpoint does not match the current database or "
            "record count. Change the checkpoint file name or check "
            "the configuration before continuing."
        )

    offset = checkpoint["next_offset"]

    if not isinstance(offset, int) or not 0 <= offset <= total:
        raise RuntimeError("Invalid offset in the checkpoint.")

    return offset


def retry_operation(operation, description):
    """Retries an operation after an error, with increasing delay."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return operation()
        except Exception:
            if attempt == MAX_RETRIES:
                raise

            delay = 2 ** attempt
            print(
                f"\nError: {description}. "
                f"Retrying {attempt + 1}/{MAX_RETRIES} "
                f"in {delay} s..."
            )
            time.sleep(delay)


# =========================
# MIGRATION
# =========================

def main():
    print("=" * 65)
    print("CHROMADB MIGRATION")
    print("=" * 65)

    print("\n[1/4] Opening local database...")

    old_client = chromadb.PersistentClient(path=SOURCE_PATH)
    old_col = old_client.get_collection(COLLECTION_NAME)
    total = old_col.count()

    print(f"Source collection: {COLLECTION_NAME}")
    print(f"Record count:      {total:,}")

    print("\n[2/4] Connecting to Docker server...")

    new_client = chromadb.HttpClient(
        host=TARGET_HOST,
        port=TARGET_PORT,
    )

    new_col = new_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata=old_col.metadata,
    )

    initial_target_count = new_col.count()

    print(f"Target records before migration: {initial_target_count:,}")

    if initial_target_count:
        print(
            "Warning: the target collection is not empty. "
            "Existing records might be overwritten by upsert."
        )

    print("\n[3/4] Copying data...")

    offset = load_checkpoint(total)
    started_at = time.monotonic()
    copied_this_run = 0

    print(f"Resuming from offset: {offset:,} / {total:,}")

    while offset < total:
        batch_started = time.monotonic()
        limit = min(BATCH_SIZE, total - offset)

        data = retry_operation(
            lambda: old_col.get(
                limit=limit,
                offset=offset,
                include=["embeddings", "documents", "metadatas"],
            ),
            "reading batch from source",
        )

        ids = data["ids"]

        if not ids:
            raise RuntimeError(
                f"Source returned an empty batch at offset {offset}. "
                "Migration aborted to prevent skipping data."
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

        retry_operation(upsert_batch, "writing batch to target")

        result = retry_operation(
            lambda: new_col.get(
                ids=ids,
                include=["metadatas"],
            ),
            "verifying batch",
        )

        found_ids = set(result["ids"])
        missing_ids = [item for item in ids if item not in found_ids]

        if missing_ids:
            raise RuntimeError(
                f"Verification failed: "
                f"{len(missing_ids)} missing identifiers. "
                "The checkpoint was not advanced."
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
            f"\rProgress: {percent:6.2f}% | "
            f"{offset:,}/{total:,} | "
            f"{speed:,.0f} records/s | "
            f"batch: {batch_time:.1f} s | "
            f"remaining: {format_time(remaining)}",
            end="",
            flush=True,
        )

    print("\n\n[4/4] Final check...")

    target_count = new_col.count()

    print(f"Source: {total:,}")
    print(f"Target: {target_count:,}")

    if target_count < total:
        raise RuntimeError(
            "Record count in the target is less than the source. "
            "Do not remove the checkpoint; investigate the cause."
        )

    if target_count > total:
        print(
            "WARNING: the target contains more records than the source. "
            "Additional records were not removed."
        )

    os.remove(CHECKPOINT_FILE)

    print("\n" + "=" * 65)
    print("MIGRATION COMPLETED")
    print(f"Verified source identifiers: {total:,}")
    print(f"Time of this session: {format_time(time.monotonic() - started_at)}")
    print("All batches were saved and verified.")
    print("=" * 65)


if __name__ == "__main__":
    main()