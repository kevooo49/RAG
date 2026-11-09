import argparse
import os
import json
import sys
import time
from pathlib import Path
import requests
import fitz
from dotenv import load_dotenv

try:
    from docx2pdf import convert
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    print("docx2pdf is missing. .docx files will be skipped. (Install: pip install docx2pdf)")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from sites import ConfigError, load_or_exit  # noqa: E402 # type: ignore

load_dotenv()
API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")


def requires_advanced_parsing(pdf_path: str) -> bool:
    try:
        with fitz.open(pdf_path) as document:
            for page in document:
                text = page.get_text("text").strip()
                if len(text) < 50:
                    return True

                images = page.get_images()
                if len(images) > 2:
                    return True

                tables = page.find_tables()
                if tables and len(tables.tables) > 0:
                    return True
        return False
    except Exception as e:
        print(f"Local analysis error for {pdf_path}: {e}. Forcing LlamaParse.")
        return True


def extract_text_locally(pdf_path: str) -> str:
    full_text = ""
    with fitz.open(pdf_path) as document:
        for page in document:
            full_text += page.get_text("text") + "\n\n"
    return full_text


def add_frontmatter(text: str, original_name: str, method: str, extension: str = "pdf", source: str = "") -> str:
    frontmatter = f"""---
plik: "{original_name}"
typ_dokumentu: {extension}
zrodlo: {source}
metoda_ekstrakcji: {method}
---

"""
    return frontmatter + text


def run_llamaparse_v2(file_path: str, api_key: str) -> str:
    if not api_key:
        raise Exception("Missing LLAMA_CLOUD_API_KEY")

    upload_url = "https://api.cloud.llamaindex.ai/api/v2/parse/upload"
    headers = {"Authorization": f"Bearer {api_key}"}

    with open(file_path, "rb") as f:
        config = {"tier": "agentic", "version": "latest"}
        response = requests.post(
            upload_url,
            headers=headers,
            files={"file": f},
            data={"configuration": json.dumps(config)},
            timeout=30
        )

    if not response.ok:
        raise Exception(f"API v2 connection error: {response.text}")

    job_id = response.json()["id"]
    status_url = f"https://api.cloud.llamaindex.ai/api/v2/parse/{job_id}?expand=markdown_full"

    max_attempts = 60
    for _ in range(max_attempts):
        res = requests.get(status_url, headers=headers, timeout=20)
        if not res.ok:
            raise Exception(f"Status check error: {res.text}")

        data = res.json()
        status = data["job"]["status"]

        if status == "COMPLETED":
            return data.get("markdown_full", "")
        elif status in ["FAILED", "CANCELLED"]:
            error_message = data["job"].get("error_message", "Unknown error")
            raise Exception(f"Job rejected by the server: {error_message}")

        time.sleep(2)

    raise Exception("LlamaParse job timeout: the process took too long.")


def process_directory(raw_folder: str, processed_folder: str, temporary_folder: str, source_domain: str):
    os.makedirs(processed_folder, exist_ok=True)
    os.makedirs(temporary_folder, exist_ok=True)

    print("Starting Advanced Parser (PyMuPDF + LlamaParse) with DOCX isolation...")
    local_count, cloud_count, docx_count = 0, 0, 0

    for file_name in os.listdir(raw_folder):
        raw_path = os.path.join(raw_folder, file_name)
        if not os.path.isfile(raw_path):
            continue

        name_without_extension, current_extension = os.path.splitext(file_name)
        current_extension = current_extension.lower().replace(".", "")
        output_path = os.path.join(processed_folder, f"{name_without_extension}.md")

        if current_extension not in ["pdf", "docx"]:
            print(f"Skipped {file_name} - unsupported format.")
            continue

        if os.path.exists(output_path) and os.path.getsize(output_path) > 150:
            continue

        path_to_analyze = raw_path
        is_temporary_file = False

        if current_extension == "docx" and DOCX_SUPPORT:
            temporary_path = os.path.join(temporary_folder, f"{name_without_extension}.pdf")
            print(f"Normalizing: {file_name} -> PDF")
            try:
                convert(raw_path, temporary_path)
                path_to_analyze = temporary_path
                is_temporary_file = True
                docx_count += 1
            except Exception as e:
                print(f"Word conversion error: {e}")
                continue

        if path_to_analyze.lower().endswith(".pdf"):
            print(f"📄 Analyzing: {os.path.basename(raw_path)}...")

            if requires_advanced_parsing(path_to_analyze):
                print("   ☁️ Sending to LlamaParse...")
                try:
                    raw_text = run_llamaparse_v2(path_to_analyze, API_KEY)
                    final_text = add_frontmatter(raw_text, file_name, "llamaparse", current_extension, source_domain)
                    cloud_count += 1
                except Exception as e:
                    print(f"   LlamaParse error: {e}")
                    if is_temporary_file and os.path.exists(path_to_analyze):
                        os.remove(path_to_analyze)
                    continue
            else:
                print("   Processing locally...")
                try:
                    raw_text = extract_text_locally(path_to_analyze)
                    final_text = add_frontmatter(raw_text, file_name, "pymupdf", current_extension, source_domain)
                    local_count += 1
                except Exception as e:
                    print(f"   Local extraction error: {e}")
                    if is_temporary_file and os.path.exists(path_to_analyze):
                        os.remove(path_to_analyze)
                    continue

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_text)

            if is_temporary_file and os.path.exists(path_to_analyze):
                os.remove(path_to_analyze)

    print("\nREPORT:")
    print(f"DOCX->PDF: {docx_count} | Local (Free): {local_count} | AI Cloud: {cloud_count}")


def main():
    ap = argparse.ArgumentParser(
        description="Raw PDF/DOCX -> Markdown (PyMuPDF locally, LlamaParse for complex files)",
        epilog="Folders come from config/sites.yaml (paths.data_root + layout). Any of them can be overridden below.")
    ap.add_argument("site", nargs="?", help="site key from config/sites.yaml, e.g. 'cok'")
    ap.add_argument("--raw", help="folder with downloaded PDF/DOCX files")
    ap.add_argument("--processed", help="output folder for .md files")
    ap.add_argument("--temp", help="scratch folder for DOCX->PDF conversion")
    ap.add_argument("--domain", help="source URL written to the frontmatter")
    ap.add_argument("--config", help="alternative sites.yaml")
    a = ap.parse_args()

    dirs = {"raw": a.raw, "processed": a.processed, "temp": a.temp, "source_domain": a.domain}
    if a.site:
        sites = load_or_exit(a.config)
        try:
            defaults = sites.document_dirs(a.site)
        except ConfigError as e:
            sys.exit(f"Configuration error: {e}")
        dirs = {k: (v if v else defaults[k]) for k, v in dirs.items()}
    missing = [k for k, v in dirs.items() if not v and k != "source_domain"]
    if missing:
        ap.error(f"give a site key (python document_parser.py cok) or --{' --'.join(missing)}")

    if not os.path.isdir(dirs["raw"]):
        sys.exit(f"Raw folder does not exist: {dirs['raw']}")
    if not API_KEY:
        print("Warning: LLAMA_CLOUD_API_KEY is not set - files needing LlamaParse will fail.")
    process_directory(raw_folder=str(dirs["raw"]), processed_folder=str(dirs["processed"]),
                      temporary_folder=str(dirs["temp"]), source_domain=dirs["source_domain"] or "")


if __name__ == "__main__":
    main()
