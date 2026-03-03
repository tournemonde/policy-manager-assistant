"""Document ingestion pipeline: parse PDFs with landingai-ade → JSON chunks."""

import json
import logging
import os
import sys
from pathlib import Path

import landingai_ade
from dotenv import load_dotenv
from landingai_ade import LandingAIADE

# --- Setup -------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAW_DIR = Path("data/1_raw")
OUT_DIR = Path("data/2_processed")
MODEL = "dpt-2-latest"


# --- Helpers -----------------------------------------------------------------

def validate_env() -> str:
    api_key = os.getenv("LANDINGAI_API_KEY")
    if not api_key:
        log.error("LANDINGAI_API_KEY not set in .env")
        sys.exit(1)
    return api_key


def parse_pdf(client: LandingAIADE, pdf_path: Path, out_dir: Path) -> int:
    out_path = out_dir / f"{pdf_path.stem}_chunks.json"

    if out_path.exists():
        log.info("  Skipping (already parsed): %s", out_path.name)
        return 0

    log.info("  Parsing: %s (%.1f KB)", pdf_path.name, pdf_path.stat().st_size / 1024)

    response = client.parse(document=pdf_path, model=MODEL)

    chunks = [chunk.model_dump() for chunk in response.chunks]

    out_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    log.info("  Saved %d chunks → %s", len(chunks), out_path.name)
    return len(chunks)


# --- Main --------------------------------------------------------------------

def main() -> None:
    log.info("landingai-ade version: %s", landingai_ade.__version__)

    api_key = validate_env()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        log.error("No PDF files found in %s", RAW_DIR)
        sys.exit(1)

    log.info("Found %d PDF(s) to process", len(pdf_files))

    client = LandingAIADE(apikey=api_key)

    total_chunks = 0
    for pdf_path in pdf_files:
        try:
            total_chunks += parse_pdf(client, pdf_path, OUT_DIR)
        except landingai_ade.APIStatusError as e:
            log.error("API error on %s: status=%s — %s", pdf_path.name, e.status_code, e.response)
        except landingai_ade.APIConnectionError as e:
            log.error("Connection error on %s: %s", pdf_path.name, e)

    log.info("Done. Total chunks across all documents: %d", total_chunks)


if __name__ == "__main__":
    main()
