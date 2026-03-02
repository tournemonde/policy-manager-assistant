"""Embed parsed chunks and upsert them into a ChromaDB vector store."""

import json
import logging
import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# --- Setup -------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/2_processed")
CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "insurance_policies"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100


# --- ChromaDB ----------------------------------------------------------------

def get_collection(chroma_dir: Path = CHROMA_DIR) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(chroma_dir))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# --- Embedding ---------------------------------------------------------------

def embed_texts(openai_client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


# --- Indexing ----------------------------------------------------------------

def index_chunks_file(
    chunks_path: Path,
    collection: chromadb.Collection,
    openai_client: OpenAI,
) -> int:
    """Parse a single *_chunks.json file and upsert its chunks into ChromaDB."""
    raw = json.loads(chunks_path.read_text())
    source_name = chunks_path.stem.replace("_chunks", "")

    valid_chunks = [c for c in raw if c.get("markdown", "").strip()]
    if not valid_chunks:
        log.warning("  No non-empty chunks found in: %s", chunks_path.name)
        return 0

    log.info(
        "  Indexing %d chunks from: %s",
        len(valid_chunks),
        chunks_path.name,
    )

    total_upserted = 0
    for i in range(0, len(valid_chunks), BATCH_SIZE):
        batch = valid_chunks[i : i + BATCH_SIZE]
        texts = [c["markdown"] for c in batch]
        embeddings = embed_texts(openai_client, texts)

        grounding_list = [c.get("grounding") or {} for c in batch]

        collection.upsert(
            ids=[c["id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "source": source_name,
                    "page": int(g.get("page") or 0),
                    "chunk_type": c.get("type") or "",
                    "box_left": float((g.get("box") or {}).get("left") or 0.0),
                    "box_top": float((g.get("box") or {}).get("top") or 0.0),
                    "box_right": float((g.get("box") or {}).get("right") or 1.0),
                    "box_bottom": float((g.get("box") or {}).get("bottom") or 1.0),
                }
                for c, g in zip(batch, grounding_list)
            ],
        )
        total_upserted += len(batch)
        log.info(
            "    Batch %d-%d upserted (%d total)",
            i + 1,
            i + len(batch),
            total_upserted,
        )

    return total_upserted


def index_all(processed_dir: Path = PROCESSED_DIR) -> None:
    """Index all *_chunks.json files found in processed_dir."""
    chunk_files = sorted(processed_dir.glob("*_chunks.json"))
    if not chunk_files:
        log.error("No chunk files found in %s", processed_dir)
        sys.exit(1)

    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    collection = get_collection()

    log.info("Found %d chunk file(s) to index", len(chunk_files))
    total = 0
    for path in chunk_files:
        total += index_chunks_file(path, collection, openai_client)

    count = collection.count()
    log.info("Done. Total upserted this run: %d | Collection size: %d", total, count)


# --- Main --------------------------------------------------------------------

def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY not set in .env")
        sys.exit(1)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    index_all()


if __name__ == "__main__":
    main()
