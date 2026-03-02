"""RAG query engine: retrieve relevant chunks from ChromaDB → GPT-4o answer with citations."""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from chatbot.index import EMBEDDING_MODEL, get_collection

# --- Setup -------------------------------------------------------------------

load_dotenv()

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o"
TOP_K = 5

SYSTEM_PROMPT = """You are an insurance policy assistant. Your job is to answer questions \
accurately using ONLY the policy excerpts provided below.

Rules:
- Base your answer solely on the provided excerpts. Do not use outside knowledge.
- For every fact or clause you mention, cite the source document and page number like this: \
(Source: <document name>, Page <n>).
- If the excerpts do not contain enough information to answer, say: \
"I don't have enough information in the provided documents to answer that."
- Be clear, helpful, and concise."""


# --- Types -------------------------------------------------------------------

@dataclass
class Citation:
    source: str
    page: int
    box_left: float = 0.0
    box_top: float = 0.0
    box_right: float = 1.0
    box_bottom: float = 1.0


@dataclass
class RAGResponse:
    answer: str
    citations: list[Citation] = field(default_factory=list)


# --- Helpers -----------------------------------------------------------------

def _clean_markdown(text: str) -> str:
    """Strip HTML anchor tags injected by landingai-ade."""
    return re.sub(r"<a[^>]*></a>\s*", "", text).strip()


def _build_context(documents: list[str], metadatas: list[dict]) -> str:
    parts: list[str] = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        clean = _clean_markdown(doc)
        parts.append(f"[EXCERPT {i} — Source: {source}, Page {page}]\n{clean}")
    return "\n\n".join(parts)


def _extract_citations(metadatas: list[dict]) -> list[Citation]:
    seen: set[tuple[str, int]] = set()
    citations: list[Citation] = []
    for meta in metadatas:
        source = meta.get("source", "unknown")
        page = int(meta.get("page") or 0)
        key = (source, page)
        if key not in seen:
            seen.add(key)
            citations.append(Citation(
                source=source,
                page=page,
                box_left=float(meta.get("box_left") or 0.0),
                box_top=float(meta.get("box_top") or 0.0),
                box_right=float(meta.get("box_right") or 1.0),
                box_bottom=float(meta.get("box_bottom") or 1.0),
            ))
    return citations


# --- Core query function -----------------------------------------------------

def query(
    question: str,
    source_filter: str | None = None,
    top_k: int = TOP_K,
    chroma_dir: Path | None = None,
) -> RAGResponse:
    """Answer a question using RAG over the indexed insurance documents.

    Args:
        question: The user's natural language question.
        source_filter: If set, restrict retrieval to this source document name.
        top_k: Number of chunks to retrieve from ChromaDB.
        chroma_dir: Override the default ChromaDB directory (used in tests).
    """
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    kwargs = {"path": chroma_dir} if chroma_dir else {}
    collection = get_collection(**kwargs)

    if collection.count() == 0:
        log.warning("ChromaDB collection is empty — run chatbot/index.py first")
        return RAGResponse(answer="The document index is empty. Please run the indexing step first.")

    # --- Retrieve ------------------------------------------------------------
    embedding = (
        openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[question])
        .data[0]
        .embedding
    )

    query_kwargs: dict = {"query_embeddings": [embedding], "n_results": top_k}
    if source_filter:
        query_kwargs["where"] = {"source": source_filter}

    results = collection.query(**query_kwargs)

    documents: list[str] = results["documents"][0]
    metadatas: list[dict] = results["metadatas"][0]

    if not documents:
        return RAGResponse(
            answer="I couldn't find any relevant passages in the documents for your question."
        )

    log.debug("Retrieved %d chunks for query: %s", len(documents), question)

    # --- Generate ------------------------------------------------------------
    context = _build_context(documents, metadatas)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Policy excerpts:\n\n{context}\n\nQuestion: {question}",
        },
    ]

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )

    answer = response.choices[0].message.content or ""
    citations = _extract_citations(metadatas)

    log.info(
        "Query answered | model=%s | chunks_used=%d | answer_len=%d",
        model,
        len(documents),
        len(answer),
    )

    return RAGResponse(answer=answer, citations=citations)


# --- CLI test ----------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    test_questions = [
        "What is not covered under car insurance?",
        "What should I do if my home is flooded?",
        "Does travel insurance cover pre-existing medical conditions?",
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        result = query(question)
        print(f"\nA: {result.answer}")
        print("\nCitations:")
        for c in result.citations:
            print(f"  • {c.source}  (page {c.page})")


if __name__ == "__main__":
    main()
