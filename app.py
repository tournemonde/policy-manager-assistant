"""Streamlit UI — Policy Manager Assistant chatbot."""

import logging
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from chatbot.index import get_collection, index_chunks_file
from chatbot.ingest import parse_pdf
from chatbot.rag import RAGResponse, Citation, query
from chatbot.render import render_page_with_highlight

# --- Setup -------------------------------------------------------------------

load_dotenv()

logging.basicConfig(level=logging.WARNING)

RAW_DIR = Path("data/1_raw")
PROCESSED_DIR = Path("data/2_processed")

# --- Page config -------------------------------------------------------------

st.set_page_config(
    page_title="Policy Manager Assistant",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)


# --- Header ------------------------------------------------------------------

st.title("🛡️ Policy Manager Assistant")
st.markdown(
    """
    The chatbot helps users understand, update, or renew their insurance policies.
    It can answer questions about **policy terms**, **coverage**, and **exclusions**
    by searching company documents.
    It can also guide users through **policy changes** or **renewal steps**.
    """
)
st.divider()


# --- Helpers -----------------------------------------------------------------

def _source_to_pdf(source: str) -> Path | None:
    """Map a ChromaDB source name back to its PDF file in data/1_raw/."""
    candidate = RAW_DIR / f"{source}.pdf"
    return candidate if candidate.exists() else None


def _display_citations(citations: list[Citation]) -> None:
    """Render source preview expanders for each citation."""
    if not citations:
        return

    for c in citations:
        pdf_path = _source_to_pdf(c.source)
        if pdf_path is None:
            continue

        label = f"📄 View in document — {c.source} · page {c.page + 1}"
        with st.expander(label, expanded=False):
            try:
                img_bytes = render_page_with_highlight(
                    pdf_path=pdf_path,
                    page_number=c.page,
                    box_left=c.box_left,
                    box_top=c.box_top,
                    box_right=c.box_right,
                    box_bottom=c.box_bottom,
                )
                st.image(img_bytes, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render page: {e}")


# --- Sidebar — document upload -----------------------------------------------

with st.sidebar:
    st.header("📂 Documents")
    st.caption("Pre-loaded policies:")
    for pdf in sorted(RAW_DIR.glob("*.pdf")):
        st.markdown(f"• {pdf.stem}")

    st.divider()
    st.subheader("Upload your own policy")
    uploaded_file = st.file_uploader(
        "Upload a PDF", type="pdf", label_visibility="collapsed"
    )

    if uploaded_file is not None:
        dest_path = RAW_DIR / uploaded_file.name
        chunks_path = PROCESSED_DIR / f"{dest_path.stem}_chunks.json"

        if not chunks_path.exists():
            with st.spinner("Parsing and indexing your document…"):
                dest_path.write_bytes(uploaded_file.getvalue())

                from landingai_ade import LandingAIADE
                ade_client = LandingAIADE(
                    apikey=os.environ["LANDINGAI_API_KEY"]
                )
                parse_pdf(ade_client, dest_path, PROCESSED_DIR)

                openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                collection = get_collection()
                index_chunks_file(chunks_path, collection, openai_client)

            st.success(f"✅ '{uploaded_file.name}' indexed and ready!")
        else:
            st.info(f"✅ '{uploaded_file.name}' already indexed.")

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --- Chat session state ------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Render history ----------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _display_citations(msg.get("citations", []))

# --- Handle new input --------------------------------------------------------

if prompt := st.chat_input("Ask about your insurance policy…"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            result: RAGResponse = query(prompt)

        st.markdown(result.answer)
        _display_citations(result.citations)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "citations": result.citations,
        }
    )
