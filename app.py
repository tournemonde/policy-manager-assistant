"""Streamlit UI — Policy Manager Assistant chatbot."""

import logging
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from chatbot.index import get_collection, index_chunks_file
from chatbot.ingest import parse_pdf
from chatbot.rag import Citation, RAGResponse, query
from chatbot.render import render_page_with_highlight
_archive_uploaded_artifacts = None
_storage_import_error: Exception | None = None
try:
    from chatbot.storage import archive_uploaded_artifacts as _archive_uploaded_artifacts
except Exception as e:  # pragma: no cover - import fallback for local/dev environments
    _storage_import_error = e

# --- Setup -------------------------------------------------------------------

load_dotenv()
logging.basicConfig(level=logging.WARNING)

RAW_DIR = Path("data/1_raw")
PROCESSED_DIR = Path("data/2_processed")

# --- Page config -------------------------------------------------------------

st.set_page_config(
    page_title="Policy Manager Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    div[data-testid="stForm"] {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid transparent !important;
        border-radius: 0.75rem !important;
        padding: 0.20rem 0.30rem !important;
    }
    div[data-testid="stForm"]:focus-within {
        border-color: #1e3a8a !important;
        box-shadow: 0 0 0 0.14rem rgba(30, 58, 138, 0.24) !important;
    }
    div[data-testid="stTextInput"] input {
        border: 1px solid transparent !important;
        border-radius: 0.6rem !important;
        box-shadow: none !important;
        background-color: transparent !important;
        outline: none !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }
    div[data-testid="stTextInput"] > div:focus-within {
        border-color: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }
    div[data-testid="stFormSubmitButton"] button {
        background-color: #111111 !important;
        border-color: #111111 !important;
        color: #ffffff !important;
        border-radius: 999px !important;
        min-height: 2.35rem !important;
        min-width: 2.35rem !important;
        font-size: 1.10rem !important;
        padding: 0 !important;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: #000000 !important;
        border-color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Helpers -----------------------------------------------------------------

def _source_to_pdf(source: str) -> Path | None:
    candidate = RAW_DIR / f"{source}.pdf"
    return candidate if candidate.exists() else None


def _available_docs() -> list[str]:
    return [p.stem for p in sorted(RAW_DIR.glob("*.pdf"))]


def _archive_upload_if_enabled(pdf_path: Path, chunks_path: Path) -> None:
    """Archive uploads to GCS in deployed environments; no-op in local dev."""
    if not os.getenv("GCS_BUCKET_NAME"):
        return

    if _archive_uploaded_artifacts is None:
        raise RuntimeError(
            "GCS archive requested but storage module is unavailable: "
            f"{_storage_import_error}"
        )

    _archive_uploaded_artifacts(pdf_path, chunks_path)


def _render_citation_expanders(citations: list[Citation]) -> None:
    """Integrated mode: collapsible expander per citation."""
    preview_pad_left, preview_col, preview_pad_right = st.columns([0.08, 0.84, 0.08], gap="small")
    for c in citations:
        pdf_path = _source_to_pdf(c.source)
        if pdf_path is None:
            continue
        label = f"📄 View in document — {c.source} · page {c.page + 1}"
        with preview_col:
            with st.expander(label, expanded=False):
                try:
                    img = render_page_with_highlight(
                        pdf_path, c.page,
                        c.box_left, c.box_top, c.box_right, c.box_bottom,
                    )
                    st.image(img, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render page: {e}")


def _render_citation_labels(citations: list[Citation]) -> None:
    """Side-by-side mode: small caption under the answer."""
    if citations:
        labels = "  ·  ".join(f"{c.source} p.{c.page + 1}" for c in citations)
        st.caption(f"📄 {labels}")


def _render_doc_panel(citations: list[Citation], nav_index: int) -> None:
    """Side-by-side mode: right panel with page navigator."""
    st.subheader("📄 Source document")

    if not citations:
        st.caption("Ask a question — the relevant document page will appear here.")
        return

    total = len(citations)
    c = citations[nav_index]
    pdf_path = _source_to_pdf(c.source)

    # --- Navigation bar
    nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
    with nav_col1:
        if st.button("◀", disabled=(nav_index == 0), use_container_width=True):
            st.session_state.citation_nav_index -= 1
            st.rerun()
    with nav_col2:
        st.markdown(
            f"<div style='text-align:center; padding-top:6px; font-size:0.85rem;'>"
            f"<b>{c.source}</b> · page {c.page + 1}"
            f"<br><span style='color:grey'>({nav_index + 1} of {total})</span></div>",
            unsafe_allow_html=True,
        )
    with nav_col3:
        if st.button("▶", disabled=(nav_index == total - 1), use_container_width=True):
            st.session_state.citation_nav_index += 1
            st.rerun()

    # --- Rendered page
    if pdf_path is None:
        st.warning(f"PDF not found: {c.source}.pdf")
        return

    try:
        img = render_page_with_highlight(
            pdf_path, c.page,
            c.box_left, c.box_top, c.box_right, c.box_bottom,
        )
        st.image(img, width=900)
    except Exception as e:
        st.warning(f"Could not render page: {e}")


def _process_prompt(prompt: str, source_filter: str | None) -> None:
    """Run one user query through RAG and update session state."""
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        with st.spinner("Searching documents…"):
            result: RAGResponse = query(prompt, source_filter=source_filter)

        st.session_state.current_citations = result.citations
        st.session_state.citation_nav_index = 0

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "citations": result.citations,
            }
        )
    except Exception as e:
        st.error(f"Something went wrong while searching: {e}")

    st.rerun()


# --- Upload confirmation dialog ----------------------------------------------

@st.dialog("Document ready")
def _upload_dialog(doc_name: str) -> None:
    if st.session_state.upload_was_new:
        st.success(f"**{doc_name}** has been parsed and indexed successfully.")
    else:
        st.info(f"**{doc_name}** was already indexed and is ready to use.")
    st.markdown(
        "Would you like to switch to this document now and start a fresh conversation?"
    )
    st.caption(
        "You can also keep your current session — "
        "the document will appear in the scope dropdown whenever you need it."
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, switch to it", use_container_width=True, type="primary"):
            st.session_state.doc_filter = doc_name
            st.session_state.messages = []
            st.session_state.current_citations = []
            st.session_state.citation_nav_index = 0
            st.session_state.pending_upload_doc = None
            st.rerun()
    with col2:
        if st.button("No, keep current session", use_container_width=True):
            st.session_state.pending_upload_doc = None
            st.rerun()


# --- Session state -----------------------------------------------------------

def _init_state() -> None:
    defaults: dict = {
        "messages": [],
        "current_citations": [],
        "citation_nav_index": 0,
        "doc_filter": "All documents",
        "layout": "Integrated",
        "pending_upload_doc": None,
        "upload_was_new": False,
        "processed_uploads": set(),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()

# Trigger upload dialog if a new document was just indexed
if st.session_state.pending_upload_doc:
    _upload_dialog(st.session_state.pending_upload_doc)


# --- Sidebar -----------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    # --- Layout toggle
    new_layout = st.radio(
        "View mode",
        ["Integrated", "Side by side"],
        index=["Integrated", "Side by side"].index(st.session_state.layout),
        horizontal=True,
    )
    if new_layout != st.session_state.layout:
        st.session_state.layout = new_layout
        st.rerun()

    st.divider()

    # --- Document scope — searchable selectbox
    st.subheader("📂 Documents")
    doc_options = ["All documents"] + _available_docs()

    current_idx = (
        doc_options.index(st.session_state.doc_filter)
        if st.session_state.doc_filter in doc_options
        else 0
    )
    selected_doc = st.selectbox(
        "Search in",
        options=doc_options,
        index=current_idx,
        placeholder="Type to search…",
    )

    if selected_doc != st.session_state.doc_filter:
        st.session_state.doc_filter = selected_doc
        st.session_state.messages = []
        st.session_state.current_citations = []
        st.session_state.citation_nav_index = 0
        st.rerun()

    st.divider()

    # --- Upload
    st.subheader("📤 Upload a policy")
    uploaded_file = st.file_uploader(
        "Upload a PDF", type="pdf", label_visibility="collapsed"
    )

    if (
        uploaded_file is not None
        and uploaded_file.name not in st.session_state.processed_uploads
    ):
        dest_path = RAW_DIR / uploaded_file.name
        chunks_path = PROCESSED_DIR / f"{dest_path.stem}_chunks.json"

        # Always save the PDF — it may have been deleted while chunks remain
        dest_path.write_bytes(uploaded_file.getvalue())

        if not chunks_path.exists():
            with st.spinner("Parsing and indexing your document…"):
                from landingai_ade import LandingAIADE
                ade_client = LandingAIADE(apikey=os.environ["LANDINGAI_API_KEY"])
                parse_pdf(ade_client, dest_path, PROCESSED_DIR)
                openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                collection = get_collection()
                index_chunks_file(chunks_path, collection, openai_client)
            st.session_state.upload_was_new = True
        else:
            st.session_state.upload_was_new = False

        # Persist user uploads to archive path in GCS (separate from baseline data).
        try:
            _archive_upload_if_enabled(dest_path, chunks_path)
        except Exception as e:
            st.warning(f"Upload archived locally, but cloud archive failed: {e}")

        # Mark as processed so reruns don't re-trigger this block
        st.session_state.processed_uploads.add(uploaded_file.name)
        st.session_state.pending_upload_doc = dest_path.stem
        st.rerun()

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_citations = []
        st.session_state.citation_nav_index = 0
        st.rerun()


# --- Derive active settings --------------------------------------------------

source_filter: str | None = (
    None
    if st.session_state.doc_filter == "All documents"
    else st.session_state.doc_filter
)
layout: str = st.session_state.layout

# If current_citations is empty (e.g. user just switched layout modes),
# recover them from the last assistant message so the doc panel isn't blank.
if not st.session_state.current_citations:
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant" and msg.get("citations"):
            st.session_state.current_citations = msg["citations"]
            break

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

if source_filter:
    st.info(f"📂 Searching in: **{source_filter}**  —  change scope in the sidebar")

st.divider()

# --- Render chat history -----------------------------------------------------
submitted_prompt: str | None = None

if layout == "Side by side":
    col_chat, col_doc = st.columns([0.98, 1.02], gap="large")

    with col_chat:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    _render_citation_labels(msg.get("citations", []))

        with st.form("chat_form_side_by_side", clear_on_submit=True):
            input_col, button_col = st.columns([0.955, 0.045], gap="small")
            with input_col:
                prompt_text = st.text_input(
                    "Ask about your insurance policy…",
                    placeholder="Ask about your insurance policy…",
                    label_visibility="collapsed",
                )
            with button_col:
                send = st.form_submit_button("↑", use_container_width=True, type="primary")
            if send and prompt_text.strip():
                submitted_prompt = prompt_text.strip()

    with col_doc:
        _render_doc_panel(
            st.session_state.current_citations,
            st.session_state.citation_nav_index,
        )

else:
    pad_left, col_center, pad_right = st.columns([0.16, 0.68, 0.16], gap="small")
    with col_center:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    _render_citation_expanders(msg.get("citations", []))

        with st.form("chat_form_integrated", clear_on_submit=True):
            input_col, button_col = st.columns([0.955, 0.045], gap="small")
            with input_col:
                prompt_text = st.text_input(
                    "Ask about your insurance policy…",
                    placeholder="Ask about your insurance policy…",
                    label_visibility="collapsed",
                )
            with button_col:
                send = st.form_submit_button("↑", use_container_width=True, type="primary")
            if send and prompt_text.strip():
                submitted_prompt = prompt_text.strip()

# --- Handle new input --------------------------------------------------------
if submitted_prompt:
    _process_prompt(submitted_prompt, source_filter)
