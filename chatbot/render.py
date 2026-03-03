"""Render a PDF page as a PNG image with a highlighted bounding box."""

from pathlib import Path

import fitz  # pymupdf

# --- Constants ---------------------------------------------------------------

HIGHLIGHT_COLOR = (1.0, 0.9, 0.0)  # yellow
HIGHLIGHT_OPACITY = 0.35
BORDER_COLOR = (0.9, 0.6, 0.0)     # amber border
RENDER_ZOOM = 2.0                   # 2× for crisp display in Streamlit


# --- Core render function ----------------------------------------------------

def render_page_with_highlight(
    pdf_path: Path,
    page_number: int,
    box_left: float,
    box_top: float,
    box_right: float,
    box_bottom: float,
    zoom: float = RENDER_ZOOM,
) -> bytes:
    """Render a single PDF page as PNG with a highlighted region.

    Args:
        pdf_path: Path to the source PDF file.
        page_number: 0-indexed page number to render.
        box_left/top/right/bottom: Normalised (0–1) bounding box coordinates.
        zoom: Render scale factor — higher = sharper but larger image.

    Returns:
        PNG image bytes ready for st.image().
    """
    doc = fitz.open(str(pdf_path))

    page_count = len(doc)
    page_idx = max(0, min(page_number, page_count - 1))
    page = doc[page_idx]

    w = page.rect.width
    h = page.rect.height

    highlight_rect = fitz.Rect(
        box_left * w,
        box_top * h,
        box_right * w,
        box_bottom * h,
    )

    # Semi-transparent fill
    page.draw_rect(
        highlight_rect,
        color=BORDER_COLOR,
        fill=HIGHLIGHT_COLOR,
        fill_opacity=HIGHLIGHT_OPACITY,
        width=1.5,
    )

    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    doc.close()
    return pix.tobytes("png")


def pdf_page_count(pdf_path: Path) -> int:
    doc = fitz.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count
