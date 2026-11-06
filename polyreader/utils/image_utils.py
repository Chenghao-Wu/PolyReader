"""Image extraction and base64 encoding utilities."""

import base64
from pathlib import Path


def image_to_base64(image_path: str) -> str:
    """Read an image file and return base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_page_image(
    pdf_path: str, page_num: int, dpi: int = 150
) -> str:
    """Render a PDF page as a PNG image and return the base64 string.

    Args:
        pdf_path: Path to the PDF file
        page_num: 1-indexed page number
        dpi: Resolution for rendering

    Returns:
        Base64-encoded PNG image string
    """
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()
    return base64.b64encode(img_bytes).decode("utf-8")


def extract_page_images_range(
    pdf_path: str, start_page: int, end_page: int, dpi: int = 150
) -> list[str]:
    """Render multiple PDF pages as base64 PNG images.

    Args:
        pdf_path: Path to the PDF
        start_page: 1-indexed start page
        end_page: 1-indexed end page (inclusive)
        dpi: Resolution

    Returns:
        List of base64-encoded PNG strings
    """
    import fitz

    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    images = []
    for i in range(start_page - 1, min(end_page, len(doc))):
        pix = doc[i].get_pixmap(matrix=mat)
        images.append(base64.b64encode(pix.tobytes("png")).decode("utf-8"))
    doc.close()
    return images
