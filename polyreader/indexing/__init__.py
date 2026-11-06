"""Paper indexing: parse a PDF, build the section tree, summarise nodes, extract metadata.

Public API:
    process_document(pdf_path, **kwargs) -> PaperIndex
    save_index(index, path) / load_index(path) / verify_index(path)
    print_tree(tree)

PaperIndex methods (the navigation tools used by extraction):
    get_page_content(doc, pages), get_page_image(doc, img_path), get_document_structure(doc)

Acknowledgement:
    The reasoning-based, vectorless index (section tree with node summaries, and the
    get_document_structure / get_page_content navigation tools) builds on
    PageIndex by VectifyAI (https://github.com/VectifyAI/PageIndex, MIT License;
    see THIRD_PARTY_NOTICES).
"""

from .paper_index import (
    IndexLoadError,
    PaperIndex,
    load_index,
    print_tree,
    process_document,
    save_index,
    verify_index,
)

__all__ = [
    "IndexLoadError",
    "PaperIndex",
    "load_index",
    "print_tree",
    "process_document",
    "save_index",
    "verify_index",
]
