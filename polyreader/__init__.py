"""polyreader: structured polymer data extraction from scientific papers.

Pipeline: PDF → ``parsing`` (Marker or MinerU) → ``indexing`` (section tree,
summaries, metadata) → ``extraction`` (entity discovery, per-sample or
full-context extraction, validation) → ``schema`` records.

    import asyncio
    from polyreader import process_document, save_index, load_index, extract, ExtractionConfig

    index = asyncio.run(process_document("paper.pdf"))
    save_index(index, "output/paper.json")
    result = asyncio.run(extract(load_index("output/paper_tree.json")))
"""

from .indexing import (
    IndexLoadError,
    PaperIndex,
    load_index,
    print_tree,
    process_document,
    save_index,
    verify_index,
)
from .extraction import (
    DocumentSet,
    ExtractionConfig,
    PolyextractConfig,
    extract,
    save_samples,
)
from .schema import (
    BlendComponent,
    ExtractionResult,
    Monomer,
    MolecularWeight,
    PaperMetadataRef,
    ProcessingStep,
    PolymerSample,
    PropertyObject,
    Provenance,
    SynthesisDetails,
)

__all__ = [
    # indexing
    "IndexLoadError", "PaperIndex", "load_index", "print_tree", "process_document",
    "save_index", "verify_index",
    # extraction
    "DocumentSet", "ExtractionConfig", "PolyextractConfig", "extract", "save_samples",
    # schema
    "BlendComponent", "ExtractionResult", "Monomer", "MolecularWeight", "PaperMetadataRef",
    "ProcessingStep", "PolymerSample", "PropertyObject", "Provenance", "SynthesisDetails",
]
