"""Polymer data extraction from indexed papers.

Usage:
    from polyreader import load_index, extract, ExtractionConfig

    index = load_index("paper_tree.json")
    result = await extract(index)
    print(result.model_dump_json(indent=2))
"""

from .orchestrator import extract, save_samples
from .config import ExtractionConfig, PolyextractConfig
from .document_set import DocumentSet
from polyreader.schema.polymer import (
    PolymerSample,
    ExtractionResult,
    PropertyObject,
    Provenance,
    Monomer,
    MolecularWeight,
    SynthesisDetails,
    BlendComponent,
    ProcessingStep,
)

__all__ = [
    "extract",
    "save_samples",
    "ExtractionConfig",
    "PolyextractConfig",
    "DocumentSet",
    "PolymerSample",
    "ExtractionResult",
    "PropertyObject",
    "Provenance",
    "Monomer",
    "MolecularWeight",
    "SynthesisDetails",
    "BlendComponent",
    "ProcessingStep",
]
