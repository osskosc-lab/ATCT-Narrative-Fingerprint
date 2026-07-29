"""ATCT Narrative Fingerprint public API."""

from .closure_states import ClosureAnalysis, analyze_closure
from .directionality import DirectionalityResult, conditional_directionality
from .document import ParsedDocument, parse_markdown_document
from .encoders import SentenceTransformerEncoder, TfidfSentenceEncoder
from .features import (
    FingerprintResult,
    analyze_text,
    compute_fingerprint,
    split_sentences,
)
from .motif_functions import MotifFunctionAnalysis, analyze_motif_functions
from .relations import RelationalAnalysis, analyze_relations
from .reporting import write_report_bundle

__all__ = [
    "FingerprintResult",
    "ClosureAnalysis",
    "DirectionalityResult",
    "MotifFunctionAnalysis",
    "ParsedDocument",
    "RelationalAnalysis",
    "SentenceTransformerEncoder",
    "TfidfSentenceEncoder",
    "analyze_text",
    "analyze_closure",
    "analyze_motif_functions",
    "analyze_relations",
    "conditional_directionality",
    "compute_fingerprint",
    "split_sentences",
    "parse_markdown_document",
    "write_report_bundle",
]

__version__ = "0.5.0"
