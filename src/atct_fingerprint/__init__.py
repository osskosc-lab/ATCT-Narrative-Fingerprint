"""ATCT Narrative Fingerprint public API."""

from .directionality import DirectionalityResult, conditional_directionality
from .document import ParsedDocument, parse_markdown_document
from .encoders import SentenceTransformerEncoder, TfidfSentenceEncoder
from .features import FingerprintResult, analyze_text, compute_fingerprint, split_sentences
from .reporting import write_report_bundle

__all__ = [
    "FingerprintResult",
    "DirectionalityResult",
    "ParsedDocument",
    "SentenceTransformerEncoder",
    "TfidfSentenceEncoder",
    "analyze_text",
    "conditional_directionality",
    "compute_fingerprint",
    "split_sentences",
    "parse_markdown_document",
    "write_report_bundle",
]

__version__ = "0.4.0"
