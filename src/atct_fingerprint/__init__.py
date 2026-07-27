"""ATCT Narrative Fingerprint public API."""

from .encoders import SentenceTransformerEncoder, TfidfSentenceEncoder
from .features import FingerprintResult, analyze_text, compute_fingerprint, split_sentences
from .reporting import write_report_bundle

__all__ = [
    "FingerprintResult",
    "SentenceTransformerEncoder",
    "TfidfSentenceEncoder",
    "analyze_text",
    "compute_fingerprint",
    "split_sentences",
    "write_report_bundle",
]

__version__ = "0.2.0"
