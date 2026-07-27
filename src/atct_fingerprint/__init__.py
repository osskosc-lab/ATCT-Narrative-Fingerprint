"""ATCT Narrative Fingerprint public API."""

from .encoders import SentenceTransformerEncoder, TfidfSentenceEncoder
from .features import FingerprintResult, analyze_text, compute_fingerprint, split_sentences

__all__ = [
    "FingerprintResult",
    "SentenceTransformerEncoder",
    "TfidfSentenceEncoder",
    "analyze_text",
    "compute_fingerprint",
    "split_sentences",
]

__version__ = "0.1.0"
