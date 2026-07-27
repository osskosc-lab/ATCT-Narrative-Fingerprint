"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .encoders import SentenceTransformerEncoder, TfidfSentenceEncoder
from .evaluation import build_encoder, evaluate_rows, load_dataset
from .features import analyze_text


def _encoder(name: str):
    if name == "e5":
        return SentenceTransformerEncoder()
    return TfidfSentenceEncoder()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atct-fingerprint",
        description="Measure order-dependent narrative structure.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="analyze one UTF-8 document")
    analyze.add_argument("path", type=Path)
    analyze.add_argument("--encoder", choices=("tfidf", "e5"), default="tfidf")
    analyze.add_argument("--seed", type=int, default=42)
    analyze.add_argument("--shuffles", type=int, default=32)

    evaluate = subparsers.add_parser(
        "evaluate", help="run the explicit train/test falsification protocol"
    )
    evaluate.add_argument("dataset", type=Path)
    evaluate.add_argument("--encoder", choices=("tfidf", "e5"), default="tfidf")
    evaluate.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "analyze":
        text = args.path.read_text(encoding="utf-8")
        result = analyze_text(
            text,
            encoder=_encoder(args.encoder),
            seed=args.seed,
            shuffle_count=args.shuffles,
        )
        payload = result.to_dict()
        payload["interpretation"] = (
            "Descriptive structure only; not an AI-authorship probability."
        )
    else:
        rows = load_dataset(args.dataset)
        payload = evaluate_rows(
            rows,
            encoder=build_encoder(args.encoder),
            seed=args.seed,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
