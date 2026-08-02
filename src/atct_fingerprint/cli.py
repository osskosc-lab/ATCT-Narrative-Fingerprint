"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .encoders import SentenceTransformerEncoder, TfidfSentenceEncoder
from .evaluation import build_encoder, evaluate_rows, load_dataset
from .features import analyze_text
from .null_models import CONTROL_NAMES
from .reporting import write_report_bundle


def _encoder(name: str):
    if name == "e5":
        return SentenceTransformerEncoder()
    return TfidfSentenceEncoder()


def _controls(value: str) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(selected).difference(CONTROL_NAMES)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown controls: {sorted(unknown)}")
    if "random" not in selected:
        raise argparse.ArgumentTypeError("controls must include random")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atct-fingerprint",
        description="Measure statistical evidence for order-conditioned coherence.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="analyze one UTF-8 document")
    analyze.add_argument("path", type=Path)
    analyze.add_argument(
        "--encoder", choices=("tfidf", "e5", "both"), default="tfidf"
    )
    analyze.add_argument("--seed", type=int, default=42)
    analyze.add_argument("--shuffles", type=int, default=200)
    analyze.add_argument(
        "--controls",
        type=_controls,
        default=CONTROL_NAMES,
        help=(
            "comma list: local,paragraph_inner,block,paragraph_order,"
            "section_order,random,reverse"
        ),
    )
    analyze.add_argument(
        "--plain-text",
        action="store_true",
        help="disable Markdown layer separation",
    )
    analyze.add_argument("--sentence-map", action="store_true")
    analyze.add_argument("--motif-analysis", action="store_true")
    analyze.add_argument("--output", type=Path)

    evaluate = subparsers.add_parser(
        "evaluate", help="run the explicit train/test secondary benchmark"
    )
    evaluate.add_argument("dataset", type=Path)
    evaluate.add_argument("--encoder", choices=("tfidf", "e5"), default="tfidf")
    evaluate.add_argument("--seed", type=int, default=42)
    return parser


def _analyze(text: str, encoder_name: str, args):
    return analyze_text(
        text,
        encoder=_encoder(encoder_name),
        seed=args.seed,
        shuffle_count=args.shuffles,
        controls=args.controls,
        parse_markdown=not args.plain_text,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "evaluate":
        rows = load_dataset(args.dataset)
        payload = evaluate_rows(
            rows,
            encoder=build_encoder(args.encoder),
            seed=args.seed,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    text = args.path.read_text(encoding="utf-8")
    if args.encoder == "both":
        lexical = _analyze(text, "tfidf", args)
        semantic = _analyze(text, "e5", args)
        payload = {
            "version": "0.7.0",
            "channels": {
                "lexical": lexical.to_dict(),
                "semantic": semantic.to_dict(),
            },
            "primary_channel": "semantic",
        }
        report_result = semantic
    else:
        report_result = _analyze(text, args.encoder, args)
        payload = report_result.to_dict()
        payload["analysis_channel"] = (
            "semantic" if args.encoder == "e5" else "lexical"
        )

    if args.output:
        payload["report_files"] = write_report_bundle(
            report_result,
            args.output,
            fingerprint_payload=payload,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
