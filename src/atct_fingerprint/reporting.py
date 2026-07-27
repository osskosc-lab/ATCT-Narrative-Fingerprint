"""Portable JSON, CSV, Markdown, and summary-PDF report output."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Mapping

from .features import FingerprintResult


def _pdf_escape(value: str) -> str:
    return (
        value.encode("ascii", "replace")
        .decode("ascii")
        .replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def _write_summary_pdf(path: Path, result: FingerprintResult) -> None:
    """Write a dependency-free, ASCII summary PDF."""

    lines = [
        "ATCT Narrative Fingerprint v0.3",
        f"Primary metric: Z_order = {result.order_z:.4f}",
        f"Gate: {result.gate}",
        f"Sentences: {result.sentence_count}",
        f"Primary history window: {result.primary_window}",
        f"Original consistency: {result.original_consistency:.4f}",
        f"Directional asymmetry: {result.reverse_directionality:.4f}",
        f"Direction method: {result.directionality.method}",
        f"Rolling long-history gain: {result.long_history_gain:.4f}",
        f"Rolling test samples: {result.long_history_prediction.test_samples}",
        f"Turning-point Z: {result.turning_point_z:.4f}",
        f"Theme cohesion: {result.theme_cohesion:.4f}",
        f"Segment diversity: {result.segment_diversity:.4f}",
        f"Structure type: {result.structure_type}",
        "",
        "Controls:",
    ]
    for name, summary in result.controls.items():
        z_text = "n/a" if summary.z is None else f"{summary.z:.4f}"
        lines.append(
            f"  {name}: mean={summary.null_mean:.4f}, "
            f"effect={summary.effect:.4f}, z={z_text}"
        )
    lines.extend(
        [
            "",
            "Interpretation boundary:",
            "Structural evidence only; not authorship probability or quality score.",
            "See report.md and UTF-8 CSV files for sentence-level Japanese text.",
        ]
    )
    commands = ["BT", "/F1 10 Tf", "50 790 Td"]
    for line in lines[:48]:
        commands.append(f"({_pdf_escape(line)}) Tj")
        commands.append("0 -14 Td")
    commands.append("ET")
    stream = "\n".join(commands).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    ]
    document = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(document))
        document.extend(f"{number} 0 obj\n".encode("ascii"))
        document.extend(obj)
        document.extend(b"\nendobj\n")
    xref = len(document)
    document.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    document.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        document.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    document.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(document)


def write_report_bundle(
    result: FingerprintResult,
    output_dir: str | Path,
    *,
    fingerprint_payload: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Write the complete v0.3 report bundle and return generated paths."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = dict(fingerprint_payload or result.to_dict())

    fingerprint_path = destination / "fingerprint.json"
    fingerprint_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sentence_path = destination / "sentence_map.csv"
    sentence_fields = [
        "index",
        "sentence_number",
        "text",
        "section",
        "history_consistency",
        "history_z",
        "history_change",
        "curvature",
        "turn_z",
        "long_history_gain",
        "direction_delta",
        "licensed_jump",
        "role",
    ]
    with sentence_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sentence_fields)
        writer.writeheader()
        writer.writerows(result.sentence_map)

    turning_path = destination / "turning_points.csv"
    turning_rows = sorted(
        result.sentence_map,
        key=lambda row: (
            float("-inf") if row["turn_z"] is None else float(row["turn_z"])
        ),
        reverse=True,
    )
    with turning_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sentence_fields)
        writer.writeheader()
        writer.writerows(turning_rows)

    motif_path = destination / "motif_pairs.csv"
    motif_fields = [
        "first_index",
        "second_index",
        "first_sentence",
        "second_sentence",
        "similarity",
        "relative_distance",
        "kind",
        "contribution",
    ]
    motif_rows = [
        pair.to_dict()
        for pair in (
            *result.motif_analysis.motif_pairs,
            *result.motif_analysis.duplicate_pairs,
        )
    ]
    with motif_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=motif_fields)
        writer.writeheader()
        writer.writerows(motif_rows)

    section_path = destination / "section_graph.csv"
    section_fields = [
        "source_id",
        "target_id",
        "source_heading",
        "target_heading",
        "transition_distance",
        "target_history_support",
        "relation",
        "licensed_jump",
    ]
    with section_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=section_fields)
        writer.writeheader()
        writer.writerows(asdict(edge) for edge in result.section_graph.edges)

    block_path = destination / "document_blocks.csv"
    block_fields = [
        "kind",
        "text",
        "start_line",
        "end_line",
        "section",
        "level",
        "role",
    ]
    block_rows = result.document_structure.get("blocks", [])
    with block_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=block_fields)
        writer.writeheader()
        writer.writerows(block_rows)

    report_path = destination / "report.md"
    controls = "\n".join(
        f"- {name}: effect={summary.effect:.4f}, "
        f"Z={'n/a' if summary.z is None else f'{summary.z:.4f}'}"
        for name, summary in result.controls.items()
    )
    report_path.write_text(
        "\n".join(
            [
                "# ATCT Narrative Fingerprint v0.3",
                "",
                f"- **Z_order:** {result.order_z:.4f}",
                f"- **判定:** {result.gate}",
                f"- **構造型:** {result.structure_type}",
                (
                    f"- **方向性（{result.directionality.method}）:** "
                    f"{result.reverse_directionality:.4f}"
                ),
                f"- **外部予測による長距離履歴利得:** {result.long_history_gain:.4f}",
                (
                    f"- **短期／長期テスト誤差:** "
                    f"{result.long_history_prediction.short_test_error:.4f} / "
                    f"{result.long_history_prediction.long_test_error:.4f}"
                ),
                f"- **転換点Z:** {result.turning_point_z:.4f}",
                f"- **主題凝集度:** {result.theme_cohesion:.4f}",
                f"- **構造的多様性:** {result.segment_diversity:.4f}",
                "",
                "## 順序対照",
                "",
                controls,
                "",
                "## 文書階層",
                "",
                (
                    f"- 本文ブロック: "
                    f"{result.document_structure.get('layer_counts', {}).get('prose', 0)}"
                ),
                (
                    f"- 数式ブロック: "
                    f"{result.document_structure.get('layer_counts', {}).get('equation', 0)}"
                ),
                f"- 節ノード: {len(result.section_graph.nodes)}",
                f"- 節遷移: {len(result.section_graph.edges)}",
                f"- Licensed Jump: {len(result.licensed_jumps)}",
                "",
                "## 解釈境界",
                "",
                result.disclaimer,
            ]
        ),
        encoding="utf-8",
    )

    pdf_path = destination / "report.pdf"
    _write_summary_pdf(pdf_path, result)
    return {
        "fingerprint": str(fingerprint_path),
        "sentence_map": str(sentence_path),
        "turning_points": str(turning_path),
        "motif_pairs": str(motif_path),
        "section_graph": str(section_path),
        "document_blocks": str(block_path),
        "markdown": str(report_path),
        "pdf": str(pdf_path),
    }
