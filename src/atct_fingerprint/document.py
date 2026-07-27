"""Markdown-aware separation of prose, equations, and document structure."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import re


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")
_THEMATIC_BREAK = re.compile(r"^\s*(?:(?:-\s*){3,}|(?:\*\s*){3,}|(?:_\s*){3,})$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_RULE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$")


@dataclass(frozen=True)
class DocumentBlock:
    kind: str
    text: str
    start_line: int
    end_line: int
    section: str
    level: int | None = None
    role: str | None = None


@dataclass(frozen=True)
class SectionSpan:
    section_id: int
    heading: str
    level: int
    parent: str | None
    sentence_indices: tuple[int, ...]
    start_sentence: int | None
    end_sentence: int | None


@dataclass(frozen=True)
class ParsedDocument:
    prose_sentences: tuple[str, ...]
    paragraph_indices: tuple[tuple[int, ...], ...]
    sections: tuple[SectionSpan, ...]
    blocks: tuple[DocumentBlock, ...]
    layer_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _equation_role(context: str, text: str) -> str:
    joined = f"{context} {text}".lower()
    role_patterns = (
        ("falsification", r"反証|停止条件|帰無|null|falsif"),
        ("hypothesis", r"仮説|命題|hypothesis|proposition"),
        ("definition", r"定義|definition|define"),
        ("conclusion", r"結論|最終|conclusion|result"),
        ("theorem", r"定理|証明|theorem|proof"),
    )
    for role, pattern in role_patterns:
        if re.search(pattern, joined):
            return role
    if r"\boxed" in text or "boxed" in joined:
        return "symbolic"
    return "equation"


def parse_markdown_document(
    text: str,
    splitter: Callable[[str], list[str]],
) -> ParsedDocument:
    """Extract semantic prose while preserving Markdown structure as metadata."""

    lines = text.splitlines()
    sentences: list[str] = []
    paragraphs: list[tuple[int, ...]] = []
    blocks: list[DocumentBlock] = []
    section_records: list[dict[str, object]] = [
        {
            "heading": "Document",
            "level": 0,
            "parent": None,
            "sentence_indices": [],
        }
    ]
    current_section = 0
    section_stack: list[tuple[int, int]] = []
    paragraph_lines: list[str] = []
    paragraph_start = 0
    fenced_kind: str | None = None
    fenced_close: str | None = None
    fenced_lines: list[str] = []
    fenced_start = 0

    def section_name() -> str:
        return str(section_records[current_section]["heading"])

    def flush_paragraph(end_line: int) -> None:
        nonlocal paragraph_lines, paragraph_start
        if not paragraph_lines:
            return
        paragraph_text = " ".join(part.strip() for part in paragraph_lines).strip()
        paragraph_sentences = splitter(paragraph_text)
        start = len(sentences)
        sentences.extend(paragraph_sentences)
        indices = tuple(range(start, len(sentences)))
        if indices:
            paragraphs.append(indices)
            section_records[current_section]["sentence_indices"].extend(indices)
        blocks.append(
            DocumentBlock(
                kind="prose",
                text=paragraph_text,
                start_line=paragraph_start,
                end_line=end_line,
                section=section_name(),
            )
        )
        paragraph_lines = []

    def add_structural_block(
        kind: str,
        value: str,
        line_number: int,
        *,
        end_line: int | None = None,
        level: int | None = None,
        role: str | None = None,
    ) -> None:
        blocks.append(
            DocumentBlock(
                kind=kind,
                text=value.strip(),
                start_line=line_number,
                end_line=end_line or line_number,
                section=section_name(),
                level=level,
                role=role,
            )
        )

    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if fenced_kind is not None:
            if stripped == fenced_close:
                fenced_lines.append(raw_line)
                content = "\n".join(fenced_lines)
                role = (
                    _equation_role(section_name(), content)
                    if fenced_kind == "equation"
                    else None
                )
                add_structural_block(
                    fenced_kind,
                    content,
                    fenced_start,
                    end_line=line_number,
                    role=role,
                )
                fenced_kind = None
                fenced_close = None
                fenced_lines = []
            else:
                fenced_lines.append(raw_line)
            continue

        if not stripped:
            flush_paragraph(line_number - 1)
            continue

        heading_match = _HEADING.match(stripped)
        if heading_match:
            flush_paragraph(line_number - 1)
            level = len(heading_match.group(1))
            heading = heading_match.group(2).strip()
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            parent = (
                str(section_records[section_stack[-1][1]]["heading"])
                if section_stack
                else None
            )
            section_records.append(
                {
                    "heading": heading,
                    "level": level,
                    "parent": parent,
                    "sentence_indices": [],
                }
            )
            current_section = len(section_records) - 1
            section_stack.append((level, current_section))
            add_structural_block(
                "heading", heading, line_number, level=level
            )
            continue

        fence = None
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence = ("code", stripped[:3])
        elif stripped in {"$$", r"\[", "["}:
            fence = (
                "equation",
                {"$$": "$$", r"\[": r"\]", "[": "]"}[stripped],
            )
        if fence:
            flush_paragraph(line_number - 1)
            fenced_kind, fenced_close = fence
            fenced_start = line_number
            fenced_lines = [raw_line]
            continue

        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            flush_paragraph(line_number - 1)
            add_structural_block(
                "equation",
                raw_line,
                line_number,
                role=_equation_role(section_name(), raw_line),
            )
            continue
        if _THEMATIC_BREAK.match(stripped):
            flush_paragraph(line_number - 1)
            add_structural_block("thematic_break", raw_line, line_number)
            continue
        if stripped.startswith(">"):
            flush_paragraph(line_number - 1)
            add_structural_block("quote", re.sub(r"^>\s?", "", stripped), line_number)
            continue
        if _LIST_ITEM.match(raw_line):
            flush_paragraph(line_number - 1)
            add_structural_block(
                "list_item", _LIST_ITEM.sub("", raw_line, count=1), line_number
            )
            continue
        if _TABLE_ROW.match(stripped) or _TABLE_RULE.match(stripped):
            flush_paragraph(line_number - 1)
            add_structural_block("table", raw_line, line_number)
            continue

        if not paragraph_lines:
            paragraph_start = line_number
        paragraph_lines.append(raw_line)

    if fenced_kind is not None:
        content = "\n".join(fenced_lines)
        role = (
            _equation_role(section_name(), content)
            if fenced_kind == "equation"
            else None
        )
        add_structural_block(
            fenced_kind,
            content,
            fenced_start,
            end_line=len(lines),
            role=role,
        )
    flush_paragraph(len(lines))

    sections: list[SectionSpan] = []
    for section_id, record in enumerate(section_records):
        indices = tuple(int(value) for value in record["sentence_indices"])
        if section_id == 0 and not indices and len(section_records) > 1:
            continue
        sections.append(
            SectionSpan(
                section_id=section_id,
                heading=str(record["heading"]),
                level=int(record["level"]),
                parent=(
                    None if record["parent"] is None else str(record["parent"])
                ),
                sentence_indices=indices,
                start_sentence=indices[0] if indices else None,
                end_sentence=indices[-1] if indices else None,
            )
        )

    layer_counts = {
        "prose": sum(block.kind == "prose" for block in blocks),
        "equation": sum(block.kind == "equation" for block in blocks),
        "structure": sum(
            block.kind not in {"prose", "equation"} for block in blocks
        ),
        "heading": sum(block.kind == "heading" for block in blocks),
        "quote": sum(block.kind == "quote" for block in blocks),
        "list_item": sum(block.kind == "list_item" for block in blocks),
    }
    return ParsedDocument(
        prose_sentences=tuple(sentences),
        paragraph_indices=tuple(paragraphs),
        sections=tuple(sections),
        blocks=tuple(blocks),
        layer_counts=layer_counts,
    )
