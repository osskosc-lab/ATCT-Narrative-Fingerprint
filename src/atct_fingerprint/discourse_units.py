"""Paragraph-aware discourse units for structural evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class DiscourseUnit:
    unit_index: int
    kind: str
    role: str
    sentence_indices: tuple[int, ...]
    text: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_discourse_units(
    sentences: Sequence[str],
    paragraphs: Sequence[Sequence[int]] | None,
    document_structure: Mapping[str, object] | None = None,
) -> tuple[DiscourseUnit, ...]:
    units: list[DiscourseUnit] = []
    for paragraph in paragraphs or ():
        indices = tuple(int(value) for value in paragraph)
        if not indices:
            continue
        role = "emphasis_or_pause" if len(indices) == 1 else "paragraph_sequence"
        units.append(
            DiscourseUnit(
                unit_index=len(units),
                kind="prose",
                role=role,
                sentence_indices=indices,
                text=" ".join(sentences[index] for index in indices),
            )
        )
    for block in (document_structure or {}).get("blocks", []):
        kind = str(block.get("kind", ""))
        if kind not in {"quote", "numbered_heading"}:
            continue
        units.append(
            DiscourseUnit(
                unit_index=len(units),
                kind=kind,
                role=(
                    "quoted_evidence"
                    if kind == "quote"
                    else "numbered_section_anchor"
                ),
                sentence_indices=(),
                text=str(block.get("text", "")),
            )
        )
    return tuple(units)

