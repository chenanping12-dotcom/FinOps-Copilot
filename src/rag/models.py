"""Stable data types shared by the retrieval stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    version: str
    effective_from: str
    page: int
    line_start: int
    line_end: int
    policy_key: str
    policy_family: str
    scope: str
    rule_type: str
    text: str
    parent_id: str
    parent_text: str

    @property
    def citation(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "page": self.page,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "chunk_id": self.chunk_id,
        }

    @property
    def citation_label(self) -> str:
        line = str(self.line_start) if self.line_start == self.line_end else f"{self.line_start}-{self.line_end}"
        return f"[{self.doc_id}, Page {self.page}, Line {line}]"

    @property
    def retrieval_text(self) -> str:
        """Text visible to retrieval, including non-gold metadata but no labels."""
        return " ".join(
            (
                self.text,
                self.version,
                self.effective_from,
                self.policy_family,
                self.scope,
                self.rule_type,
            )
        )


@dataclass
class RetrievalHit:
    chunk: ChunkRecord
    score: float
    rank: int
    method: str
    components: dict[str, float] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def citation(self) -> str:
        return self.chunk.citation_label
