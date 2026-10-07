"""Postings and PostingsList representations for Lexis inverted index."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from lexis.index.compression import GapEncoder, VariableByteEncoder, GammaEncoder, UncompressedEncoder


@dataclass
class Posting:
    doc_id: int
    tf: int
    positions: List[int] = field(default_factory=list)
    title_tf: int = 0
    body_tf: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "tf": self.tf,
            "positions": self.positions,
            "title_tf": self.title_tf,
            "body_tf": self.body_tf,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Posting":
        return cls(
            doc_id=data["doc_id"],
            tf=data["tf"],
            positions=data.get("positions", []),
            title_tf=data.get("title_tf", 0),
            body_tf=data.get("body_tf", 0),
        )


@dataclass
class PostingsList:
    term: str
    doc_frequency: int = 0
    postings: List[Posting] = field(default_factory=list)

    def add_occurrence(self, doc_id: int, position: int, zone: str = "body"):
        """Add an occurrence of the term in doc_id at position with given zone."""
        if not self.postings or self.postings[-1].doc_id != doc_id:
            posting = Posting(doc_id=doc_id, tf=1, positions=[position])
            if zone == "title":
                posting.title_tf = 1
            else:
                posting.body_tf = 1
            self.postings.append(posting)
            self.doc_frequency += 1
        else:
            curr = self.postings[-1]
            curr.tf += 1
            curr.positions.append(position)
            if zone == "title":
                curr.title_tf += 1
            else:
                curr.body_tf += 1

    def doc_ids(self) -> List[int]:
        return [p.doc_id for p in self.postings]

    def compress_doc_ids(self, scheme: str = "variable_byte") -> bytes:
        """Compress doc_ids using the specified compression scheme."""
        ids = self.doc_ids()
        if not ids:
            return b""
        gaps = GapEncoder.encode(ids)
        if scheme == "uncompressed":
            return UncompressedEncoder.encode(ids)
        elif scheme == "gap":
            return UncompressedEncoder.encode(gaps)
        elif scheme == "variable_byte":
            return VariableByteEncoder.encode(gaps)
        elif scheme == "gamma":
            return GammaEncoder.encode(gaps)
        else:
            raise ValueError(f"Unknown compression scheme: {scheme}")

    def decompress_doc_ids(self, data: bytes, scheme: str = "variable_byte", count: Optional[int] = None) -> List[int]:
        """Decompress doc_ids from compressed bytes."""
        if not data:
            return []
        if scheme == "uncompressed":
            return UncompressedEncoder.decode(data)
        elif scheme == "gap":
            gaps = UncompressedEncoder.decode(data)
            return GapEncoder.decode(gaps)
        elif scheme == "variable_byte":
            gaps = VariableByteEncoder.decode(data)
            return GapEncoder.decode(gaps)
        elif scheme == "gamma":
            c = count if count is not None else self.doc_frequency
            gaps = GammaEncoder.decode(data, c)
            return GapEncoder.decode(gaps)
        else:
            raise ValueError(f"Unknown compression scheme: {scheme}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "term": self.term,
            "df": self.doc_frequency,
            "postings": [p.to_dict() for p in self.postings],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PostingsList":
        plist = cls(term=data["term"], doc_frequency=data.get("df", 0))
        plist.postings = [Posting.from_dict(p) for p in data.get("postings", [])]
        if not plist.doc_frequency:
            plist.doc_frequency = len(plist.postings)
        return plist
