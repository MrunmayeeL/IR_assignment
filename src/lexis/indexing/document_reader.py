"""Document streaming reader for Reuters corpus."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple, List


@dataclass
class Document:
    doc_name: str
    title: str
    body: str
    filepath: Optional[str] = None


class DocumentReader:
    """Streams documents from corpus directory with deterministic ordering."""

    @staticmethod
    def parse_file(filepath: Path) -> Document:
        doc_name = filepath.stem  # e.g., 'doc_00001'
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        if not lines:
            return Document(doc_name=doc_name, title="", body="", filepath=str(filepath))

        title = lines[0].strip()
        body = "".join(lines[1:]).strip()
        return Document(doc_name=doc_name, title=title, body=body, filepath=str(filepath))

    @classmethod
    def stream_corpus(cls, corpus_dir: str) -> Iterator[Document]:
        """Stream documents sorted deterministically by document name."""
        corpus_path = Path(corpus_dir)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

        files = sorted(
            [p for p in corpus_path.iterdir() if p.is_file() and p.suffix == ".txt"],
            key=lambda p: p.stem,
        )

        for file_path in files:
            yield cls.parse_file(file_path)

    @classmethod
    def get_document_files(cls, corpus_dir: str) -> List[Path]:
        corpus_path = Path(corpus_dir)
        return sorted(
            [p for p in corpus_path.iterdir() if p.is_file() and p.suffix == ".txt"],
            key=lambda p: p.stem,
        )
