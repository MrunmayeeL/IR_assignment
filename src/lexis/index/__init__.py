"""Index data structures and compression algorithms."""

from lexis.index.compression import (
    GapEncoder,
    VariableByteEncoder,
    GammaEncoder,
    UncompressedEncoder,
    DictionaryCompressor,
)
from lexis.index.skip_pointers import (
    SkipList,
    SkipStats,
    intersect_with_skips,
    intersect_without_skips,
)
from lexis.index.postings import Posting, PostingsList
from lexis.index.dictionary import Dictionary, DocumentRegistry, TermInfo
from lexis.index.positional_index import PositionalIndex

__all__ = [
    "GapEncoder",
    "VariableByteEncoder",
    "GammaEncoder",
    "UncompressedEncoder",
    "DictionaryCompressor",
    "SkipList",
    "SkipStats",
    "intersect_with_skips",
    "intersect_without_skips",
    "Posting",
    "PostingsList",
    "Dictionary",
    "DocumentRegistry",
    "TermInfo",
    "PositionalIndex",
]
