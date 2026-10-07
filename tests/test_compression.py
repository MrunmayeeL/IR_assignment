"""Unit tests for postings compression algorithms."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexis.index.compression import (
    GapEncoder,
    VariableByteEncoder,
    GammaEncoder,
    UncompressedEncoder,
    DictionaryCompressor,
)


def test_gap_encoding():
    numbers = [5, 12, 19, 45, 100]
    gaps = GapEncoder.encode(numbers)
    assert gaps == [5, 7, 7, 26, 55]

    decoded = GapEncoder.decode(gaps)
    assert decoded == numbers


def test_variable_byte_encoding():
    numbers = [1, 5, 128, 500, 16384]
    encoded = VariableByteEncoder.encode(numbers)
    decoded = VariableByteEncoder.decode(encoded)
    assert decoded == numbers


def test_gamma_encoding():
    numbers = [1, 2, 9, 13, 1024]
    encoded = GammaEncoder.encode(numbers)
    decoded = GammaEncoder.decode(encoded, count=len(numbers))
    assert decoded == numbers


def test_uncompressed_encoding():
    numbers = [10, 20, 30, 40]
    encoded = UncompressedEncoder.encode(numbers)
    assert len(encoded) == 16  # 4 ints * 4 bytes
    decoded = UncompressedEncoder.decode(encoded)
    assert decoded == numbers


def test_dictionary_compression():
    terms = ["oil", "prices", "surge", "yesterday"]
    buffer, offsets = DictionaryCompressor.compress(terms)
    lengths = [len(t.encode("utf-8")) for t in terms]
    decompressed = DictionaryCompressor.decompress(buffer, offsets, lengths)
    assert decompressed == terms
