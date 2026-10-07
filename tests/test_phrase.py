"""Unit tests for positional phrase queries."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexis.index.positional_index import PositionalIndex
from lexis.preprocessing.tokenizer import Tokenizer
from lexis.query.query_engine import QueryEngine


def test_phrase_query():
    index = PositionalIndex()
    reg = index.document_registry
    d1 = reg.register("doc_01")
    d2 = reg.register("doc_02")
    d3 = reg.register("doc_03")

    # doc_01: "the federal reserve announced" -> federal(1), reserve(2) consecutive
    index.add_term_occurrence("federal", d1, 1)
    index.add_term_occurrence("reserve", d1, 2)

    # doc_02: "reserve of federal bank" -> reserve(0), federal(2) not consecutive
    index.add_term_occurrence("reserve", d2, 0)
    index.add_term_occurrence("federal", d2, 2)

    # doc_03: "federal foreign reserve" -> federal(0), reserve(2) separated
    index.add_term_occurrence("federal", d3, 0)
    index.add_term_occurrence("foreign", d3, 1)
    index.add_term_occurrence("reserve", d3, 2)

    engine = QueryEngine(index, Tokenizer())
    res = engine.search_exact('"federal reserve"')
    assert res == ["doc_01"]


def test_three_word_phrase():
    index = PositionalIndex()
    reg = index.document_registry
    d1 = reg.register("doc_01")
    d2 = reg.register("doc_02")

    # doc_01: "bank of japan" -> bank(0), of(1), japan(2)
    index.add_term_occurrence("bank", d1, 0)
    index.add_term_occurrence("of", d1, 1)
    index.add_term_occurrence("japan", d1, 2)

    # doc_02: "japan bank of"
    index.add_term_occurrence("japan", d2, 0)
    index.add_term_occurrence("bank", d2, 1)
    index.add_term_occurrence("of", d2, 2)

    engine = QueryEngine(index, Tokenizer())
    res = engine.search_exact('"bank of japan"')
    assert res == ["doc_01"]
