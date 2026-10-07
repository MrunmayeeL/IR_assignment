"""Unit tests for proximity queries."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexis.index.positional_index import PositionalIndex
from lexis.preprocessing.tokenizer import Tokenizer
from lexis.query.query_engine import QueryEngine


def test_proximity_within_window():
    index = PositionalIndex()
    reg = index.document_registry
    d1 = reg.register("doc_01")
    d2 = reg.register("doc_02")
    d3 = reg.register("doc_03")

    # doc_01: economy at pos 2, growth at pos 5 -> diff = 3 <= 5
    index.add_term_occurrence("economy", d1, 2)
    index.add_term_occurrence("growth", d1, 5)

    # doc_02: growth at pos 10, economy at pos 12 -> diff = 2 <= 5 (reverse order)
    index.add_term_occurrence("growth", d2, 10)
    index.add_term_occurrence("economy", d2, 12)

    # doc_03: economy at pos 0, growth at pos 8 -> diff = 8 > 5
    index.add_term_occurrence("economy", d3, 0)
    index.add_term_occurrence("growth", d3, 8)

    engine = QueryEngine(index, Tokenizer())
    res = engine.search_exact("economy /5 growth")
    assert sorted(res) == ["doc_01", "doc_02"]
