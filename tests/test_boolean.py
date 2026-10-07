"""Unit tests for Boolean operators, parentheses, and precedence."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexis.index.positional_index import PositionalIndex
from lexis.preprocessing.tokenizer import Tokenizer
from lexis.query.query_engine import QueryEngine


def create_synthetic_index():
    index = PositionalIndex()
    reg = index.document_registry
    d1 = reg.register("doc_01")
    d2 = reg.register("doc_02")
    d3 = reg.register("doc_03")
    d4 = reg.register("doc_04")

    # doc_01: oil, opec
    index.add_term_occurrence("oil", d1, 0)
    index.add_term_occurrence("opec", d1, 1)

    # doc_02: wheat, corn
    index.add_term_occurrence("wheat", d2, 0)
    index.add_term_occurrence("corn", d2, 1)

    # doc_03: oil, wheat, corn, soviet
    index.add_term_occurrence("oil", d3, 0)
    index.add_term_occurrence("wheat", d3, 1)
    index.add_term_occurrence("corn", d3, 2)
    index.add_term_occurrence("soviet", d3, 3)

    # doc_04: gold, silver, price
    index.add_term_occurrence("gold", d4, 0)
    index.add_term_occurrence("silver", d4, 1)
    index.add_term_occurrence("price", d4, 2)

    return index


def test_boolean_and():
    index = create_synthetic_index()
    engine = QueryEngine(index, Tokenizer())
    res = engine.search_exact("oil AND opec")
    assert res == ["doc_01"]


def test_boolean_or():
    index = create_synthetic_index()
    engine = QueryEngine(index, Tokenizer())
    res = engine.search_exact("opec OR gold")
    assert sorted(res) == ["doc_01", "doc_04"]


def test_boolean_not():
    index = create_synthetic_index()
    engine = QueryEngine(index, Tokenizer())
    res = engine.search_exact("wheat AND corn AND NOT soviet")
    assert res == ["doc_02"]


def test_boolean_parentheses():
    index = create_synthetic_index()
    engine = QueryEngine(index, Tokenizer())
    res = engine.search_exact("(gold OR silver) AND price")
    assert res == ["doc_04"]

    res2 = engine.search_exact("(oil OR gold) AND (opec OR price)")
    assert sorted(res2) == ["doc_01", "doc_04"]
