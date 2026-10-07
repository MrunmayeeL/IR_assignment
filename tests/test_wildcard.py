"""Unit tests for wildcard queries and permuterm index."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexis.index.positional_index import PositionalIndex
from lexis.query.wildcard import PermutermIndex
from lexis.preprocessing.tokenizer import Tokenizer
from lexis.query.query_engine import QueryEngine


def test_permuterm_expansion():
    vocab = ["economy", "economic", "economist", "agriculture", "agricultural", "export", "import", "petrochemical", "petrochemicals"]
    p_idx = PermutermIndex(vocab)

    # Prefix
    assert sorted(p_idx.expand_pattern("econom*")) == ["economic", "economist", "economy"]

    # Suffix
    assert sorted(p_idx.expand_pattern("*port")) == ["export", "import"]

    # Infix / multi-wildcard
    assert sorted(p_idx.expand_pattern("pet*chem*")) == ["petrochemical", "petrochemicals"]


def test_wildcard_evaluation():
    index = PositionalIndex()
    reg = index.document_registry
    d1 = reg.register("doc_01")
    d2 = reg.register("doc_02")
    d3 = reg.register("doc_03")

    index.add_term_occurrence("economic", d1, 0)
    index.add_term_occurrence("economy", d2, 0)
    index.add_term_occurrence("agriculture", d3, 0)

    engine = QueryEngine(index, Tokenizer())
    res = engine.search_exact("econom*")
    assert sorted(res) == ["doc_01", "doc_02"]
