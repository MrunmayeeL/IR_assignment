"""Unit tests for Levenshtein distance and spelling correction."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexis.index.positional_index import PositionalIndex
from lexis.query.spelling import levenshtein_distance, SpellingCorrector


def test_levenshtein_distance():
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("singaproe", "singapore") == 2
    assert levenshtein_distance("belisved", "believed") == 1
    assert levenshtein_distance("same", "same") == 0


def test_spelling_corrections():
    index = PositionalIndex()
    reg = index.document_registry
    d1 = reg.register("doc_01")
    d2 = reg.register("doc_02")

    # Add occurrences of vocabulary words
    for _ in range(10):
        index.add_term_occurrence("believed", d1, 0)
    for _ in range(5):
        index.add_term_occurrence("believes", d2, 0)

    # Re-save dictionary total_tf
    index.dictionary.add_or_update("believed", 1, total_tf=10)
    index.dictionary.add_or_update("believes", 1, total_tf=5)

    corrector = SpellingCorrector(index)
    assert not corrector.is_in_vocabulary("belisved")

    sug = corrector.suggest_correction("belisved", max_distance=2)
    assert sug == "believed"
