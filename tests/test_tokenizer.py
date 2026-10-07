"""Unit tests for tokenization, normalization, stemming, and stopwords."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexis.preprocessing.normalizer import Normalizer
from lexis.preprocessing.stemmer import PorterStemmer
from lexis.preprocessing.stopwords import StopwordFilter
from lexis.preprocessing.tokenizer import Tokenizer


def test_normalizer():
    norm = Normalizer(case_folding=True)
    assert norm.normalize("FEDERAL RESERVE") == "federal reserve"
    assert norm.normalize("  Spaces  ") == "spaces"


def test_stemmer():
    stemmer = PorterStemmer(enabled=True)
    assert stemmer.stem("economies") == "economi"
    assert stemmer.stem("agreements") == "agreement"
    assert stemmer.stem("meeting") == "meet"

    stemmer_disabled = PorterStemmer(enabled=False)
    assert stemmer_disabled.stem("economies") == "economies"


def test_stopwords():
    sw = StopwordFilter(enabled=True)
    assert sw.is_stopword("the")
    assert sw.is_stopword("and")
    assert not sw.is_stopword("inflation")

    sw_disabled = StopwordFilter(enabled=False)
    assert not sw_disabled.is_stopword("the")


def test_tokenizer_document():
    tok = Tokenizer(
        normalizer=Normalizer(case_folding=True),
        stemmer=PorterStemmer(enabled=False),
        stopword_filter=StopwordFilter(enabled=False),
    )
    title = "Oil Prices Surge"
    body = "Crude oil jumped yesterday."
    tokens = tok.tokenize_document(title=title, body=body)

    # Check terms
    terms = [t.term for t in tokens]
    assert "oil" in terms
    assert "prices" in terms
    assert "crude" in terms

    # Check zones
    assert tokens[0].zone == "title"
    assert tokens[1].zone == "title"
    assert tokens[2].zone == "title"
    assert tokens[3].zone == "body"

    # Check continuous positions
    assert tokens[0].position == 0
    assert tokens[1].position == 1
    assert tokens[2].position == 2
    assert tokens[3].position == 3
