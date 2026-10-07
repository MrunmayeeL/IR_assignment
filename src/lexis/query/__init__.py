"""Query execution, parsing, wildcard, phrase, proximity, and spelling modules."""

from lexis.query.lexer import QueryLexer, QueryToken, TokenType
from lexis.query.parser import (
    QueryParser,
    Node,
    TermNode,
    PhraseNode,
    ProximityNode,
    WildcardNode,
    NotNode,
    AndNode,
    OrNode,
)
from lexis.query.boolean import BooleanEvaluator
from lexis.query.phrase import PhraseEvaluator
from lexis.query.proximity import ProximityEvaluator
from lexis.query.wildcard import WildcardEvaluator, PermutermIndex
from lexis.query.spelling import SpellingCorrector, levenshtein_distance
from lexis.query.query_engine import QueryEngine

__all__ = [
    "QueryLexer",
    "QueryToken",
    "TokenType",
    "QueryParser",
    "Node",
    "TermNode",
    "PhraseNode",
    "ProximityNode",
    "WildcardNode",
    "NotNode",
    "AndNode",
    "OrNode",
    "BooleanEvaluator",
    "PhraseEvaluator",
    "ProximityEvaluator",
    "WildcardEvaluator",
    "PermutermIndex",
    "SpellingCorrector",
    "levenshtein_distance",
    "QueryEngine",
]
