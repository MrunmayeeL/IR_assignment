"""Unified query engine for Lexis."""

from typing import List, Optional, Tuple, Set
from lexis.index.positional_index import PositionalIndex
from lexis.preprocessing.tokenizer import Tokenizer
from lexis.query.parser import QueryParser
from lexis.query.boolean import BooleanEvaluator
from lexis.query.spelling import SpellingCorrector


class QueryEngine:
    """Unified search engine interface supporting exact Boolean/phrase and spelling correction."""

    def __init__(self, index: PositionalIndex, tokenizer: Tokenizer):
        self.index = index
        self.tokenizer = tokenizer
        self.boolean_evaluator = BooleanEvaluator(index, tokenizer)
        self.spelling_corrector = SpellingCorrector(index)

    def search_exact(self, query_str: str) -> List[str]:
        """Execute an exact retrieval query (Boolean, Phrase, Proximity, Wildcard).
        
        Returns sorted list of document names.
        """
        ast = QueryParser.parse_query(query_str)
        return self.boolean_evaluator.execute(ast)

    def check_query_spelling(self, query_str: str) -> List[Tuple[str, Optional[str]]]:
        """Check query terms for spelling mistakes and return (term, suggestion) pairs."""
        terms = self.tokenizer.tokenize_terms(query_str)
        results = []
        for term in terms:
            if not self.spelling_corrector.is_in_vocabulary(term):
                suggestion = self.spelling_corrector.suggest_correction(term)
                results.append((term, suggestion))
            else:
                results.append((term, term))
        return results
