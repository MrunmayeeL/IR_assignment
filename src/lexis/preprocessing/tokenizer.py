"""Tokenizer implementation for Lexis.

Produces structured Token objects with positional and zone information.
Ensures identical tokenization and normalization across documents and queries.
"""

import re
from dataclasses import dataclass
from typing import List, Optional
from lexis.preprocessing.normalizer import Normalizer
from lexis.preprocessing.stemmer import PorterStemmer
from lexis.preprocessing.stopwords import StopwordFilter


@dataclass
class Token:
    term: str
    raw_term: str
    position: int
    zone: str  # 'title' or 'body'


class Tokenizer:
    # Match alphanumeric sequences
    TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")

    def __init__(
        self,
        normalizer: Optional[Normalizer] = None,
        stemmer: Optional[PorterStemmer] = None,
        stopword_filter: Optional[StopwordFilter] = None,
        min_len: int = 1,
    ):
        self.normalizer = normalizer if normalizer is not None else Normalizer()
        self.stemmer = stemmer if stemmer is not None else PorterStemmer(enabled=False)
        self.stopword_filter = stopword_filter if stopword_filter is not None else StopwordFilter(enabled=False)
        self.min_len = min_len

    def tokenize_document(
        self,
        title: Optional[str] = "",
        body: Optional[str] = "",
    ) -> List[Token]:
        """Tokenize a full document preserving positional stream across title and body.
        
        Tokens in title are assigned zone='title'.
        Tokens in body are assigned zone='body'.
        Positions are continuous 0, 1, 2, ... across the unified document token stream
        so phrase and proximity queries match exact text order.
        """
        tokens: List[Token] = []
        pos = 0

        title_text = self.normalizer.normalize(title)
        for match in self.TOKEN_PATTERN.finditer(title_text):
            raw = match.group()
            term = self._process_term(raw)
            if term is not None:
                tokens.append(Token(term=term, raw_term=raw, position=pos, zone="title"))
            pos += 1

        body_text = self.normalizer.normalize(body)
        for match in self.TOKEN_PATTERN.finditer(body_text):
            raw = match.group()
            term = self._process_term(raw)
            if term is not None:
                tokens.append(Token(term=term, raw_term=raw, position=pos, zone="body"))
            pos += 1

        return tokens

    def tokenize_text(self, text: str, zone: str = "body") -> List[Token]:
        """Tokenize raw text (e.g. queries) into a list of Tokens."""
        tokens: List[Token] = []
        norm_text = self.normalizer.normalize(text)
        pos = 0
        for match in self.TOKEN_PATTERN.finditer(norm_text):
            raw = match.group()
            term = self._process_term(raw)
            if term is not None:
                tokens.append(Token(term=term, raw_term=raw, position=pos, zone=zone))
            pos += 1
        return tokens

    def tokenize_terms(self, text: str) -> List[str]:
        """Convenience method returning just a list of processed term strings."""
        tokens = self.tokenize_text(text)
        return [t.term for t in tokens]

    def _process_term(self, raw_term: str) -> Optional[str]:
        """Apply case folding, stopword removal, and stemming."""
        if len(raw_term) < self.min_len:
            return None
        term = raw_term.lower()
        if self.stopword_filter.is_stopword(term):
            return None
        if self.stemmer.enabled:
            term = self.stemmer.stem(term)
        return term
